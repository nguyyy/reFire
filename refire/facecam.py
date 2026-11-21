"""Find the streamer's facecam per source (VOD) -- punch pins to it, thumb cuts it out.

Sample frames across the footage the cut uses, detect faces (YuNet), cluster face centers;
the facecam is the cluster that's there most often with a steady face size (in-game faces
come and go and change size). Its rectangle is the persistent straight-edged box around
that cluster, else the median face box expanded. Results cache in <run_dir>/facecam.json.

    cams = Facecams(manifest_path)          # timeline.tsv beside it if dumped, else clips
    cams.detect(0)                          # -> Facecam
    cams.face_present(0, t0, t1)            # fraction of 3 ROI frames with a face

`source_id` / times follow refire.timeline_map (R3): VOD index, transcript clock.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .timeline_map import VOD_BASE, TimelineMap

MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
MODEL_PATH = Path.home() / ".cache" / "refire" / "models" / "face_detection_yunet_2023mar.onnx"

SAMPLE_EVERY, MIN_SAMPLES, MAX_SAMPLES = 20.0, 30, 120
DECODE_W = 640
MIN_SCORE = 0.8
EPS = 0.05                  # dbscan radius, normalized frame units
MIN_CONFIDENCE = 0.6        # below this punch skips the source, thumb searches the whole frame
EXPAND_W, EXPAND_H = 3.0, 3.2
SNAP = 0.01


@dataclass
class Facecam:
    box: list[float]            # x0, y0, x1, y1 normalized
    face_center: list[float]
    region: str
    confidence: float           # facecam cluster frame coverage (R4)
    samples: int

    @property
    def usable(self) -> bool:
        return self.confidence >= MIN_CONFIDENCE


# --------------------------------------------------------------------------- detector

def _model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "refire"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
            raise RuntimeError(f"yunet checksum mismatch from {MODEL_URL}")
        MODEL_PATH.write_bytes(data)
    return MODEL_PATH


class YuNet:
    """frame (bgr) -> [(x, y, w, h, score)] normalized to that frame."""

    def __init__(self, score: float = MIN_SCORE):
        import cv2
        self._net = cv2.FaceDetectorYN.create(str(_model()), "", (320, 320), score)

    def __call__(self, img) -> list[tuple[float, float, float, float, float]]:
        h, w = img.shape[:2]
        self._net.setInputSize((w, h))
        _, faces = self._net.detect(img)
        if faces is None:
            return []
        return [(f[0] / w, f[1] / h, f[2] / w, f[3] / h, float(f[-1])) for f in faces]


# --------------------------------------------------------------------------- pure core

def dbscan(pts: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    """labels per point, -1 = noise. ponytail: O(n^2) distances, fine for <= a few k points."""
    n = len(pts)
    labels = np.full(n, -1)
    if not n:
        return labels
    near = np.linalg.norm(pts[:, None] - pts[None], axis=2) <= eps
    core = near.sum(1) >= min_pts
    c = 0
    for i in range(n):
        if labels[i] != -1 or not core[i]:
            continue
        labels[i], stack = c, [i]
        while stack:
            j = stack.pop()
            if not core[j]:
                continue
            for k in np.flatnonzero(near[j] & (labels == -1)):
                labels[k] = c
                stack.append(k)
        c += 1
    return labels


def pick_cluster(dets: list[tuple[int, float, float, float, float]], n_frames: int):
    """dets: (frame, x, y, w, h). -> (member dets, coverage) of the facecam cluster, or None.
    ponytail: score = coverage / (1 + 4 * size cv); tune on real VODs if a game face wins."""
    if not dets:
        return None
    a = np.array(dets, float)
    centers = a[:, 1:3] + a[:, 3:5] / 2
    labels = dbscan(centers, EPS, max(3, n_frames // 10))
    best = None
    for c in set(labels.tolist()) - {-1}:
        m = a[labels == c]
        coverage = len(set(m[:, 0].astype(int))) / n_frames
        cv = float(m[:, 3].std() / max(m[:, 3].mean(), 1e-6))
        score = coverage / (1 + 4 * cv)
        if best is None or score > best[0]:
            best = (score, m, coverage)
    return None if best is None else (best[1], best[2])


def _line_rows(strong: np.ndarray, rows, c0: int, c1: int, thr: float):
    """first row (in scan order) whose strong-edge fraction over columns c0:c1 >= thr."""
    for y in rows:
        if strong[y, c0:c1].mean() >= thr:
            return y
    return None


def edge_rect(frames: list[np.ndarray], face: list[float], thr: float = 0.6):
    """axis-aligned box of persistent straight edges around `face` (x0, y0, x1, y1), or None.
    sides with no line run to the frame border (corner cams have only two drawn edges)."""
    import cv2
    if len(frames) < 3:
        return None
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f for f in frames]
    persist = np.mean([cv2.Canny(g, 50, 150) > 0 for g in gray], axis=0)
    strong = cv2.dilate((persist >= 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    h, w = strong.shape
    fx0, fy0, fx1, fy1 = int(face[0] * w), int(face[1] * h), int(np.ceil(face[2] * w)), int(np.ceil(face[3] * h))
    gx, gy = max(2, (fx1 - fx0) // 4), max(2, (fy1 - fy0) // 4)   # a border sits clear of the head
    top = _line_rows(strong, range(fy0 - gy, -1, -1), fx0, fx1, thr)
    bot = _line_rows(strong, range(fy1 + gy, h), fx0, fx1, thr)
    left = _line_rows(strong.T, range(fx0 - gx, -1, -1), fy0, fy1, thr)
    right = _line_rows(strong.T, range(fx1 + gx, w), fy0, fy1, thr)
    found = [s is not None for s in (left, top, right, bot)]
    if sum(found) < 2:
        return None
    x0, y0 = left or 0, top or 0
    x1, y1 = right if right is not None else w - 1, bot if bot is not None else h - 1
    # each drawn edge must run the box's whole side, not just past the face (rejects ui bits)
    full = [strong[y0, x0:x1 + 1].mean(), strong[y1, x0:x1 + 1].mean(),
            strong[y0:y1 + 1, x0].mean(), strong[y0:y1 + 1, x1].mean()]
    if any(f and v < thr for f, v in zip((top is not None, bot is not None,
                                          left is not None, right is not None), full)):
        return None
    return [x0 / w, y0 / h, (x1 + 1) / w, (y1 + 1) / h]


def expand(face: list[float]) -> list[float]:
    cx, cy = (face[0] + face[2]) / 2, (face[1] + face[3]) / 2
    hw, hh = (face[2] - face[0]) * EXPAND_W / 2, (face[3] - face[1]) * EXPAND_H / 2
    return [max(0.0, cx - hw), max(0.0, cy - hh), min(1.0, cx + hw), min(1.0, cy + hh)]


def snap(box: list[float]) -> list[float]:
    x0, y0, x1, y1 = box
    return [0.0 if x0 < SNAP else x0, 0.0 if y0 < SNAP else y0,
            1.0 if x1 > 1 - SNAP else x1, 1.0 if y1 > 1 - SNAP else y1]


def region(box: list[float]) -> str:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    row = ("top", "middle", "bottom")[min(int(cy * 3), 2)]
    return row + "-" + ("left", "center", "right")[min(int(cx * 3), 2)]


def from_frames(frames: list[np.ndarray], detector) -> Facecam | None:
    """the whole algorithm minus decoding. None when no face cluster exists."""
    dets = [(i, x, y, w, h) for i, f in enumerate(frames)
            for x, y, w, h, s in detector(f) if s >= MIN_SCORE]
    hit = pick_cluster(dets, len(frames))
    if hit is None:
        return None
    m, coverage = hit
    x, y, w, h = np.median(m[:, 1:5], axis=0)
    face = [float(x), float(y), float(x + w), float(y + h)]
    box = snap(edge_rect(frames, face) or expand(face))
    r4 = lambda v: [round(float(c), 4) for c in v]   # noqa: E731
    return Facecam(r4(box), r4([x + w / 2, y + h / 2]), region(box), round(coverage, 3), len(frames))


# --------------------------------------------------------------------------- run io

def sample_times(ranges: list[tuple[str, float, float]], every=SAMPLE_EVERY,
                 lo=MIN_SAMPLES, hi=MAX_SAMPLES) -> list[tuple[str, float]]:
    """evenly over the concatenated used footage: (media, media seconds)."""
    total = sum(b - a for _, a, b in ranges)
    if total <= 0:
        return []
    n = int(min(hi, max(lo, total // every)))
    out = []
    for k in range(n):
        want, acc = (k + 0.5) * total / n, 0.0
        for media, a, b in ranges:
            if want <= acc + (b - a):
                out.append((media, a + want - acc))
                break
            acc += b - a
    return out


def read_frames(samples: list[tuple[str, float]], width: int = DECODE_W, caps: dict | None = None) -> list[np.ndarray]:
    """`caps`: a dict to keep captures open across calls (caller releases); None = open + close here."""
    import cv2
    out, own = [], caps is None
    caps = {} if own else caps
    try:
        for media, t in samples:
            cap = caps.get(media) or caps.setdefault(media, cv2.VideoCapture(str(media)))
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, f = cap.read()
            if ok:
                out.append(cv2.resize(f, (width, round(f.shape[0] * width / f.shape[1])))
                           if width and f.shape[1] != width else f)
    finally:
        if own:
            for c in caps.values():
                c.release()
    return out


class Facecams:
    def __init__(self, manifest_path, timeline_path=None, override: dict | None = None, detector=None):
        self.manifest_path = Path(manifest_path)
        self.run_dir = self.manifest_path.parent.parent          # R2: dir holding ae/manifest.json
        self.cache_path = self.run_dir / "facecam.json"
        self.override = override or {}
        self._detector = detector
        self._ranges = self._load_ranges(timeline_path)
        self._cams: dict[int, Facecam] = {}
        self._caps: dict = {}             # open captures for face_present (reopening costs ~100 ms)

    @property
    def detector(self):
        if self._detector is None:
            self._detector = YuNet()
        return self._detector

    def _load_ranges(self, timeline_path):
        """(source_id, media, src t0, src t1, offset) of the footage the cut uses."""
        tl = Path(timeline_path or self.manifest_path.parent / "timeline.tsv")
        if tl.exists():
            return [(s.source_id, s.media, s.src_in, s.src_out, s.offset)
                    for s in TimelineMap.load(self.manifest_path, tl).ours()]
        m = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        srcs = m.get("sources") or [{"path": m["source"], "offset": 0.0}]
        return [(int(c["start"] // VOD_BASE), srcs[c.get("src", 0)]["path"], c["start"], c["end"],
                 srcs[c.get("src", 0)].get("offset", 0.0)) for c in m.get("clips", [])]

    def _box_override(self, source_id: int):
        o = self.override.get(source_id) or self.override.get(str(source_id)) or self.override
        return o.get("box") if isinstance(o, dict) else None

    def detect(self, source_id: int) -> Facecam | None:
        if source_id in self._cams:
            return self._cams[source_id]
        box = self._box_override(source_id)
        if box:
            cam = Facecam(list(box), [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], region(box), 1.0, 0)
        else:
            cache = json.loads(self.cache_path.read_text(encoding="utf-8")) if self.cache_path.exists() else {}
            if str(source_id) in cache:
                cam = Facecam(**cache[str(source_id)]) if cache[str(source_id)] else None
            else:
                rs = [(media, a - off, b - off) for sid, media, a, b, off in self._ranges if sid == source_id]
                cam = from_frames(read_frames(sample_times(rs)), self.detector)
                cache[str(source_id)] = asdict(cam) if cam else None
                self.cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        self._cams[source_id] = cam
        return cam

    def frames(self, source_id: int, times, width: int = DECODE_W) -> list[np.ndarray]:
        """decoded frames at source times (transcript clock) that the cut uses; width 0 = full res."""
        samples = []
        for t in times:
            hit = next(((media, t - off) for sid, media, a, b, off in self._ranges
                        if sid == source_id and a <= t < b), None)
            if hit:
                samples.append(hit)
        return read_frames(samples, width, caps=self._caps) if samples else []

    def face_present(self, source_id: int, t0: float, t1: float, n: int = 3) -> float:
        """fraction of n frames in [t0, t1] (source time) with a face inside the facecam box."""
        cam = self.detect(source_id)
        if cam is None:
            return 0.0
        frames = self.frames(source_id, [t0 + (t1 - t0) * (k + 0.5) / n for k in range(n)])
        if not frames:
            return 0.0
        x0, y0, x1, y1 = cam.box
        hits = 0
        for f in frames:
            h, w = f.shape[:2]
            roi = f[int(y0 * h):int(np.ceil(y1 * h)), int(x0 * w):int(np.ceil(x1 * w))]
            hits += bool(roi.size) and any(d[4] >= MIN_SCORE for d in self.detector(roi))
        return hits / len(frames)
