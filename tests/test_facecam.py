"""facecam detection on synthetic frames with an injected detector (no model, no video)."""
import json

import cv2
import numpy as np
import pytest

from refire import facecam
from refire.facecam import Facecams, dbscan, from_frames, region, sample_times

H, W = 360, 640
CAM = [0.0, 0.62, 0.27, 1.0]                  # bottom-left facecam, as in zy0x's layout


def _frames(n=40, border=True, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        f = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (0, 0), 6)
        f = cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX)        # changing "gameplay"
        if border:
            x1, y0 = int(CAM[2] * W), int(CAM[1] * H)
            cv2.rectangle(f, (-5, y0), (x1, H + 5), (255, 255, 255), 2)
        out.append(f)
    return out


class FakeDetector:
    """facecam face in 90% of frames with a steady size; a game face of wild size in some."""

    def __init__(self, seed=0):
        self.rng, self.i = np.random.default_rng(seed), 0

    def __call__(self, img):
        self.i += 1
        if img.shape[:2] != (H, W):            # roi call from face_present
            return [(0.4, 0.3, 0.2, 0.3, 0.95)]
        out = []
        if self.i % 10:
            j = self.rng.normal(0, 0.004, 2)
            out.append((0.09 + j[0], 0.72 + j[1], 0.06, 0.11, 0.93))
        if self.i % 3 == 0:
            s = self.rng.uniform(0.03, 0.2)
            out.append((0.6, 0.3, s, s * 1.5, 0.9))
        out.append((0.8, 0.1, 0.05, 0.05, 0.5))  # below the score floor
        return out


def test_dbscan_two_blobs_and_noise():
    rng = np.random.default_rng(1)
    pts = np.vstack([rng.normal([0.1, 0.8], 0.005, (20, 2)), rng.normal([0.7, 0.3], 0.005, (15, 2)),
                     [[0.5, 0.5]]])
    lab = dbscan(pts, 0.05, 3)
    assert len(set(lab[:20])) == 1 and len(set(lab[20:35])) == 1 and lab[0] != lab[20]
    assert lab[-1] == -1


def test_finds_the_drawn_cam_rectangle():
    cam = from_frames(_frames(), FakeDetector())
    assert cam.box == pytest.approx(CAM, abs=0.015)
    assert cam.region == "bottom-left"
    assert cam.confidence == pytest.approx(0.9, abs=0.03)           # coverage = frames with the cam face
    assert cam.face_center == pytest.approx([0.12, 0.775], abs=0.01)
    assert cam.samples == 40


def test_no_border_falls_back_to_expanded_face_box():
    cam = from_frames(_frames(border=False), FakeDetector())
    # median face 0.09..0.15 x 0.72..0.83 -> x3.0 / x3.2 around its center
    assert cam.box[0] == pytest.approx(0.03, abs=0.01)
    assert cam.box[2] == pytest.approx(0.12 + 0.09, abs=0.01)
    assert cam.box[1] == pytest.approx(0.775 - 0.176, abs=0.01)
    assert cam.box[3] == pytest.approx(0.775 + 0.176, abs=0.01)


def test_no_faces_is_none():
    assert from_frames(_frames(5), lambda f: []) is None


def test_region_thirds():
    assert region([0, 0.62, 0.27, 1]) == "bottom-left"
    assert region([0, 0.3, 0.2, 0.6]) == "middle-left"
    assert region([0.75, 0, 1, 0.3]) == "top-right"
    assert region([0.4, 0.4, 0.6, 0.6]) == "middle-center"


def test_sample_times_clamped_and_spread():
    s = sample_times([("a", 0, 100)])
    assert len(s) == 30 and s[0] == ("a", pytest.approx(100 / 60))
    s = sample_times([("a", 0, 40), ("b", 50, 4050)])                   # 4040 s -> capped at 120
    assert len(s) == 120 and s[0][0] == "a" and all(m == "b" for m, _ in s[1:])
    assert sample_times([]) == []


def _run(tmp_path, clips, sources):
    ae = tmp_path / "ae"
    ae.mkdir()
    (ae / "manifest.json").write_text(json.dumps({"clips": clips, "sources": sources}), encoding="utf-8")
    return ae / "manifest.json"


def test_run_level_detect_caches_and_overrides(tmp_path, monkeypatch):
    mp = _run(tmp_path, [{"start": 110, "end": 150, "src": 0}], [{"path": "A.mov", "offset": 100}])
    seen = []
    monkeypatch.setattr(facecam, "read_frames", lambda s, width=640: seen.extend(s) or _frames())
    cams = Facecams(mp, detector=FakeDetector())
    cam = cams.detect(0)
    assert cam.region == "bottom-left" and cam.usable
    assert seen[0][0] == "A.mov" and 10 <= seen[0][1] < 50              # media time = src - offset
    cached = json.loads((tmp_path / "facecam.json").read_text())
    assert cached["0"]["box"] == cam.box
    # a fresh instance reads the cache instead of decoding
    monkeypatch.setattr(facecam, "read_frames", lambda *a, **k: pytest.fail("decoded again"))
    assert Facecams(mp, detector=FakeDetector()).detect(0).box == cam.box
    # style override wins, globally or per source
    o = Facecams(mp, override={"box": [0.7, 0, 1, 0.3]}).detect(0)
    assert o.box == [0.7, 0, 1, 0.3] and o.region == "top-right" and o.confidence == 1.0
    assert Facecams(mp, override={"0": {"box": [0, 0, 0.2, 0.2]}}).detect(0).box == [0, 0, 0.2, 0.2]


def test_face_present_checks_the_roi(tmp_path, monkeypatch):
    mp = _run(tmp_path, [{"start": 110, "end": 150, "src": 0}], [{"path": "A.mov", "offset": 100}])
    cams = Facecams(mp, override={"box": CAM}, detector=FakeDetector())
    got = []
    monkeypatch.setattr(facecam, "read_frames", lambda s, width=640, caps=None: got.extend(s) or _frames(len(s)))
    assert cams.face_present(0, 120, 123) == 1.0
    assert [round(t, 2) for _, t in got] == [20.5, 21.5, 22.5]
    assert cams.face_present(0, 300, 303) == 0.0                        # not in the cut
    cams._detector = lambda img: []
    assert cams.face_present(0, 120, 123) == 0.0
