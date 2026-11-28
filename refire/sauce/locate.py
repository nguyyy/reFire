"""Find a zoom's target on screen (04-sauce.md 5.4) with OWLv2 (Apache-2.0), ONNX on the CPU.

Spike S9 (Nov 27): the base ensemble model, fp32, takes ~11 s a 1080p frame on this CPU (int8 was
7.5 s and boxed worse); zooms are rare in Nguy's style (~0.1/min), so that's a few seconds a video
and no torch install. The director names the target in words; a detection under MIN_SCORE, or a
target that has left the zoomed window by the zoom's last frame, means no zoom (spec: skip).
Zooms are a hard cut to a static window (Nguy, Nov 27: no keyframed ease), so no tracking yet.
ponytail: static window; add OpenCV's ViT tracker + keyed Position if moving targets get skipped a lot.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = "onnx-community/owlv2-base-patch16-ensemble-ONNX"
CACHE = Path.home() / ".cache" / "refire" / "models" / "owlv2"
MIN_SCORE = 0.2                 # OWLv2 sigmoid score a target needs (generic queries on his frames: 0.04-0.3)
MAX_SCALE, MIN_SCALE = 2.0, 1.3 # spec: capped at 200% (1080p source); under 130% isn't worth a cut
MARGIN = 1.6                    # the target fills 1/MARGIN of the zoomed frame's width or height
_SESS = None


def _load():
    global _SESS
    if _SESS is None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        mp = hf_hub_download(REPO, "onnx/model.onnx", local_dir=CACHE)
        tok = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json", local_dir=CACHE))
        pc = json.loads(Path(hf_hub_download(REPO, "preprocessor_config.json", local_dir=CACHE)).read_text())
        _SESS = (ort.InferenceSession(mp, providers=["CPUExecutionProvider"]), tok, pc)
    return _SESS


def detect(bgr: np.ndarray, query: str) -> tuple[float, list[float]]:
    """(score, [x0, y0, w, h] px) of the best box for `query` in a BGR frame."""
    import cv2
    sess, tok, pc = _load()
    e = tok.encode(query)
    L = 16
    ids = np.zeros((1, L), np.int64)
    am = np.zeros((1, L), np.int64)
    n = min(L, len(e.ids))
    ids[0, :n], am[0, :n] = e.ids[:n], 1
    h, w = bgr.shape[:2]
    side = max(h, w)
    S = pc["size"]["height"]
    pad = np.full((side, side, 3), 0.5, np.float32)          # owlv2 pads bottom/right with gray, then resizes
    pad[:h, :w] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    x = (cv2.resize(pad, (S, S)) - np.array(pc["image_mean"], np.float32)) / np.array(pc["image_std"], np.float32)
    names = [o.name for o in sess.get_outputs()]
    out = sess.run(None, {"input_ids": ids, "pixel_values": x.transpose(2, 0, 1)[None], "attention_mask": am})
    logits, boxes = out[names.index("logits")][0][:, 0], out[names.index("pred_boxes")][0]
    k = int(logits.argmax())
    cx, cy, bw, bh = (boxes[k] * side).tolist()
    return float(1 / (1 + np.exp(-logits[k]))), [cx - bw / 2, cy - bh / 2, bw, bh]


def window(box: list[float], W: int, H: int, max_scale: float = MAX_SCALE) -> tuple[list[float], float] | None:
    """target box -> (anchor = the window's center, normalized; scale) with the window kept inside the
    frame (no black edges), or None when the target is too big to be worth a zoom."""
    x0, y0, bw, bh = box
    s = min(max_scale, W / max(1.0, bw * MARGIN), H / max(1.0, bh * MARGIN))
    if s < MIN_SCALE:
        return None
    half = 0.5 / s
    cx = min(max((x0 + bw / 2) / W, half), 1 - half)
    cy = min(max((y0 + bh / 2) / H, half), 1 - half)
    return [round(cx, 4), round(cy, 4)], round(s, 3)


def inside(box: list[float], center: list[float], scale: float, W: int, H: int) -> bool:
    """is the box's center still within the zoomed window?"""
    cx, cy = (box[0] + box[2] / 2) / W, (box[1] + box[3] / 2) / H
    half = 0.5 / scale
    return abs(cx - center[0]) <= half and abs(cy - center[1]) <= half
