"""Alpha cutouts with BiRefNet (MIT, official ONNX exports) on onnxruntime.

zy0x cutout per spec: inset the facecam crop 1-2% (drops the cam border), BiRefNet alpha,
1 px feather. `lite` (swin tiny, 224 MB) is the CPU default; `portrait` (973 MB) is better
on people and wants a GPU box. Weights fetch once to ~/.cache/refire/models, sha256 pinned.
"""
from __future__ import annotations

import hashlib
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np

MODELS_DIR = Path.home() / ".cache" / "refire" / "models"
_REL = "https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/"
MODELS = {
    "lite": ("BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
             "5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333"),
    "portrait": ("BiRefNet-portrait-epoch_150.onnx", None),     # ponytail: pin on first fetch
}
SIZE = 1024
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def _weights(model: str) -> Path:
    name, sha = MODELS[model]
    p = MODELS_DIR / name
    if not p.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        tmp, h = p.with_suffix(".part"), hashlib.sha256()
        with urllib.request.urlopen(urllib.request.Request(_REL + name, headers={"User-Agent": "refire"}),
                                    timeout=600) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                h.update(chunk)
                f.write(chunk)
        if sha and h.hexdigest() != sha:
            tmp.unlink()
            raise RuntimeError(f"checksum mismatch for {name}")
        tmp.rename(p)
    return p


@lru_cache(maxsize=2)
def _session(model: str):
    import onnxruntime as ort
    prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    return ort.InferenceSession(str(_weights(model)), providers=prov)


def alpha(img_bgr: np.ndarray, model: str = "lite") -> np.ndarray:
    """foreground alpha (uint8, image size) for a BGR image."""
    import cv2
    h, w = img_bgr.shape[:2]
    x = cv2.resize(img_bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255.0
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
    s = _session(model)
    logits = s.run(None, {s.get_inputs()[0].name: np.ascontiguousarray(x)})[0][0, 0]
    a = 0.5 * (1.0 + np.tanh(0.5 * logits))                  # sigmoid without overflow
    return (cv2.resize(a, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1) * 255).astype(np.uint8)


def inset(img: np.ndarray, frac: float) -> np.ndarray:
    """trim `frac` off every side (facecam border remnants)."""
    h, w = img.shape[:2]
    dy, dx = int(round(h * frac)), int(round(w * frac))
    return img[dy:h - dy, dx:w - dx]


def cutout(img_bgr: np.ndarray, inset_frac: float = 0.015, feather: float = 1.0, model: str = "lite",
           alpha_fn=None) -> np.ndarray:
    """BGRA cutout: inset, BiRefNet alpha, `feather` px gaussian on the edge. `alpha_fn` injects
    a stand-in model for tests."""
    import cv2
    img = inset(img_bgr, inset_frac)
    a = (alpha_fn or (lambda im: alpha(im, model)))(img)
    if feather:
        a = cv2.GaussianBlur(a, (0, 0), feather)
    return np.dstack([img, a])
