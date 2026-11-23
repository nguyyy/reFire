"""zy0x's expression from MediaPipe Face Landmarker blendshapes (03-thumb section 3, the E and O terms
of the face score). shock = jawOpen + eyeWide + browInnerUp; hype = smile + jawOpen; laughing = smile
+ jawOpen + eyeSquint; deadpan = little activation + mouthPress; skeptical = brow asymmetry +
mouthPress. `facing` is the O term: eyes open, not looking down, head not tilted far (the critic kept
flagging "looking down"). The model (3.7 MB, Google's official) is fetched once into the cache.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL = Path.home() / ".cache" / "refire" / "models" / "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
             "float16/1/face_landmarker.task")
TILT_OK, TILT_SPAN = 12.0, 25.0      # head pitch/yaw (deg) free up to 12, then the O term falls to 0 by 37
LOOK_DOWN = 0.4                      # eyeLookDown above this costs O (eyes on his lap, not the camera)
BLINK_OK = 0.5                       # eyeBlink past this scales the expression toward 0 (shut at 1.0)


@lru_cache(maxsize=1)
def landmarker():
    """the face landmarker, or None when mediapipe or the model can't be had (callers fall back)."""
    try:
        import mediapipe  # noqa: F401
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
    except ImportError:
        return None
    if not MODEL.exists():
        import urllib.request
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL)
        except OSError:
            return None
    opts = vision.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=str(MODEL)),
                                        output_face_blendshapes=True, output_facial_transformation_matrixes=True,
                                        num_faces=1)
    return vision.FaceLandmarker.create_from_options(opts)


def read(bgr: np.ndarray) -> dict | None:
    """-> {"bs": {blendshape: 0-1}, "pitch": deg, "yaw": deg}, or None (no face / no model)."""
    lm = landmarker()
    if lm is None:
        return None
    import cv2
    import mediapipe as mp
    rgb = np.ascontiguousarray(cv2.cvtColor(np.ascontiguousarray(bgr[:, :, :3]), cv2.COLOR_BGR2RGB))
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_blendshapes:
        return None
    bs = {c.category_name: float(c.score) for c in res.face_blendshapes[0]}
    pitch = yaw = 0.0
    if res.facial_transformation_matrixes:
        R = np.asarray(res.facial_transformation_matrixes[0])[:3, :3]
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, float(R[1, 2])))))
        yaw = math.degrees(math.asin(max(-1.0, min(1.0, float(R[0, 2])))))
    return {"bs": bs, "pitch": pitch, "yaw": yaw}


def _m(bs: dict, *names: str) -> float:
    return float(np.mean([bs.get(n, 0.0) for n in names]))


def score(expression: str, bs: dict) -> float:
    """how well the blendshapes match the asked expression, 0-1 (the spec's recipes), times an
    eyes-open factor: closed eyes never sell a thumbnail (a laughing squint outscored them)."""
    blink = _m(bs, "eyeBlinkLeft", "eyeBlinkRight")
    return _recipe(expression, bs) * (1 - min(1.0, max(0.0, blink - BLINK_OK) / (1 - BLINK_OK)))


def _recipe(expression: str, bs: dict) -> float:
    jaw, smile = bs.get("jawOpen", 0.0), _m(bs, "mouthSmileLeft", "mouthSmileRight")
    wide, squint = _m(bs, "eyeWideLeft", "eyeWideRight"), _m(bs, "eyeSquintLeft", "eyeSquintRight")
    brow_in, press = bs.get("browInnerUp", 0.0), _m(bs, "mouthPressLeft", "mouthPressRight")
    if expression == "shock":
        return (jaw + wide + brow_in) / 3
    if expression == "hype":
        return (smile + jaw) / 2
    if expression == "laughing":
        return (smile + jaw + squint) / 3
    if expression == "deadpan":
        return 0.5 * (1 - max(jaw, smile, wide, brow_in)) + 0.5 * press
    if expression == "skeptical":
        asym = abs(bs.get("browOuterUpLeft", 0.0) - bs.get("browOuterUpRight", 0.0)) + \
            abs(bs.get("browDownLeft", 0.0) - bs.get("browDownRight", 0.0))
        return 0.5 * min(1.0, asym) + 0.5 * press
    return (jaw + wide + brow_in) / 3                    # unknown: the house default is shock


def facing(r: dict) -> float:
    """the O term, 0-1: eyes open, eyes on the camera (not down), head not tilted far."""
    bs = r["bs"]
    open_ = 1 - _m(bs, "eyeBlinkLeft", "eyeBlinkRight")
    down = max(0.0, _m(bs, "eyeLookDownLeft", "eyeLookDownRight") - LOOK_DOWN) / (1 - LOOK_DOWN)
    tilt = max(abs(r["pitch"]), abs(r["yaw"]))
    head = 1 - min(1.0, max(0.0, tilt - TILT_OK) / TILT_SPAN)
    return max(0.0, open_ * (1 - down) * head)
