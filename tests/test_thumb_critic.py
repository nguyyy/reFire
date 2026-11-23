"""thumb critic: bounded moves applied to spec.json (no model, no render)."""
import json

import numpy as np
from PIL import Image

from refire.thumb import critic


def _spec(tmp_path):
    a = tmp_path / "assets"
    a.mkdir()
    Image.fromarray(np.full((300, 200, 4), 255, np.uint8)).save(a / "card.png")
    Image.fromarray(np.full((500, 400, 4), 255, np.uint8)).save(a / "face.png")
    return {"scene": {"cards": [{"src": "assets/card.png", "x": 100.0, "y": 50.0, "w": 400.0, "z": -70, "rot": [0, 8, 0]}]},
            "layers": [{"kind": "face", "src": "assets/face.png", "x": 760.0, "y": 70.0, "w": 520.0,
                        "face_box": [900.0, 150.0, 200.0, 250.0]}]}


def test_moves_are_clamped():
    m = critic.clamp({"dx": 999, "dy": -999, "scale": 3})
    assert m == {"dx": critic.MOVE["dx"], "dy": -critic.MOVE["dy"], "scale": critic.MOVE["scale"][1]}


def test_left_move_keeps_its_center_when_scaling(tmp_path):
    spec = _spec(tmp_path)
    c = spec["scene"]["cards"][0]
    cx = c["x"] + c["w"] / 2
    critic.move_left(spec, tmp_path, {"dx": 0, "dy": 0, "scale": 1.2})
    assert abs(c["w"] - 480) < 0.1 and abs(c["x"] + c["w"] / 2 - cx) < 0.2


def test_left_move_carries_the_characters_face(tmp_path):
    spec = _spec(tmp_path)
    spec["scene"]["cards"][0]["face"] = [300, 180]
    critic.move_left(spec, tmp_path, {"dx": 40, "dy": -10, "scale": 1.0})
    assert spec["scene"]["cards"][0]["face"] == [340, 170]


def test_his_anger_mark_and_quote_follow_his_face(tmp_path):
    spec = _spec(tmp_path)
    spec["layers"] += [{"kind": "emote", "x": 850.0, "y": 60.0, "w": 120}, {"kind": "quote", "x": 600.0, "y": 10, "w": 400}]
    critic.move_face(spec, tmp_path, {"dx": -30, "dy": 0, "scale": 1.0})
    assert spec["layers"][1]["x"] == 820 and spec["layers"][2]["x"] == 570 and spec["layers"][2]["y"] == 10


def test_face_never_lifts_its_cut_edge_off_the_bottom(tmp_path):
    spec = _spec(tmp_path)
    critic.move_face(spec, tmp_path, {"dx": -20, "dy": -150, "scale": 0.85})
    L = spec["layers"][0]
    assert L["y"] + L["w"] * 500 / 400 >= 720 - 0.5                 # pushed back down to the frame bottom
    assert abs(L["face_box"][2] - 170) < 0.1


def test_review_applies_the_verdict_and_records_issues(tmp_path):
    (tmp_path / "A").mkdir()
    spec = _spec(tmp_path / "A")
    (tmp_path / "A" / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    Image.new("RGB", (1280, 720)).save(tmp_path / "A" / "thumb.png")
    renders = []
    import refire.thumb.render as R
    orig, R.render = R.render, lambda p, browser=None: renders.append(p)
    try:
        out = critic.review(tmp_path, ask=lambda text, Out: Out.model_validate({"candidates": [
            {"id": "A", "score": 70, "issues": ["head sliced by the cam edge"], "left": {"dx": -60}}]}))
    finally:
        R.render = orig
    assert out[0]["moved"] and renders                               # moved -> re-rendered
    rec = json.loads((tmp_path / "A" / "critic.json").read_text(encoding="utf-8"))
    assert rec["issues"] == ["head sliced by the cam edge"] and rec["left"]["dx"] == -60
