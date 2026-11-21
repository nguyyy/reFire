"""deterministic thumbnail checks + report (refire.thumb.checks / report), OCR injected."""
import json

import numpy as np
from PIL import Image

from refire.thumb import checks, report


def cand(root, name, face_box, hook_box, color, accent="gold", text=("BIG", "HOOK?"), extra=0):
    d = root / name
    (d / "assets").mkdir(parents=True)
    Image.new("RGB", (1280, 720), color).save(d / "thumb.png")
    Image.new("RGBA", (480, 420), (0, 0, 0, 255)).save(d / "assets" / "face.png")
    json.dump({"x": hook_box[0], "y": hook_box[1], "w": hook_box[2], "h": hook_box[3]}, open(d / "assets" / "hook.json", "w"))
    layers = [{"kind": "face", "src": "assets/face.png", "x": 700, "y": 80, "w": 600, "face_box": face_box}]
    layers += [{"kind": "glyph", "src": "x", "x": 0, "y": 0, "w": 10}] * extra
    spec = {"canvas": [1280, 720], "layers": layers, "text": {"lines": list(text), "accent_phrase": text[0], "accent": accent}}
    (d / "spec.json").write_text(json.dumps(spec))
    return d


def test_checks_pass_and_fail(tmp_path):
    ok = cand(tmp_path, "A", [880, 200, 200, 300], [40, 500, 700, 180], (30, 60, 200))          # blue bg, gold accent
    res = {r["name"]: r for r in checks.candidate(ok, ocr=lambda im: "BIG HOOK?")}
    assert all(r["ok"] for r in res.values()), res
    bad = cand(tmp_path, "B", [880, 200, 120, 100], [600, 230, 700, 490], (200, 160, 30), extra=4)  # gold on gold, over the face
    res = {r["name"]: r for r in checks.candidate(bad, ocr=lambda im: "BG HK")}
    assert not res["hook readable small"]["ok"] and not res["face size"]["ok"]
    assert not res["badge zone"]["ok"] and not res["accent vs background"]["ok"] and not res["clutter"]["ok"]
    assert not res["face clear"]["ok"]


def test_dominant_hue_and_phash():
    blue = np.zeros((40, 40, 3), np.uint8); blue[..., 2] = 220
    assert abs(checks.dominant_hue(blue) - 240) < 2


def test_report_orders_passing_candidates_first(tmp_path):
    cand(tmp_path / "thumb", "A", [880, 200, 120, 100], [600, 500, 700, 200], (200, 160, 30))   # fails
    b = cand(tmp_path / "thumb", "B", [880, 200, 200, 300], [40, 500, 700, 180], (30, 60, 200))
    Image.fromarray(np.random.default_rng(1).integers(0, 255, (720, 1280, 3), dtype=np.uint8)).save(b / "thumb.png")
    (tmp_path / "thumb" / "concepts.json").write_text(json.dumps({"concepts": [
        {"id": "A", "type": "reaction", "title": "a", "payoff_beats": [3]},
        {"id": "B", "type": "reveal", "title": "What did he roll?", "payoff_beats": [5, 6]}]}))
    out = report.build(tmp_path, ocr=lambda im: "BIG HOOK?")
    md = out.read_text(encoding="utf-8")
    assert md.index("1. **B**") < md.index("2. **A**") and "slot 1" in md
    assert "What did he roll?" in md and "Paid off by:** 5, 6" in md and "three concept types ok" in md
