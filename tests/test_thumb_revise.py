import json

import pytest
from PIL import Image

from refire.thumb import revise


def cand(tmp_path):
    d = tmp_path / "A"
    (d / "assets").mkdir(parents=True)
    Image.new("RGBA", (400, 500)).save(d / "assets" / "face.png")
    spec = {"layers": [{"kind": "face", "src": "assets/face.png", "x": 700, "y": 100, "w": 600,
                        "face_box": [850, 200, 250, 300]}],
            "text": {"lines": ["BIG", "HOOK?"], "accent_phrase": "HOOK?", "accent": "gold"}}
    (d / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    return d, spec


def test_pointer_edits_set_append_remove_and_report_bad_paths():
    spec = {"text": {"lines": ["A", "B"]}, "layers": [{"x": 1}, {"x": 2}]}
    errs = revise.apply_edits(spec, [{"path": "/text/lines/1", "value": "C"}, {"path": "/layers/-", "value": {"x": 3}},
                                     {"op": "remove", "path": "/layers/0"}, {"path": "/text/annotation", "value": "(x)"},
                                     {"path": "/nope/deep", "value": 1}, {"path": "/layers/9/x", "value": 0}])
    assert spec == {"text": {"lines": ["A", "C"], "annotation": "(x)"}, "layers": [{"x": 2}, {"x": 3}]}
    assert len(errs) == 2 and "/nope/deep" in errs[0] and "/layers/9/x" in errs[1]


def test_edit_moves_zy0x_and_refuses_missing_assets_and_a_lost_accent(tmp_path):
    d, spec = cand(tmp_path)
    errs = revise.edit(spec, d, {"face": {"dx": -100, "dy": 0, "scale": 1.0}, "left": {"dx": 0, "dy": 0, "scale": 1.0},
                                 "edits": [{"op": "set", "path": "/text/lines", "value": ["NEW", "HOOK?"]}]})
    assert errs == [] and spec["layers"][0]["x"] == 600 and spec["text"]["lines"] == ["NEW", "HOOK?"]
    errs = revise.edit(spec, d, {"face": {}, "left": {}, "edits": [
        {"path": "/layers/0/src", "value": "assets/other.png"}, {"path": "/text/lines", "value": ["NO", "ACCENT"]}]})
    assert any("no such asset" in e for e in errs) and any("accent_phrase" in e for e in errs)


def test_noop_moves_leave_the_spec_alone(tmp_path):
    d, spec = cand(tmp_path)
    before = json.dumps(spec)
    assert revise.edit(spec, d, {"face": {"dx": 0, "dy": 0, "scale": 1.0}, "left": {"dx": 0, "dy": 0, "scale": 1.0},
                                 "edits": []}) == []
    assert json.dumps(spec) == before                 # move_face would re-snap the cut edge otherwise


def test_revise_retries_once_then_writes_renders_and_logs(tmp_path, monkeypatch):
    d, _ = cand(tmp_path)
    from refire.thumb import checks, render, report
    monkeypatch.setattr(render, "render", lambda p, **k: p.parent / "thumb.png")
    monkeypatch.setattr(checks, "candidate", lambda d, ocr=None: [{"name": "face size", "ok": True, "value": 1, "need": 1}])
    monkeypatch.setattr(report, "build", lambda *a, **k: None)
    answers = iter([{"edits": [{"path": "/text/lines", "value": ["NO", "ACCENT"]}]},          # loses the accent
                    {"note": "shorter hook", "edits": [{"path": "/text/lines", "value": ["HOOK?", "NOW"]}]}])
    prompts = []

    def ask(text, Out):
        prompts.append(text)
        return Out.model_validate(next(answers))

    r = revise.revise(d, "shorter hook", ask=ask, log=lambda *a: None)
    assert "broke these rules" in prompts[1] and r["note"] == "shorter hook"
    assert json.loads((d / "spec.json").read_text(encoding="utf-8"))["text"]["lines"] == ["HOOK?", "NOW"]
    assert json.loads((d / "spec.prev.json").read_text(encoding="utf-8"))["text"]["lines"] == ["BIG", "HOOK?"]
    assert json.loads((d / "revise.jsonl").read_text(encoding="utf-8").splitlines()[0])["instruction"] == "shorter hook"


def test_revise_gives_up_after_the_retry(tmp_path):
    d, _ = cand(tmp_path)
    bad = {"edits": [{"path": "/nope/x", "value": 1}]}
    with pytest.raises(ValueError, match="thumb revise"):
        revise.revise(d, "x", ask=lambda t, Out: Out.model_validate(bad), log=lambda *a: None)
    assert not (d / "spec.prev.json").exists()
