import json

from refire.cli import _progress_writer


def test_progress_writer_throttles_and_finalizes(tmp_path):
    pf = tmp_path / "progress.json"
    w = _progress_writer(pf)

    w(0.0, "starting")
    assert json.loads(pf.read_text())["pct"] == 0

    # same integer pct AND same message -> throttled, file not rewritten
    w(0.004, "starting")                    # int(0.4) == 0
    assert json.loads(pf.read_text())["msg"] == "starting"

    # same pct but a NEW message -> writes: the scout moves through 13 chapter
    # messages inside 3 pct points, and the panel must not show a stale line.
    w(0.008, "scouting chapter 2/13")
    d = json.loads(pf.read_text())
    assert d["pct"] == 0 and d["msg"] == "scouting chapter 2/13"

    # pct change -> writes
    w(0.5, "half")
    d = json.loads(pf.read_text())
    assert d == {"pct": 50, "msg": "half", "done": False}

    # done always writes even when pct is unchanged
    w(0.5, "error: boom", done=True)
    d = json.loads(pf.read_text())
    assert d["done"] is True and d["msg"] == "error: boom"
