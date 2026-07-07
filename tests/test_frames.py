"""Frame sampling plan + sheet layout math + VLM caption cache (all pure/mocked)."""
import json

import numpy as np

from refire import vlm
from refire.frames import _grid, _label, _parse_showinfo, _thin_frames


def test_parse_showinfo_pulls_pts():
    err = ("[Parsed_showinfo_1 @ 0x1] n:   0 pts:  12345 pts_time:12.345 duration...\n"
           "random ffmpeg chatter\n"
           "[Parsed_showinfo_1 @ 0x1] n:   1 pts:  99999 pts_time:100.5 pos...\n")
    assert _parse_showinfo(err) == [12.345, 100.5]
    assert _parse_showinfo("") == []


def test_thin_frames_min_interval_and_floor():
    # two cuts too close together -> second dropped; long tail gap -> backfilled
    got = _thin_frames([10.0, 15.0, 30.0], duration=430.0,
                       min_interval_s=20.0, floor_gap_s=180.0, max_frames=600)
    assert 10.0 in got and 15.0 not in got and 30.0 in got
    assert any(t > 30.0 for t in got)          # the 30..430 gap got uniform fill
    assert got == sorted(got)
    gaps = [b - a for a, b in zip([0.0] + got, got + [430.0])]
    assert max(gaps) <= 180.0


def test_thin_frames_caps_max():
    got = _thin_frames([float(t) for t in range(0, 10000, 25)], duration=10000.0,
                       max_frames=100)
    assert len(got) <= 100


def test_grid_layout():
    assert _grid(4, 3, 3) == [(0, 0), (0, 1), (0, 2), (1, 0)]
    assert len(_grid(99, 3, 3)) == 9


def test_label_formats():
    assert _label(65) == "01:05"
    assert _label(3725) == "1:02:05"


def test_caption_frames_cache_and_resume(tmp_path, monkeypatch):
    import ollama
    calls = []

    def fake_chat(model, messages):
        calls.append(messages[0]["images"][0])
        return {"message": {"content": "player dies to boss\nextra line"}}

    monkeypatch.setattr(ollama, "chat", fake_chat)
    monkeypatch.setattr(vlm, "vlm_available", lambda m: True)
    cache = tmp_path / "caps.json"
    frames = [{"t": 10.0, "path": "a.jpg"}, {"t": 20.0, "path": "b.jpg"}]
    caps = vlm.caption_frames(frames, cache)
    assert caps == {10.0: "player dies to boss", 20.0: "player dies to boss"}
    assert len(calls) == 2 and cache.exists()
    # second run: fully cached -> zero model calls, survives a new frame appearing
    caps2 = vlm.caption_frames(frames + [{"t": 30.0, "path": "c.jpg"}], cache)
    assert len(calls) == 3 and caps2[30.0] == "player dies to boss"


def test_caption_frames_rejects_degenerate_and_selfheals(tmp_path, monkeypatch):
    import ollama
    from refire.vlm import _degenerate
    assert _degenerate("@@@@@@@@@@@@@@@@") and _degenerate("!!!!!!!!")
    assert not _degenerate("player dies to boss") and not _degenerate("ok")

    monkeypatch.setattr(ollama, "chat",
                        lambda model, messages: {"message": {"content": "@" * 29}})
    monkeypatch.setattr(vlm, "vlm_available", lambda m: True)
    cache = tmp_path / "caps.json"
    # a collapsed model caches nothing (not the garbage)
    assert vlm.caption_frames([{"t": 1.0, "path": "a.jpg"}], cache) == {}
    assert json.loads(cache.read_text()) == {}

    # a pre-poisoned cache is dropped on load, so the frame gets re-attempted
    cache.write_text(json.dumps({"1.0": "@" * 29}))
    calls = []

    def good(model, messages):
        calls.append(1)
        return {"message": {"content": "menu screen"}}

    monkeypatch.setattr(ollama, "chat", good)
    caps = vlm.caption_frames([{"t": 1.0, "path": "a.jpg"}], cache)
    assert caps == {1.0: "menu screen"} and len(calls) == 1


def test_caption_frames_missing_model_degrades(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm, "vlm_available", lambda m: False)
    caps = vlm.caption_frames([{"t": 1.0, "path": "x.jpg"}], tmp_path / "c.json")
    assert caps == {}


def test_caption_frames_midrun_crash_keeps_partial(tmp_path, monkeypatch):
    import ollama
    state = {"n": 0}

    def flaky(model, messages):
        state["n"] += 1
        if state["n"] > 1:
            raise RuntimeError("ollama fell over")
        return {"message": {"content": "menu screen"}}

    monkeypatch.setattr(ollama, "chat", flaky)
    monkeypatch.setattr(vlm, "vlm_available", lambda m: True)
    cache = tmp_path / "caps.json"
    caps = vlm.caption_frames([{"t": 1.0, "path": "a.jpg"},
                               {"t": 2.0, "path": "b.jpg"}], cache)
    assert caps == {1.0: "menu screen"}
    assert json.loads(cache.read_text()) == {"1.0": "menu screen"}
