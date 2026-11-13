"""refire auto: settings picked per VOD, base-edit render options, the final check."""
import json

import pytest

from refire import assemble, auto, render
from refire.subtitles import lines_ass


def _diff(root, vod, end):
    d = root / vod
    d.mkdir(parents=True)
    (d / "clip_diff.json").write_text(json.dumps(
        {"final": [{"seq_in": 0.0, "seq_out": end / 2}, {"seq_in": end / 2, "seq_out": end}]}))


def test_learned_duration_is_the_median_finished_runtime_capped_by_the_vod(tmp_path):
    for vod, end in [("1", 500), ("2", 900), ("3", 1100)]:
        _diff(tmp_path, vod, end)
    assert auto.learned_duration(tmp_path, vod_len=4 * 3600) == (
        900, "median runtime of your 3 finished edits")
    t, why = auto.learned_duration(tmp_path, vod_len=1200)      # 20 min stream
    assert t == 600 and "capped" in why
    assert auto.learned_duration(tmp_path / "none")[0] == auto.DEFAULT_DURATION


def test_vod_info_reads_the_chat_downloads_video_block(tmp_path):
    p = tmp_path / "1.chat.json"
    p.write_text(json.dumps({"video": {"title": "BIG STREAM", "game": "Genshin Impact",
                                       "length": 7200}, "comments": []}))
    assert auto.vod_info(p) == {"title": "BIG STREAM", "game": "Genshin Impact",
                                "length": 7200.0}
    assert auto.vod_info(tmp_path / "missing.json") == {}


def test_plan_is_a_base_edit_with_measured_stage_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(auto, "pick_encoder", lambda: "h264_nvenc")
    pl = auto.plan({"title": "T", "game": "Genshin Impact", "length": 3 * 3600}, tmp_path)
    kw = pl["kwargs"]
    assert kw["game"] == "genshin impact" and kw["brief"] is None and kw["render"]
    assert not any(kw[k] for k in ("emotes", "sfx", "music", "cards", "motion_zoom",
                                   "boundary", "trim", "critics", "markers"))
    assert kw["size"] == (1920, 1080) and kw["encoder"] == "h264_nvenc"
    assert kw["duration_s"] == auto.DEFAULT_DURATION and set(pl["why"]) >= {"duration_s", "brief"}
    assert auto.plan({}, tmp_path, duration=600, brief="the boss")["kwargs"]["duration_s"] == 600


def test_verify_fails_hard_on_canvas_or_audio_and_notes_runtime(tmp_path, monkeypatch):
    mp4 = tmp_path / "final.mp4"
    assert auto.verify(mp4, 900)[1] == [f"{mp4} was not written"]
    mp4.write_bytes(b"")
    monkeypatch.setattr(auto, "probe", lambda p: {"duration_s": 1400.0, "width": 1920,
                                                  "height": 1080, "audio": True})
    _p, bad, notes = auto.verify(mp4, 900)
    assert bad == [] and "outside" in notes[0]
    monkeypatch.setattr(auto, "probe", lambda p: {"duration_s": 900.0, "width": 1280,
                                                  "height": 720, "audio": False})
    _p, bad, notes = auto.verify(mp4, 900)
    assert len(bad) == 2 and notes == []


def test_lines_ass_burns_the_fixed_lines_as_given():
    ass = lines_ass([{"text": "remember THAT", "start": 0.0, "end": 0.6},
                     {"text": "Kinich wins", "start": 0.6, "end": 1.0},
                     {"text": "", "start": 1.0, "end": 1.2}])
    ev = [ln for ln in ass.splitlines() if ln.startswith("Dialogue:")]
    assert len(ev) == 2
    assert ev[0].endswith(r"{\k30}remember {\k30}THAT") and r"{\k20}Kinich" in ev[1]


def _calls(monkeypatch, **kw):
    calls = []
    monkeypatch.setattr(render, "_run", lambda cmd: calls.append(cmd))
    render.render_clip("v.mp4", {"start": 0.0, "end": 5.0}, "c.ass", "out.mp4",
                       motion_zoom=False, **kw)
    return calls


def test_render_clip_size_and_nvenc_quality(monkeypatch):
    calls = _calls(monkeypatch, size=(1920, 1080), encoder="h264_nvenc")
    scale = next(c for c in calls if "-vf" in c)
    assert "scale=1920:1080" in scale[scale.index("-vf") + 1]
    assert all("-cq" in c for c in calls)
    assert not any("-cq" in c for c in _calls(monkeypatch))       # libx264 unchanged


def test_render_clips_burns_given_captions_instead_of_words(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(assemble, "render_clip",
                        lambda video, seg, ass, out, **k: seen.update(
                            ass=ass.read_text(encoding="utf-8"), size=k.get("size")))
    monkeypatch.setattr(assemble, "assemble", lambda clips, out, music=None: out)
    words = [{"text": "kinich", "start": 1.0, "end": 1.5, "emph": False}]
    out = assemble.render_clips("v.mp4", tmp_path, [{"start": 0.0, "end": 3.0}], words,
                                captions=[[{"text": "Kinich", "start": 1.0, "end": 1.5}]],
                                size=(1920, 1080), out_name="final.mp4", fade=0.0)
    assert out.name == "final.mp4" and seen["size"] == (1920, 1080)
    assert "Kinich" in seen["ass"] and "kinich" not in seen["ass"]


def test_loudnorm_filter_measures_then_applies_linear():
    assert auto.loudnorm_filter().endswith(":print_format=json")
    f = auto.loudnorm_filter({"input_i": "-19.1", "input_tp": "-3.5", "input_lra": "7.4",
                              "input_thresh": "-29.4", "target_offset": "0.2"})
    assert "I=-14.0" in f and "measured_I=-19.1" in f and "linear=true" in f


def test_join_words_glues_split_numbers_only():
    from refire.subtitles import join_words
    assert join_words(["i", "spent", "5", ",000!"]) == "i spent 5,000!"
    assert join_words(["1", ".5", "seconds"]) == "1.5 seconds"
    assert join_words(["wait", ",", "what", "..."]) == "wait , what ..."   # untouched
    assert join_words([]) == ""


def test_glossary_reuses_an_earlier_vods_list_without_the_llm(tmp_path, monkeypatch):
    from refire import glossary
    old = tmp_path / "111"
    old.mkdir()
    (old / "glossary_genshin-impact.json").write_text(json.dumps(["Mitya", "Nahida"]))
    monkeypatch.setattr(glossary, "_llm_terms", lambda *a: pytest.fail("llm called"))
    got = glossary.game_glossary("Genshin Impact", cache_dir=tmp_path / "222", extra="Valeri")
    assert got == ["Valeri", "Mitya", "Nahida"]
    assert json.loads((tmp_path / "222" / "glossary_genshin-impact.json").read_text()) == ["Mitya", "Nahida"]
    assert (tmp_path / "glossary_genshin-impact.json").exists()      # shared copy


def test_ticker_reports_elapsed_until_stopped():
    import time
    from refire.pipeline import _Ticker
    seen = []
    t = _Ticker(seen.append, "finding speech", every=0.05)
    time.sleep(0.18)
    t.stop()
    n = len(seen)
    time.sleep(0.12)
    assert seen[0] == "finding speech" and seen[1].startswith("finding speech (0m00s)")
    assert len(seen) == n >= 3
