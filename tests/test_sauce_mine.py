"""sauce mine (04-sauce.md section 3): project events, detectors, statistics. No Premiere, no video."""
import numpy as np

from refire.sauce import detect, mine, project


def item(**k):
    base = {"track": "V2", "t": 10.0, "end": 14.0, "name": "", "media": "", "src_in": 0.0, "src_out": 4.0,
            "adjustment": False, "nested": False, "fx": {}}
    return {**base, **k}


def tr(anchor, keys, static=100.0):
    return {"AE.ADBE Geometry2": {"Anchor Point": {"v": anchor, "k": []}, "Scale Height": {"v": static, "k": keys}}}


def test_his_punch_preset_is_one_segment_with_its_ramps():
    keys = [(0.0, 100.0), (0.4667, 190.0), (2.95, 200.0), (3.4833, 100.0)]
    s = project.segments(keys, 100.0, 3.72)
    assert len(s) == 1 and s[0]["ramp_in"] == 0.467 and s[0]["ramp_out"] == 0.533
    assert s[0]["first"] == 190.0 and s[0]["peak"] == 200.0 and s[0]["t1"] == 3.483


def test_a_trimmed_layer_keeps_only_its_visible_punches_and_a_static_scale_is_a_hard_cut():
    keys = [(-1.1, 100.0), (-0.633, 190.0), (0.0, 100.0), (0.667, 190.0), (5.767, 200.0), (6.3, 100.0)]
    s = project.segments(keys, 100.0, 6.32)
    assert len(s) == 1 and s[0]["t0"] == 0.0 and s[0]["ramp_in"] == 0.667
    assert project.segments([], 150.0, 2.0) == [{"t0": 0.0, "t1": 2.0, "ramp_in": 0.0, "ramp_out": 0.0,
                                                 "first": 150.0, "peak": 150.0}]
    assert project.segments([], 100.0, 2.0) == []


def test_classify_reads_his_tracks():
    boom = project.classify(item(track="A3", end=11.25, media="C:/x/vine-boom.mp3",
                                 fx={"vol": {"Level": {"v": 0.03528355062, "k": []}}}))
    assert boom[0]["type"] == "boom" and boom[0]["db"] == -14.0
    assert project.classify(item(track="A2", end=11.0, media="C:/x/lie-detector-meme.mp3"))[0]["type"] == "sfx:lie-detector-meme"
    assert project.classify(item(track="A4", end=200.0, media="C:/x/song.mp3")) == []          # music
    assert project.classify(item(track="V1", media="C:/run/a/b/ae/proxies/p_dnxhr_sq_1.000-22.740.mov")) == []
    emo = project.classify(item(media="C:/x/huhcat.gif", fx={"AE.ADBE Motion": {"Scale": {"v": 240.0, "k": []},
                                                                                "Position": {"v": [0.5, 0.5], "k": []}}}))
    assert emo[0]["type"] == "emote:huhcat" and emo[0]["scale"] == 240.0 and emo[0]["file"] == "C:/x/huhcat.gif"
    keys = [(0.0, 100.0), (0.4, 190.0), (3.0, 200.0), (3.5, 100.0)]
    assert project.classify(item(adjustment=True, fx=tr([0.0, 1.0], keys)))[0]["type"] == "punch"
    z = project.classify(item(adjustment=True, fx=tr([0.5, 0.5], keys)))[0]
    assert z["type"] == "zoom" and z["t"] == 10.0 and z["end"] == 13.5
    fx = {**tr([0.5, 0.5], [], 45.0), "AE.Impact_Camera_Shake_FX": {}}
    assert project.classify(item(adjustment=True, fx=fx))[0]["type"] == "fx"


def test_premiere_volume_is_linear_with_plus_15_db_at_one():
    assert project.db(0.0316227766) == -15.0 and project.db(0.0562341325) == -10.0 and project.db(0.17782794) == 0.0


def test_the_boom_matcher_finds_a_quiet_copy_in_noise():
    rng = np.random.default_rng(0)
    tpl = rng.standard_normal(800).astype(np.float32)
    x = rng.standard_normal(30000).astype(np.float32) * 0.5
    for at in (3000, 17000):
        x[at:at + 800] += 0.6 * tpl
    got = detect.peaks(detect.ncc(x, tpl), 1000, 0.5, 0.4)
    assert [round(t, 3) for t, _ in got] == [3.0, 17.0]
    assert np.allclose(detect._xcorr(x, tpl, block=4096)[:100], np.correlate(x[:899], tpl, "valid"), atol=1e-3)


def vid(events, words=(), dur=300.0, project=True):
    return {"key": "v", "title": "t", "duration": dur, "project": {"events": events} if project else None,
            "events": events, "words": list(words)}


def test_boom_timing_measures_the_pause_after_the_line():
    words = [{"text": "slime", "start": 9.0, "end": 9.5}, {"text": "next", "start": 12.0, "end": 12.4}]
    b = mine.boom_timing(vid([{"t": 9.8, "end": 11.05, "type": "boom"}], words))
    assert b == [{"t": 9.8, "word": "slime", "delta": 0.3, "over_speech": False}]
    b = mine.boom_timing(vid([{"t": 12.2, "end": 13.4, "type": "boom"}], words))
    assert b[0]["over_speech"] and b[0]["word"] == "next"
    padded = [{"text": "slime", "start": 9.0, "end": 11.5}]          # whisper billed the pause to the word
    b = mine.boom_timing(vid([{"t": 10.0, "end": 11.25, "type": "boom"}], padded))
    assert b == [{"t": 10.0, "word": "slime", "delta": 0.3, "over_speech": False}]


def test_profile_rates_gaps_and_cooccurrence():
    ev = [{"t": 5.0, "end": 6.25, "type": "boom", "source": "project", "db": -14.0},
          {"t": 5.0, "end": 9.0, "type": "punch", "ramp_in": 0.4, "ramp_out": 0.5, "scale": 190.0, "peak": 200.0},
          {"t": 100.0, "end": 101.25, "type": "boom", "source": "project", "db": -10.0},
          {"t": 100.05, "end": 102.0, "type": "emote:huhcat", "scale": 240.0, "position": [0.5, 0.5], "source": "project",
           "opacity_keys": [(0.0, 100.0), (1.9, 0.0)], "file": "x"}]
    p = mine.profile([vid(ev)])
    assert p["rates"]["boom"]["cold_open_per_min"] == 1.0 and p["rates"]["boom"]["body_per_min"] == round(1 / 4, 3)
    assert p["gaps"]["boom"]["median"] == 95.0
    assert p["cooccur"]["boom+punch"] == {"rate": 0.5, "n": 2} and p["cooccur"]["boom+emote"]["rate"] == 0.5
    assert p["emote"]["library"][0]["name"] == "huhcat" and p["boom"]["db"]["median"] == -12.0


def test_exemplar_windows_mark_the_event_and_labels_come_from_one_call():
    words = [{"text": w, "start": float(i), "end": i + 0.5} for i, w in enumerate("I did not just die to a slime".split())]
    assert mine.window(words, 4.0, w=2.0) == "not just [*] die to"
    rows = mine.exemplars([vid([{"t": 4.0, "end": 5.0, "type": "boom"}], words)])
    calls = []

    def ask(text, Out):
        calls.append(text)
        return Out.model_validate({"labels": [{"i": 0, "why": "dies to a slime right after denying it"}]})

    mine.label(rows, ask=ask)
    assert len(calls) == 1 and rows[0]["why"].startswith("dies")
    mine.label(rows, ask=ask)                                  # already labelled: no second call
    assert len(calls) == 1


def test_report_renders_with_detected_and_project_videos():
    ev = [{"t": 5.0, "end": 6.25, "type": "boom", "source": "project"}]
    vids = [vid(ev), vid([{"t": 50.0, "end": 51.25, "type": "boom", "source": "detected"}], project=False)]
    md = mine.report(vids, mine.profile(vids), mine.exemplars(vids),
                     {"t": {"boom": {"truth": 1, "found": 1, "false": 0},
                            "emote": {"truth": 0, "in_library": 0, "found": 0, "false": 0}}})
    assert "# Sauce style report" in md and "| t | 1/1 | 0 |" in md and "detected" in md
