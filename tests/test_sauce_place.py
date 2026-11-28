"""sauce P2 (04-sauce.md 5.2-5.3): director/critic plumbing with stubbed calls, the deterministic solver, ops."""
from refire.sauce import place, plan

PROF = {"boom": {"delta": {"median": 0.0}}, "emote": {"hold": {"median": 1.4}, "library": []},
        "rates": {"boom": {"body_per_min": 1.0}, "emote": {"body_per_min": 0.2}},
        "curve": {"boom": [{"t0": 0, "t1": 60, "per_min": 2.0, "n": 0}], "emote": [{"t0": 0, "t1": 60, "per_min": 0.5, "n": 0}]}}


BUSY = PROF | {"curve": {}, "rates": {"boom": {"body_per_min": 2.0}, "emote": {"body_per_min": 2.0}}}


def phrase(i, t0, t1, text="line"):
    return {"id": i, "t0": t0, "t1": t1, "text": text, "words": [{"text": "w", "t0": t0, "t1": t1}],
            "f": {"z_loud_peak": 0.0, "laugh_dur": 0.0, "lex": False, "caps": False}, "s": 0.0}


def ev(i, ty="boom", conf=0.8, asset=None):
    return {"phrase": i, "type": ty, "asset": asset, "confidence": conf, "reason": "r"}


def test_booms_snap_to_the_word_end_with_whispers_padding_clamped():
    p = phrase(0, 10.0, 12.0)                              # a 2 s "word" = a pause billed to it
    assert place.hit_time(p, 0.0) == 10.7 and place.hit_time(phrase(0, 10.0, 10.4), 0.1) == 10.5


def test_tau_density_and_no_overlapping_booms():
    ph = {i: phrase(i, 70.0 + 0.5 * i, 70.3 + 0.5 * i) for i in range(4)} | {9: phrase(9, 200, 200.3)}
    kept, dropped = place.solve([ev(0, conf=0.9), ev(1, conf=0.95), ev(9, conf=0.5)], ph, PROF, place.config())
    assert [e["phrase"] for e in kept] == [1] and {d["why"].split()[0] for d in dropped} == {"another", "confidence"}
    many = {i: phrase(i, 70.0 + 3 * i, 70.3 + 3 * i) for i in range(6)}
    kept, dropped = place.solve([ev(i, conf=0.7) for i in range(6)], many, PROF, place.config({"soft_cap": 99}))
    assert len(kept) == 2 and all("density" in d["why"] for d in dropped)   # ceil(1.5 x 1.0/min x 1 min) = 2


def test_a_paired_emote_lands_on_its_boom_and_a_newer_emote_cuts_the_older_short():
    ph = {0: phrase(0, 10.0, 10.3), 1: phrase(1, 10.9, 11.2)}
    kept, _ = place.solve([ev(0), ev(0, "emote", 0.7, "huhcat"), ev(1, "emote", 0.75, "sajj")], ph,
                          BUSY, place.config(), total=600.0)
    emo = {e["asset"]: e for e in kept if e["type"] == "emote"}
    boom = next(e for e in kept if e["type"] == "boom")
    assert emo["huhcat"]["t"] == boom["t"] == 10.3 and emo["huhcat"]["end"] == emo["sajj"]["t"] == 11.2
    assert [e["id"] for e in kept] == ["b01", "e01", "e02"]
    ph[1] = phrase(1, 10.5, 10.6)                          # the newer one 0.3 s in: too short a flash, dropped
    kept, dropped = place.solve([ev(0, "emote", 0.8, "huhcat"), ev(1, "emote", 0.7, "sajj")], ph,
                                BUSY, place.config(), total=600.0)
    assert [e["asset"] for e in kept] == ["huhcat"] and "on screen" in dropped[0]["why"]


def test_ops_tag_every_item_and_set_his_boom_level(tmp_path):
    class TM:
        ticks_per_frame = 4233600000

        def ticks(self, t):
            return round(t * 60) * self.ticks_per_frame
    kept = [{"id": "b01", "type": "boom", "t": 1.0, "end": 2.25}, {"id": "e01", "type": "emote", "asset": "huhcat", "t": 1.0, "end": 2.4}]
    o = place.ops(kept, TM(), {"huhcat": tmp_path / "huhcat.mov"}, place.config())["ops"]
    assert o[0] == {"op": "sauce_clear", "prefix": "refire:sauce:", "video": [1, 4], "audio": [2]}
    assert o[1]["audio"] and o[1]["track"] == 2 and abs(o[1]["level"] - 0.035481) < 1e-5 and o[1]["name"] == "refire:sauce:b01"
    assert not o[2]["audio"] and o[2]["track"] == 4 and o[2]["file"].endswith("huhcat.mov") and int(o[2]["start"]) == 60 * 4233600000


def test_library_skips_screenshots_and_carries_moods():
    prof = {"emote": {"library": [{"name": "huhcat", "file": "x.gif"}, {"name": "Screenshot 1", "file": "y.png"},
                                  {"name": "gone", "file": None}]}}
    lib = place.library(prof, [{"type": "emote", "asset": "huhcat", "why": "confused by the drop"}])
    assert [r["name"] for r in lib] == ["huhcat"] and lib[0]["moods"] == "confused by the drop"


def test_retrieve_prefers_rare_shared_words_and_covers_every_emote():
    rows = [{"type": "boom", "asset": None, "text": "the slime killed me again"},
            {"type": "boom", "asset": None, "text": "the the the the"},
            {"type": "emote", "asset": "sajj", "text": "nothing in common"}]
    got = plan.retrieve(rows, ["killed by a slime"], k=5)
    assert got[0]["text"].startswith("the slime") and any(r["asset"] == "sajj" for r in got)


def test_direct_retries_once_and_the_critic_can_only_lower_or_drop():
    data = {"phrases": [phrase(0, 1, 2, "I did not die"), phrase(1, 3, 4, "died")], "context": {}}
    lib = [{"name": "huhcat"}]
    prof = {"rates": {"boom": {"body_per_min": 1.0}, "emote": {"body_per_min": 0.2}}, "emote": {"hold": {"median": 1.4}}}
    answers = iter([{"events": [{"phrase": 1, "type": "emote", "asset": "nope", "confidence": 0.9}]},
                    {"events": [{"phrase": 1, "type": "boom", "confidence": 0.9, "reason": "dies right after"},
                                {"phrase": 0, "type": "emote", "asset": "huhcat", "confidence": 0.7}]},
                    {"verdicts": [{"i": 0, "keep": True, "confidence": 0.95}, {"i": 1, "keep": False, "confidence": 0.9}]}])
    texts = []

    def ask(text, Out):
        texts.append(text)
        return Out.model_validate(next(answers))

    evs = plan.direct(data, prof, lib, [], ask=ask, log=lambda *a: None)
    assert "not in the library" in texts[1] and [e["confidence"] for e in evs] == [0.9, 0.0]


def test_log_lists_placed_and_dropped_with_reasons():
    md = place.log_md([{"t": 61.5, "type": "boom", "confidence": 0.8, "text": "I'm him.", "reason": "hypocrisy"}],
                      [{"t": 5.0, "type": "emote", "asset": "sajj", "confidence": 0.4, "text": "x", "why": "confidence 0.4 < tau 0.6"}],
                      place.config())
    assert '- 01:01.50 boom (0.80) "I\'m him." - hypocrisy' in md and "emote+sajj (0.40)" in md and "tau 0.6" in md


ZPROF = PROF | {"rates": {**PROF["rates"], "zoom": {"body_per_min": 0.12}}, "curve": {**PROF["curve"], "zoom": []}}


def test_zooms_start_on_the_phrase_hold_2_to_7_s_stop_at_a_cut_and_skip_punches():
    ph = {0: phrase(0, 100.0, 101.0), 1: phrase(1, 200.0, 201.0), 2: phrase(2, 300.0, 301.0)}
    z = lambda i, d: {**ev(i, "zoom"), "target": "the boss", "duration": d}           # noqa: E731
    kept, dropped = place.solve([z(0, 10.0), z(1, 3.0), z(2, 3.0)], ph, ZPROF, place.config({"density": 99}),
                                total=400.0, cuts=[201.5], punches=[(299.0, 302.0)])
    assert [(e["t"], e["end"]) for e in kept] == [(100.0, 107.0)]                    # clamped to 7 s
    why = {d["phrase"]: d["why"] for d in dropped}
    assert "cut 1.5s in" in why[1] and "punch" in why[2]


def test_rare_types_get_a_whole_video_budget_not_one_per_bin():
    ph = {i: phrase(i, 100.0 + 60 * i, 101.0 + 60 * i) for i in range(5)}
    kept, dropped = place.solve([{**ev(i, "zoom"), "target": "x", "duration": 3} for i in range(5)], ph, ZPROF,
                                place.config(), total=600.0)
    assert len(kept) == place.budget(ZPROF, "zoom", 600.0, 1.5) == 2                   # ceil(1.5 x 0.12/min x 10 min)
    assert all("whole video" in d["why"] for d in dropped)


def test_zoom_window_stays_on_screen_and_skips_targets_too_big_to_zoom():
    from refire.sauce import locate
    c, s = locate.window([1800, 1000, 100, 60], 1920, 1080)                            # a small box in the corner
    assert s == 2.0 and c == [0.75, 0.75]                                              # clamped: no black edge
    assert locate.window([200, 100, 1400, 800], 1920, 1080) is None                     # under 130%: not worth it
    assert locate.inside([1400, 780, 40, 40], c, s, 1920, 1080) and not locate.inside([100, 100, 40, 40], c, s, 1920, 1080)


def test_zoom_ops_are_a_centered_transform_on_v2():
    class TM:
        ticks_per_frame = 4233600000

        def ticks(self, t):
            return round(t * 60) * self.ticks_per_frame
    kept = [{"id": "z01", "type": "zoom", "t": 2.0, "end": 5.0, "anchor": [0.7, 0.3], "scale": 1.85}]
    o = place.ops(kept, TM(), {}, place.config())["ops"]
    assert o[1]["op"] == "insert_adjustment" and o[1]["v"] == 1 and o[1]["name"] == "refire:sauce:z01"
    assert o[2]["params"][:4] == [[0, [0.7, 0.3]], [1, [0.5, 0.5]], [2, True], [3, 185.0]]


def test_zoom_check_is_one_call_over_every_crop(tmp_path):
    calls = []

    def ask(text, Out):
        calls.append(text)
        return Out.model_validate({"verdicts": [{"i": 0, "shows": True}, {"i": 1, "shows": False, "note": "UI only"}]})

    got = place.verify_zooms([{"target": "the boss"}, {"target": "the 143k number"}], tmp_path, "m", "cli", ask)
    assert len(calls) == 1 and "zoom_1.jpg - target: the 143k number" in calls[0]
    assert got == {0: (True, ""), 1: (False, "UI only")}
