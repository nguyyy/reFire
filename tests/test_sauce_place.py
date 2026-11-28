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
                          BUSY, place.config())
    emo = {e["asset"]: e for e in kept if e["type"] == "emote"}
    boom = next(e for e in kept if e["type"] == "boom")
    assert emo["huhcat"]["t"] == boom["t"] == 10.3 and emo["huhcat"]["end"] == emo["sajj"]["t"] == 11.2
    assert [e["id"] for e in kept] == ["b01", "e01", "e02"]
    ph[1] = phrase(1, 10.5, 10.6)                          # the newer one 0.3 s in: too short a flash, dropped
    kept, dropped = place.solve([ev(0, "emote", 0.8, "huhcat"), ev(1, "emote", 0.7, "sajj")], ph,
                                BUSY, place.config())
    assert [e["asset"] for e in kept] == ["huhcat"] and "on screen" in dropped[0]["why"]


def test_ops_tag_every_item_and_set_his_boom_level(tmp_path):
    class TM:
        ticks_per_frame = 4233600000

        def ticks(self, t):
            return round(t * 60) * self.ticks_per_frame
    kept = [{"id": "b01", "type": "boom", "t": 1.0, "end": 2.25}, {"id": "e01", "type": "emote", "asset": "huhcat", "t": 1.0, "end": 2.4}]
    o = place.ops(kept, TM(), {"huhcat": tmp_path / "huhcat.mov"}, place.config())["ops"]
    assert o[0] == {"op": "sauce_clear", "prefix": "refire:sauce:", "video": [4], "audio": [2]}
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
