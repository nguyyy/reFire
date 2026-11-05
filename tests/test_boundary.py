"""Boundary stage: thread-head starts, reaction-tail ends, lesson caps, merges."""
from refire import boundary
from refire.learn.diff import Context


def W(spec):
    return [{"text": t, "start": a, "end": b} for t, a, b in spec]


WORDS = W([("Earlier.", 0.0, 0.5),
           ("I", 5.0, 5.2), ("told", 5.3, 5.6), ("him", 5.7, 6.0), ("so", 6.1, 6.3),
           ("he", 6.35, 6.6), ("died.", 6.7, 7.0),
           ("No", 7.05, 7.3), ("way", 7.4, 7.6), ("dude.", 7.7, 8.0),
           ("Later.", 20.0, 20.5)])
LOUD = [(7.5, 2.5), (12.5, 0.0), (17.5, 0.0)]


def clip(i, a, b):
    return {"i": i, "start": a, "end": b, "keep": [[a, b]]}


def test_back_to_thread_head_and_tail_through_reaction():
    ctx = Context(WORDS, loud=LOUD)
    out, changed, log = boundary.refine([clip(0, 6.35, 7.0)], ctx)
    assert out[0]["start"] == 5.0            # 'he' opened mid-thought -> head after silence
    assert out[0]["end"] == 8.0              # loud + speech on -> end of the reaction line
    assert changed == {0} and log["back"] == 1 and log["tail"] == 1


def test_style_modes_and_cold_open_are_left_alone():
    ctx = Context(WORDS, loud=LOUD)
    out, _c, _l = boundary.refine([clip(0, 6.35, 7.0)], ctx, snap="transient",
                                  keep_build=False)
    assert (out[0]["start"], out[0]["end"]) == (6.35, 7.0)
    cold = {**clip(0, 6.35, 7.0), "section": "Cold Open"}
    assert boundary.refine([cold], ctx)[1] == set()


def test_caps_come_from_lessons_and_merge_needs_a_merge_lesson():
    ctx = Context(WORDS, loud=LOUD)
    tight = {"labels": {"start_late": {"count": 3, "p75_abs": 0.5},
                        "end_early": {"count": 3, "p75_abs": 0.1}}}
    out, changed, _l = boundary.refine([clip(0, 6.35, 7.0)], ctx, tight)
    assert changed == set()                  # every candidate is past the learned caps
    quiet = Context(WORDS)
    two = [clip(0, 10.0, 12.0), clip(1, 13.0, 15.0)]
    assert boundary.refine(two, quiet)[1] == set()
    merged = {"labels": {"merged": {"count": 2, "p75_abs": 2.0}}}
    out, changed, log = boundary.refine(two, quiet, merged)
    assert out[0]["end"] == 13.0 and log["merge"] == 1


def test_refine_sections_writes_back_in_place():
    sec = [{"title": "a", "clips": [{"start": 6.35, "end": 7.0}]}]
    log = boundary.refine_sections(sec, WORDS, Context(WORDS, loud=LOUD), {})
    assert sec[0]["clips"][0]["start"] == 5.0 and log["back"] == 1
