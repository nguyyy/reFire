"""Intra-clip trim: utterances, removal test, word-safe cut points, cap, exclude."""
from refire import trim
from refire.learn.diff import Context
from refire.select import compress_silence


def W(spec):
    return [{"text": t, "start": a, "end": b} for t, a, b in spec]


WORDS = W([("Did", 0.0, 0.3), ("you", 0.4, 0.6), ("see?", 0.7, 1.0),
           ("um", 2.0, 2.3), ("anyways", 2.4, 2.9),
           ("Yes!", 4.0, 4.5),
           ("loading", 6.0, 6.5), ("again", 6.6, 7.0),
           ("NO", 8.0, 8.4)])
LOUD = [(2.5, 0.0), (7.5, 2.0)]


def clip():
    return {"i": 0, "start": 0.0, "end": 9.0, "keep": [[0.0, 9.0]]}


def fake(labels):
    def judge(system, user):
        assert "S1 [" in user and "Reply ONLY with JSON" in system
        return {"spans": [{"id": k, **v} for k, v in labels.items()]}
    return judge


def test_utterances_split_on_silence():
    sp = trim.utterances([clip()], WORDS)
    assert [s["text"] for s in sp] == ["Did you see?", "um anyways", "Yes!", "loading again", "NO"]


def test_removal_test_needs_label_no_dependents_low_energy_and_cap():
    labels = {"S2": {"label": "filler", "needed_by": [], "confidence": 0.9},
              "S1": {"label": "setup", "needed_by": ["S3"], "confidence": 0.9},
              "S4": {"label": "off_topic", "needed_by": [], "confidence": 0.9},
              "S3": {"label": "filler", "needed_by": ["S5"], "confidence": 0.9}}
    ctx = Context(WORDS, loud=LOUD)
    out, log = trim.trim_clips([clip()], ctx, judge=fake(labels), max_frac=1.0)
    cut = out[0]["cut"]
    assert len(cut) == 1 and log["cut"] == 1     # S3 needed later, S4 is loud (z 2.0)
    a, b = cut[0]
    assert 1.0 < a < 2.0 and 2.9 < b < 4.0       # inside the silences, no word clipped
    assert all(not (a < w["start"] < b) or w["text"] in ("um", "anyways") for w in WORDS)
    out2, log2 = trim.trim_clips([clip()], ctx, judge=fake(labels), max_frac=0.0)
    assert "cut" not in out2[0] and log2["cut"] == 0


def test_pause_before_punchline_is_kept():
    labels = {"S4": {"label": "filler", "needed_by": [], "confidence": 0.9}}
    ctx = Context(WORDS, loud=[(6.5, 0.0), (8.2, 3.0)])
    out, _ = trim.trim_clips([clip()], ctx, judge=fake(labels), max_frac=1.0)
    a, b = out[0]["cut"][0]
    assert b < 7.2                                # the gap before the loud 'NO' survives


def test_failed_judge_keeps_everything():
    def boom(system, user):
        raise RuntimeError("ollama down")
    out, log = trim.trim_clips([clip()], Context(WORDS), judge=boom)
    assert "cut" not in out[0] and log["cut"] == 0


def test_compress_silence_exclude_drops_the_span_and_its_words():
    keep, retimed, dur = compress_silence(WORDS, 0.0, 9.0, pad=0.3, exclude=[[1.05, 3.95]])
    assert all(not (1.05 <= a and b <= 3.95) for a, b in keep)
    assert "um" not in [w["text"] for w in retimed]
    full = compress_silence(WORDS, 0.0, 9.0, pad=0.3)[2]
    assert dur < full
