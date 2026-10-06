from refire.select import budget_select, parse_duration


def test_parse_duration_forms():
    assert parse_duration("1200") == 1200.0
    assert parse_duration("20m") == 1200.0
    assert parse_duration("90s") == 90.0
    assert parse_duration("1.5h") == 5400.0
    assert parse_duration("20:00") == 1200.0
    assert parse_duration("1:00:00") == 3600.0


def _clip(start, end, score):
    return {"start": start, "end": end, "score": score}


def test_budget_fills_to_target_highest_score_first_chrono_out():
    clips = [
        _clip(0, 60, 5.0),     # 60s
        _clip(300, 360, 9.0),  # 60s, best
        _clip(120, 180, 7.0),  # 60s
        _clip(600, 660, 3.0),  # 60s, weakest
    ]
    picked, warning = budget_select(clips, target_s=120, tol=0.25)
    assert warning is None
    # greedy picks the two highest scores (9,7) = 120s, then stops
    assert {c["score"] for c in picked} == {9.0, 7.0}
    # output is chronological
    assert [c["start"] for c in picked] == [120, 300]


def test_budget_warns_on_shortfall_without_padding():
    clips = [_clip(0, 30, 8.0), _clip(100, 130, 6.0)]   # only 60s total
    picked, warning = budget_select(clips, target_s=600, tol=0.25)
    assert len(picked) == 2          # keeps everything it has
    assert warning is not None and "not padding" in warning


def test_budget_order_score():
    clips = [_clip(0, 60, 5.0), _clip(300, 360, 9.0)]
    picked, _ = budget_select(clips, target_s=120, tol=0.25, order="score")
    assert [c["score"] for c in picked] == [9.0, 5.0]


# --- the local scorer must never take down a run ---

def test_a_wedged_local_scorer_scores_zero_instead_of_killing_the_run(monkeypatch):
    """`narrative.cast` is not wrapped in try/except, and by the time it runs the
    director's and critic's Claude calls are already spent with outline.json unwritten.
    A dead/hung ollama must degrade to a 0 score, not raise."""
    import ollama

    from refire import score

    class _Wedged:
        def __init__(self, *a, **k):
            pass

        def chat(self, **k):
            raise TimeoutError("runner deadlocked on VRAM")

    monkeypatch.setattr(ollama, "Client", _Wedged)
    got = score.score_chunk({"text": "some words"}, brief="b", model="m")
    assert got["llm_score"] == 0.0
    assert got["reason"] == "scorer_unavailable"


def test_the_scorer_call_is_given_a_timeout(monkeypatch):
    """No timeout is what let a wedged runner hang the pipeline indefinitely."""
    import ollama

    from refire import score

    seen = {}

    class _Client:
        def __init__(self, *a, **k):
            seen.update(k)

        def chat(self, **k):
            return {"message": {"content": '{"score": 7, "reason": "ok"}'}}

    monkeypatch.setattr(ollama, "Client", _Client)
    got = score.score_chunk({"text": "w"}, brief="b", model="m")
    assert seen.get("timeout") == score.SCORE_TIMEOUT_S
    assert got["llm_score"] == 7.0


# --- cut points: where a shot can start and stop ---
# sentence snap is the real floor on shot length, so dense reels need boundaries that are
# still breaths but finer than a sentence

import pytest

from refire.select import (TRANSIENT_FLOOR_S, snap_to_phrase, snap_to_sentences,
                           snap_to_transient, speech_intervals)


def _run_words(spec):
    """spec = [(start, end, text), ...] -> whisper-shaped words."""
    return [{"text": t, "start": s, "end": e} for s, e, t in spec]


# two speech runs: 0.0-2.0 and 5.0-7.0 (a 3s silence splits them), no punctuation
_TWO_RUNS = _run_words([(0.0, 0.5, "one"), (0.6, 1.1, "two"), (1.2, 2.0, "three"),
                        (5.0, 5.5, "four"), (5.6, 6.1, "five"), (6.2, 7.0, "six")])


def test_phrase_snap_lands_on_the_breath():
    assert speech_intervals(_TWO_RUNS) == [(0.0, 2.0), (5.0, 7.0)]
    # a span starting mid-run pulls back to the run head and out to the run tail
    assert snap_to_phrase(_TWO_RUNS, 0.8, 1.5) == (0.0, 2.0)


def test_phrase_snap_never_lands_mid_word():
    """The whole safety claim: every boundary it returns is a word edge."""
    edges = {w["start"] for w in _TWO_RUNS} | {w["end"] for w in _TWO_RUNS}
    for a, b in [(0.8, 1.5), (5.2, 6.0), (0.0, 7.0), (1.9, 5.1)]:
        x, y = snap_to_phrase(_TWO_RUNS, a, b)
        assert x in edges or x == a
        assert y in edges or y == b


def test_phrase_snap_is_tighter_than_a_sentence_snap():
    """The point of the mode: a shot the sentence snap would inflate stays short."""
    ws = _run_words([(0.0, 0.5, "a"), (0.6, 1.1, "b"), (1.2, 1.8, "c"),
                     (1.9, 2.4, "d"), (2.5, 6.0, "end.")])
    sent = snap_to_sentences(ws, 0.6, 1.3)
    phr = snap_to_phrase(ws, 0.6, 1.3)
    assert sent[1] == 6.0                      # dragged out to the terminator
    assert phr[1] < sent[1]                    # phrase stops at the breath instead


def test_phrase_snap_respects_its_pad():
    # run head is 5s away, a 0.5s pad must not drag the cut back to it
    assert snap_to_phrase(_TWO_RUNS, 6.5, 6.8, max_pad=0.5)[0] == 6.5


def test_transient_out_point_lands_before_the_peak_resolves():
    ws = [{"text": "quiet", "start": 0.0, "end": 1.0, "rms": 10.0},
          {"text": "SCREAM", "start": 1.1, "end": 2.0, "rms": 900.0},
          {"text": "after", "start": 2.1, "end": 3.0, "rms": 12.0}]
    _a, b = snap_to_transient(ws, 0.0, 3.0, truncate=0.3)
    assert b == pytest.approx(1.7)             # the peak ends at 2.0, cut 0.3 early


def test_transient_never_produces_a_glitch_length_shot():
    ws = [{"text": "BANG", "start": 0.0, "end": 0.4, "rms": 900.0},
          {"text": "then", "start": 0.5, "end": 1.5, "rms": 5.0}]
    a, b = snap_to_transient(ws, 0.0, 1.5, truncate=0.3)
    assert b - a >= TRANSIENT_FLOOR_S          # 0.4-0.3 = 0.1s would be a glitch


def test_transient_degrades_to_phrase_without_audio():
    """A missing/unreadable wav leaves no rms; the cut must still be clean, not random."""
    assert snap_to_transient(_TWO_RUNS, 0.8, 1.5) == snap_to_phrase(_TWO_RUNS, 0.8, 1.5,
                                                                    max_pad=1.0)


def test_transient_searches_only_the_tail_for_its_peak():
    """A loud word early in a long span is the build-up, not the moment to cut on."""
    ws = ([{"text": "EARLY", "start": 0.0, "end": 0.5, "rms": 900.0}]
          + [{"text": f"w{i}", "start": 1.0 + i, "end": 1.8 + i, "rms": 10.0 + i}
             for i in range(10)])
    _a, b = snap_to_transient(ws, 0.0, 11.0, truncate=0.3)
    assert b > 5.0                             # not dragged back to the 0.5s peak
