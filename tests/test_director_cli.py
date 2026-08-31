"""`_complete_cli`: recovering an answer that Claude Code split across assistant turns.

The failure this guards is silent and expensive. When thinking eats the output budget the
message stops on `max_tokens` and the CLI continues in a SECOND assistant message -- but
the `result` event carries only that final turn, so a JSON answer arrives as a mid-object
fragment, the parse raises, and the whole run degrades to a cut nobody asked for. The
event shapes below are copied from the real session log of run `vivid-lynx`.
"""
import json
import subprocess
from types import SimpleNamespace

from refire import director

WHOLE = json.dumps({
    "central_idea": "he insists it will be a calm stream",
    "cold_open": [{"start_s": 10.0, "end_s": 12.0}],
    "beats": [{"title": "A", "role": "hook", "intent": "i", "query": "q",
               "start_s": 1.0, "end_s": 2.0,
               "segments": [{"start_s": 1.0, "end_s": 1.5}]}],
})
SPLIT = len(WHOLE) - 40          # tear the JSON mid-object, like a real max_tokens stop


def _assistant(text, stop):
    return json.dumps({"type": "assistant",
                       "message": {"content": [{"type": "text", "text": text}],
                                   "stop_reason": stop}})


def _result(text):
    return json.dumps({"type": "result", "result": text, "stop_reason": "end_turn"})


def _fake_cli(monkeypatch, lines):
    """Stub the `claude` binary: stdout replays `lines`, exit 0. Returns the recorded argv."""
    seen = {}

    class FakeProc:
        returncode = 0

        def __init__(self):
            self.stdin = SimpleNamespace(write=lambda s: seen.setdefault("stdin", s),
                                         close=lambda: None)
            self.stdout = iter(l + "\n" for l in lines)

        def wait(self):
            return 0

        def kill(self):
            seen["killed"] = True

    monkeypatch.setattr("shutil.which", lambda name: r"C:\fake\claude.exe")
    monkeypatch.setattr(subprocess, "Popen",
                        lambda cmd, **kw: (seen.setdefault("cmd", cmd), FakeProc())[1])
    return seen


def test_answer_split_across_turns_is_rejoined(monkeypatch, tmp_path):
    """The vivid-lynx failure: result holds only the tail, the head is in turn 1."""
    head, tail = WHOLE[:SPLIT], WHOLE[SPLIT:]
    _fake_cli(monkeypatch, [_assistant(head, "max_tokens"),
                            _assistant(tail, "end_turn"),
                            _result(tail)])            # <- result = final turn ONLY
    got = director._complete_cli("m", "sys", [{"text": "u"}], director.Outline,
                                 trace=tmp_path)
    assert got.central_idea == "he insists it will be a calm stream"
    assert len(got.cold_open) == 1 and got.beats[0].title == "A"
    # the trace must show it was two turns, or the next person debugging this is blind
    dump = next(tmp_path.glob("*-claude-response.txt")).read_text(encoding="utf-8")
    assert "TURNS: 2" in dump and "max_tokens" in dump


def test_complete_result_is_used_unchanged(monkeypatch, tmp_path):
    """Healthy single-turn call: `result` parses, so the join is never consulted."""
    _fake_cli(monkeypatch, [_assistant(WHOLE, "end_turn"), _result(WHOLE)])
    got = director._complete_cli("m", "sys", [{"text": "u"}], director.Outline,
                                 trace=tmp_path)
    assert got.beats[0].title == "A"


def test_result_wins_over_turn_narration(monkeypatch):
    """Drill-down mode (tools="Read") lets the model narrate between tool calls. That prose
    must never outrank a `result` that already parses."""
    _fake_cli(monkeypatch, [_assistant("Let me read {chapter_01}.", "end_turn"),
                            _assistant(WHOLE, "end_turn"),
                            _result(WHOLE)])
    got = director._complete_cli("m", "sys", [{"text": "u"}], director.Outline)
    assert got.beats[0].title == "A"


def test_unparseable_everywhere_raises(monkeypatch):
    """Nothing parseable in either place -> raise, so the caller aborts rather than
    quietly shipping the flat cut."""
    _fake_cli(monkeypatch, [_assistant("no json here", "end_turn"), _result("nope")])
    try:
        director._complete_cli("m", "sys", [{"text": "u"}], director.Outline)
    except RuntimeError as e:
        assert "no parseable answer" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


# --- the beat-count cap that triggered the whole thing --------------------
def test_pace_cannot_inflate_the_beat_ask_without_bound():
    assert director._n_beats(960.0, 1.0) == 20        # the natural count
    assert director._n_beats(960.0, 0.35) == 30       # was 57 -> 57k thinking tokens
    assert director._n_beats(3600.0, 1.0) == 75       # long-form is NOT clamped
    assert director._n_beats(960.0, 2.0) == 10        # slow pace still thins it out
    assert director._n_beats(30.0, 1.0) == 2          # floor holds
