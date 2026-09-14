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


def test_think_tick_is_an_inplace_ascii_heartbeat():
    """The subscription CLI streams `thinking_delta` with the text BLANKED, so these
    empty chunks are the only live signal during a multi-minute think. They must not
    spam lines (a 7-min call ticks ~260 times) and must not carry a glyph the cp1252
    console can't encode -- that raises mid-call and kills the run."""
    from refire.director import _think_tick

    t = _think_tick("director", 143.0, 1)
    assert t.startswith(chr(13))          # carriage return: overwrite, don't scroll
    assert chr(10) not in t               # never a newline
    assert t.isascii()                    # cp1252-safe
    assert "2m23s" in t and "director" in t

    assert "0m07s" in _think_tick("d", 7.4, 0)      # zero-padded seconds
    assert "7m18s" in _think_tick("d", 438.0, 0)    # the real director-call length
    # spinner advances and wraps at 4
    spins = [_think_tick("d", 1.0, b).strip()[-1] for b in range(5)]
    assert len(set(spins[:4])) == 4 and spins[4] == spins[0]


def test_blank_thinking_deltas_drive_a_live_tick(monkeypatch, capsys):
    """The whole point: the subscription CLI sends `thinking_delta` with the text
    blanked, on a ~1.6s heartbeat. The old `and d.get("thinking")` guard dropped every
    one, so the terminal showed nothing for ~6 of a director call's 7 minutes."""
    import json as _json
    import subprocess
    from pydantic import BaseModel
    from refire import director

    class Ans(BaseModel):
        answer: str

    def ev(delta):
        return _json.dumps({"type": "stream_event", "event": {"delta": delta}}) + "\n"

    lines = ([ev({"type": "thinking_delta", "thinking": ""}) for _ in range(5)]
             + [ev({"type": "text_delta", "text": '{"answer": '}),
                ev({"type": "text_delta", "text": '"done"}'}),
                _json.dumps({"type": "result", "result": '{"answer": "done"}',
                             "stop_reason": "end_turn"}) + "\n"])

    class _Proc:
        returncode = 0

        def __init__(self, *a, **k):
            self.stdin, self.stdout = _io(), iter(lines)

        def wait(self):
            return 0

    class _io:
        def write(self, _): pass
        def close(self): pass

    monkeypatch.setattr(director.shutil if hasattr(director, "shutil") else __import__("shutil"),
                        "which", lambda _: "claude")
    monkeypatch.setattr(subprocess, "Popen", _Proc)

    got = director._complete_cli("m", "sys", [{"type": "text", "text": "u"}], Ans)
    assert got.answer == "done"

    out = capsys.readouterr().out
    assert "--- thinking ---" in out and "--- writing ---" in out
    assert "thinking... 0m00s" in out          # the tick rendered, not swallowed
    assert out.count(chr(13)) >= 5             # one in-place tick per blank delta


REVIEW_JSON = json.dumps({
    "approved": True, "notes": "reads well",
    "outline": {"central_idea": "x", "beats": [
        {"title": "A", "role": "hook", "intent": "i", "query": "q",
         "start_s": 1.0, "end_s": 2.0}]},
})

_LOG = {"central_idea": "x", "beats": [
    {"title": "A", "intent": "i", "start": 3.0, "end": 5.0, "score": 9.0, "text": "hi"}]}


def test_director_opens_a_named_session_at_the_chosen_effort(monkeypatch):
    """Nothing set an effort level before; and without --session-id the review rounds have
    nothing to resume, which is what made every round re-send the whole stream map."""
    seen = _fake_cli(monkeypatch, [_result(WHOLE)])
    director.outline("[0s] map", "brief", "title", 600.0,
                     effort="max", session_id="11111111-2222-3333-4444-555555555555")
    cmd = seen["cmd"]
    assert cmd[cmd.index("--effort") + 1] == "max"
    assert cmd[cmd.index("--session-id") + 1] == "11111111-2222-3333-4444-555555555555"
    assert "--resume" not in cmd                 # the director OPENS it
    assert "--system-prompt-file" in cmd


def test_review_resumes_the_session_and_drops_the_map(monkeypatch):
    """The whole point: on a resume the ~240KB map is already in the conversation, so the
    round sends only the realized cut. The rubric rides in the user turn because
    --system-prompt-file cannot be swapped on a resumed session."""
    smap = ("[0s] a very long stream map line" + chr(10)) * 500
    seen = _fake_cli(monkeypatch, [_result(REVIEW_JSON)])
    rv = director.review(smap, "brief", "title", _LOG, 600.0,
                         session_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    assert rv.approved is True
    cmd = seen["cmd"]
    assert cmd[cmd.index("--resume") + 1] == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    assert "--system-prompt-file" not in cmd
    assert "--session-id" not in cmd
    assert smap not in seen["stdin"]             # the map is NOT re-sent
    assert "CURRENT ROUGH CUT" in seen["stdin"]  # ...but the new cut is
    assert "senior video editor" in seen["stdin"]   # ...and so is the rubric


def test_a_failed_resume_falls_back_to_a_full_call(monkeypatch):
    """A stale session must never cost a run that already spent a director call."""
    smap = ("[0s] map line" + chr(10)) * 500
    calls = []

    real = director._complete_cli

    def flaky(*a, **kw):
        calls.append(kw.get("resume", False))
        if kw.get("resume"):
            raise RuntimeError("no conversation found with session ID")
        return real(*a, **kw)

    monkeypatch.setattr(director, "_complete_cli", flaky)
    seen = _fake_cli(monkeypatch, [_result(REVIEW_JSON)])
    rv = director.review(smap, "brief", "title", _LOG, 600.0, session_id="dead-session")
    assert rv.approved is True
    assert calls == [True, False]                # tried warm, then re-sent everything
    assert smap in seen["stdin"]                 # the fallback carries the full map
