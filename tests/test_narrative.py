"""Casting logic (no network/Ollama): stub the director outline + embed/score."""
from types import SimpleNamespace

import pytest

from refire import narrative


def _beat(title, start_s=None, end_s=None):
    # default: no bounds -> casting falls back to query retrieval (the old path)
    return SimpleNamespace(title=title, intent=f"intent {title}",
                           query=f"query {title}", start_s=start_s, end_s=end_s)


def _outline(*titles):
    return SimpleNamespace(central_idea="the story",
                           beats=[_beat(t) for t in titles])


def _chunks():
    # sim decreases with index (query is [1,0]); score also decreases with index, so
    # casting is deterministic: earliest unused chunk wins each beat.
    return [{"start": i * 100.0, "end": i * 100.0 + 60.0, "text": f"c{i}",
             "embedding": [1.0 - i * 0.1, i * 0.1], "chat_z": 0.0, "s": 10 - i}
            for i in range(6)]


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr("refire.retrieve.embed", lambda texts, **k: [[1.0, 0.0]])
    monkeypatch.setattr("refire.score.score_chunk",
                        lambda c, brief="", model="": {"llm_score": float(c["s"]),
                                                       "reason": "r"})


# --- fallback path (no director bounds) -> retrieves chunks, like the old behavior ---

def test_one_clip_per_beat_in_order_no_reuse(stubbed):
    ol = _outline("A", "B", "C")
    sections, log, warning = narrative.cast(ol, _chunks(), [], target_s=1000.0,
                                            model="m", tol=0.25)
    assert [s["title"] for s in sections] == ["A", "B", "C"]      # story order kept
    assert all(len(s["clips"]) == 1 for s in sections)            # one clip per beat
    starts = [s["clips"][0]["start"] for s in sections]
    assert len(set(starts)) == 3                                  # no chunk reused
    assert log["central_idea"] == "the story"
    assert len(log["beats"]) == 3


def test_under_budget_warns_never_pads(stubbed):
    # 3 clips * 60s = 180s, target 1000s -> short -> warning, still 3 clips
    sections, log, warning = narrative.cast(_outline("A", "B", "C"), _chunks(), [],
                                            target_s=1000.0, model="m", tol=0.25)
    assert warning is not None and len(sections) == 3


def test_over_budget_trims_lowest_score(stubbed):
    # 180s of material, target 100s (ceiling 125 at tol .25) -> shed the lowest-scored beat
    # down to the tolerance ceiling, not the bare target (the goal, not a hard cap)
    sections, log, warning = narrative.cast(_outline("A", "B", "C"), _chunks(), [],
                                            target_s=100.0, model="m", tol=0.25)
    total = sum(s["clips"][0]["end"] - s["clips"][0]["start"] for s in sections)
    assert total <= 125.0 and len(sections) == 2   # one 60s beat dropped, 120s ships
    titles = [s["title"] for s in sections]
    assert "A" in titles and "C" not in titles     # highest-scored survives, lowest dropped


# --- primary path: the director picked explicit in/out timestamps ---

def _words(text):
    # one word per token; 1s apart, 0.8s long; `.?!` terminate sentences
    return [{"text": t, "start": i * 1.0, "end": i * 1.0 + 0.8}
            for i, t in enumerate(text.split())]


def test_cast_uses_director_bounds_without_retrieval(monkeypatch):
    calls = {"embed": 0}
    monkeypatch.setattr("refire.retrieve.embed",
                        lambda *a, **k: (calls.update(embed=calls["embed"] + 1), [[1.0, 0.0]])[1])
    monkeypatch.setattr("refire.score.score_chunk",
                        lambda c, brief="", model="": {"llm_score": 9.0, "reason": "r"})
    words = _words("a b c. d e f. g h i.")   # sentences end at idx 2 (2.8), 5 (5.8), 8 (8.8)
    ol = SimpleNamespace(central_idea="x", beats=[
        SimpleNamespace(title="T", intent="i", query="q", start_s=3.0, end_s=5.8)])
    sections, log, warning = narrative.cast(ol, [], words, target_s=1000.0, model="m")
    assert calls["embed"] == 0                       # retrieval skipped entirely
    clip = sections[0]["clips"][0]
    assert clip["start"] == pytest.approx(3.0)       # snapped to "d e f." sentence start
    assert clip["end"] == pytest.approx(5.8)         # ...and its terminator "f."
    assert log["beats"][0]["dir_start"] == 3.0       # director bounds recorded in the log


def test_lazy_chunks_callable_is_not_invoked_on_the_bounds_path(monkeypatch):
    """`chunks` may be a callable so the caller can defer a full embedding pass.

    On the normal path (director gave usable bounds) nothing needs candidate
    retrieval, so the provider must never be called -- that deferral is the whole
    point, and a refactor that eagerly resolves it silently costs a local embedding
    pass over the entire VOD on every run.
    """
    monkeypatch.setattr("refire.score.score_chunk",
                        lambda c, brief="", model="": {"llm_score": 9.0, "reason": "r"})
    words = _words("a b c. d e f. g h i.")

    def _boom():
        raise AssertionError("resolved chunks despite valid director bounds")

    ol = SimpleNamespace(central_idea="x", beats=[
        SimpleNamespace(title="T", intent="i", query="q", start_s=3.0, end_s=5.8)])
    sections, _log, _w = narrative.cast(ol, _boom, words, target_s=1000.0, model="m")
    assert sections[0]["clips"][0]["start"] == pytest.approx(3.0)


def test_lazy_chunks_callable_resolves_once_on_the_fallback_path(stubbed):
    """When a beat DOES need retrieval, the provider is called -- and only once."""
    calls = {"n": 0}

    def _provider():
        calls["n"] += 1
        return _chunks()

    ol = SimpleNamespace(central_idea="x", beats=[
        SimpleNamespace(title="A", intent="i", query="q", start_s=5.0, end_s=5.0),
        SimpleNamespace(title="B", intent="i", query="q", start_s=9.0, end_s=1.0)])
    sections, _log, _w = narrative.cast(ol, _provider, [], target_s=1000.0, model="m")
    assert len(sections) == 2
    assert calls["n"] == 1        # memoized across beats, not re-embedded per beat


def test_invalid_bounds_falls_back_to_retrieval(stubbed):
    # end_s <= start_s is unusable -> retrieve by query (chunk-based old path)
    ol = SimpleNamespace(central_idea="x", beats=[
        SimpleNamespace(title="A", intent="i", query="q", start_s=5.0, end_s=5.0)])
    sections, log, warning = narrative.cast(ol, _chunks(), [], target_s=1000.0, model="m")
    assert sections[0]["clips"][0]["start"] == 0.0   # cast the top retrieved chunk
    assert log["beats"][0]["dir_start"] is None      # logged as a fallback


def test_trim_to_budget_protects_the_arc_spine():
    """Over budget -> shed redundant low-score escalations, NOT the spine or the connective
    setup, even though the setup scores lowest of all."""
    from refire import narrative
    beats = [
        {"role": "hook", "score": 5.0, "start": 0.0, "end": 30.0},
        {"role": "setup", "score": 3.0, "start": 30.0, "end": 60.0},   # lowest score
        {"role": "escalation", "score": 9.0, "start": 60.0, "end": 90.0},
        {"role": "escalation", "score": 4.0, "start": 90.0, "end": 120.0},
        {"role": "climax", "score": 8.0, "start": 120.0, "end": 150.0},
        {"role": "button", "score": 6.0, "start": 150.0, "end": 180.0},
    ]  # 180s total, target 100 (ceiling 125 at tol .25) -> must drop 2x 30s beats
    kept, dropped = narrative._trim_to_budget(beats, target_s=100.0, tol=0.25)
    roles = [b["role"] for b in kept]
    assert roles.count("escalation") == 0        # both escalations shed first
    assert {"hook", "setup", "climax", "button"} <= set(roles)   # spine + setup survive
    assert sum(b["end"] - b["start"] for b in kept) <= 125.0     # within ceiling
    assert all(d["role"] == "escalation" for d in dropped)       # the shed beats surface


def test_trim_to_budget_ships_overshoot_within_tolerance():
    """A cut modestly over target is NOT trimmed -- the target is a goal, not a hard cap."""
    from refire import narrative
    beats = [{"role": "escalation", "score": 1.0, "start": s, "end": s + 20.0}
             for s in (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)]   # 120s
    kept, dropped = narrative._trim_to_budget(beats, target_s=100.0, tol=0.25)  # ceiling 125
    assert len(kept) == 6 and not dropped   # 120 <= 125, nothing dropped despite low scores


def test_stream_map_is_sentence_level():
    from refire import director
    words = [{"text": t, "start": s, "end": s + 0.4} for t, s in
             [("Hello", 0.0), ("world.", 0.5), ("Next", 65.0), ("one.", 65.5)]]
    # Timestamps are labelled in SECONDS (with an 's'), never mm:ss -- the director must
    # copy them straight into start_s/end_s, and [09:06] would be misread as 9.06s.
    assert director.stream_map(words) == "[0s] Hello world.\n[65s] Next one."


def test_stream_map_uses_seconds_past_an_hour():
    """A 2h+ moment must read as its second value, not a colon form that collapses to ~9s
    (the bug that made a 3.5h stream's cut land in the first 48s)."""
    from refire import director
    words = [{"text": "late.", "start": 8097.0, "end": 8097.4}]   # 2h14m57s
    assert director.stream_map(words) == "[8097s] late."


# --- director backends parse into the new Beat schema ---


def _fake_anthropic(monkeypatch, responses, output_format=None):
    """Stub anthropic.Anthropic; stream() yields `responses` ((pydantic|None), stop_reason)
    in order as JSON text blocks and records each call's kwargs into the returned `calls`.

    The director prompts for JSON and validates it itself (the strict structured-output
    endpoint 400s on the editorial schema's depth) over a streamed call (large max_tokens
    needs streaming), so a None response = an empty text block (thinking ate the budget),
    a plain `str` = that raw answer text verbatim (for a truncated reply), and a model =
    its `model_dump_json()`.

    Also hides the `claude` binary so the default `backend="cli"` takes its real
    CLI-missing fallthrough to this API stub instead of shelling out for real.
    """
    anthropic = pytest.importorskip("anthropic")
    from refire import director  # noqa: F401

    monkeypatch.setattr("shutil.which", lambda name: None)

    calls = []
    seq = iter(responses)

    class _Block:
        type = "text"

        def __init__(self, text):
            self.text = text

    class FakeStream:
        def __init__(self, msg):
            self._msg = msg

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __iter__(self):   # _complete streams deltas live; no events = silent stream
            return iter(())

        def get_final_message(self):
            return self._msg

    class FakeMessages:
        def stream(self, **kw):
            calls.append(kw)
            parsed, stop = next(seq)
            text = ("" if parsed is None else
                    parsed if isinstance(parsed, str) else parsed.model_dump_json())
            return FakeStream(SimpleNamespace(content=[_Block(text)], stop_reason=stop))

    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda *a, **k: SimpleNamespace(messages=FakeMessages()))
    return calls


def test_director_outline_plumbing(monkeypatch):
    """outline() returns the parsed Outline via a streamed adaptive-thinking call, with
    the stream map cache-marked (the retry + review rounds read it warm)."""
    from refire import director
    want = director.Outline(central_idea="x", beats=[
        director.Beat(title="A", intent="i", query="q", start_s=1.0, end_s=2.0)])
    calls = _fake_anthropic(monkeypatch, [(want, "end_turn")])
    got = director.outline("[00:00] hi", "brief", "title", 1800.0)   # 30-min target
    assert got.central_idea == want.central_idea and got.beats[0].title == "A"
    assert calls[0]["max_tokens"] == director.MAX_TOKENS
    # Opus 4.8: adaptive only; summarized display streams thinking to the console
    assert calls[0]["thinking"] == {"type": "adaptive", "display": "summarized"}
    blocks = calls[0]["messages"][0]["content"]
    assert "[00:00] hi" in blocks[0]["text"]              # map in the first block
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}


def test_director_retries_without_thinking_on_none(monkeypatch):
    """parsed_output None (thinking ate the budget) -> retry once with no thinking."""
    from refire import director
    want = director.Outline(central_idea="x", beats=[
        director.Beat(title="A", intent="i", query="q", start_s=1.0, end_s=2.0)])
    calls = _fake_anthropic(monkeypatch, [(None, "max_tokens"), (want, "end_turn")])
    got = director.outline("[00:00] hi", "brief", "title", 600.0)
    assert got.beats[0].title == "A"
    assert len(calls) == 2
    assert calls[0]["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert calls[1]["thinking"] == {"type": "disabled"}   # retry turns thinking off


def test_director_retries_on_truncated_max_tokens(monkeypatch):
    """Text present but stop_reason=max_tokens -> thinking ate the budget and the JSON got
    cut off mid-object. The API path has no auto-continuation to stitch it back, so it
    must take the same thinking-off retry as the empty case rather than parse a fragment."""
    from refire import director
    want = director.Outline(central_idea="x", beats=[
        director.Beat(title="A", intent="i", query="q", start_s=1.0, end_s=2.0)])
    cut = want.model_dump_json()[:-30]                    # a mid-object fragment
    calls = _fake_anthropic(monkeypatch, [(cut, "max_tokens"), (want, "end_turn")])
    got = director.outline("[00:00] hi", "brief", "title", 600.0)
    assert got.beats[0].title == "A"
    assert len(calls) == 2
    assert calls[1]["thinking"] == {"type": "disabled"}


def test_director_raises_when_no_outline(monkeypatch):
    """Both attempts empty -> raise with the stop_reason instead of returning None."""
    from refire import director
    _fake_anthropic(monkeypatch, [(None, "max_tokens"), (None, "max_tokens")])
    with pytest.raises(RuntimeError, match="max_tokens"):
        director.outline("[00:00] hi", "brief", "title", 600.0)


def test_complete_cli_parses_stream_json(monkeypatch):
    """The claude-CLI backend drives `claude -p`, streams deltas, parses the final
    `result` event's JSON (tolerating fences + interleaved non-JSON stderr lines),
    and never leaks ANTHROPIC_API_KEY into the subprocess env."""
    import io
    import json as _json

    from refire import director
    want = director.Outline(central_idea="x", beats=[
        director.Beat(title="A", intent="i", query="q", start_s=1.0, end_s=2.0)])
    lines = [
        _json.dumps({"type": "stream_event", "event": {
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "hm"}}}),
        _json.dumps({"type": "stream_event", "event": {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "{"}}}),
        "claude: some stderr noise",   # stderr merges into stdout; parser must skip it
        _json.dumps({"type": "result", "subtype": "success", "is_error": False,
                     "stop_reason": "end_turn",
                     "result": "```json\n" + want.model_dump_json() + "\n```"}),
    ]
    seen = {}

    class _FakeProc:
        def __init__(self):
            self.stdin = io.StringIO()
            self.stdout = iter(ln + "\n" for ln in lines)
            self.returncode = 0

        def wait(self):
            return self.returncode

    def fake_popen(cmd, **kw):
        seen["cmd"], seen["env"] = cmd, kw.get("env")
        return _FakeProc()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr("shutil.which", lambda name: "claude")
    monkeypatch.setattr("subprocess.Popen", fake_popen)
    got = director.outline("[0s] hi", "brief", "title", 600.0, backend="cli")
    assert got.beats[0].title == "A"
    # system prompt goes via --system-prompt-file (a bare --system-prompt with newlines
    # gets split by Windows argv parsing and silently corrupts the later flags)
    assert "-p" in seen["cmd"] and "--system-prompt-file" in seen["cmd"]
    assert "--system-prompt" not in seen["cmd"]
    sp_path = seen["cmd"][seen["cmd"].index("--system-prompt-file") + 1]
    assert sp_path.endswith(".txt")
    assert "--tools" in seen["cmd"] and "--setting-sources" in seen["cmd"]
    assert "ANTHROPIC_API_KEY" not in seen["env"]   # bill the subscription, not the key


def test_cli_backend_falls_through_to_api_when_missing(monkeypatch, capsys):
    """No `claude` on PATH -> the default cli backend quietly uses the API path."""
    from refire import director
    want = director.Outline(central_idea="x", beats=[
        director.Beat(title="A", intent="i", query="q", start_s=1.0, end_s=2.0)])
    calls = _fake_anthropic(monkeypatch, [(want, "end_turn")])   # also hides the CLI
    got = director.outline("[0s] hi", "brief", "title", 600.0)   # default backend="cli"
    assert got.beats[0].title == "A" and len(calls) == 1
    assert "claude CLI unavailable" in capsys.readouterr().out


def test_cast_reports_progress_per_beat(stubbed):
    seen = []
    narrative.cast(_outline("A", "B"), _chunks(), [], target_s=1000.0, model="m",
                   progress=lambda f, m: seen.append(m))
    assert len(seen) == 2 and "A" in seen[0] and "B" in seen[1]


def test_resolve_model_keeps_installed(monkeypatch):
    import ollama
    from refire import pipeline
    models = SimpleNamespace(models=[
        SimpleNamespace(model="llama3.1:8b", size=5_000_000_000),
        SimpleNamespace(model="nomic-embed-text:latest", size=274_000_000)])
    monkeypatch.setattr(ollama, "list", lambda: models)
    assert pipeline._resolve_ollama_model("llama3.1:8b") == "llama3.1:8b"
    assert pipeline._resolve_ollama_model("llama3.1") == "llama3.1:8b"   # base-name match


def test_resolve_model_autopicks_largest_non_embed(monkeypatch):
    import ollama
    from refire import pipeline
    models = SimpleNamespace(models=[
        SimpleNamespace(model="llama3.1:8b", size=5_000_000_000),
        SimpleNamespace(model="phi3:mini", size=2_000_000_000),
        SimpleNamespace(model="nomic-embed-text:latest", size=274_000_000)])
    monkeypatch.setattr(ollama, "list", lambda: models)
    # qwen2.5:14b absent -> biggest chat model, never the embedder
    assert pipeline._resolve_ollama_model("qwen2.5:14b") == "llama3.1:8b"


# --- editor-review loop: the critic reads the realized cut, then revises ---

def test_realized_script_renders_beats_in_order():
    """_realized_script feeds the critic the ACTUAL transcript, in order (the loop signal)."""
    from refire import director
    log = {"central_idea": "the idea", "beats": [
        {"title": "Open", "intent": "hook", "start": 0.0, "end": 5.0,
         "score": 9.0, "text": "first line"},
        {"title": "Close", "intent": "button", "start": 65.0, "end": 70.0,
         "score": 8.0, "text": "last line"}]}
    s = director._realized_script(log)
    assert s.index("Open") < s.index("Close")          # story order preserved
    assert "first line" in s and "last line" in s      # realized transcript, not the plan
    # source stamps say where each moment sat in the STREAM -- what the critic needs to
    # re-cut a beat
    assert "source 00:00-00:05" in s and "source 01:05-01:10" in s
    # ...and the cut position says where the VIEWER is when they reach it. Beat 2 plays at
    # 00:05 in the finished video even though it came from 01:05 of the stream; without
    # this the critic cannot see a sag, only a list of moments.
    assert "cut 00:00-00:05" in s and "cut 00:05-00:10" in s
    assert "PACING AUDIT" in s
    assert "the idea" in s


def test_review_returns_revised_outline_with_cached_map_first(monkeypatch):
    """review() parses a Review; the stream map is the FIRST (cache-marked) block so
    round 2 reads round 1's cache, and the realized cut follows in the second block."""
    from refire import director
    revised = director.Outline(central_idea="y", beats=[
        director.Beat(title="B", intent="i", query="q", start_s=1.0, end_s=2.0)])
    want = director.Review(approved=False, notes="flat; needs a closing button",
                           outline=revised)
    calls = _fake_anthropic(monkeypatch, [(want, "end_turn")], output_format=director.Review)
    log = {"central_idea": "x", "beats": [
        {"title": "A", "intent": "i", "start": 3.0, "end": 5.0, "score": 9.0,
         "text": "hi there"}]}
    got = director.review("[0s] the map", "brief", "title", log, 600.0)
    assert got.approved is False and got.outline.beats[0].title == "B"
    assert calls[0]["max_tokens"] == director.MAX_TOKENS
    blocks = calls[0]["messages"][0]["content"]
    assert "[0s] the map" in blocks[0]["text"]                # stable map first...
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert "hi there" in blocks[1]["text"]                    # ...realized cut after


def test_local_director_parses_ollama_json(monkeypatch):
    """outline_local() parses an Ollama JSON reply into an Outline (no server)."""
    import json

    ollama = pytest.importorskip("ollama")
    from refire import director

    payload = {"central_idea": "x", "beats": [
        {"title": "A", "intent": "i", "query": "q", "start_s": 1.0, "end_s": 2.0}]}
    monkeypatch.setattr(ollama, "chat",
                        lambda **kw: {"message": {"content": json.dumps(payload)}})
    got = director.outline_local("[00:00] hi", "brief", "title", 90.0)
    assert got.central_idea == "x" and got.beats[0].title == "A"


# --- editorial schema (full Beat) + payoff-aware bounds + cut plan ---

def test_beat_parses_full_and_minimal():
    """Full editorial Beat parses; a minimal one still does (defaults fill the rest)."""
    from refire import director
    full = director.Beat(title="T", role="climax", intent="i", viewer_question="q?",
                         turn="t", transition_in="after setup", texture="chaotic",
                         energy=5, setup_start_s=10.0, payoff_start_s=14.0,
                         reaction_end_s=17.0, start_s=12.0, end_s=16.0, query="q")
    assert full.role == "climax" and full.energy == 5 and full.payoff_start_s == 14.0
    lean = director.Beat(title="T", start_s=1.0, end_s=2.0)        # old shape, no editorial
    assert lean.role == "" and lean.energy == 3 and lean.reaction_end_s is None
    ol = director.Outline(central_idea="c", beats=[lean])
    assert ol.story_shape == "" and ol.viewer_promise == ""        # additive defaults


def _ed_beat(**kw):
    base = dict(title="T", role="hook", intent="i", viewer_question="q?", turn="t",
                transition_in="tr", texture="funny", energy=4, setup_start_s=None,
                payoff_start_s=None, reaction_end_s=None, start_s=0.0, end_s=1.0, query="q")
    base.update(kw)
    return SimpleNamespace(**base)


def _fixed_score(monkeypatch):
    monkeypatch.setattr("refire.score.score_chunk",
                        lambda c, brief="", model="": {"llm_score": 9.0, "reason": "r"})


def test_cast_logs_editorial_fields(monkeypatch):
    """cast() carries story_shape/viewer_promise + per-beat editorial fields into the log."""
    _fixed_score(monkeypatch)
    words = _words("a b c. d e f. g h i.")
    ol = SimpleNamespace(central_idea="idea", story_shape="confidence -> chaos",
                         viewer_promise="watch him spiral", ending_needed="absurd build",
                         beats=[_ed_beat(start_s=0.0, end_s=1.0)])
    sections, log, warning = narrative.cast(ol, [], words, target_s=1000.0, model="m")
    assert log["story_shape"] == "confidence -> chaos"
    assert log["viewer_promise"] == "watch him spiral"
    b = log["beats"][0]
    assert b["role"] == "hook" and b["viewer_question"] == "q?" and b["energy"] == 4


def test_cast_extends_end_for_reaction(monkeypatch):
    """reaction_end_s widens the out-point so the cut keeps the reaction tail."""
    _fixed_score(monkeypatch)
    words = _words("a b c. d e f. g h i.")     # terminators at 2.8, 5.8, 8.8
    # end_s alone snaps to 2.8 (sentence 1); reaction_end_s=4.0 pushes into sentence 2 -> 5.8
    ol = SimpleNamespace(central_idea="x", beats=[
        _ed_beat(start_s=0.0, end_s=1.0, reaction_end_s=4.0)])
    sections, log, warning = narrative.cast(ol, [], words, target_s=1000.0, model="m")
    assert sections[0]["clips"][0]["end"] == pytest.approx(5.8)


def test_final_beat_phrase_fallback_ends_on_pause(monkeypatch):
    """The final beat ends on a clean breath when whisper dropped the closing punctuation."""
    _fixed_score(monkeypatch)
    words = [{"text": "uh", "start": 0.0, "end": 0.5},
             {"text": "wait", "start": 0.5, "end": 1.0},     # speech run ends here (2s gap)
             {"text": "what", "start": 3.0, "end": 3.5}]
    ol = SimpleNamespace(central_idea="x", beats=[_ed_beat(start_s=0.0, end_s=0.9)])
    sections, log, warning = narrative.cast(ol, [], words, target_s=1000.0, model="m")
    assert sections[0]["clips"][0]["end"] == pytest.approx(1.0)   # the pause, not mid-word 0.9


def test_realized_script_includes_role_and_viewer_question():
    from refire import director
    log = {"central_idea": "idea", "story_shape": "x -> y", "beats": [
        {"title": "Open", "role": "hook", "intent": "grab", "viewer_question": "how?",
         "transition_in": "", "energy": 5, "start": 0.0, "end": 5.0, "score": 9.0,
         "text": "first line"}]}
    s = director._realized_script(log)
    assert "[hook]" in s and "how?" in s and "first line" in s and "x -> y" in s


def test_cut_plan_md_renders_beats():
    """cut_plan_md renders the editor's notebook: shape, role-tagged beats, mm:ss, notes."""
    log = {"central_idea": "the idea", "story_shape": "confidence -> chaos",
           "viewer_promise": "watch him spiral", "rounds": 1,
           "review": [{"round": 1, "approved": True, "notes": "tight"}],
           "beats": [{"title": "Cold Open", "role": "hook", "intent": "show chaos",
                      "viewer_question": "how did we get here?", "transition_in": "",
                      "texture": "chaotic", "energy": 5, "start": 134.0, "end": 156.0,
                      "score": 9.0, "reason": "what am I looking at"}]}
    md = narrative.cut_plan_md(log)
    assert "# Cut Plan" in md
    assert "Cold Open" in md and "(hook)" in md
    assert "02:14-02:36" in md                          # mm:ss clip bounds
    assert "confidence -> chaos" in md
    assert "Round 1 (approved): tight" in md


# --- line-level cutting: the director's segments become jump cuts within a beat ---

def _seg(a, b):
    return SimpleNamespace(start_s=a, end_s=b)


def test_beat_segments_parse_and_default_empty():
    from refire import director
    b = director.Beat(title="T", start_s=1.0, end_s=9.0,
                      segments=[{"start_s": 1.0, "end_s": 3.0},
                                {"start_s": 7.0, "end_s": 9.0}])
    assert b.segments[1].end_s == 9.0
    assert director.Beat(title="T", start_s=1.0, end_s=2.0).segments == []


def test_cast_segments_make_multiple_clips_and_cut_the_filler(monkeypatch):
    """A beat with segments renders as several tight clips (jump cuts); the rambling
    between them is gone from the kept text, and `dur` counts kept footage only."""
    _fixed_score(monkeypatch)
    words = _words("a b c. d e f. g h i. j k l. m n o.")   # terminators 2.8/5.8/8.8/11.8/14.8
    ol = SimpleNamespace(central_idea="x", beats=[
        _ed_beat(start_s=0.0, end_s=14.8,
                 segments=[_seg(0.0, 2.8), _seg(12.0, 14.8)])])
    sections, log, warning = narrative.cast(ol, [], words, target_s=1000.0, model="m")
    clips = sections[0]["clips"]
    assert len(clips) == 2
    assert clips[0]["end"] == pytest.approx(2.8)
    assert clips[1]["start"] == pytest.approx(12.0)
    b = log["beats"][0]
    assert b["dur"] == pytest.approx(5.6)                  # 2.8s + 2.8s kept, not 14.8s
    assert "d e f" not in b["text"]                        # the middle ramble is cut
    assert "a b c." in b["text"] and "m n o." in b["text"]


def test_cast_merges_segments_that_touch_after_snapping(monkeypatch):
    """Segments that snap to (near-)adjacent bounds fuse into one clip -- no stutter cut."""
    _fixed_score(monkeypatch)
    words = _words("a b c. d e f. g h i.")
    ol = SimpleNamespace(central_idea="x", beats=[
        _ed_beat(start_s=0.0, end_s=5.8,
                 segments=[_seg(0.0, 2.8), _seg(3.0, 5.8)])])   # 0.2s apart after snap
    sections, log, warning = narrative.cast(ol, [], words, target_s=1000.0, model="m")
    clips = sections[0]["clips"]
    assert len(clips) == 1
    assert clips[0]["end"] == pytest.approx(5.8)


def test_anchor_widening_is_clamped(monkeypatch):
    """A stray reaction_end_s can't silently re-inflate a deliberately tight cut: widening
    caps at WIDEN_SLOP past the director's own out-point (final beat gets more room)."""
    _fixed_score(monkeypatch)
    words = [{"text": "w", "start": float(i), "end": i + 0.8} for i in range(60)]
    ol = SimpleNamespace(central_idea="x", beats=[
        _ed_beat(start_s=0.0, end_s=10.0, reaction_end_s=40.0),   # wants +30s of "reaction"
        _ed_beat(start_s=50.0, end_s=55.0)])                      # final beat
    sections, log, warning = narrative.cast(ol, [], words, target_s=1000.0, model="m")
    assert sections[0]["clips"][0]["end"] == pytest.approx(10.0 + narrative.WIDEN_SLOP)


def test_realized_script_shows_cuts_and_dropped_beats():
    from refire import director
    log = {"central_idea": "i", "beats": [
        {"title": "T", "role": "hook", "intent": "i", "start": 0.0, "end": 30.0,
         "segments": [[0.0, 5.0], [20.0, 30.0]], "dur": 15.0, "score": 9.0, "text": "x"}],
        "dropped": [{"title": "Ley-Line Grind", "role": "escalation", "dur": 45.0}]}
    s = director._realized_script(log)
    assert "2 cuts" in s and "15s kept" in s        # the critic sees the edit, not the span
    assert "Ley-Line Grind" in s and "dropped" in s  # ...and what the budget trim shed


def test_cut_plan_md_shows_segment_cuts():
    log = {"central_idea": "i", "beats": [
        {"title": "T", "role": "climax", "intent": "i", "start": 0.0, "end": 60.0,
         "segments": [[0.0, 10.0], [40.0, 60.0]], "dur": 30.0, "score": 9.0}]}
    md = narrative.cut_plan_md(log)
    assert "2 segments, 30s kept" in md


# --- cold open: the one clip allowed to play out of chronological order ---

def _co_setup(monkeypatch):
    monkeypatch.setattr("refire.score.score_chunk",
                        lambda c, brief="", model="": {"llm_score": 9.0, "reason": "r"})
    return _words("a b c. " * 40)          # 120 words, 1s apart -> 0..120s


def _co_outline(cold, first_start=0.0):
    """Two beats plus an optional teaser. The second beat is the one being teased."""
    return SimpleNamespace(
        central_idea="x", cold_open=cold,
        beats=[SimpleNamespace(title="Open", intent="i", query="q",
                               start_s=first_start, end_s=first_start + 11.8),
               SimpleNamespace(title="Peak", intent="i", query="q",
                               start_s=60.0, end_s=71.8)])


def test_cold_open_plays_first_out_of_stream_order(monkeypatch):
    words = _co_setup(monkeypatch)
    ol = _co_outline(SimpleNamespace(start_s=63.0, end_s=66.0))
    sections, log, _ = narrative.cast(ol, [], words, target_s=1000.0, model="m")
    assert [s["title"] for s in sections] == ["Cold Open", "Open", "Peak"]
    co = sections[0]["clips"][0]
    # it is a FLASH-FORWARD: source-wise it comes from after the beat that follows it
    assert co["start"] == pytest.approx(63.0) and co["start"] > sections[1]["clips"][0]["start"]
    assert log["cold_open"][0]["dur"] == pytest.approx(co["end"] - co["start"])


def test_cold_open_is_clamped_to_a_teaser(monkeypatch):
    words = _co_setup(monkeypatch)
    ol = _co_outline(SimpleNamespace(start_s=63.0, end_s=110.0))   # a whole scene
    sections, _, _ = narrative.cast(ol, [], words, target_s=1000.0, model="m")
    co = sections[0]["clips"][0]
    assert co["end"] - co["start"] <= narrative.COLD_OPEN_MAX_S + 2.0   # +snap pad


def test_cold_open_the_cut_never_delivers_is_dropped(monkeypatch):
    """A teaser of footage no beat contains is a promise the video breaks."""
    words = _co_setup(monkeypatch)
    ol = _co_outline(SimpleNamespace(start_s=95.0, end_s=98.0))   # inside no beat
    sections, log, _ = narrative.cast(ol, [], words, target_s=1000.0, model="m")
    assert [s["title"] for s in sections] == ["Open", "Peak"]
    assert log["cold_open"] == []


def test_cold_open_next_to_the_opening_beat_is_a_stutter_not_a_promise(monkeypatch):
    words = _co_setup(monkeypatch)
    # first beat starts at 54s; teasing 63s is 9s ahead of itself
    ol = _co_outline(SimpleNamespace(start_s=63.0, end_s=66.0), first_start=54.0)
    sections, _, _ = narrative.cast(ol, [], words, target_s=1000.0, model="m")
    assert [s["title"] for s in sections] == ["Open", "Peak"]


def test_no_cold_open_leaves_the_cut_untouched(monkeypatch):
    words = _co_setup(monkeypatch)
    ol = _co_outline(None)
    sections, log, _ = narrative.cast(ol, [], words, target_s=1000.0, model="m")
    assert [s["title"] for s in sections] == ["Open", "Peak"]
    assert log["cold_open"] == []


def test_cold_open_shifts_the_pacing_clock(monkeypatch):
    """Beats play later in the finished video than their kept footage alone implies."""
    from refire.pacing import cut_timeline
    log = {"cold_open": {"start": 63.0, "end": 68.0, "dur": 5.0},
           "beats": [{"title": "A", "dur": 30.0, "start": 0.0, "end": 30.0, "text": "x"}]}
    assert cut_timeline(log)[0]["at"] == 5.0


# --- honest beat durations -------------------------------------------------

def _gapped_words():
    """Two sentences with 20s of dead air between them -- the shape `compress_silence`
    exists to remove, and which the renderer WILL remove before anyone sees the cut."""
    return ([{"text": t, "start": i * 1.0, "end": i * 1.0 + 0.8}
             for i, t in enumerate(["a", "b", "c."])]
            + [{"text": t, "start": 23.0 + i, "end": 23.8 + i}
               for i, t in enumerate(["d", "e", "f."])])


def test_beat_dur_is_the_compressed_length_the_renderer_will_ship(monkeypatch):
    """`dur` drives the pacing audit, the critic's realized script and the budget trimmer.

    Measuring the raw spans made every beat read longer than it ships (one real run
    audited a 16:33 cut as 21:28), so the audit flagged beats that were never long and
    the trimmer shed beats it had room for. The gap must not count.
    """
    monkeypatch.setattr("refire.score.score_chunk",
                        lambda c, brief="", model="": {"llm_score": 9.0, "reason": "r"})
    words = _gapped_words()
    ol = SimpleNamespace(central_idea="x", beats=[
        SimpleNamespace(title="T", intent="i", query="q", start_s=0.0, end_s=25.8)])

    _s, tight, _w = narrative.cast(ol, [], words, target_s=1000.0, model="m")
    _s, raw, _w = narrative.cast(ol, [], words, target_s=1000.0, model="m",
                                 deadspace=False)

    span = raw["beats"][0]["end"] - raw["beats"][0]["start"]
    # --no-deadspace: nothing is compressed downstream either, so the span IS the length
    assert raw["beats"][0]["dur"] == pytest.approx(span)
    # default: the 20s of dead air never reaches the viewer, so it must not be counted
    assert tight["beats"][0]["dur"] < span - 15.0


# --- the role budget reaches the prompts -----------------------------------

def test_both_system_prompts_survive_format_with_the_role_budget_spliced_in():
    """The budget table is injected into strings that are later `.format(n=...)`ed.

    A literal brace in that table raises at call time, inside the one code path that
    costs a Claude call to reach -- so it gets asserted here instead.
    """
    from refire import director

    roles = director._role_note(director.CUT_SHRINK)
    for system in (director.pick_system("a brief"), director.pick_system(""),
                   director._REVIEW_SYSTEM):
        rendered = system.format(n=20, roles=roles)   # must not raise
        assert "escalation 39-117s" in rendered       # ...and the budget actually arrived
        assert "climax 58-169s" in rendered


def test_the_director_is_given_role_budgets_in_span_seconds_not_finished_seconds():
    """ROLE_BUDGET is finished video; the director picks spans that get compressed.

    Handed the table unscaled it would pick a 90s span for a 90s ceiling and ship ~69s --
    under-filling every beat by exactly the compression it cannot see.
    """
    from refire import director
    from refire.pacing import ROLE_BUDGET

    lo, hi = ROLE_BUDGET["escalation"]
    assert director._role_note(1.0) == director._role_note()        # 1.0 is the identity
    assert f"escalation {lo}-{hi}s" in director._role_note(1.0)     # canonical, finished
    scaled = director._role_note(0.5)                               # spans, inflated
    assert f"escalation {lo * 2}-{hi * 2}s" in scaled


def test_the_header_asks_for_more_span_than_the_finished_target():
    """A 14-minute ask used to ship under 11: the director aimed its SPANS at the target,
    and compression then took ~23% off. It has to be told both numbers."""
    from refire import director

    h = director._header("b", "t", 840.0, shrink=0.77)
    assert "~840s of FINISHED video" in h
    assert "~1090s" in h            # 840 / 0.77 -- the span budget to actually hit it
    assert "23%" in h               # ...and why it is bigger
    # shrink=1.0 means "no compression downstream", so no second number to explain
    assert "SPAN BUDGET" not in director._header("b", "t", 840.0)


def test_cast_reports_the_shrink_this_stream_realized(monkeypatch):
    """The self-calibration signal: review rounds get this instead of the constant prior."""
    monkeypatch.setattr("refire.score.score_chunk",
                        lambda c, brief="", model="": {"llm_score": 9.0, "reason": "r"})
    words = _gapped_words()
    ol = SimpleNamespace(central_idea="x", beats=[
        SimpleNamespace(title="T", intent="i", query="q", start_s=0.0, end_s=25.8)])

    _s, log, _w = narrative.cast(ol, [], words, target_s=1000.0, model="m")
    span = log["beats"][0]["end"] - log["beats"][0]["start"]
    assert log["shrink"] == pytest.approx(log["beats"][0]["dur"] / span, abs=1e-3)
    assert 0 < log["shrink"] < 1                      # the gap cost real time

    # --no-deadspace: nothing is compressed, so a span IS its finished length
    _s, raw, _w = narrative.cast(ol, [], words, target_s=1000.0, model="m",
                                 deadspace=False)
    assert raw["shrink"] == pytest.approx(1.0)


def test_shrink_is_none_rather_than_invented_when_there_is_nothing_to_measure():
    assert narrative._realized_shrink([]) is None


# --- direction (--style) and cut speed (--pace) --------------------------

def test_the_styled_system_prompt_also_survives_format():
    """`_STYLE_RULE` is spliced into a string that is later `.format(n=..., roles=...)`ed.

    A literal brace in it would raise inside the one code path that costs a Claude call
    to reach -- the same trap the unstyled prompts are guarded against above.
    """
    from refire import director

    roles = director._role_note(director.CUT_SHRINK)
    for system in (director.pick_system("a brief", "highlight reel"),
                   director.pick_system(None, "highlight reel")):
        rendered = system.format(n=20, roles=roles)   # must not raise
        assert "escalation 39-117s" in rendered


def test_dense_styles_are_told_the_shot_length_they_are_graded_on():
    """`pacing.audit` grades mean shot length against `AVG_SHOT_S * pace`, but only when
    build-up is deliberately skipped. Left unstated the director picked 14s-mean segments
    against a 4.2s ceiling -- a floor with no ceiling is how a dense style ships slow
    shots. Told exactly where measured, so instruction and enforcement cannot drift.
    """
    from refire import director
    from refire.pacing import AVG_SHOT_S

    dense = director.pick_system("b", "fast", keep_build=False, pace=0.35)
    assert f"AVERAGE about {AVG_SHOT_S * 0.35:.1f} seconds" in dense
    assert "at least ~1.05 seconds" in dense            # floor still scales too
    dense.format(n=20, roles="x")                       # brace-free, or the run dies

    # keep_build=True is where the audit measures the OPPOSITE flag, so stating a
    # ceiling there would constrain a cut nothing checks.
    assert "AVERAGE about" not in director.pick_system("b", "fast", keep_build=True)
    assert "AVERAGE about" not in director.pick_system("b")


def test_a_style_replaces_the_arc_mandate_instead_of_arguing_with_it():
    """Without a style the prompt forbids the alternative outright ("that is a highlight
    reel, not a story"), so asking for one would be arguing with the instructions."""
    from refire import director

    default = director.pick_system("a brief")
    styled = director.pick_system("a brief", "highlight reel, loudest first")
    assert "highlight reel, not a story" in default
    assert "highlight reel, not a story" not in styled
    assert "DIRECTION above" in styled
    assert "DIRECTION above" not in default


def test_the_styled_prompt_keeps_the_ending_and_the_beat_schema():
    """Only the STRUCTURE rule is swapped. The ending-must-breathe clause is not about
    structure, and dropping the schema would break every beat the director emits."""
    from refire import director

    styled = director.pick_system("a brief", "highlight reel")
    assert "reaction_end_s" in styled and "BREATHE" in styled
    for field in ("start_s", "segments", "payoff_start_s", "texture", "energy"):
        assert field in styled


def test_style_is_identity_when_absent():
    """Every existing run passes style=None; those prompts must not move at all."""
    from refire import director

    assert director.pick_system("b") == director._SYSTEM + director._MAP_LEGEND
    assert director.pick_system("b", None) == director.pick_system("b")
    assert director.pick_system(None, None) == director.pick_system(None)


def test_the_stream_driven_mode_survives_the_structure_swap():
    """pick_system slices its no-brief variant out of the prompt by text; the swap must
    not break that surgery."""
    from refire import director

    free = director.pick_system(None, "highlight reel")
    assert free.startswith("You are a video editor cutting a Twitch")
    assert "NO editorial brief was given" in free
    assert "DIRECTION above" in free


def test_the_header_carries_the_direction_next_to_the_brief():
    """One line in _header is the whole free-text mechanism -- outline, review and the
    local director all route through it."""
    from refire import director

    h = director._header("hu tao pulls", "t", 600.0, style="fast, favor loud")
    assert "EDITOR'S BRIEF: hu tao pulls" in h
    assert "DIRECTION (how to cut it): fast, favor loud" in h
    assert "DIRECTION" not in director._header("hu tao pulls", "t", 600.0)


def test_pace_scales_the_budget_the_director_is_told():
    """--style says "fast cuts" in words; --pace moves the numbers the prompt states
    literally, which is the only thing an adjective has no leverage over."""
    from refire import director

    assert director._role_note(1.0, 1.0) == director._role_note(1.0)   # identity
    assert "climax 36-104s" in director._role_note(1.0, 0.8)           # 45,130 x 0.8
    assert "climax 68-195s" in director._role_note(1.0, 1.5)


def test_the_budget_stops_shrinking_where_the_beat_count_is_capped():
    """Below the knee the beat COUNT is clamped (`BEAT_INFLATION_CAP`), so shrinking the
    budgets any further can only ship a shorter video -- which is what 8:53 against a
    16:00 target was. `pacing.fill_pace` holds them at the floor instead."""
    from refire import director
    from refire.pacing import BEAT_INFLATION_CAP

    knee = 1.0 / BEAT_INFLATION_CAP
    assert director._role_note(1.0, 0.35) == director._role_note(1.0, knee)
    assert "climax 30-87s" in director._role_note(1.0, 0.35)   # 45,130 x 2/3, not x 0.35


def test_beat_count_and_budget_fill_the_target():
    """The invariant the two knobs exist to preserve: count x per-beat budget = runtime,
    at any pace. Broken, `--pace 0.35` could not reach a 16-minute target even with every
    beat at its ceiling."""
    from refire import director
    from refire.pacing import fill_pace, scaled_budget

    for target in (600.0, 960.0, 1800.0):
        for pace in (0.35, 0.5, 1.0, 1.5):
            budget = scaled_budget(fill_pace(pace))
            mid = sum((lo + hi) / 2 for lo, hi in budget.values()) / len(budget)
            fill = director._n_beats(target, pace) * mid
            assert 0.75 * target <= fill <= 1.35 * target, (target, pace, fill)


def test_pace_moves_the_beat_count_with_the_budgets():
    """Shrinking the per-beat budgets alone would just ship a SHORTER video; the count
    has to scale with them or the two fight (the SEC_PER_BEAT note records that bug)."""
    from refire import director

    assert director._n_beats(900.0, 1.0) == director._n_beats(900.0)
    assert director._n_beats(900.0, 0.6) > director._n_beats(900.0)
    assert director._n_beats(900.0, 1.5) < director._n_beats(900.0)
    assert director._n_beats(60.0, 0.05) >= 2        # floor holds at any pace
    assert director._n_beats(900.0, 0.0) >= 2        # ...and does not divide by zero


# --- montage stack: the cold open as several unexplained moments ---------

def _stack_outline(*spans, first_start=0.0, energies=(3, 3)):
    """Two beats plus a cold_open LIST. Beat 2 (60-71.8s) is where the stack is lifted."""
    return SimpleNamespace(
        central_idea="x",
        cold_open=[SimpleNamespace(start_s=a, end_s=b) for a, b in spans],
        beats=[SimpleNamespace(title="Open", intent="i", query="q", energy=energies[0],
                               start_s=first_start, end_s=first_start + 11.8),
               SimpleNamespace(title="Peak", intent="i", query="q", energy=energies[1],
                               start_s=60.0, end_s=71.8)])


def test_stack_plays_every_moment_in_one_section(monkeypatch):
    words = _co_setup(monkeypatch)
    ol = _stack_outline((62.0, 64.0), (66.0, 68.0), (70.0, 71.5))
    sections, log, _ = narrative.cast(ol, [], words, target_s=1000.0, model="m", stack=8)
    assert sections[0]["title"] == "Cold Open"
    # one section, several clips -- build_manifest renders that as jump cuts, which is
    # exactly what a stack is
    assert len(sections[0]["clips"]) == 3
    assert len(log["cold_open"]) == 3


def test_stack_moments_are_flashes_not_scenes(monkeypatch):
    words = _co_setup(monkeypatch)
    ol = _stack_outline((62.0, 71.0), (66.0, 71.5))     # both asked for whole scenes
    sections, _, _ = narrative.cast(ol, [], words, target_s=1000.0, model="m", stack=8)
    for c in sections[0]["clips"]:
        assert c["end"] - c["start"] <= narrative.STACK_MAX_S + 0.5   # + the phrase pad


def test_stack_escalates_so_the_best_moment_lands_last(monkeypatch):
    """Weakest first is the whole point -- a stack that opens on its best moment has
    nowhere to climb and reads as a normal intro."""
    words = _co_setup(monkeypatch)
    # 2s lifted from the energy-5 beat, 66s lifted from the energy-2 beat
    ol = _stack_outline((62.0, 64.0), (2.0, 4.0), first_start=0.0, energies=(2, 5))
    sections, _, _ = narrative.cast(ol, [], words, target_s=1000.0, model="m", stack=8)
    starts = [c["start"] for c in sections[0]["clips"]]
    assert starts[-1] == pytest.approx(62.0)        # the energy-5 moment is LAST
    assert starts[0] == pytest.approx(2.0)


def test_stack_drops_a_moment_the_cut_never_delivers(monkeypatch):
    words = _co_setup(monkeypatch)
    ol = _stack_outline((62.0, 64.0), (95.0, 97.0))   # 95s is inside no beat
    sections, _, _ = narrative.cast(ol, [], words, target_s=1000.0, model="m", stack=8)
    assert len(sections[0]["clips"]) == 1


def test_stack_is_capped_at_the_requested_count(monkeypatch):
    words = _co_setup(monkeypatch)
    ol = _stack_outline(*[(60.0 + i, 61.5 + i) for i in range(8)])
    sections, _, _ = narrative.cast(ol, [], words, target_s=1000.0, model="m", stack=3)
    assert len(sections[0]["clips"]) == 3


def test_stack_zero_is_the_teaser_path_untouched(monkeypatch):
    """The default must not become "a 1-moment stack" -- it keeps the teaser's own
    ceiling and its sentence snap."""
    words = _co_setup(monkeypatch)
    ol = _stack_outline((63.0, 74.0))
    sections, _, _ = narrative.cast(ol, [], words, target_s=1000.0, model="m")
    co = sections[0]["clips"][0]
    assert co["end"] - co["start"] > narrative.STACK_MAX_S          # teaser, not a flash
    assert co["end"] - co["start"] <= narrative.COLD_OPEN_MAX_S + 2.0


def test_stack_shifts_the_pacing_clock_by_its_whole_length():
    from refire.pacing import cut_timeline

    log = {"cold_open": [{"start": 62.0, "end": 64.0, "dur": 2.0},
                         {"start": 66.0, "end": 68.0, "dur": 2.0}],
           "beats": [{"title": "A", "dur": 30.0, "start": 0.0, "end": 30.0, "text": "x"}]}
    assert cut_timeline(log)[0]["at"] == 4.0
    # an outline from before cold_open was a list still measures
    old = {"cold_open": {"start": 62.0, "end": 67.0, "dur": 5.0}, "beats": log["beats"]}
    assert cut_timeline(old)[0]["at"] == 5.0


# --- the Outline schema tolerates both shapes ----------------------------

def test_outline_accepts_a_bare_cold_open_object():
    """An older outline.json, or a model that ignores the list shape, must not fail the
    whole call."""
    from refire.director import Outline

    beats = [{"title": "A", "start_s": 0.0, "end_s": 10.0}]
    assert Outline(central_idea="x", cold_open={"start_s": 1.0, "end_s": 4.0},
                   beats=beats).cold_open[0].start_s == 1.0
    assert Outline(central_idea="x", cold_open=None, beats=beats).cold_open == []
    assert Outline(central_idea="x", beats=beats).cold_open == []
    assert len(Outline(central_idea="x", beats=beats,
                       cold_open=[{"start_s": 1.0, "end_s": 2.0},
                                  {"start_s": 5.0, "end_s": 6.0}]).cold_open) == 2


# --- the stack rewrites the director's teaser paragraph ------------------

def test_stack_swaps_the_teaser_paragraph_for_a_stack_rule():
    from refire import director

    plain = director.pick_system("b")
    stacked = director.pick_system("b", stack=8)
    assert "flash-forward teaser, 3-6 seconds" in plain
    assert "flash-forward teaser, 3-6 seconds" not in stacked
    assert "MONTAGE STACK of about 8 moments" in stacked
    assert "the single best moment LAST" in stacked
    # the swap must not eat the beat schema that follows it
    assert "Then break it into BEATS" in stacked
    assert plain.format(n=5, roles="r") and stacked.format(n=5, roles="r")   # brace-free


def test_stack_without_a_style_keeps_the_arc():
    """--stack is a cold-open shape, not a licence to drop the story arc; only --style
    swaps that."""
    from refire import director

    stacked = director.pick_system("b", stack=8)
    assert "highlight reel, not a story" in stacked
    assert "MONTAGE STACK" in stacked


def test_stack_survives_the_stream_driven_intro():
    from refire import director

    s = director.pick_system(None, style="fast", stack=6)
    assert s.startswith("You are a video editor cutting a Twitch gaming stream into one "
                        "focused, cohesive short. NO editorial brief")
    assert "MONTAGE STACK of about 6 moments" in s


# --- cut-point modes reach the cast --------------------------------------

def _snap_words(n=60):
    """One long sentence every 6s, with an RMS peak on the last word of each."""
    ws = []
    for i in range(n):
        t = i * 1.0
        last = (i % 6) == 5
        ws.append({"text": ("end." if last else f"w{i}"), "start": t, "end": t + 0.8,
                   "rms": 900.0 if last else 10.0})
    return ws


def _snap_outline():
    return SimpleNamespace(central_idea="x", cold_open=[],
                           beats=[SimpleNamespace(title="A", intent="i", query="q",
                                                  start_s=6.0, end_s=17.0,
                                                  reaction_end_s=23.0)])


def _cast_with(monkeypatch, **kw):
    monkeypatch.setattr("refire.score.score_chunk",
                        lambda c, brief="", model="": {"llm_score": 9.0, "reason": "r"})
    return narrative.cast(_snap_outline(), [], _snap_words(), target_s=1000.0,
                          model="m", **kw)


def test_phrase_and_transient_cut_shorter_than_sentence(monkeypatch):
    base = _cast_with(monkeypatch)[1]["beats"][0]["dur"]
    phrase = _cast_with(monkeypatch, snap="phrase")[1]["beats"][0]["dur"]
    trans = _cast_with(monkeypatch, snap="transient")[1]["beats"][0]["dur"]
    assert phrase <= base and trans < base


def test_transient_drops_the_reaction_tail(monkeypatch):
    """reaction_end_s exists to stop a cut feeling abrupt -- which is exactly what a
    premature cut is FOR, so transient mode must not honour it."""
    sent_end = _cast_with(monkeypatch)[1]["beats"][0]["end"]
    trans_end = _cast_with(monkeypatch, snap="transient")[1]["beats"][0]["end"]
    assert sent_end >= 23.0            # the reaction tail was kept
    assert trans_end < sent_end        # ...and deliberately dropped


def test_sentence_mode_is_the_untouched_default(monkeypatch):
    a = _cast_with(monkeypatch)[1]
    b = _cast_with(monkeypatch, snap="sentence", order="chrono", stack=0)[1]
    assert a == b


def test_the_final_cut_of_the_video_is_never_truncated(monkeypatch):
    """The ending is the most visible cut there is; truncating it is an abrupt stop,
    not a style."""
    monkeypatch.setattr("refire.score.score_chunk",
                        lambda c, brief="", model="": {"llm_score": 9.0, "reason": "r"})
    ol = SimpleNamespace(central_idea="x", cold_open=[], beats=[
        SimpleNamespace(title="A", intent="i", query="q", start_s=6.0, end_s=17.0),
        SimpleNamespace(title="B", intent="i", query="q", start_s=24.0, end_s=35.0)])
    words = _snap_words()
    _s, log, _w = narrative.cast(ol, [], words, target_s=1000.0, model="m",
                                 snap="transient")
    # the last beat's out-point still lands on a sentence terminator ("end." at x.8)
    assert log["beats"][-1]["end"] == pytest.approx(35.8)


# --- play order ----------------------------------------------------------

def _ordered_outline():
    """Four beats whose stream order is deliberately the WORST tonal order: two calm
    then two chaotic."""
    mk = lambda t, s, tex, en: SimpleNamespace(   # noqa: E731
        title=t, intent="i", query="q", start_s=s, end_s=s + 5.8,
        texture=tex, energy=en)
    return SimpleNamespace(central_idea="x", cold_open=[], beats=[
        mk("calm1", 0.0, "sincere", 1), mk("calm2", 12.0, "sincere", 2),
        mk("wild1", 24.0, "chaotic", 5), mk("wild2", 36.0, "chaotic", 4)])


def _order_titles(monkeypatch, **kw):
    monkeypatch.setattr("refire.score.score_chunk",
                        lambda c, brief="", model="": {"llm_score": 9.0, "reason": "r"})
    sections, _log, _w = narrative.cast(_ordered_outline(), [], _snap_words(),
                                        target_s=1000.0, model="m", **kw)
    return [s["title"] for s in sections]


def test_chrono_is_still_the_default(monkeypatch):
    assert _order_titles(monkeypatch) == ["calm1", "calm2", "wild1", "wild2"]
    assert _order_titles(monkeypatch, order="chrono") == _order_titles(monkeypatch)


def test_director_order_keeps_the_outline_sequence(monkeypatch):
    """Same list here, but it must come from the outline rather than from a re-sort --
    a reordered outline would otherwise be silently undone."""
    monkeypatch.setattr("refire.score.score_chunk",
                        lambda c, brief="", model="": {"llm_score": 9.0, "reason": "r"})
    ol = _ordered_outline()
    ol.beats = [ol.beats[2], ol.beats[0], ol.beats[3], ol.beats[1]]
    sections, _l, _w = narrative.cast(ol, [], _snap_words(), target_s=1000.0, model="m",
                                      order="director")
    assert [s["title"] for s in sections] == ["wild1", "calm1", "wild2", "calm2"]


def test_whiplash_opens_on_the_peak_and_alternates_texture(monkeypatch):
    titles = _order_titles(monkeypatch, order="whiplash")
    assert titles[0] == "wild1"                       # highest energy opens
    assert titles != ["calm1", "calm2", "wild1", "wild2"]
    textures = ["chaotic" if t.startswith("wild") else "sincere" for t in titles]
    # no two same-texture beats back to back -- an alternative always existed here
    assert all(a != b for a, b in zip(textures, textures[1:])), titles


def test_whiplash_scores_the_bigger_jolt_higher():
    from refire.narrative import _mismatch

    calm = {"texture": "sincere", "energy": 1}
    wild = {"texture": "chaotic", "energy": 5}
    assert _mismatch(calm, wild) > _mismatch(calm, {"texture": "sincere", "energy": 2})


# --- the critic must not stuff a styled cut back into shape --------------
# Silencing the pacing FLAG is not enough: the rubric asks for the same thing in prose,
# and prose is what the critic re-plans from. Two of these items are literal orders to
# add footage back ("RESTORE a representative run", "Extend its end").

_PROTECT = ["does any clip start without enough context",   # SETUP CLARITY
            "RESTORE a representative run",                 # MISSING BUILD-UP
            "does any clip end BEFORE its payoff"]          # PAYOFF COMPLETION
_MIRROR = ["- NO SETUP NEEDED:", "- DENSITY:", "- PREMATURE CUTS ARE INTENDED:"]


def test_review_rubric_keeps_protecting_moments_by_default():
    from refire import director

    s = director._review_system()
    assert s == director._REVIEW_SYSTEM
    assert all(x in s for x in _PROTECT)
    assert not any(x in s for x in _MIRROR)


def test_review_rubric_flips_when_build_up_is_deliberately_skipped():
    from refire import director

    s = director._review_system(keep_build=False)
    assert not any(x in s for x in _PROTECT), "the critic will re-stuff the cut"
    assert all(x in s for x in _MIRROR)
    # the ending is the one moment that still has to resolve
    assert "The FINAL beat is the one exception" in s
    # and the rubric still has to survive .format()
    assert s.format(n=5, roles="r")


def test_outline_prompt_stops_demanding_the_whole_build_up():
    from refire import director

    plain = director.pick_system("b")
    dense = director.pick_system("b", style="fast", keep_build=False)
    assert "must keep a representative RUN" in plain
    assert "must keep a representative RUN" not in dense
    assert "jump to the punchline with no build" not in dense
    assert "not thorough, it is slack" in dense


def test_segment_floor_scales_with_pace():
    """A 3s floor stated next to a paced budget contradicts it -- a 1-2.5s stack moment
    is below the minimum the same prompt just set."""
    from refire import director

    assert "at least ~3 seconds" in director.pick_system("b")
    assert "at least ~3 seconds" in director.pick_system("b", pace=1.0)   # identity
    assert "at least ~1.05 seconds" in director.pick_system("b", pace=0.35)
    assert "at least ~1 seconds" in director.pick_system("b", pace=0.1)   # floored at 1s


def test_the_style_clause_outranks_the_whole_rubric_not_one_item():
    """Naming only ARC leaves a model dutifully applying the other twelve items."""
    import refire.director as d

    seen = {}
    d._complete_cli = lambda m, sys, c, cls, tag="", **k: (
        seen.__setitem__(tag, sys), (_ for _ in ()).throw(RuntimeError("x")))[1]
    log = {"central_idea": "x", "cold_open": [], "beats": [
        {"title": "A", "role": "climax", "energy": 5, "dur": 40.0, "start": 0.0,
         "end": 40.0, "text": "w " * 60, "segments": [[0.0, 40.0]]}]}
    try:
        d.review("MAP", "b", "t", log, 480.0, style="dense clip reel", keep_build=False)
    except Exception:
        pass
    s = seen["review"]
    assert "the direction wins" in s and "the rubric is what is wrong" in s
    assert "never widen, extend or re-add footage" in s
