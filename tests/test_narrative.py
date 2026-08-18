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
    and a model = its `model_dump_json()`.

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
            text = "" if parsed is None else parsed.model_dump_json()
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
    assert log["cold_open"]["dur"] == pytest.approx(co["end"] - co["start"])


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
    assert log["cold_open"] is None


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
    assert log["cold_open"] is None


def test_cold_open_shifts_the_pacing_clock(monkeypatch):
    """Beats play later in the finished video than their kept footage alone implies."""
    from refire.pacing import cut_timeline
    log = {"cold_open": {"start": 63.0, "end": 68.0, "dur": 5.0},
           "beats": [{"title": "A", "dur": 30.0, "start": 0.0, "end": 30.0, "text": "x"}]}
    assert cut_timeline(log)[0]["at"] == 5.0
