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
    # 180s of material, target 100s -> trim lowest-scored beats under budget
    sections, log, warning = narrative.cast(_outline("A", "B", "C"), _chunks(), [],
                                            target_s=100.0, model="m", tol=0.25)
    total = sum(s["clips"][0]["end"] - s["clips"][0]["start"] for s in sections)
    assert total <= 100.0 and len(sections) < 3
    assert "A" in [s["title"] for s in sections]   # highest-scored beat survives


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


def test_invalid_bounds_falls_back_to_retrieval(stubbed):
    # end_s <= start_s is unusable -> retrieve by query (chunk-based old path)
    ol = SimpleNamespace(central_idea="x", beats=[
        SimpleNamespace(title="A", intent="i", query="q", start_s=5.0, end_s=5.0)])
    sections, log, warning = narrative.cast(ol, _chunks(), [], target_s=1000.0, model="m")
    assert sections[0]["clips"][0]["start"] == 0.0   # cast the top retrieved chunk
    assert log["beats"][0]["dir_start"] is None      # logged as a fallback


def test_stream_map_is_sentence_level():
    from refire import director
    words = [{"text": t, "start": s, "end": s + 0.4} for t, s in
             [("Hello", 0.0), ("world.", 0.5), ("Next", 65.0), ("one.", 65.5)]]
    assert director.stream_map(words) == "[00:00] Hello world.\n[01:05] Next one."


# --- director backends parse into the new Beat schema ---

def test_max_tokens_reserves_room_after_thinking():
    from refire import director
    # always thinking budget + reserved answer room, monotonic, capped
    assert director._max_tokens(90.0) > director.THINK_BUDGET
    assert director._max_tokens(2700.0) >= director._max_tokens(1200.0)
    assert director._max_tokens(99999.0) <= 24000           # capped
    assert director._out_room(90.0) >= 4000                 # answer always gets room


def _fake_anthropic(monkeypatch, responses, output_format=None):
    """Stub anthropic.Anthropic; parse() yields `responses` (parsed_output, stop_reason)
    in order and records each call's kwargs into the returned `calls` list."""
    anthropic = pytest.importorskip("anthropic")
    from refire import director
    expected = output_format or director.Outline

    calls = []
    seq = iter(responses)

    class FakeMessages:
        def parse(self, **kw):
            assert kw["output_format"] is expected
            calls.append(kw)
            parsed, stop = next(seq)
            return SimpleNamespace(parsed_output=parsed, stop_reason=stop)

    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda *a, **k: SimpleNamespace(messages=FakeMessages()))
    return calls


def test_director_outline_plumbing(monkeypatch):
    """outline() returns the parsed Outline and passes a target-scaled max_tokens."""
    from refire import director
    want = director.Outline(central_idea="x", beats=[
        director.Beat(title="A", intent="i", query="q", start_s=1.0, end_s=2.0)])
    calls = _fake_anthropic(monkeypatch, [(want, "end_turn")])
    got = director.outline("[00:00] hi", "brief", "title", 1800.0)   # 30-min target
    assert got is want and got.beats[0].title == "A"
    assert calls[0]["max_tokens"] == director._max_tokens(1800.0)
    assert calls[0]["thinking"]["type"] == "enabled"      # explicit budget, not adaptive


def test_director_retries_without_thinking_on_none(monkeypatch):
    """parsed_output None (thinking ate the budget) -> retry once with no thinking."""
    from refire import director
    want = director.Outline(central_idea="x", beats=[
        director.Beat(title="A", intent="i", query="q", start_s=1.0, end_s=2.0)])
    calls = _fake_anthropic(monkeypatch, [(None, "max_tokens"), (want, "end_turn")])
    got = director.outline("[00:00] hi", "brief", "title", 600.0)
    assert got is want
    assert len(calls) == 2
    assert "thinking" in calls[0] and "thinking" not in calls[1]   # retry drops thinking


def test_director_raises_when_no_outline(monkeypatch):
    """Both attempts empty -> raise with the stop_reason instead of returning None."""
    from refire import director
    _fake_anthropic(monkeypatch, [(None, "max_tokens"), (None, "max_tokens")])
    with pytest.raises(RuntimeError, match="max_tokens"):
        director.outline("[00:00] hi", "brief", "title", 600.0)


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
    assert "[00:00-00:05]" in s and "[01:05-01:10]" in s
    assert "the idea" in s


def test_review_returns_revised_outline_and_reserves_notes_room(monkeypatch):
    """review() parses a Review, feeds the realized transcript, and reserves answer room
    for its notes on top of the outline budget."""
    from refire import director
    revised = director.Outline(central_idea="y", beats=[
        director.Beat(title="B", intent="i", query="q", start_s=1.0, end_s=2.0)])
    want = director.Review(approved=False, notes="flat; needs a closing button",
                           outline=revised)
    calls = _fake_anthropic(monkeypatch, [(want, "end_turn")], output_format=director.Review)
    log = {"central_idea": "x", "beats": [
        {"title": "A", "intent": "i", "start": 3.0, "end": 5.0, "score": 9.0,
         "text": "hi there"}]}
    got = director.review("[00:00] hi there", "brief", "title", log, 600.0)
    assert got is want and got.outline.beats[0].title == "B"
    assert calls[0]["max_tokens"] == min(24000, director._max_tokens(600.0) + 1500)
    assert "hi there" in calls[0]["messages"][0]["content"]   # critic reads the realized cut


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
