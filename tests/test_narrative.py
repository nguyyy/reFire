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

def test_director_outline_plumbing(monkeypatch):
    """outline() returns the parsed Outline without a live API call."""
    anthropic = pytest.importorskip("anthropic")
    from refire import director

    want = director.Outline(central_idea="x", beats=[
        director.Beat(title="A", intent="i", query="q", start_s=1.0, end_s=2.0)])

    class FakeMessages:
        def parse(self, **kw):
            assert kw["output_format"] is director.Outline
            return SimpleNamespace(parsed_output=want)

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: FakeClient())
    got = director.outline("[00:00] hi", "brief", "title", 90.0)
    assert got is want and got.beats[0].title == "A"


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
