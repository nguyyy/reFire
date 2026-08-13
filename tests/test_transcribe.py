"""Transcriber plugin seam: backend dispatch + backend-keyed cache (no GPU/cloud)."""
import json

import pytest

from refire import transcribe as T


def test_unknown_backend_raises(tmp_path):
    with pytest.raises(SystemExit):
        T.transcribe(tmp_path / "a.wav", backend="nope")


def test_deepgram_without_key_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    with pytest.raises(SystemExit):                     # missing key -> clear error
        T.transcribe(tmp_path / "a.wav", backend="deepgram")


def test_deepgram_with_key_is_deferred(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "x")
    with pytest.raises(NotImplementedError):            # seam wired, pass not implemented
        T.transcribe(tmp_path / "a.wav", backend="deepgram")


def test_local_dispatch_writes_source_and_caches(monkeypatch, tmp_path):
    fake = [{"text": "hi", "start": 0.0, "end": 0.4}]
    monkeypatch.setitem(T._BACKENDS, "local", lambda *a, **k: fake)
    cache = tmp_path / "transcript.json"

    got = T.transcribe(tmp_path / "a.wav", cache_path=cache, backend="local")
    assert got == fake
    src = json.loads((tmp_path / "transcript.source.json").read_text())
    assert src["backend"] == "local"

    # second call serves the cache -> the (now exploding) backend must NOT run
    def _boom(*a, **k):
        raise AssertionError("re-transcribed despite a valid cache")
    monkeypatch.setitem(T._BACKENDS, "local", _boom)
    assert T.transcribe(tmp_path / "a.wav", cache_path=cache, backend="local") == fake

    # switching backend invalidates the cache (different engine) -> re-dispatch
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    with pytest.raises(SystemExit):                     # deepgram dispatched, no key
        T.transcribe(tmp_path / "a.wav", cache_path=cache, backend="deepgram")


def test_model_is_part_of_the_cache_key(monkeypatch, tmp_path):
    """A different whisper model must re-transcribe, not serve the old model's words.

    Without this, switching to a faster model silently returns the previous model's
    transcript -- so every speed/quality comparison measures nothing.
    """
    fake = [{"text": "hi", "start": 0.0, "end": 0.4}]
    monkeypatch.setitem(T._BACKENDS, "local", lambda *a, **k: fake)
    cache = tmp_path / "transcript.json"
    T.transcribe(tmp_path / "a.wav", cache_path=cache, model_size="large-v3")
    assert json.loads((tmp_path / "transcript.source.json").read_text())["model"] == "large-v3"

    # same model -> cache hit (backend must not run)
    def _boom(*a, **k):
        raise AssertionError("re-transcribed despite a valid cache")
    monkeypatch.setitem(T._BACKENDS, "local", _boom)
    assert T.transcribe(tmp_path / "a.wav", cache_path=cache, model_size="large-v3") == fake

    # different model -> cache MISS, backend re-runs
    other = [{"text": "yo", "start": 0.0, "end": 0.3}]
    monkeypatch.setitem(T._BACKENDS, "local", lambda *a, **k: other)
    assert T.transcribe(tmp_path / "a.wav", cache_path=cache,
                        model_size="large-v3-turbo") == other


def test_legacy_cache_without_sidecar_still_hits(monkeypatch, tmp_path):
    """Pre-existing caches have no .source.json; they can only be local/large-v3."""
    fake = [{"text": "hi", "start": 0.0, "end": 0.4}]
    cache = tmp_path / "transcript.json"
    cache.write_text(json.dumps(fake), encoding="utf-8")
    (tmp_path / "transcript.glossary.json").write_text('""', encoding="utf-8")

    def _boom(*a, **k):
        raise AssertionError("re-transcribed a valid legacy cache")
    monkeypatch.setitem(T._BACKENDS, "local", _boom)
    assert T.transcribe(tmp_path / "a.wav", cache_path=cache,
                        model_size="large-v3") == fake


def test_snap_to_glossary_fixes_near_misses_only():
    from refire.transcribe import snap_to_glossary
    w = lambda t: {"text": t, "start": 0.0, "end": 1.0}  # noqa: E731
    out = [x["text"] for x in snap_to_glossary(
        [w("kinech"), w("Arlequino,"), w("Nuvillette."), w("kitchen"), w("running")],
        "Kinich, Arlecchino, Neuvillette")]
    assert out == ["Kinich", "Arlecchino,", "Neuvillette.", "kitchen", "running"]
