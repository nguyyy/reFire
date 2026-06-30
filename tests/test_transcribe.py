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
