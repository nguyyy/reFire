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


def test_speech_regions_offsets_slices_rejoins_edges_and_caches(monkeypatch, tmp_path):
    """Silero runs per wav slice, so each slice's sample offsets must be lifted by where the
    slice starts, and a region a slice edge cut in two rejoined -- otherwise every region
    past the first 10 minutes lands at the wrong time and long lines split."""
    import sys
    import wave
    from types import SimpleNamespace

    wav = tmp_path / "audio.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\0\0" * 16000 * 3)
    per_slice = iter([[{"start": 8000, "end": 16000}],     # 0.5-1.0 in slice 0
                      [{"start": 0, "end": 4000}],         # 1.0-1.25: continues it
                      [{"start": 8000, "end": 12000}]])    # 2.5-2.75: a new region
    fake = SimpleNamespace(VadOptions=lambda **k: k,       # no onnx/ctranslate2 import
                           get_speech_timestamps=lambda x, o, sampling_rate: next(per_slice))
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", fake)
    monkeypatch.setattr(T, "VAD_CHUNK_S", 1.0)
    cache = tmp_path / "speech.json"

    assert T.speech_regions(wav, cache_path=cache) == [[0.5, 1.25], [2.5, 2.75]]
    # served from cache: the exhausted iterator raises if Silero runs again
    assert T.speech_regions(wav, cache_path=cache) == [[0.5, 1.25], [2.5, 2.75]]
    assert T.speech_regions(tmp_path / "missing.wav") == []

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


def test_release_whisper_actually_frees_the_model(monkeypatch):
    """A Whisper model still cached after `make` transcribes pushes the 12GB scout model into
    shared memory (4.9 tok/s vs 159). The release must drop the LAST reference, not just
    empty a dict that something else still points into."""
    import sys
    import weakref
    from types import SimpleNamespace

    class FakeModel:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeModel))
    T._whisper.cache_clear()
    ref = weakref.ref(T._whisper("large-v3-turbo", "cuda", "float16"))
    assert ref() is not None            # cached: this is what outlived transcription
    T.release_whisper()
    assert ref() is None


def test_snap_to_glossary_fixes_near_misses_only():
    from refire.transcribe import snap_to_glossary
    w = lambda t: {"text": t, "start": 0.0, "end": 1.0}  # noqa: E731
    out = [x["text"] for x in snap_to_glossary(
        [w("kinech"), w("Arlequino,"), w("Nuvillette."), w("kitchen"), w("running")],
        "Kinich, Arlecchino, Neuvillette")]
    assert out == ["Kinich", "Arlecchino,", "Neuvillette.", "kitchen", "running"]
