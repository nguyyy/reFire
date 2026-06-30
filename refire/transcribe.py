"""Transcribe audio to word-level timestamps with faster-whisper (local GPU)."""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import Callable, TypedDict


class Word(TypedDict):
    text: str
    start: float
    end: float


def _register_cuda_dlls() -> None:
    """On Windows, add pip-installed NVIDIA bin dirs to the DLL search path so
    ctranslate2 can resolve cuBLAS/cuDNN/CUDA-runtime. No-op elsewhere."""
    if sys.platform != "win32":
        return
    # faster-whisper (ctranslate2) + numpy each ship libiomp5md.dll; allow the
    # duplicate rather than crashing. Must be set before the heavy import.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    for site in __import__("site").getsitepackages():
        for bin_dir in glob.glob(os.path.join(site, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(bin_dir)
            except OSError:
                pass
            # ctranslate2 resolves cuBLAS via PATH at runtime, so prepend too
            if bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def _wav_duration(wav_path: str | Path) -> float:
    """Seconds of audio in a WAV (stdlib), or 0.0 if unreadable."""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate() or 1)
    except (wave.Error, OSError, ZeroDivisionError):
        return 0.0


def transcribe(
    wav_path: str | Path,
    cache_path: str | Path | None = None,
    model_size: str = "large-v3",
    device: str = "cuda",
    hotwords: str = "",
    progress: Callable[[float], None] | None = None,
    backend: str = "local",
) -> list[Word]:
    """Return word-level transcript via the chosen `backend`. Caches to cache_path JSON.

    `backend` selects the transcriber plugin ("local" faster-whisper, default and free;
    "deepgram" cloud — wired but deferred). All backends normalize to the same
    {text,start,end} word schema. `hotwords` is a comma-joined game glossary that biases
    decoding toward proper nouns. The cache is keyed by BOTH the glossary AND the backend
    (a `.source.json` sidecar records backend/model), so switching either re-transcribes
    instead of serving a stale or wrong-engine transcript. `progress(frac)` (0..1), if
    given, is called as decoding advances.
    """
    cache_path = Path(cache_path) if cache_path else None
    sidecar = cache_path.with_suffix(".glossary.json") if cache_path else None
    source = cache_path.with_suffix(".source.json") if cache_path else None
    if cache_path and cache_path.exists():
        prev = sidecar.read_text(encoding="utf-8") if sidecar and sidecar.exists() else '""'
        # legacy caches predate the source sidecar -> treat them as the local backend
        src_backend = (json.loads(source.read_text(encoding="utf-8")).get("backend")
                       if source and source.exists() else "local")
        if json.loads(prev) == hotwords and src_backend == backend:
            return json.loads(cache_path.read_text(encoding="utf-8"))

    fn = _BACKENDS.get(backend)
    if fn is None:
        raise SystemExit(f"unknown transcriber '{backend}' "
                         f"(choices: {', '.join(sorted(_BACKENDS))})")
    words = fn(wav_path, hotwords=hotwords, model_size=model_size, device=device,
               progress=progress)

    if cache_path:
        cache_path.write_text(json.dumps(words), encoding="utf-8")
        sidecar.write_text(json.dumps(hotwords), encoding="utf-8")
        source.write_text(json.dumps(
            {"backend": backend, "model": model_size, "hotwords": hotwords,
             "ts": time.time()}), encoding="utf-8")
    return words


def _stt_local(wav_path, hotwords="", model_size="large-v3", device="cuda",
               progress=None, **_kw) -> list[Word]:
    """faster-whisper local GPU backend (default; free/private)."""
    _register_cuda_dlls()
    from faster_whisper import WhisperModel  # local import: heavy, GPU-only

    model = WhisperModel(model_size, device=device, compute_type="float16")
    # hotwords is the bounded bias path; passing initial_prompt too would (a) make
    # faster-whisper ignore hotwords and (b) feed an uncapped prompt that can push
    # the decoder past Whisper's 448-position limit. condition_on_previous_text=False
    # keeps positions bounded so a hallucinated/runaway segment can't overflow either.
    segments, _ = model.transcribe(
        str(wav_path), word_timestamps=True,
        hotwords=hotwords or None, condition_on_previous_text=False)

    dur = _wav_duration(wav_path) if progress else 0.0
    words: list[Word] = []
    for seg in segments:            # faster-whisper yields lazily as it decodes
        for w in (seg.words or []):
            words.append({"text": w.word.strip(), "start": w.start, "end": w.end})
        if progress and dur:
            progress(min(1.0, seg.end / dur))
    return words


def _stt_deepgram(wav_path, hotwords="", progress=None, **_kw) -> list[Word]:
    """Deepgram cloud backend -- seam wired, transcription pass deferred.

    The plugin is fully plumbed (flag -> backend dispatch -> .source.json cache key) and
    the key is read from DEEPGRAM_API_KEY (.env). The actual Deepgram call is intentionally
    left for later. To finish: POST `wav_path` to Deepgram (e.g. nova-3, smart_format,
    punctuate, keyterm=hotwords), map the returned words to the {text,start,end} schema,
    and chunk multi-hour audio under the upload-size limit then recombine with offsets.
    """
    import os
    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key:
        raise SystemExit(
            "DEEPGRAM_API_KEY not set -- add it to your .env to use --transcriber deepgram "
            "(or use --transcriber local, the default).")
    raise NotImplementedError(
        "Deepgram transcriber not implemented yet (key found, backend wired). "
        "Use --transcriber local for now.")


# Transcriber plugin registry: name -> backend fn(wav_path, hotwords, ..., progress).
_BACKENDS = {"local": _stt_local, "deepgram": _stt_deepgram}
