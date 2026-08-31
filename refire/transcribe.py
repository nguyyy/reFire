"""Transcribe audio to word-level timestamps with faster-whisper (local GPU)."""
from __future__ import annotations

import difflib
import glob
import json
import os
import sys
import time
import wave
from functools import lru_cache
from pathlib import Path
from typing import Callable, TypedDict


class Word(TypedDict):
    text: str
    start: float
    end: float


# large-v3-turbo has 4 decoder layers to large-v3's 32 at near-identical WER, and is the
# single biggest lever on a 6h VOD. Pass model_size="large-v3" to get the old quality back.
DEFAULT_WHISPER_MODEL = "large-v3-turbo"
# 8 fits alongside turbo's weights on an 8GB card with room to spare; raise it if you have
# more VRAM (throughput scales with it), or drop compute_type to int8_float16 to make room.
DEFAULT_BATCH_SIZE = 8


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
    model_size: str = DEFAULT_WHISPER_MODEL,
    device: str = "cuda",
    hotwords: str = "",
    progress: Callable[[float], None] | None = None,
    backend: str = "local",
    batch_size: int = DEFAULT_BATCH_SIZE,
    compute_type: str = "float16",
) -> list[Word]:
    """Return word-level transcript via the chosen `backend`. Caches to cache_path JSON.

    `backend` selects the transcriber plugin ("local" faster-whisper, default and free;
    "deepgram" cloud — wired but deferred). All backends normalize to the same
    {text,start,end} word schema. `hotwords` is a comma-joined game glossary that biases
    decoding toward proper nouns. The cache is keyed by the glossary, the backend AND the
    model (a `.source.json` sidecar records backend/model), so changing any of them
    re-transcribes instead of serving a stale or wrong-engine transcript. `progress(frac)`
    (0..1), if given, is called as decoding advances.
    """
    cache_path = Path(cache_path) if cache_path else None
    sidecar = cache_path.with_suffix(".glossary.json") if cache_path else None
    source = cache_path.with_suffix(".source.json") if cache_path else None
    if cache_path and cache_path.exists():
        prev = sidecar.read_text(encoding="utf-8") if sidecar and sidecar.exists() else '""'
        # legacy caches predate the source sidecar -> treat them as local/large-v3, the
        # only thing they could have been.
        src = (json.loads(source.read_text(encoding="utf-8"))
               if source and source.exists() else {})
        # The model MUST be part of the key: without it, switching to a faster model
        # silently serves the old model's transcript and every A/B is a lie.
        if (json.loads(prev) == hotwords
                and src.get("backend", "local") == backend
                and src.get("model", "large-v3") == model_size):
            return json.loads(cache_path.read_text(encoding="utf-8"))

    fn = _BACKENDS.get(backend)
    if fn is None:
        raise SystemExit(f"unknown transcriber '{backend}' "
                         f"(choices: {', '.join(sorted(_BACKENDS))})")
    words = snap_to_glossary(
        fn(wav_path, hotwords=hotwords, model_size=model_size, device=device,
           progress=progress, batch_size=batch_size, compute_type=compute_type),
        hotwords)

    if cache_path:
        cache_path.write_text(json.dumps(words), encoding="utf-8")
        sidecar.write_text(json.dumps(hotwords), encoding="utf-8")
        source.write_text(json.dumps(
            {"backend": backend, "model": model_size, "hotwords": hotwords,
             "ts": time.time()}), encoding="utf-8")
    return words


_STRIP = " \t\"'.,!?:;()[]…-"
_CUTOFF = 0.72   # "arlequino" vs "Arlecchino" scores 0.74; below this it's a real word


def snap_to_glossary(words: list[Word], hotwords: str) -> list[Word]:
    """Rewrite near-miss proper nouns to their glossary spelling, in place.

    Hotwords only *bias* the decoder -- once it has committed to "kinech" nothing
    downstream fixes it, and the caption, the LLM director and the search index all
    inherit the error. Guards against snapping ordinary English: a candidate must
    share the first letter, be >=5 chars, and clear _CUTOFF similarity.
    """
    by_initial: dict[str, list[str]] = {}
    canon: dict[str, str] = {}
    for term in hotwords.split(","):
        for tok in term.split():                 # "Hu Tao" -> per-token matching
            if len(tok) >= 5 and tok.lower() not in canon:
                canon[tok.lower()] = tok
                by_initial.setdefault(tok[0].lower(), []).append(tok.lower())
    if not canon:
        return words

    for w in words:
        core = w["text"].strip(_STRIP)
        key = core.lower()
        if len(core) < 5 or key in canon:
            continue
        hit = difflib.get_close_matches(key, by_initial.get(key[0], ()), n=1,
                                        cutoff=_CUTOFF)
        if hit:
            w["text"] = w["text"].replace(core, canon[hit[0]], 1)
    return words


@lru_cache(maxsize=1)
def _whisper(model_size: str, device: str, compute_type: str):
    """One WhisperModel per (size, device, precision), reused across calls.

    The constructor loads gigabytes of weights onto the GPU. `make` transcribes one long
    wav and never noticed, but `loud` calls this once per picked moment -- dozens of
    20-second windows -- where reloading per call costs far more than the decoding does.
    maxsize=1: a different precision evicts the old model rather than holding two on the
    GPU at once.
    """
    from faster_whisper import WhisperModel
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def _stt_local(wav_path, hotwords="", model_size=DEFAULT_WHISPER_MODEL, device="cuda",
               progress=None, batch_size=DEFAULT_BATCH_SIZE, compute_type="float16",
               **_kw) -> list[Word]:
    """faster-whisper local GPU backend (default; free/private)."""
    _register_cuda_dlls()
    # local imports: heavy, GPU-only
    from faster_whisper import BatchedInferencePipeline

    model = _whisper(model_size, device, compute_type)
    # hotwords is the bounded bias path; passing initial_prompt too would (a) make
    # faster-whisper ignore hotwords and (b) feed an uncapped prompt that can push
    # the decoder past Whisper's 448-position limit. condition_on_previous_text=False
    # keeps positions bounded so a hallucinated/runaway segment can't overflow either.
    #
    # Batched + VAD is the whole speed story on a multi-hour VOD: sequential decoding
    # leaves the GPU idle between windows, and without VAD we pay full decode price for
    # the hours of dead air a Twitch stream contains. faster-whisper maps VAD-clipped
    # timestamps back onto absolute source seconds, so every downstream consumer
    # (perception, casting, ae_export) is unaffected.
    segments, _ = BatchedInferencePipeline(model=model).transcribe(
        str(wav_path), batch_size=batch_size, word_timestamps=True,
        hotwords=hotwords or None, vad_filter=True, condition_on_previous_text=False)

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
