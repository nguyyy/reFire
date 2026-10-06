"""Transcribe audio to word-level timestamps with faster-whisper (local GPU)."""
from __future__ import annotations

import difflib
import gc
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


# large-v3-turbo has 4 decoder layers vs 32 at about the same WER, biggest speed win on a
# 6h vod. model_size="large-v3" for the old quality
DEFAULT_WHISPER_MODEL = "large-v3-turbo"
# 8 fits next to turbo on an 8GB card. raise it with more vram, or use int8_float16 for room
DEFAULT_BATCH_SIZE = 8


def _register_cuda_dlls() -> None:
    """On Windows, add pip-installed NVIDIA bin dirs to the DLL search path so
    ctranslate2 can resolve cuBLAS/cuDNN/CUDA-runtime. No-op elsewhere."""
    if sys.platform != "win32":
        return
    # faster-whisper and numpy both ship libiomp5md.dll, allow the dupe instead of crashing.
    # has to be set before the import
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    for site in __import__("site").getsitepackages():
        for bin_dir in glob.glob(os.path.join(site, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(bin_dir)
            except OSError:
                pass
            # ctranslate2 finds cuBLAS via PATH at runtime so prepend it too
            if bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def _wav_duration(wav_path: str | Path) -> float:
    """Seconds of audio in a WAV (stdlib), or 0.0 if unreadable."""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate() or 1)
    except (wave.Error, OSError, ZeroDivisionError):
        return 0.0


# whisper's own batched vad threshold + silence gap so "speech" means the same thing here.
# no pad, compress_silence adds its own
VAD_OPTS = {"threshold": 0.5, "min_silence_duration_ms": 160,
            "min_speech_duration_ms": 250, "speech_pad_ms": 0}
VAD_CHUNK_S = 600.0   # read wav in 10 min slices, a 6h stream is ~1.4GB as float32


def speech_regions(wav_path: str | Path, cache_path: str | Path | None = None) -> list[list[float]]:
    """Absolute [start, end] seconds where Silero hears speech, words or not. [] if unreadable.

    The word list is not a speech map. Whisper is handed some speech and returns no words
    for it -- measured across 7 real cuts, 239s of the 1762s `compress_silence` removed held
    Silero speech at this very threshold (a 22s quest exchange in one beat). That audio was
    cut as silence. ~18s per 4h stream on CPU; cached, keyed on `VAD_OPTS`.
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        try:
            got = json.loads(cache_path.read_text(encoding="utf-8"))
            if got["opts"] == VAD_OPTS:
                return got["regions"]
        except (ValueError, KeyError, TypeError):
            pass   # corrupt/old cache -> rescan
    import numpy as np
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    regions: list[list[float]] = []
    try:
        with wave.open(str(wav_path), "rb") as wf:
            sr, step = wf.getframerate(), int(VAD_CHUNK_S * wf.getframerate())
            for pos in range(0, wf.getnframes(), step):
                x = np.frombuffer(wf.readframes(step), dtype=np.int16).astype(np.float32) / 32768.0
                for c in get_speech_timestamps(x, VadOptions(**VAD_OPTS), sampling_rate=sr):
                    s, e = (pos + c["start"]) / sr, (pos + c["end"]) / sr
                    if regions and s - regions[-1][1] < 0.05:   # rejoin across a slice edge
                        regions[-1][1] = round(e, 3)
                    else:
                        regions.append([round(s, 3), round(e, 3)])
    except (wave.Error, OSError, ValueError):
        return []
    if cache_path:
        cache_path.write_text(json.dumps({"opts": VAD_OPTS, "regions": regions}),
                              encoding="utf-8")
    return regions


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
        # old caches have no source sidecar, they can only be local/large-v3
        src = (json.loads(source.read_text(encoding="utf-8"))
               if source and source.exists() else {})
        # model has to be in the key or switching models serves the old transcript
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
_CUTOFF = 0.72   # "arlequino" vs "Arlecchino" is 0.74, below this it's a real word


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


def release_whisper() -> None:
    """Drop the cached WhisperModel so its VRAM is free before the next ollama load.

    The cache outlives transcription for the whole process. On a 16GB card, gpt-oss:20b
    (12GB) loaded beside it spills into shared memory while `ollama ps` still says 100% GPU:
    the scout window measured 4.9 tok/s with Whisper cached vs 159 after this -- a run that
    looked frozen on "scouting" with the GPU pinned.
    """
    _whisper.cache_clear()
    gc.collect()


def _stt_local(wav_path, hotwords="", model_size=DEFAULT_WHISPER_MODEL, device="cuda",
               progress=None, batch_size=DEFAULT_BATCH_SIZE, compute_type="float16",
               **_kw) -> list[Word]:
    """faster-whisper local GPU backend (default; free/private)."""
    _register_cuda_dlls()
    # local imports, heavy + gpu only
    from faster_whisper import BatchedInferencePipeline

    model = _whisper(model_size, device, compute_type)
    # hotwords is the bounded bias path. adding initial_prompt would make faster-whisper ignore
    # hotwords and could push past whisper's 448 position limit. condition_on_previous_text=False
    # keeps a runaway segment from overflowing too.
    #
    # batched + vad is where the speed comes from on a long vod: no idle gpu between windows and
    # no decoding hours of dead air. timestamps map back to source seconds so nothing downstream
    # changes
    segments, _ = BatchedInferencePipeline(model=model).transcribe(
        str(wav_path), batch_size=batch_size, word_timestamps=True,
        hotwords=hotwords or None, vad_filter=True, condition_on_previous_text=False)

    dur = _wav_duration(wav_path) if progress else 0.0
    words: list[Word] = []
    for seg in segments:            # yields lazily as it decodes
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


# transcriber backends: name -> fn(wav_path, hotwords, ..., progress)
_BACKENDS = {"local": _stt_local, "deepgram": _stt_deepgram}
