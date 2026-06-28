"""Transcribe audio to word-level timestamps with faster-whisper (local GPU)."""
from __future__ import annotations

import glob
import json
import os
import sys
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
) -> list[Word]:
    """Return word-level transcript. Reads/writes cache_path JSON if given.

    `hotwords` is a comma-joined game glossary that biases decoding toward proper
    nouns. The cache is keyed by the glossary: a transcript made with a different
    (or no) glossary is ignored so a stale "who tao" is never served back.
    `progress(frac)` (0..1), if given, is called as segments stream in.
    """
    cache_path = Path(cache_path) if cache_path else None
    sidecar = cache_path.with_suffix(".glossary.json") if cache_path else None
    if cache_path and cache_path.exists():
        prev = sidecar.read_text(encoding="utf-8") if sidecar and sidecar.exists() else '""'
        if json.loads(prev) == hotwords:
            return json.loads(cache_path.read_text(encoding="utf-8"))

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

    if cache_path:
        cache_path.write_text(json.dumps(words), encoding="utf-8")
        sidecar.write_text(json.dumps(hotwords), encoding="utf-8")
    return words
