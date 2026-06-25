"""Transcribe audio to word-level timestamps with faster-whisper (local GPU)."""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import TypedDict


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


def transcribe(
    wav_path: str | Path,
    cache_path: str | Path | None = None,
    model_size: str = "large-v3",
    device: str = "cuda",
) -> list[Word]:
    """Return word-level transcript. Reads/writes cache_path JSON if given."""
    if cache_path and Path(cache_path).exists():
        return json.loads(Path(cache_path).read_text(encoding="utf-8"))

    _register_cuda_dlls()
    from faster_whisper import WhisperModel  # local import: heavy, GPU-only

    model = WhisperModel(model_size, device=device, compute_type="float16")
    segments, _ = model.transcribe(str(wav_path), word_timestamps=True)

    words: list[Word] = []
    for seg in segments:
        for w in (seg.words or []):
            words.append({"text": w.word.strip(), "start": w.start, "end": w.end})

    if cache_path:
        Path(cache_path).write_text(json.dumps(words), encoding="utf-8")
    return words
