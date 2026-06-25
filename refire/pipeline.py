"""Chain the segment-detection stages, caching artifacts in a run directory."""
from __future__ import annotations

import json
from pathlib import Path

from .audio import extract_audio
from .chat import chat_signal
from .chunk import make_chunks
from .rank import rank_select
from .score import DEFAULT_MODEL, score_chunk
from .transcribe import transcribe


def run(
    video: str | Path,
    chat: str | Path,
    run_dir: str | Path,
    model: str = DEFAULT_MODEL,
    w_llm: float = 0.6,
    w_chat: float = 0.4,
    top_n: int | None = None,
    threshold: float | None = None,
) -> Path:
    """Run detection end-to-end; return path to segments.json."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    wav = run_dir / "audio.wav"
    transcript_cache = run_dir / "transcript.json"
    out = run_dir / "segments.json"

    if not wav.exists():
        extract_audio(video, wav)
    words = transcribe(wav, cache_path=transcript_cache)
    if not words:
        out.write_text("[]", encoding="utf-8")
        return out

    duration = words[-1]["end"]
    signal = chat_signal(chat, duration)
    chunks = make_chunks(words, signal)

    scored = []
    for ch in chunks:
        res = score_chunk(ch, model=model)
        scored.append({**ch, **res})

    segments = rank_select(scored, w_llm=w_llm, w_chat=w_chat,
                           top_n=top_n, threshold=threshold)
    out.write_text(json.dumps(segments, indent=2), encoding="utf-8")
    return out
