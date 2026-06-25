"""Score a transcript chunk for clip-worthiness with a local Ollama LLM."""
from __future__ import annotations

import json

from .chunk import Chunk

DEFAULT_MODEL = "qwen2.5:14b"

_SYSTEM = (
    "You rate moments from a Twitch gaming stream for a highlights compilation. "
    "Given a transcript snippet, rate how clip-worthy it is for a highlight reel "
    "on a 1-10 scale (10 = must-clip: clutch play, big win/loss, hype, funny "
    "reaction). Reply ONLY with JSON: {\"llm_score\": <1-10>, \"reason\": \"<short>\"}."
)


def score_chunk(chunk: Chunk, model: str = DEFAULT_MODEL) -> dict:
    """Return {'llm_score': float, 'reason': str}. Parse failure -> score 0."""
    import ollama  # local import: optional heavy dep

    resp = ollama.chat(
        model=model,
        format="json",
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": chunk["text"]},
        ],
    )
    try:
        data = json.loads(resp["message"]["content"])
        score = float(data["llm_score"])
        return {"llm_score": score, "reason": str(data.get("reason", ""))}
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return {"llm_score": 0.0, "reason": "parse_error"}
