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

# Brief-aware scoring: a fully-anchored scale (not just the top end) so scores
# spread out instead of compressing into the middle, and off-topic content scores
# low even when it's loud.
_SYSTEM_BRIEF = (
    "You curate clips for a focused highlight video. The EDITOR'S BRIEF states the "
    "subject and vibe the video should capture. Given a transcript snippet, rate it "
    "1-10 for how well it serves the brief, judging BOTH on-topic relevance AND "
    "standalone watchability. Anchor the scale strictly: "
    "1 = off-topic or dead air; 3 = loosely related, forgettable; "
    "5 = on-topic but flat; 7 = on-topic and engaging; "
    "9 = squarely on-brief and genuinely gripping or funny; "
    "10 = unmissable, the clip this video exists for. "
    "Be decisive and use the full range. Reply ONLY with JSON: "
    "{\"score\": <1-10>, \"relevance\": <1-10>, \"reason\": \"<short>\"}."
)


def score_chunk(chunk: Chunk, brief: str = "", model: str = DEFAULT_MODEL) -> dict:
    """Return {'llm_score': float, 'reason': str} (+ 'relevance' when a brief is given).

    brief="" keeps the original topic-agnostic clip-worthiness scoring (legacy
    detect path); a brief switches to focused relevance+quality scoring. Parse
    failure -> score 0.
    """
    import ollama  # local import: optional heavy dep

    if brief:
        system, key = _SYSTEM_BRIEF, "score"
        user = f"BRIEF: {brief}\n\nTRANSCRIPT:\n{chunk['text']}"
    else:
        system, key = _SYSTEM, "llm_score"
        user = chunk["text"]

    resp = ollama.chat(
        model=model,
        format="json",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    try:
        data = json.loads(resp["message"]["content"])
        out = {"llm_score": float(data[key]), "reason": str(data.get("reason", ""))}
        if brief and "relevance" in data:
            out["relevance"] = float(data["relevance"])
        return out
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return {"llm_score": 0.0, "reason": "parse_error"}
