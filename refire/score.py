"""Score a transcript chunk for clip-worthiness with a local Ollama LLM."""
from __future__ import annotations

import json

from .chunk import Chunk

# default to a model that fits an 8GB gpu. qwen2.5:14b scores better but OOMs here, use --model
DEFAULT_MODEL = "llama3.1:8b"

# a stuck local scorer can't cost a whole run. ollama.chat has no timeout, so a runner stuck
# on vram pressure would hang mid-cast after the claude calls are already paid for. one
# rating takes seconds, this leaves room for a big model's cold start
SCORE_TIMEOUT_S = 180.0

# ollama unloads a model after 5 idle minutes and the director/critic calls run longer than
# that, so every cast paid a ~13s cold load (measured on the qwen3 30b scorer). make() unloads
# it when the run ends so the next run's whisper gets the card back
KEEP_ALIVE = "30m"

_SYSTEM = (
    "You rate moments from a Twitch gaming stream for a highlights compilation. "
    "Given a transcript snippet, rate how clip-worthy it is for a highlight reel "
    "on a 1-10 scale (10 = must-clip: clutch play, big win/loss, hype, funny "
    "reaction). Reply ONLY with JSON: {\"llm_score\": <1-10>, \"reason\": \"<short>\"}."
)

# brief-aware: fully anchored scale so scores spread out instead of bunching in the middle,
# and off-topic stuff scores low even when it's loud
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
    import ollama  # optional heavy dep

    if brief:
        system, key = _SYSTEM_BRIEF, "score"
        user = f"BRIEF: {brief}\n\nTRANSCRIPT:\n{chunk['text']}"
    else:
        system, key = _SYSTEM, "llm_score"
        user = chunk["text"]

    try:
        resp = ollama.Client(timeout=SCORE_TIMEOUT_S).chat(
            model=model,
            format="json",
            keep_alive=KEEP_ALIVE,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except Exception as e:                      # timeout, dead runner, refused socket
        # never raise, narrative.cast isn't wrapped so it would kill a run that already paid for the
        # director + critic. same as a parse failure, beat scores 0 and the cut survives
        print(f"[score] local scorer unavailable ({type(e).__name__}); scoring 0")
        return {"llm_score": 0.0, "reason": "scorer_unavailable"}
    try:
        data = json.loads(resp["message"]["content"])
        out = {"llm_score": float(data[key]), "reason": str(data.get("reason", ""))}
        if brief and "relevance" in data:
            out["relevance"] = float(data["relevance"])
        return out
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return {"llm_score": 0.0, "reason": "parse_error"}


def warm_scorer(model: str = DEFAULT_MODEL) -> None:
    """Load the scorer model ahead of the first cast (run it in a thread while Claude thinks,
    the card is idle then). Same default options as `score_chunk` -- a different num_ctx
    would make ollama reload anyway. Best effort: a cold cast still works, just slower."""
    try:
        import ollama
        ollama.Client(timeout=SCORE_TIMEOUT_S).generate(model=model, keep_alive=KEEP_ALIVE)
    except Exception:
        pass


def release_scorer(model: str = DEFAULT_MODEL) -> None:
    """Unload the scorer if it is still resident (see KEEP_ALIVE). Best effort."""
    try:
        import ollama
        client = ollama.Client(timeout=30)
        # an unload request for a model that isn't loaded can load it first, so check
        if any(m.model == model for m in client.ps().models):
            client.generate(model=model, keep_alive=0)
    except Exception:
        pass
