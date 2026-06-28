"""Brief-driven retrieval: rank cached chunks by relevance to a free-text brief.

Stage 1 of the two-stage focus pipeline. Embeds the brief and every chunk with a
local Ollama embedding model, ranks by cosine similarity, and returns the top-k
candidates for LLM scoring. A brief with no concrete subject ("the funniest
moments") has no semantic anchor -> cosine is flat, so those fall back to chat
liveliness (z-score) to source a lively candidate pool.
"""
from __future__ import annotations

import math

EMBED_MODEL = "nomic-embed-text"
CHAT_FRAC = 0.2     # share of the candidate pool reserved for chat-lively chunks


def embed(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Embed each string -> a vector. Ollama embeddings API (local import)."""
    import ollama  # optional heavy dep
    try:
        ollama.show(model)            # cheap presence check
    except ollama.ResponseError:
        ollama.pull(model)            # first run: fetch it (idempotent)
    return [[float(x) for x in ollama.embeddings(model=model, prompt=t)["embedding"]]
            for t in texts]


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return num / (na * nb) if na and nb else 0.0


def retrieve_with_vec(query_vec: list[float], chunks: list[dict], k: int,
                      chat_frac: float = CHAT_FRAC) -> list[dict]:
    """Pure ranking: top-k chunks for the brief, by cosine + a chat safety net.

    Each chunk carries 'embedding' (and optionally 'chat_z'). Most slots go to the
    highest cosine similarity (on-topic); a small `chat_frac` slice is reserved for
    the liveliest-chat chunks so a pure-vibe brief ("funniest moments"), whose
    cosine ranking is near-meaningless, still surfaces lively candidates for the LLM
    scorer to judge. Returns chunks with an added 'sim' (cosine to the brief).
    """
    if not chunks:
        return []
    k = max(1, min(k, len(chunks)))
    sims = [_cosine(query_vec, c["embedding"]) for c in chunks]
    by_sim = sorted(range(len(chunks)), key=lambda i: sims[i], reverse=True)
    n_chat = min(max(1, int(k * chat_frac)), k - 1) if k > 1 else 0
    picked = list(by_sim[: k - n_chat])
    seen = set(picked)
    for i in sorted(range(len(chunks)), key=lambda i: chunks[i].get("chat_z", 0.0),
                    reverse=True):
        if len(picked) >= k:
            break
        if i not in seen:
            picked.append(i)
            seen.add(i)
    return [{**chunks[i], "sim": round(float(sims[i]), 4)} for i in picked]


def retrieve(brief: str, chunks: list[dict], k: int, model: str = EMBED_MODEL) -> list[dict]:
    """Top-k chunks most relevant to `brief` (embeds the brief, then ranks)."""
    if not chunks:
        return []
    qv = embed([brief], model=model)[0]
    return retrieve_with_vec(qv, chunks, k)


def _demo() -> None:
    chunks = [
        {"text": "a", "embedding": [1.0, 0.0], "chat_z": 0.0},   # most on-topic
        {"text": "b", "embedding": [0.0, 1.0], "chat_z": 9.0},   # off-topic but loud
        {"text": "c", "embedding": [0.9, 0.1], "chat_z": 0.0},
        {"text": "d", "embedding": [0.8, 0.2], "chat_z": 0.0},
    ]
    top = retrieve_with_vec([1.0, 0.0], chunks, 4)
    texts = [c["text"] for c in top]
    assert texts[0] == "a", texts          # cosine leads (chat doesn't dominate)
    assert "b" in texts, texts             # but the loud chunk still makes the pool
    print("retrieve ok")


if __name__ == "__main__":
    _demo()
