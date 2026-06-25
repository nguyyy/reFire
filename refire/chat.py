"""Turn a Twitch chat replay into a message-rate z-score signal over time.

Input: TwitchDownloader chat JSON. Shape (only fields we use):
    {"comments": [{"content_offset_seconds": 12.5}, ...]}
A chat-rate spike is the strongest cheap signal that something exciting happened.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _message_offsets(chat_json: str | Path) -> list[float]:
    """Extract per-message offset seconds. Missing/empty -> []."""
    p = Path(chat_json)
    if not p.exists() or p.stat().st_size == 0:
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    comments = data.get("comments", []) if isinstance(data, dict) else data
    out = []
    for c in comments:
        t = c.get("content_offset_seconds")
        if t is not None:
            out.append(float(t))
    return out


def chat_signal(
    chat_json: str | Path,
    duration: float,
    window: float = 5.0,
) -> list[tuple[float, float]]:
    """Bin messages into `window`-second buckets, return (t_center, zscore).

    Empty/missing chat -> all-zero signal (transcript-only fallback).
    """
    n_bins = max(1, int(np.ceil(duration / window)))
    centers = [(i + 0.5) * window for i in range(n_bins)]

    offsets = _message_offsets(chat_json)
    if not offsets:
        return [(c, 0.0) for c in centers]

    counts = np.zeros(n_bins)
    for t in offsets:
        idx = min(int(t // window), n_bins - 1)
        counts[idx] += 1

    std = counts.std()
    z = (counts - counts.mean()) / std if std > 0 else np.zeros(n_bins)
    return list(zip(centers, z.tolist()))
