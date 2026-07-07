"""Fuse what the stream SAYS with how it FEELS and LOOKS into one moment map.

The v1 director read a bare transcript. A human editor also hears the room get loud,
sees chat explode, and watches the screen. This module fuses those channels -- transcript
sentences, chat-rate z-score (chat.py), audio-loudness z-score (RMS over the run's wav),
and optional local-VLM scene captions (vlm.py) -- into the "moment map v2" text the
director reads, plus `top_windows` (the highest-signal spans, used to pick which footage
gets contact-sheeted for Claude's eyes).

Everything here is pure fusion over precomputed signals: no models, no ffmpeg, so it's
unit-testable. Empty/missing signals degrade the map to exactly the v1 transcript map.
"""
from __future__ import annotations

import wave
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np

# z-score thresholds for the (chat!)/(chat!!) and (loud)/(LOUD) line marks
CHAT_HI = (1.5, 3.0)
LOUD_HI = (1.5, 3.0)


def _sentences(words: list[dict]) -> list[tuple[int, float, str]]:
    """Transcript words -> (start_s, end_s, sentence) tuples, breaking on .?!

    The shared splitter behind `director.stream_map` (v1) and `moment_map` (v2), so both
    maps agree on line boundaries and timestamps.
    """
    out: list[tuple[int, float, str]] = []
    cur: list[str] = []
    sent_start = 0
    sent_end = 0.0
    for w in words:
        if not cur:
            sent_start = int(w["start"])
        cur.append(w["text"])
        sent_end = float(w["end"])
        if w["text"][-1:] in ".?!":
            out.append((sent_start, sent_end, " ".join(cur)))
            cur = []
    if cur:   # trailing words with no terminal punctuation
        out.append((sent_start, sent_end, " ".join(cur)))
    return out


def loudness_signal(wav_path: str | Path, window: float = 5.0) -> list[tuple[float, float]]:
    """Bucket the run's mono 16k WAV into `window`-second RMS z-scores.

    Same (t_center, z) shape as `chat.chat_signal` so the two fuse symmetrically.
    Missing/unreadable wav -> [] (map degrades to no loudness marks).
    """
    wav_path = Path(wav_path)
    if not wav_path.exists():
        return []
    try:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        sig = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    except (wave.Error, OSError, ValueError):
        return []
    if sig.size == 0 or sr <= 0:
        return []
    step = int(sr * window)
    n_bins = int(np.ceil(sig.size / step))
    rms = np.empty(n_bins)
    for i in range(n_bins):
        seg = sig[i * step:(i + 1) * step]
        rms[i] = np.sqrt(np.mean(seg * seg)) if seg.size else 0.0
    std = rms.std()
    z = (rms - rms.mean()) / std if std > 0 else np.zeros(n_bins)
    centers = [(i + 0.5) * window for i in range(n_bins)]
    return list(zip(centers, z.tolist()))


class _ZLookup:
    """max-z-over-a-time-span lookup on a (t_center, z) signal (sorted by t)."""

    def __init__(self, signal: list[tuple[float, float]] | None):
        signal = signal or []
        self.t = [t for t, _ in signal]
        self.z = [z for _, z in signal]
        # bucket half-width, inferred from spacing so spans catch adjacent buckets
        self.half = (self.t[1] - self.t[0]) / 2.0 if len(self.t) > 1 else 2.5

    def max_in(self, a: float, b: float) -> float:
        if not self.t:
            return 0.0
        lo = bisect_left(self.t, a - self.half)
        hi = bisect_right(self.t, b + self.half)
        return max(self.z[lo:hi], default=0.0)


def _marks(chat: _ZLookup, loud: _ZLookup, a: float, b: float) -> str:
    """The line-prefix marks for a sentence spanning [a, b]. '' when nothing spikes."""
    out = ""
    cz = chat.max_in(a, b)
    if cz >= CHAT_HI[1]:
        out += "(chat!!)"
    elif cz >= CHAT_HI[0]:
        out += "(chat!)"
    lz = loud.max_in(a, b)
    if lz >= LOUD_HI[1]:
        out += "(LOUD)"
    elif lz >= LOUD_HI[0]:
        out += "(loud)"
    return out


def moment_map(
    words: list[dict],
    chat_z: list[tuple[float, float]] | None = None,
    audio_z: list[tuple[float, float]] | None = None,
    captions: dict[float, str] | None = None,
) -> str:
    """The moment map v2: the v1 transcript map enriched with signal marks + scene lines.

    Lines look like `[546s] (chat!!)(LOUD) NO WAY dude`; VLM scene captions interleave in
    time order as their own `[550s] [scene: player dies to boss]` lines. With no signals
    and no captions this returns exactly `director.stream_map(words)`.
    """
    chat = _ZLookup(chat_z)
    loud = _ZLookup(audio_z)
    scenes = sorted((captions or {}).items())   # [(t, caption)]
    si = 0
    lines: list[str] = []
    for a, b, text in _sentences(words):
        while si < len(scenes) and scenes[si][0] <= a:
            t, cap = scenes[si]
            lines.append(f"[{int(t)}s] [scene: {cap}]")
            si += 1
        m = _marks(chat, loud, a, b)
        lines.append(f"[{a}s] {m + ' ' if m else ''}{text}")
    for t, cap in scenes[si:]:
        lines.append(f"[{int(t)}s] [scene: {cap}]")
    return "\n".join(lines)


def top_windows(
    chat_z: list[tuple[float, float]] | None,
    audio_z: list[tuple[float, float]] | None,
    k: int = 8,
    span_s: float = 120.0,
) -> list[tuple[float, float]]:
    """The k highest-combined-signal non-overlapping (t0, t1) windows, sorted by t0.

    Combined per-bucket signal = max(0, chat_z) + max(0, audio_z); each window scores the
    sum over `span_s`. Greedy pick keeps windows disjoint. Empty signals -> [].
    """
    sig: dict[float, float] = {}
    for series in (chat_z, audio_z):
        for t, z in series or []:
            sig[t] = sig.get(t, 0.0) + max(0.0, z)
    if not sig:
        return []
    ts = sorted(sig)
    step = ts[1] - ts[0] if len(ts) > 1 else 5.0
    vals = np.array([sig[t] for t in ts])
    w = max(1, int(round(span_s / step)))
    # sliding-window sums; candidate i spans buckets [i, i+w)
    sums = np.convolve(vals, np.ones(w), mode="valid") if len(vals) >= w \
        else np.array([vals.sum()])
    order = np.argsort(sums)[::-1]
    picked: list[tuple[float, float]] = []
    for i in order:
        if len(picked) >= k or sums[i] <= 0:
            break
        t0 = ts[int(i)] - step / 2.0
        t1 = t0 + w * step
        if any(t0 < b and a < t1 for a, b in picked):
            continue
        picked.append((max(0.0, t0), t1))
    return sorted(picked)


def excerpt(map_text: str, a: float, b: float) -> str:
    """The map lines whose `[Ns]` stamp falls inside [a, b] (unstamped lines ride along)."""
    from .director import _parse_stamp
    out: list[str] = []
    keeping = False
    for line in map_text.splitlines():
        t = _parse_stamp(line)
        if t is not None:
            keeping = a <= t <= b
        if keeping:
            out.append(line)
    return "\n".join(out)


def _merge_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sort + merge overlapping/touching (a, b) spans."""
    out: list[tuple[float, float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def chapter_guide(chapters) -> str:
    """The coarse table of contents the story pass always reads (pure)."""
    lines = ["CHAPTER GUIDE (what each stretch of the stream is about):"]
    for i, c in enumerate(chapters, 1):
        lines.append(f"{i:02d}. [{int(c.start_s)}s-{int(c.end_s)}s] {c.title}"
                     + (f" -- {c.summary}" if c.summary else ""))
        for n in c.notable_moments:
            lines.append(f"      * [{int(n.t_s)}s] {n.why}")
    return "\n".join(lines)


def digest(chapters, map_text: str, windows: list[tuple[float, float]] | None = None,
           pad_s: float = 60.0, budget_chars: int = 500_000) -> str:
    """A context-sized director input for maps too big to send whole (pure).

    Chapter guide first (the whole stream, coarse), then full-resolution map excerpts for
    the regions that matter -- the top-signal `windows` plus every chapter's notable
    moments +-pad_s -- merged and emitted in stream order until `budget_chars` runs out.
    """
    parts = [chapter_guide(chapters), ""]
    spans = [(max(0.0, a - pad_s), b + pad_s) for a, b in windows or []]
    for c in chapters:
        spans += [(max(0.0, n.t_s - pad_s), n.t_s + pad_s) for n in c.notable_moments]
    used = sum(len(p) for p in parts)
    parts.append("FULL-RESOLUTION EXCERPTS (exact transcript around the key moments; "
                 "anchor start_s/end_s/segments to THESE lines):")
    for a, b in _merge_spans(spans):
        ex = excerpt(map_text, a, b)
        if not ex:
            continue
        block = f"\n--- excerpt [{int(a)}s-{int(b)}s] ---\n{ex}"
        if used + len(block) > budget_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


def save_signals(path: str | Path, chat_z, audio_z) -> None:
    """Cache both bucket series to run/<vod>/signals.json (cheap to recompute, but the
    file doubles as an inspectable artifact -- 'did the pipeline hear that scream?')."""
    import json
    Path(path).write_text(
        json.dumps({"chat": chat_z or [], "audio": audio_z or []}), encoding="utf-8")


def load_signals(path: str | Path):
    """-> (chat_z, audio_z) from `save_signals`, or (None, None) if absent/corrupt."""
    import json
    p = Path(path)
    if not p.exists():
        return None, None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return ([tuple(x) for x in d.get("chat", [])],
                [tuple(x) for x in d.get("audio", [])])
    except (ValueError, TypeError):
        return None, None
