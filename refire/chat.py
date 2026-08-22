"""Turn a Twitch chat replay into a message-rate z-score signal -- and into vocabulary.

Input: TwitchDownloader chat JSON. Shape (the fields we use):
    {"comments": [{"content_offset_seconds": 12.5,
                   "message": {"body": "KINICH PULL",
                               "fragments": [{"text": ..., "emoticon": null|{...}}]}}]}
A chat-rate spike is the strongest cheap signal that something exciting happened.
Chat *text* is a second, unrelated signal: viewers spell proper nouns correctly at the
exact second the streamer says them, which is precisely what the transcriber gets wrong.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=4)
def _load(path: str, _stamp: tuple = ()) -> list[dict]:
    """Parsed comment list, memoized per (path, mtime, size). Missing/empty/bad -> [].

    A 6h chat replay is tens of MB of JSON and three callers want it; parse it once.
    `_stamp` is part of the cache key only -- see `_comments`.
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return []
    comments = data.get("comments", []) if isinstance(data, dict) else data
    return comments if isinstance(comments, list) else []


def _comments(chat_json: str | Path) -> list[dict]:
    p = Path(chat_json)
    try:
        st = p.stat()
        stamp = (st.st_mtime_ns, st.st_size)
    except OSError:
        return []
    return _load(str(p), stamp)


def _message_offsets(chat_json: str | Path) -> list[float]:
    """Extract per-message offset seconds. Missing/empty -> []."""
    out = []
    for c in _comments(chat_json):
        t = c.get("content_offset_seconds")
        if t is not None:
            out.append(float(t))
    return out


def chat_signal(
    chat_json: str | Path,
    duration: float,
    window: float = 5.0,
    offset: float = 0.0,
) -> list[tuple[float, float]]:
    """Bin messages into `window`-second buckets, return (t_center, zscore).

    Empty/missing chat -> all-zero signal (transcript-only fallback). `offset` is where
    the media starts in the stream (a `--start` window): chat offsets are absolute, the
    rest of the pipeline is zero-based off the downloaded file.
    """
    n_bins = max(1, int(np.ceil(duration / window)))
    centers = [(i + 0.5) * window for i in range(n_bins)]

    offsets = _message_offsets(chat_json)
    if not offsets:
        return [(c, 0.0) for c in centers]

    counts = np.zeros(n_bins)
    for t in offsets:
        t -= offset
        if t < 0:                      # before the window we downloaded
            continue
        idx = min(int(t // window), n_bins - 1)
        counts[idx] += 1

    std = counts.std()
    z = (counts - counts.mean()) / std if std > 0 else np.zeros(n_bins)
    return list(zip(centers, z.tolist()))


_TOKEN = re.compile(r"[A-Za-z][A-Za-z'\-]{3,}")
# Titlecase words that are ordinary English, not names. An ordinary word in the hotword
# list drags near-miss real words onto it (see genshin.txt's header), so these must go.
# The capitalization-ratio guard below catches the long tail; this covers the words chat
# capitalizes so consistently that the ratio can't tell them from a name.
_STOP = frozenset("""
that this what when where they them then than there their your yours just like about
have hard here holy hell fuck shit damn bruh chat guys good nice wait real true yeah
yeah nope stop dont cant wont didnt does dude okay omfg lmao lmfao gonna wanna
what's thats it's i'm you're we're he's she's don't can't didn't
""".split())


def _fragment_texts(comment: dict):
    """Non-emote text of one message. Emote fragments are dropped: 'KEKW' and 'Pepega'
    are chat vocabulary, never something the streamer says aloud."""
    msg = comment.get("message") or {}
    frags = msg.get("fragments")
    if isinstance(frags, list) and frags:
        for f in frags:
            if isinstance(f, dict) and f.get("emoticon") is None and f.get("text"):
                yield str(f["text"])
    elif msg.get("body"):
        yield str(msg["body"])


def chat_terms(chat_json: str | Path, min_count: int = 3, limit: int = 60,
               cap_ratio: float = 0.6) -> list[str]:
    """Mine the VOD's own chat for proper nouns the transcriber will mangle.

    A token qualifies when it is seen >= `min_count` times and is capitalized in at least
    `cap_ratio` of those (names are; "that" and "just" are not, which is what separates
    them without a giant stopword list). ALL-CAPS-only tokens are excluded -- that's chat
    slang and emote spam, not a name. Returned in frequency order, most-said first.

    Free and local: no LLM, no network. Complements `glossary.game_glossary`, whose local
    model simply doesn't know characters released after its training cutoff -- but chat
    was typing their names all stream.

    NOTE: this is EVIDENCE FOR AN LLM, not a decoder bias list. Measured on a real VOD it
    returns real names (Mualani, Ceekay, ShihiroMori) mixed with 7TV/BTTV emotes (Gayge,
    Smoge -- TwitchDownloader only tags *native* Twitch emotes, so the fragment filter
    can't see those) and ordinary words (Tasty, Check). A reader with judgment ignores the
    junk; whisper's `hotwords` cannot, and genshin.txt's header spells out what an ordinary
    word in that list does. So this feeds `caption_fix`, NOT `transcribe`.
    """
    total: Counter = Counter()
    capped: Counter = Counter()
    for c in _comments(chat_json):
        for text in _fragment_texts(c):
            for tok in _TOKEN.findall(text):
                key = tok.lower()
                if key in _STOP:
                    continue
                total[key] += 1
                if tok[0].isupper() and not tok.isupper():
                    capped[key] += 1
    hits = [(n, k) for k, n in total.items()
            if n >= min_count and capped[k] / n >= cap_ratio]
    hits.sort(key=lambda kn: (-kn[0], kn[1]))
    # canonical spelling = Titlecase; the decoder and snap_to_glossary both want a form,
    # and `emphasis.style_text` lowercases captions anyway.
    return [k.capitalize() if k.islower() else k for _, k in hits[:limit]]


def chat_names(chat_json: str | Path, min_count: int = 2, limit: int = 40) -> list[str]:
    """Display names of the stream's regulars, most-active first.

    Higher precision than `chat_terms` -- a display name is a proper noun by construction,
    no emote/English ambiguity -- and it covers the case a game glossary structurally
    cannot: PEOPLE. "zhegef" is the streamer's friend Zajef; no Genshin term list will ever
    hold that, but the people around the stream are right here.
    """
    names: Counter = Counter()
    for c in _comments(chat_json):
        n = ((c.get("commenter") or {}).get("display_name") or "").strip()
        if len(n) >= 3:
            names[n] += 1
    return [n for n, k in names.most_common() if k >= min_count][:limit]


def chat_lines(chat_json: str | Path, t0: float, t1: float, limit: int = 25,
               offset: float = 0.0) -> list[str]:
    """Non-emote message bodies sent in [t0, t1). `offset` is where the media starts in
    the stream (a `--start` window), since chat offsets are absolute. Newest dropped
    first once `limit` is hit -- the earliest reactions are the ones on topic."""
    out: list[str] = []
    for c in _comments(chat_json):
        t = c.get("content_offset_seconds")
        if t is None:
            continue
        t = float(t) - offset
        if t < t0:
            continue
        if t >= t1:
            break                       # comments are chronological
        body = " ".join(_fragment_texts(c)).strip()
        if body:
            out.append(body)
        if len(out) >= limit:
            break
    return out


def _demo() -> None:
    import tempfile
    chat = [{"content_offset_seconds": i,
             "message": {"body": b, "fragments": [{"text": b, "emoticon": None}]}}
            for i, b in enumerate(["Kinich pull", "Kinich!", "that Kinich",
                                   "that that", "that", "KEKW KEKW KEKW"])]
    chat.append({"content_offset_seconds": 9,
                 "message": {"body": "PogChamp",
                             "fragments": [{"text": "PogChamp", "emoticon": {"id": "1"}}]}})
    for c in chat:
        c["commenter"] = {"display_name": "Zajef"}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.json"
        p.write_text(json.dumps({"comments": chat}), encoding="utf-8")
        terms = chat_terms(p, min_count=3)
        assert terms == ["Kinich"], terms          # 'that' stopped, KEKW all-caps, emote skipped
        assert chat_names(p) == ["Zajef"]
        assert chat_lines(p, 0, 2) == ["Kinich pull", "Kinich!"]
        assert chat_lines(p, 0, 2, offset=1.0) == ["Kinich!", "that Kinich"]
        assert chat_terms(Path(d) / "missing.json") == []
        assert chat_names(Path(d) / "missing.json") == []
    print("chat ok")


if __name__ == "__main__":
    _demo()
