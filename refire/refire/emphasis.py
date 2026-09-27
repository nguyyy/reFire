"""Flag emphasized words so captions render lowercase by default, CAPS on hype.

Emphasis = a curated interjection (lol/wtf/...), a word the transcript already
wrote in ALL CAPS, or a word that was audibly yelled (RMS spike vs the clip).
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .transcribe import Word

EMPHASIS_WORDS = {
    "lol", "lmao", "lmfao", "rofl", "wtf", "omg", "omfg", "wth", "bruh", "bro",
    "pog", "pogchamp", "poggers", "insane", "crazy", "clutch", "ez", "gg",
    "holy", "damn", "yo", "wow", "woah", "whoa", "no", "nooo", "yes", "yess",
    "lets", "let's", "huge", "what", "sheesh", "nani", "actually",
}

LOUD_K = 1.8          # word RMS must exceed median*K (or the 85th pct) to be "loud"
_CLEAN = str.maketrans("", "", ".,!?\"'")


def _is_keyword(text: str) -> bool:
    return text.lower().translate(_CLEAN) in EMPHASIS_WORDS


def _is_allcaps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)


def word_rms(wav_path: str | Path, words: list[Word]) -> list[float]:
    """Per-word audio RMS off the mono 16k WAV (stdlib `wave`). All-zeros if unreadable.

    The transient signal, at word resolution. `perception.loudness_signal` buckets the
    same wav at 5s for the moment map's (loud) marks, which is far too coarse to cut on;
    this is the fine-grained view, and it is the one primitive both the emphasis flags
    and `select.snap_to_transient` need -- so it lives here once rather than being
    re-derived per caller.
    """
    wav_path = Path(wav_path)
    if not wav_path.exists() or not words:
        return [0.0] * len(words)
    try:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        sig = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    except (wave.Error, OSError, ValueError):
        return [0.0] * len(words)

    out = []
    for w in words:
        a, b = int(w["start"] * sr), int(w["end"] * sr)
        seg = sig[a:b]
        out.append(float(np.sqrt(np.mean(seg * seg))) if seg.size else 0.0)
    return out


def loud_word_flags(wav_path: str | Path, words: list[Word]) -> list[bool]:
    """Per-word True if its audio RMS spikes above the clip baseline.

    Returns all-False if the wav is missing/unreadable so emphasis silently degrades to
    keyword-only.
    """
    rms = word_rms(wav_path, words)
    arr = np.asarray(rms)
    voiced = arr[arr > 0]
    if voiced.size == 0:
        return [False] * len(words)
    thr = max(float(np.median(voiced)) * LOUD_K, float(np.percentile(voiced, 85)))
    return [r > thr for r in rms]


def annotate_emphasis(words: list[Word], wav_path: str | Path) -> list[Word]:
    """Add `emph` to each word in place: keyword OR all-caps OR yelled.

    Also stamps the raw `rms` it measured, so a later pass that needs to cut ON the audio
    peak (`select.snap_to_transient`) reads it off the words instead of re-opening and
    re-scanning a multi-hour wav.
    """
    rms = word_rms(wav_path, words)
    arr = np.asarray(rms)
    voiced = arr[arr > 0]
    thr = (max(float(np.median(voiced)) * LOUD_K, float(np.percentile(voiced, 85)))
           if voiced.size else float("inf"))
    for w, r in zip(words, rms):
        w["rms"] = r
        w["emph"] = bool(r > thr or _is_keyword(w["text"]) or _is_allcaps(w["text"]))
    return words


def style_text(text: str, emph: bool) -> str:
    """Lowercase by default; UPPERCASE when the word is emphasized."""
    return text.upper() if emph else text.lower()


def _demo() -> None:
    assert _is_keyword("LOL!") and not _is_keyword("character")
    assert _is_allcaps("WTF") and not _is_allcaps("Hi")
    assert style_text("Insane", True) == "INSANE"
    assert style_text("Character", False) == "character"
    # missing wav -> no loudness, but keyword still flags
    ws = [{"text": "lol", "start": 0.0, "end": 0.2},
          {"text": "ok", "start": 0.3, "end": 0.5}]
    annotate_emphasis(ws, "does_not_exist.wav")
    assert ws[0]["emph"] and not ws[1]["emph"]
    print("emphasis ok")


if __name__ == "__main__":
    _demo()
