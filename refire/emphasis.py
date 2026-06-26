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


def loud_word_flags(wav_path: str | Path, words: list[Word]) -> list[bool]:
    """Per-word True if its audio RMS spikes above the clip baseline.

    Reads the mono 16k WAV (stdlib `wave`). Returns all-False if the file is
    missing/unreadable so emphasis silently degrades to keyword-only.
    """
    wav_path = Path(wav_path)
    if not wav_path.exists() or not words:
        return [False] * len(words)
    try:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        sig = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    except (wave.Error, OSError, ValueError):
        return [False] * len(words)

    rms = []
    for w in words:
        a, b = int(w["start"] * sr), int(w["end"] * sr)
        seg = sig[a:b]
        rms.append(float(np.sqrt(np.mean(seg * seg))) if seg.size else 0.0)
    arr = np.asarray(rms)
    voiced = arr[arr > 0]
    if voiced.size == 0:
        return [False] * len(words)
    thr = max(float(np.median(voiced)) * LOUD_K, float(np.percentile(voiced, 85)))
    return [r > thr for r in rms]


def annotate_emphasis(words: list[Word], wav_path: str | Path) -> list[Word]:
    """Add `emph` to each word in place: keyword OR all-caps OR yelled."""
    loud = loud_word_flags(wav_path, words)
    for w, hot in zip(words, loud):
        w["emph"] = bool(hot or _is_keyword(w["text"]) or _is_allcaps(w["text"]))
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
