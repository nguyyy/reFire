"""Polish events found in a rendered video that has no project (04-sauce.md section 3, extended to
Nguy's exports + his channel's uploads). Each detector is checked against the four edits whose
project gives the exact events (mine.validate).

- booms: the vine-boom file itself, normalized cross-correlation on the mixed audio
- emotes: his emotes sit dead center (Motion 0.5:0.5 in every edit), so each library image is
  matched (masked by its alpha) at a few display sizes against the frame's center, 2 frames a second
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

SR = 16000
BOOM_FILE = Path.home() / "Desktop" / "editing pack" / "sfx" / "actualSFXPackCurated" / "pops" / "vine-boom.mp3"
BOOM_HEAD = 0.6            # s of the file matched: the hit + body, which survive a trim or a fade out
BOOM_MIN = 0.5             # normalized correlation a boom clears (tuned in mine.validate)
BOOM_SEP = 0.4             # s; two hits closer than this are one


def audio(path, sr: int = SR) -> np.ndarray:
    """mono float32 at `sr` via ffmpeg."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def _xcorr(x: np.ndarray, tpl: np.ndarray, block: int = 1 << 20) -> np.ndarray:
    """sum(x[i:i+n] * tpl) for every valid i, by fft in overlapping blocks (no scipy here)."""
    n = len(tpl)
    m = len(x) - n + 1
    if m <= 0:
        return np.zeros(0)
    size = 1 << int(np.ceil(np.log2(block + n)))
    T = np.conj(np.fft.rfft(tpl.astype(np.float64), size))
    out = np.empty(m)
    for s in range(0, m, block):
        seg = x[s:s + block + n - 1].astype(np.float64)
        r = np.fft.irfft(np.fft.rfft(seg, size) * T, size)
        k = min(block, m - s)
        out[s:s + k] = r[:k]
    return out


def ncc(x: np.ndarray, tpl: np.ndarray) -> np.ndarray:
    """normalized cross-correlation of tpl against every offset of x (1 = an exact, scaled copy)."""
    tpl = tpl - tpl.mean()
    n = len(tpl)
    num = _xcorr(x, tpl)
    c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    c1 = np.concatenate([[0.0], np.cumsum(x.astype(np.float64))])
    energy = (c[n:] - c[:-n]) - (c1[n:] - c1[:-n]) ** 2 / n
    return num / (np.sqrt(np.maximum(energy, 1e-12)) * np.linalg.norm(tpl))


def peaks(score: np.ndarray, sr: int, thr: float, sep: float) -> list[tuple[float, float]]:
    """(seconds, score) of local maxima >= thr, at least `sep` apart (highest first wins)."""
    idx = np.where(score >= thr)[0]
    out: list[tuple[float, float]] = []
    for i in idx[np.argsort(-score[idx])]:
        t = i / sr
        if all(abs(t - o) >= sep for o, _ in out):
            out.append((t, float(score[i])))
    return sorted(out)


def booms(path_or_audio, thr: float = BOOM_MIN, sr: int = SR) -> list[tuple[float, float]]:
    """(start of the vine boom file, score) for every boom in a video."""
    x = audio(path_or_audio, sr) if not isinstance(path_or_audio, np.ndarray) else path_or_audio
    tpl = audio(BOOM_FILE, sr)[: int(BOOM_HEAD * sr)]
    return peaks(ncc(x, tpl), sr, thr, BOOM_SEP)


# --------------------------------------------------------------------------- emotes

EMOTE_DIR = Path.home() / "Desktop" / "editing pack" / "icons" / "emote"   # his curated emotes
EMOTE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
FW, FH = 480, 270                                            # frames are matched at this size
HEIGHTS = (56, 80, 110, 150)                                 # displayed emote heights at 270p (his: ~200-600 px at 1080p)
EMOTE_FPS = 2.0
EMOTE_MIN = 0.8                                              # tuned in mine.validate (precision first)
BLUR = 1.5                                                   # px sigma on both sides: a 1-2 px offset from dead center stops mattering


def library(extra: list | None = None, d: Path = EMOTE_DIR) -> dict[str, Path]:
    """emote name -> file: his emote folder + any emote file an edit used that lives elsewhere."""
    out = {p.stem: p for p in sorted(Path(d).glob("*")) if p.suffix.lower() in EMOTE_EXT} if Path(d).exists() else {}
    for p in extra or []:
        if Path(p).exists() and Path(p).suffix.lower() in EMOTE_EXT:
            out.setdefault(Path(p).stem, Path(p))
    return out


def _templates(lib: dict[str, Path]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(name, flat pixel indices around the frame center, zero-mean unit template) per emote x gif
    frame (up to 4) x display height, in color, alpha-masked."""
    import cv2
    from PIL import Image, ImageSequence
    out = []
    for name, p in lib.items():
        try:
            fr = [f.convert("RGBA") for f in ImageSequence.Iterator(Image.open(p))]
        except Exception:                                    # an unreadable file is just not in the library
            continue
        for f in fr[:: max(1, len(fr) // 4)][:4]:
            a = np.asarray(f)
            for h in HEIGHTS:
                w = max(8, round(h * a.shape[1] / a.shape[0]))
                if w > FW or h > FH:
                    continue
                r = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
                rgb = cv2.GaussianBlur(r[:, :, :3].astype(np.float32), (0, 0), BLUR)
                ys, xs = np.nonzero(r[:, :, 3] > 128)
                if len(ys) < 0.2 * h * w:
                    continue
                v = rgb[ys, xs].reshape(-1)
                v = v - v.mean()
                if v.std() < 8:                              # flat = matches anything
                    continue
                Y, X = ys + (FH - h) // 2, xs + (FW - w) // 2
                idx = ((Y * FW + X)[:, None] * 3 + np.arange(3)).reshape(-1)
                out.append((name, idx, v / np.linalg.norm(v)))
    return out


def rgb_frames(path, fps: float = EMOTE_FPS):
    """(t, blurred float32 FWxFHx3 frame, flattened) via ffmpeg."""
    import cv2
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={FW}:{FH}",
                          "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    n, i = FW * FH * 3, 0
    try:
        while (buf := p.stdout.read(n)) and len(buf) == n:
            f = np.frombuffer(buf, np.uint8).reshape(FH, FW, 3).astype(np.float32)
            yield i / fps, cv2.GaussianBlur(f, (0, 0), BLUR).reshape(-1)
            i += 1
    finally:
        p.kill()


def emotes(path, lib: dict[str, Path] | None = None, thr: float = EMOTE_MIN, fps: float = EMOTE_FPS) -> list[dict]:
    """[{t, end, name, score}]: one library emote at the frame center for >= 2 samples in a row
    (his emotes hold >= 1.1 s). ponytail: exact-center masked ncc; an emote off center is missed."""
    tpls = _templates(lib or library())
    hits = []
    for t, f in rgb_frames(path, fps):
        best = (0.0, "")
        for name, idx, v in tpls:
            c = f[idx]
            c = c - c.mean()
            n = np.linalg.norm(c)
            s = float(c @ v / n) if n > 1e-6 else 0.0
            if s > best[0]:
                best = (s, name)
        hits.append((t, *best))
    runs: list[dict] = []
    for t, s, name in hits:                                  # consecutive samples of one emote = one event
        if s < thr * 0.85:
            continue
        if runs and runs[-1]["name"] == name and t - runs[-1]["end"] <= 1.5 / fps:
            r = runs[-1]
            r["end"], r["score"], r["n"] = t, max(r["score"], s), r["n"] + 1
        else:
            runs.append({"t": t, "end": t, "name": name, "score": s, "n": 1})
    return [{k: r[k] for k in ("t", "end", "name", "score")} for r in runs if r["n"] >= 2 and r["score"] >= thr]
