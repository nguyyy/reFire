"""Loudest-moments compilation across many VODs, without downloading any of them whole.

`make` reads a stream for STORY: the full video down, six hours transcribed, a Claude
director and critic. None of that is how you find screaming. This path downloads audio
only (~440MB per 6h stream against ~15GB for the mp4), scores it for loudness SPIKES, and
fetches video for just the seconds that survive -- a few minutes of transfer per stream
instead of a few hours.

The cut spans several VODs, so clips are laid on one VIRTUAL timeline: VOD i sits at
`i * VOD_BASE` seconds. Word times, clip times and each source's offset all carry that
base, so `clip.start - sources[src].offset` still lands inside that clip's own downloaded
window -- which is the only arithmetic the Premiere panel does. Everything downstream
(`srt.build_srt`, `srt.recaption`, `reFirePpro.jsx`) stays single-source-shaped and needed
no changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import perception
from .audio import extract_audio
from .ingest import DL_THREADS, ensure_audio, ensure_vod
from .score import DEFAULT_MODEL
from .select import SILENCE_PAD, budget_select, snap_to_phrase
from .transcribe import DEFAULT_BATCH_SIZE, DEFAULT_WHISPER_MODEL

# Spacing between VODs on the virtual timeline. Longer than any stream by a wide margin,
# and round enough that a manifest timestamp still reads as "VOD 3, 41 minutes in".
VOD_BASE = 100_000.0

BUCKET_S = 1.0        # loudness resolution. perception's 5s default sizes the director's
                      # moment map; it is far too coarse to place a cut on.
BASELINE_S = 60.0     # rolling median span: "how loud has it been AROUND here"
CLIP_S = 14.0         # total shot length
LEAD_S = 4.0          # of which this much is run-up, before the spike window
PAD_S = 5.0           # extra seconds downloaded each side, so the snap has room to move
PER_VOD = 12          # candidate windows scanned out of each stream


def spike_signal(wav_path, window: float = BUCKET_S,
                 baseline_s: float = BASELINE_S) -> list[tuple[float, float]]:
    """Loudness z-score MINUS its own rolling median -> [(t_center, spike)].

    Same shape as `perception.loudness_signal`, so it drops straight into
    `perception.top_windows`. The subtraction is the entire point: on raw level, a stretch
    that is merely loud THROUGHOUT -- a boss fight, a hot-mixed menu, a music bed --
    outranks a sudden scream, because a window sum wins on duration rather than on peak.
    Against a local baseline only the DEPARTURE from the surrounding level scores, which
    is what a person means by "the loudest moment".

    Missing/unreadable wav -> [], exactly like the signal it wraps.
    """
    sig = perception.loudness_signal(wav_path, window=window)
    if not sig:
        return []
    v = np.array([z for _, z in sig], dtype=np.float64)
    w = max(3, int(round(baseline_s / window)) | 1)      # odd -> the median is centered
    if v.size <= w:
        return [(t, float(z)) for t, z in sig]           # too short to have a baseline
    base = np.median(sliding_window_view(np.pad(v, w // 2, mode="edge"), w), axis=-1)
    return [(t, float(a - b)) for (t, _), a, b in zip(sig, v, base)]


def _score_windows(sig, wins, lead_s: float) -> list[dict]:
    """(t0, t1) windows -> budget_select rows, each extended back by `lead_s` of run-up.

    Score is the window's own spike mass (negative buckets contribute nothing, matching
    `top_windows`), so the ranking that picked the windows is the one that fills the
    duration budget across VODs.
    """
    ts = np.array([t for t, _ in sig])
    zs = np.clip([z for _, z in sig], 0.0, None)
    out = []
    for a, b in wins:
        lo, hi = np.searchsorted(ts, a), np.searchsorted(ts, b)
        out.append({"a": max(0.0, a - lead_s), "b": b, "score": float(zs[lo:hi].sum())})
    return out


def scan(vod_id, cache_dir="vods", per_vod: int = PER_VOD, clip_s: float = CLIP_S,
         lead_s: float = LEAD_S, baseline_s: float = BASELINE_S,
         threads: int = DL_THREADS) -> list[dict]:
    """One VOD -> its top spike windows, audio-only. Returns budget_select rows.

    The wav lands at `run/<vod>/audio.wav` -- the same cache `make` uses, and the audio-
    only rendition is the same audio, so a later `make` on this VOD reuses it instead of
    re-extracting it from a 15GB mp4.
    """
    run_dir = Path("run") / str(vod_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    wav = run_dir / "audio.wav"
    if not wav.exists():          # the download is the expensive half -- only if needed
        extract_audio(ensure_audio(vod_id, cache_dir, threads), wav)
    sig = spike_signal(wav, baseline_s=baseline_s)
    if not sig:
        print(f"[scan] {vod_id}: no readable audio -- skipped")
        return []
    perception.save_signals(run_dir / "spikes.json", sig)   # inspectable, not make's cache
    span = max(2.0, clip_s - lead_s)
    rows = _score_windows(sig, perception.top_windows(sig, k=per_vod, span_s=span), lead_s)
    print(f"[scan] {vod_id}: {sig[-1][0] / 3600:.1f}h scanned, {len(rows)} candidates")
    return rows


def loud(
    vod_ids,
    duration_s: float,
    out_dir: str | Path | None = None,
    cache_dir: str | Path = "vods",
    per_vod: int = PER_VOD,
    clip_s: float = CLIP_S,
    lead_s: float = LEAD_S,
    pad_s: float = PAD_S,
    baseline_s: float = BASELINE_S,
    game: str = "",
    terms: str = "",
    words_per_line: int = 3,
    deadspace: bool = True,
    silence_pad: float = SILENCE_PAD,
    model: str = DEFAULT_MODEL,
    transcriber: str = "local",
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    compute_type: str = "float16",
    threads: int = DL_THREADS,
) -> Path:
    """Several VOD ids -> one Premiere manifest of their loudest moments. Returns its path.

    Audio-only per stream to find the spikes, then a windowed video download per surviving
    moment, transcribed on its own (minutes of audio, not hours) so the captions exist
    without whisper ever running over a whole VOD.
    """
    from .ae_export import build_manifest
    from .emphasis import annotate_emphasis
    from .glossary import game_glossary
    from .pipeline import _run_name
    from .srt import write_srt
    from .transcribe import transcribe

    vod_ids = [str(v) for v in vod_ids]
    out_dir = Path(out_dir) if out_dir else Path("run") / _run_name("loud")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {out_dir.name}  ({len(vod_ids)} VODs, target {duration_s / 60:.0f}m)")

    cands = []
    for i, vid in enumerate(vod_ids):
        base = i * VOD_BASE
        for row in scan(vid, cache_dir, per_vod, clip_s, lead_s, baseline_s, threads):
            cands.append({**row, "vod": vid, "base": base,
                          "start": base + row["a"], "end": base + row["b"]})
    if not cands:
        raise SystemExit("No loud moments found -- nothing to cut.")

    picked, warning = budget_select(cands, duration_s, order="chrono")
    print(f"[pick] {len(picked)} moments of {len(cands)} candidates, "
          f"~{sum(c['end'] - c['start'] for c in picked) / 60:.1f}m before tightening")

    hotwords = ", ".join(game_glossary(game, model=model, cache_dir=out_dir, extra=terms))
    clips, sources, words = [], [], []
    for i, c in enumerate(picked):
        # ponytail: the downloader crops on INTEGER seconds, so a window's offset is the
        # floor/ceil of what we asked for -- exact, as long as --trim-mode stays Exact
        # (its default). If captions ever drift by about a second, that is the thing to
        # check first; the fix is reading the real start back off the downloaded file.
        dl_a = float(int(max(0.0, c["a"] - pad_s)))
        dl_b = float(int(c["b"] + pad_s) + 1)
        print(f"[fetch] {i + 1}/{len(picked)}  vod {c['vod']} "
              f"{dl_a / 60:.1f}m ({dl_b - dl_a:.0f}s)")
        video, _chat = ensure_vod(c["vod"], cache_dir, start=dl_a, end=dl_b,
                                  want_chat=False, threads=threads)
        wdir = out_dir / "windows" / f"{i:03d}"
        wav = wdir / "audio.wav"
        if not wav.exists():
            extract_audio(video, wav)
        w = transcribe(wav, cache_path=wdir / "transcript.json", hotwords=hotwords,
                       backend=transcriber, model_size=whisper_model,
                       batch_size=batch_size, compute_type=compute_type)
        # Per-window RMS thresholds: inside an already-loud moment only the genuinely
        # louder words should shout in caps.
        annotate_emphasis(w, wav)
        off = c["base"] + dl_a       # the window file starts at t=0; lift it onto the
        w = [{**x, "start": x["start"] + off, "end": x["end"] + off} for x in w]
        # Snap the real in/out INSIDE the padded download -- that is what pad_s bought.
        # Clamped to the window: snapping may move the cut, but it must never point the
        # manifest at footage this file does not contain.
        a, b = snap_to_phrase(w, c["start"], c["end"], max_pad=pad_s)
        a, b = max(a, off), min(b, off + (dl_b - dl_a))
        clips.append({"start": a, "end": max(b, a + 1.0), "src": i})
        sources.append((video, off))
        words.extend(w)

    mp = build_manifest(sources[0][0], out_dir, words, [{"title": "", "clips": clips}],
                        words_per_line, motion_zoom=False, deadspace=deadspace,
                        silence_pad=silence_pad, cards=False, proxy=False,
                        sources=sources)
    srt = write_srt(mp)
    print(f"Manifest: {mp}\nCaptions: {srt}")
    if warning:
        print("WARNING:", warning)
    return mp
