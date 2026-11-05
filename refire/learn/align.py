"""Map a finished Premiere timeline back to run-local VOD seconds.

"VOD seconds" here means the run's own clock: 0-based in the downloaded window for a
`--start/--end` run (the clock transcript.json, signals.json and the manifest share),
with `window_start` reported so absolute stream time is one addition away.

Per media item, in order of trust:
1. the path is one of the manifest's `sources[]` (proxies, raw-VOD parts, a `loud` run's
   virtual timeline at i*100000s): VOD = offset + source time
2. a reFire proxy name `p_<profile>_<a>-<b>.mov`: VOD = a + source time (works with the
   manifest gone -- ae_export cuts proxies from a = clip start - PROXY_PAD)
3. the manifest's raw `source`: VOD = source time
4. a `rough*.mp4`: rough time -> manifest clip -> VOD, through each clip's `keep` spans
5. anything else (memes, SFX, graphics, other VODs) -> overlays, context not cut decisions
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from . import prproj

PROXY_RE = re.compile(r"p_[a-z0-9_]+?_(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\.(?:mov|mp4)$", re.I)
PROXY_PAD, PROXY_TAIL = 2.0, 10.0      # ae_export's values (unchanged since 2026-08)
WINDOW_RE = re.compile(r"@(\d+)-(\d+|end)$")


def norm(p) -> str:
    return str(p or "").replace("\\", "/").lower()


def window_start(cache_dir) -> float:
    m = WINDOW_RE.search(Path(cache_dir).name)
    return float(m.group(1)) if m else 0.0


def ordered_clips(manifest: dict) -> list[dict]:
    """Clips in layout order (sections, then clip_indices) -- the order build() and the
    rough render lay them out. Same rule as srt._ordered_clips."""
    from ..srt import _ordered_clips
    return _ordered_clips(manifest)


def proposal(manifest: dict) -> list[dict]:
    """reFire's cut as VOD spans: one entry per manifest clip, in play order.

    `keep` is what actually reached V1 (build() places one trackItem per kept span);
    `start`/`end` is the clip envelope before dead-air removal.
    """
    out = []
    for i, c in enumerate(ordered_clips(manifest)):
        a, b = float(c["start"]), float(c["end"])
        keep = [[a + x, a + y] for x, y in (c.get("keep") or [[0.0, b - a]])]
        out.append({"i": i, "start": a, "end": b, "keep": keep,
                    "role": c.get("role", ""), "src": c.get("src")})
    return out


def proposal_from_proxies(paths, words=None, voiced=None, pad: float = 0.3) -> list[dict]:
    """Rebuild a lost manifest's cut from the proxies a project imported.

    One proxy per clip, cut at [start - PROXY_PAD, end + PROXY_TAIL], so the clip itself
    is recoverable from the name. `keep` is re-derived with the same compress_silence the
    manifest used when a transcript is available; without one the clip is kept whole.
    Play order is assumed chronological (the default --order).
    """
    from ..select import compress_silence
    spans = set()
    for p in paths:
        m = PROXY_RE.search(Path(str(p).replace("\\", "/")).name)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            spans.add((round(a + PROXY_PAD if a > 0 else a, 3), round(b - PROXY_TAIL, 3)))
    out = []
    for i, (a, b) in enumerate(sorted(spans)):
        keep = [[a, b]]
        if words:
            k, _r, _d = compress_silence(words, a, b, pad=pad, voiced=voiced)
            keep = [[a + x, a + y] for x, y in k]
        out.append({"i": i, "start": a, "end": b, "keep": keep, "role": "", "src": None})
    return out


def _rough_map(manifest: dict) -> list[tuple[float, float, float]]:
    """[(rough_t0, rough_t1, vod_t0)] pieces of the ffmpeg rough cut's timeline."""
    out, acc = [], 0.0
    for c in ordered_clips(manifest):
        a = float(c["start"])
        for x, y in (c.get("keep") or [[0.0, float(c["end"]) - a]]):
            out.append((acc, acc + (y - x), a + x))
            acc += y - x
    return out


def _segment(it, vod_a, vod_b, method) -> dict:
    sp = prproj.speed(it)
    return {"vod_start": round(vod_a, 3), "vod_end": round(vod_b, 3),
            "seq_in": round(it["seq_in"], 3), "seq_out": round(it["seq_out"], 3),
            "track": it["track"], "speed": round(sp, 4), "method": method,
            "media": it["media"]}


def map_items(items: list[dict], manifest: dict | None, run_dir=None,
              run_key: str = "") -> dict:
    """Final-timeline items -> {segments, overlays, flags, methods}.

    `run_key` ("<cache dir>/<run name>", lowercase-insensitive) limits proxy-name
    mapping to THIS run's proxies -- a project can hold several runs' cuts, and a proxy
    from another VOD would otherwise land on this VOD's clock.
    """
    manifest = manifest or {}
    srcs = {norm(s["path"]): float(s.get("offset", 0.0))
            for s in manifest.get("sources") or []}
    raw = norm(manifest.get("source")) if manifest.get("source") else None
    rough = _rough_map(manifest) if manifest.get("clips") else []
    rk = norm(run_key).replace("\\", "/")
    segs, overlays, flags, methods = [], [], [], {}
    for it in items:
        m = norm(it["media"])
        name = Path(m).name
        vod = None
        if m in srcs:
            vod, how = srcs[m] + it["src_in"], "sources"
        elif PROXY_RE.search(name) and (not rk or f"/run/{rk}/" in m):
            vod, how = float(PROXY_RE.search(name).group(1)) + it["src_in"], "proxy_name"
        elif raw and m == raw:
            vod, how = it["src_in"], "raw_vod"
        elif name.startswith("rough") and name.endswith(".mp4") and rough \
                and (not run_dir or norm(Path(run_dir)) in m):
            for t0, t1, v0 in rough:            # an item can span several rough pieces
                a, b = max(it["src_in"], t0), min(it["src_out"], t1)
                if b > a:
                    segs.append(_segment(it, v0 + a - t0, v0 + b - t0, "rough"))
            methods["rough"] = methods.get("rough", 0) + 1
            continue
        if vod is None:
            overlays.append({"media": it["media"], "seq_in": round(it["seq_in"], 3),
                             "seq_out": round(it["seq_out"], 3), "track": it["track"],
                             "kind": it["kind"]})
            continue
        s = _segment(it, vod, vod + (it["src_out"] - it["src_in"]), how)
        if abs(s["speed"] - 1.0) > 0.01:
            flags.append({"flag": "speed", "seq_in": s["seq_in"], "speed": s["speed"]})
        if it["media"] and not os.path.exists(it["media"]):
            flags.append({"flag": "missing_media", "seq_in": s["seq_in"],
                          "media": it["media"]})
        segs.append(s)
        methods[how] = methods.get(how, 0) + 1
    segs.sort(key=lambda s: (s["seq_in"], s["track"]))
    return {"segments": segs, "overlays": overlays, "flags": flags, "methods": methods}


def load_manifest(run_dir) -> dict | None:
    p = Path(run_dir) / "ae" / "manifest.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
