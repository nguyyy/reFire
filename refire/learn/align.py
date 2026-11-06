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
    title = {}
    for sec in manifest.get("sections") or []:
        for k in sec.get("clip_indices") or []:
            title[k] = sec.get("title", "")
    clips = manifest.get("clips") or []
    idx = {id(c): k for k, c in enumerate(clips)}
    out = []
    for i, c in enumerate(ordered_clips(manifest)):
        a, b = float(c["start"]), float(c["end"])
        keep = [[a + x, a + y] for x, y in (c.get("keep") or [[0.0, b - a]])]
        out.append({"i": i, "start": a, "end": b, "keep": keep,
                    "role": c.get("role", ""), "src": c.get("src"),
                    "section": title.get(idx.get(id(c)), "")})
    return out


def proposal_from_proxies(paths, words=None, voiced=None, pad: float = 0.3) -> list[dict]:
    """Rebuild a lost manifest's cut from the proxies a project imported.

    One proxy per clip, cut at [start - PROXY_PAD, end + PROXY_TAIL], so the clip itself
    is recoverable from the name. `keep` is re-derived with the same compress_silence the
    manifest used when a transcript is available; without one the clip is kept whole.

    Play order is the order `paths` first names each proxy: pass the project's bin media
    in document order. The panel imports proxies in manifest order and Premiere keeps
    it -- on the runs that still have a manifest the bin order is the play order exactly,
    and August builds played in story order, not chronologically. A clip lying inside a
    later-playing clip is a teaser: section 'Cold Open', as the manifest would say (the
    same rule picks exactly the manifests' cold opens).
    """
    from ..select import compress_silence
    spans: dict[tuple[float, float], None] = {}           # insertion-ordered set
    for p in paths:
        m = PROXY_RE.search(Path(str(p).replace("\\", "/")).name)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            spans.setdefault((round(a + PROXY_PAD if a > 0 else a, 3), round(b - PROXY_TAIL, 3)))
    order = list(spans)
    out = []
    for i, (a, b) in enumerate(order):
        keep = [[a, b]]
        if words:
            k, _r, _d = compress_silence(words, a, b, pad=pad, voiced=voiced)
            keep = [[a + x, a + y] for x, y in k]
        teaser = any(j > i and x <= a + 0.5 and b <= y + 0.5 for j, (x, y) in enumerate(order))
        out.append({"i": i, "start": a, "end": b, "keep": keep, "role": "", "src": None,
                    "section": "Cold Open" if teaser else ""})
    return out


_TOK = re.compile(r"[a-z0-9']+")
CUE_SLACK = 60.0     # most dead air one gap between two cues of a clip can have lost


def _toks(s: str) -> list[str]:
    return _TOK.findall(s.lower())


def locate_cues(cues, words, spans) -> list[float | None]:
    """reFire caption cue [(cut t0, t1, text)] -> VOD time of its first word, or None.

    Text lookup in the transcript, in cut order. A cue that continues the previous
    located cue (VOD advanced by its cut advance plus at most CUE_SLACK of removed dead
    air) takes the nearest such hit; otherwise it needs a single hit inside a proxy clip
    (2+ words) or in the whole VOD (4+ words). A 3+ word cue may drop an end word, since
    the build's transcript and today's re-transcription differ here and there.
    """
    T, TS = [], []
    for w in words:
        for t in _toks(w["text"]):
            T.append(t)
            TS.append(float(w["start"]))
    first: dict[str, list[int]] = {}
    for i, t in enumerate(T):
        first.setdefault(t, []).append(i)

    def hits(q):
        return [p for p in first.get(q[0], []) if T[p:p + len(q)] == q]

    out, prev = [], None
    for a, _b, text in cues:
        q = _toks(text)
        if not q:
            out.append(None)
            continue
        ps = hits(q)
        if not ps and len(q) >= 3:
            ps = sorted(set(hits(q[1:])) | {p - 1 for p in hits(q[:-1]) if p > 0})
        vs = [TS[p] for p in ps]
        pick = None
        if prev is not None:
            pa, pv = prev
            near = [v for v in vs if -0.6 <= (v - pv) - (a - pa) <= CUE_SLACK]
            if near:
                pick = min(near, key=lambda v: (v - pv) - (a - pa))
        if pick is None:
            inside = [v for v in vs if any(x - 1 <= v <= y + 1 for x, y in spans)]
            if len(q) >= 2 and len(inside) == 1:
                pick = inside[0]
            elif len(q) >= 4 and len(vs) == 1:
                pick = vs[0]
        if pick is not None:
            prev = (a, pick)
        out.append(pick)
    return out


CUE_DENSITY = 0.5    # cues/s; reFire's full captions run ~1.2/s, the Aug emphasis-only ~0.17


def _cue_words(cues, loc, spans) -> list[dict]:
    """Located cues as word-like spans in VOD time; an unlocated cue borrows the offset
    of the cue before it (no dead air is cut inside a run of speech)."""
    out, prev = [], None
    for (a, b, t), v in zip(cues, loc):
        if v is None:
            if prev is None:
                continue
            v = a + (prev[1] - prev[0])
            if not any(x - 1 <= v <= y + 1 for x, y in spans):
                continue
        else:
            prev = (a, v)
        out.append({"text": t, "start": v, "end": v + (b - a)})
    return sorted(out, key=lambda w: w["start"])


def use_captions(prop: list[dict], cues, words, voiced=None,
                 pad: float = 0.3) -> tuple[list[dict], dict]:
    """Check a rebuilt proposal against the captions.srt its build shipped, and take the
    dead-air keep the captions support.

    The cues are reFire's own cut in cut time (Premiere keeps an imported .srt's cues
    inside the project, so they outlive the run folder) and they are the build's own
    transcript. Each located cue goes to the proxy clip holding it -- in an overlap the
    previous cue's clip wins, so a teaser does not steal its source clip's lines -- and a
    clip sits at the median cut time of its longest run of cues: `order_agree_pct` is
    the share of consecutive clips whose caption positions increase in `prop` order.

    Keep: today's re-transcription can hear speech the build's didn't and vice versa
    (a clip the build heard nothing in was kept whole). With dense captions an
    alternative keep is cut from today's words plus the cue spans, clips no cue reached
    kept whole -- on the two runs with a manifest it reproduces the real keep (100%
    overlap, 2-6s extra) -- and whichever total is closer to the captions' length wins.
    """
    from ..select import compress_silence
    spans = [(c["start"], c["end"]) for c in prop]
    loc = locate_cues(cues, words, spans)
    runs: dict[int, list[list[float]]] = {}
    prev_k = None
    for (a, _b, _t), v in zip(cues, loc):
        if v is None:
            continue
        ks = [k for k, (x, y) in enumerate(spans) if x - 1 <= v <= y + 1]
        if not ks:
            prev_k = None
            continue
        k = prev_k if prev_k in ks else min(ks, key=lambda k: spans[k][1] - spans[k][0])
        if k != prev_k:
            runs.setdefault(k, []).append([])
        runs[k][-1].append(a)
        prev_k = k
    pos = {k: sorted(max(rs, key=len))[len(max(rs, key=len)) // 2] for k, rs in runs.items()}
    seq = [pos[k] for k in sorted(pos)]
    agree = sum(seq[n] < seq[n + 1] for n in range(len(seq) - 1)) / max(1, len(seq) - 1)
    end = max((b for _a, b, _t in cues), default=0.0)
    total = lambda p: sum(y - x for c in p for x, y in c["keep"])
    n_loc = sum(v is not None for v in loc)
    check = {"cues": len(cues), "located_pct": round(100 * n_loc / max(1, len(cues)), 1),
             "clips_reached": len(runs), "clips": len(prop),
             "order_agree_pct": round(100 * agree, 1),
             "captions_end_s": round(end, 1), "rebuilt_s": round(total(prop), 1),
             "keep_from": "transcript"}
    if not end or len(cues) / end < CUE_DENSITY:
        return prop, check
    union = sorted(list(words) + _cue_words(cues, loc, spans), key=lambda w: w["start"])
    alt = []
    for k, c in enumerate(prop):
        keep = [[c["start"], c["end"]]]
        if k in runs:
            kk, _r, _d = compress_silence(union, c["start"], c["end"], pad=pad, voiced=voiced)
            keep = [[c["start"] + x, c["start"] + y] for x, y in kk]
        alt.append(dict(c, keep=keep))
    check["with_cues_s"] = round(total(alt), 1)
    if abs(total(alt) - end) < abs(total(prop) - end):
        check["keep_from"] = "transcript + captions"
        return alt, check
    return prop, check


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
