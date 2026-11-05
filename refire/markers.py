"""Review markers: flag the cut points a human will most likely have to fix.

Every clip boundary gets a risk (probability the editor moves it), learned from past
finished edits: `refire learn` stores each proposed clip's boundary features and whether
the human moved that edge (clip_diff.json `boundaries`); `fit` turns those rows into a
naive-Bayes model. With no history yet the measured 2026-11 priors below are used.

OFF by default (`make --markers`). Measured 2026-11, leave-one-out over 4 finished
edits, cold-open teasers excluded: precision 0.24-0.42 against 0.30 for marking every
edge, not monotonic in the threshold -- no better than chance yet. (An earlier 0.57 was
an artifact: teaser edges are short and mid-sentence, and "hit" edits through the VOD
time they share with their beats while sitting at the start of the video.) The model
retrains from every `refire learn`, so re-check with `refire eval --all --markers`.

Features are cheap and deterministic, measured over 196 kept clips in 4 finished edits
(edit rate when the feature is on vs off):
  start: speech within 1s before the cut  16% vs 4%  (the clip opens mid-thought)
  end:   speech within 1s after the cut   18% vs 7%
         loudness z>=1 around the end     22% vs 10% (and almost never an overstay)
         last word has no .?!             29% vs 11%
A referent word opening the clip ("he", "that", "wait") barely moved it (12% vs 9%).

Markers ride on manifest clips as `clip["markers"] = [{at, name, comment, risk, ...}]`
(`at` = seconds from the clip head on its dead-air-free timeline), so the Premiere and AE
builds place them from wherever the clip actually lands. markers.csv is the fallback.
ASCII only: these strings end up on a cp1252 console.
"""
from __future__ import annotations

import csv
import math
import re
from bisect import bisect_left
from pathlib import Path

START_F = ("speech_before_03", "speech_before_1", "referent", "loud_before")
END_F = ("speech_after_03", "speech_after_1", "no_terminal", "loud_end",
         "chat_after_1", "chat_after_2")
# one model per fix direction: speech right before a cut predicts EXTENDING the start but
# anti-predicts trimming it, so a single "edited" label washes both out
KINDS = {"start_late": START_F, "start_early": START_F,
         "end_early": END_F, "end_late": END_F}
REF = re.compile(r"^(he|she|they|it|that|this|those|these|him|her|them|again|wait|so|"
                 r"and|but|because|cause|then|also|which)\b", re.I)
# fallback when no finished edits exist yet: `fit` over the 2026-11 data (196 clips, 4
# VODs), as prior + {feature: (P(on|fix), P(on|no fix))}
DEFAULT: dict = {
    "start_late": {"prior": 0.084, "p": {
        "speech_before_03": (0.529, 0.291),
        "speech_before_1": (0.765, 0.486),
        "referent": (0.294, 0.229),
        "loud_before": (0.176, 0.297),
    }},
    "start_early": {"prior": 0.079, "p": {
        "speech_before_03": (0.375, 0.307),
        "speech_before_1": (0.375, 0.523),
        "referent": (0.375, 0.222),
        "loud_before": (0.375, 0.278),
    }},
    "end_early": {"prior": 0.126, "p": {
        "speech_after_03": (0.28, 0.269),
        "speech_after_1": (0.64, 0.551),
        "no_terminal": (0.16, 0.108),
        "loud_end": (0.36, 0.24),
        "chat_after_1": (0.4, 0.329),
        "chat_after_2": (0.12, 0.096),
    }},
    "end_late": {"prior": 0.074, "p": {
        "speech_after_03": (0.2, 0.277),
        "speech_after_1": (0.6, 0.559),
        "no_terminal": (0.2, 0.107),
        "loud_end": (0.2, 0.26),
        "chat_after_1": (0.333, 0.339),
        "chat_after_2": (0.067, 0.102),
    }},
}
THRESHOLD = 0.26      # --marker-threshold default: ~11/video in the 2026-11 LOO eval
NEAR_S = 3.0          # a marker "hits" an edit within this many VOD seconds
MIN_ROWS = 60         # fewer training boundaries than this -> DEFAULT
COLD_OPEN = "Cold Open"   # narrative.cast's section title for the teaser / stack


def features(ctx, ks: float, ke: float) -> tuple[dict, dict, dict]:
    """(start feats, end feats, info for the reason text) for a clip shown [ks, ke].
    `ctx` is a learn.diff.Context (words + loudness + chat)."""
    W, S = ctx.words, ctx.starts
    k = bisect_left(S, ks)
    prev = W[k - 1] if k > 0 else None
    gap_b = ks - prev["end"] if prev else 99.0
    first = " ".join(w["text"] for w in W[k:k + 2]).strip(" ,.!?\"'")
    j = bisect_left(S, ke)
    nxt = W[j] if j < len(W) else None
    gap_a = nxt["start"] - ke if nxt else 99.0
    last = W[j - 1]["text"] if 0 < j <= len(W) else ""
    lz_s = (ctx._stat(ctx.loud, ks - 3, ks + 1) or {}).get("max") or 0.0
    lz_e = (ctx._stat(ctx.loud, ke - 1, ke + 3) or {}).get("max") or 0.0
    cz_e = (ctx._stat(ctx.chat, ke, ke + 8) or {}).get("max") or 0.0
    sf = {"speech_before_03": gap_b < 0.3, "speech_before_1": gap_b < 1.0,
          "referent": bool(REF.match(first)), "loud_before": lz_s >= 1.0}
    ef = {"speech_after_03": gap_a < 0.3, "speech_after_1": gap_a < 1.0,
          "no_terminal": bool(last) and last[-1:] not in ".?!", "loud_end": lz_e >= 1.0,
          "chat_after_1": cz_e >= 1.0, "chat_after_2": cz_e >= 2.0}
    info = {"gap_before": gap_b, "gap_after": gap_a, "first": first[:24],
            "loud_start": lz_s, "loud_end": lz_e, "chat_end": cz_e}
    return sf, ef, info


def training_rows(diffs, exclude=()) -> list[tuple[str, dict, bool]]:
    """(kind, feats, fixed?) per kept clip edge per fix direction, from clip_diff.json."""
    out = []
    for d in diffs:
        if d["vod"] in exclude:
            continue
        for b in d.get("boundaries") or []:
            for kind in KINDS:
                side = kind.split("_")[0]
                out.append((kind, b[f"{side}_feats"], b[f"{side}_label"] == kind))
    return out


def fit(rows) -> dict:
    """Naive Bayes per fix direction, Laplace-smoothed. Falls back to DEFAULT below
    MIN_ROWS boundaries -- that few, one VOD's quirks would be the whole model."""
    if len(rows) < MIN_ROWS * len(KINDS) and DEFAULT:
        return DEFAULT
    model = {}
    for kind, names in KINDS.items():
        rs = [(f, y) for k, f, y in rows if k == kind]
        pos = [f for f, y in rs if y]
        neg = [f for f, y in rs if not y]
        p = {n: (round((sum(f.get(n, False) for f in pos) + 1) / (len(pos) + 2), 3),
                 round((sum(f.get(n, False) for f in neg) + 1) / (len(neg) + 2), 3))
             for n in names}
        model[kind] = {"prior": round((len(pos) + 1) / (len(rs) + 2), 3), "p": p}
    return model


def risk(model: dict, kind: str, feats: dict) -> tuple[float, str | None]:
    """(P(the editor makes this fix), the feature that raised it most)."""
    m = model[kind]
    lo = math.log(m["prior"] / (1 - m["prior"]))
    best, top = 0.0, None
    for n, (pe, po) in m["p"].items():
        on = feats.get(n, False)
        lr = math.log(pe / po) if on else math.log((1 - pe) / (1 - po))
        lo += lr
        if on and lr > best:
            best, top = lr, n
    return 1 / (1 + math.exp(-lo)), top


def _gap(g: float, where: str) -> str:
    # the visible edge carries compress_silence's breath pad, so the neighbouring word can
    # overlap it -- that reads as "runs straight into", not as a negative gap
    if g < 0.1:
        return "speech runs straight into the cut" if where == "before" else \
            "speech runs straight past the cut"
    return f"speech {g:.1f}s {where} the cut"


def _reason(top: str | None, kind: str, info: dict) -> str:
    if kind == "start_early":
        return "start may include lead-in -- consider trimming in"
    if kind == "end_late":
        return "end may overstay -- consider trimming out"
    return {
        "speech_before_03": "start may be mid-thought -- " + _gap(info["gap_before"], "before"),
        "speech_before_1": "start may be mid-thought -- " + _gap(info["gap_before"], "before"),
        "referent": f"opens on '{info['first']}' -- the setup may be earlier",
        "loud_before": f"start may miss the build-up -- loud z={info['loud_start']:.1f} just before",
        "speech_after_03": "end may cut off speech -- " + _gap(info["gap_after"], "after"),
        "speech_after_1": "end may cut off speech -- " + _gap(info["gap_after"], "after"),
        "no_terminal": "end lands mid-sentence -- the payoff may continue",
        "loud_end": f"end may cut off the reaction -- loudness z={info['loud_end']:.1f}",
        "chat_after_1": f"end may cut off the reaction -- chat still z={info['chat_end']:.1f}",
        "chat_after_2": f"end may cut off the reaction -- chat still z={info['chat_end']:.1f}",
    }.get(top, "start may be too late" if kind == "start_late" else "end may be too early")


def clip_markers(clips, ctx, model, threshold: float = THRESHOLD) -> int:
    """Attach `markers` to each clip dict ({start, end, keep?}) in place. Returns count.
    One marker per edge at most: the riskiest fix direction for that edge."""
    n = 0
    for c in clips:
        if c.get("section") == COLD_OPEN:      # teasers are short and mid-line on purpose
            c.setdefault("markers", [])
            continue
        keep = c.get("keep") or [[c["start"], c["end"]]]
        ks, ke = keep[0][0], keep[-1][1]
        sf, ef, info = features(ctx, ks, ke)
        out = []
        for side, f, vod in (("start", sf, ks), ("end", ef, ke)):
            p, kind, top = max((*risk(model, k, f)[:1], k, risk(model, k, f)[1])
                               for k in KINDS if k.startswith(side))
            if p >= threshold:
                out.append({"side": side, "kind": kind, "vod": round(vod, 2),
                            "risk": round(p, 2), "name": f"reFire: {kind.replace('_', ' ')}",
                            "comment": _reason(top, kind, info)})
        c["markers"] = (c.get("markers") or []) + out
        n += len(out)
    return n


def marker_score(markers, records) -> dict:
    """How useful the markers were against what the human actually changed.
    precision = markers within NEAR_S of an edit; recall = fix_cost seconds with a marker
    within NEAR_S of the edited span."""
    edits = [r for r in records if r["label"] != "reordered"]

    def near(t, r):
        return r["vod_start"] - NEAR_S <= t <= r["vod_end"] + NEAR_S

    hits = sum(any(near(m["vod"], r) for r in edits) for m in markers)
    tot = sum(abs(r["delta_s"]) for r in edits)
    cov = sum(abs(r["delta_s"]) for r in edits if any(near(m["vod"], r) for m in markers))
    return {"n": len(markers), "precision": round(hits / len(markers), 3) if markers else None,
            "recall_s": round(cov / tot, 3) if tot else None}


def write_csv(manifest: dict, path) -> Path:
    """markers.csv in Premiere master-timeline time (clips laid end to end, as srt.py)."""
    from .srt import _ordered_clips
    path = Path(path)
    rows, at = [], 0.0
    for c in _ordered_clips(manifest):
        dur = c.get("dur") or (c["end"] - c["start"])
        for m in c.get("markers") or []:
            t = at + min(max(m.get("at", 0.0), 0.0), dur)
            rows.append([f"{int(t) // 60:02d}:{t % 60:06.3f}", round(t, 3), m.get("vod", ""),
                         m.get("side", ""), m.get("risk", ""), m.get("comment", "")])
        at += dur
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timecode", "master_s", "vod_s", "kind", "risk", "reason"])
        w.writerows(rows)
    return path


def annotate_manifest(manifest_path, ctx, model, threshold: float = THRESHOLD) -> int:
    """Add boundary markers to a built manifest's clips (in place) + write markers.csv.

    `at` is seconds from the clip head on its tight timeline: 0 for the start, 0.5s
    before the tail for the end, so it reads inside the right clip in either host."""
    import json
    mp = Path(manifest_path)
    man = json.loads(mp.read_text(encoding="utf-8"))
    n = 0
    section = {k: sec.get("title", "") for sec in man.get("sections") or []
               for k in sec.get("clip_indices") or []}
    for ci, c in enumerate(man.get("clips") or []):
        a = float(c["start"])
        keep = [[a + x, a + y] for x, y in (c.get("keep") or [[0.0, c["end"] - a]])]
        tmp = {"start": a, "end": float(c["end"]), "keep": keep,
               "section": section.get(ci, "")}
        n += clip_markers([tmp], ctx, model, threshold)
        dur = c.get("dur") or (c["end"] - a)
        for m in tmp["markers"]:
            m["at"] = 0.0 if m["side"] == "start" else round(max(0.0, dur - 0.5), 3)
        keep_rel = c.get("keep") or [[0.0, c["end"] - a]]
        for m in c.get("markers") or []:           # critic notes carry a VOD time
            if m.get("side") == "critic":
                m["at"] = round(sum(max(0.0, min(kb, m["vod"] - a) - ka)
                                    for ka, kb in keep_rel), 3)
                tmp["markers"].append(m)
                n += 1
        for x, y in c.get("cut") or []:            # each trimmed span is worth a glance
            at = sum(max(0.0, min(kb, x - a) - ka) for ka, kb in
                     (c.get("keep") or [[0.0, c["end"] - a]]))
            tmp["markers"].append({"side": "trim", "kind": "trim", "vod": round(x, 2),
                                   "risk": 0.5, "at": round(at, 3),
                                   "name": "reFire: trimmed",
                                   "comment": f"trimmed {y - x:.1f}s of filler -- check the flow"})
            n += 1
        if tmp["markers"]:
            c["markers"] = tmp["markers"]
    mp.write_text(json.dumps(man, indent=2), encoding="utf-8")
    write_csv(man, mp.parent / "markers.csv")
    return n
