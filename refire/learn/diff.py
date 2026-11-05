"""reFire's proposed cut vs the human's final cut, both in VOD seconds -> labelled edits.

Pure: span lists in, records out. Every record carries `delta_s`, the signed change in
kept footage (+ = the human kept more than reFire proposed, - = cut it), so a label's
median reads directly ("end_early median +3.8s").

Units: reFire proposes CLIPS (manifest clips, each a contiguous envelope whose dead air
was already jump-cut into `keep` spans). The human's final is a set of VOD spans. The
final's coverage is unioned, each coverage piece is attached to the clip(s) it touches,
and the clip's visible boundary (first/last keep span) is compared to the attached
envelope. Coverage touching two clips is a merge; coverage touching none is `added`.

| label | delta | meaning |
|---|---|---|
| start_late / start_early | + / - | start extended earlier / trimmed later |
| end_early / end_late | + / - | end extended / trimmed |
| interior_cut | - | a span inside a kept clip removed (filler) |
| restored | + | dead air reFire cut inside a clip, put back |
| dropped | - | whole clip removed |
| added | + | footage reFire never proposed (missed moment) |
| merged | + | two clips joined, the gap between them kept |
| reordered | 0 | same footage, different place in the order |

fix_cost = sum(|d| for boundary edits) + interior_cut + dropped + added, where `added`
also includes restored dead air and merge gaps: all three are seconds of VOD the human
used that reFire did not propose (the brief's t_added). Reorders cost 0 seconds.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from statistics import median

EPS = 0.12        # premiere snaps in/out to frames; below this two edges are the same
MIN_SPAN = 0.25   # edits smaller than this are frame jitter, not decisions
REACH = 12.0      # coverage this close outside a clip is that clip's boundary edit
CONTEXT_S = 10.0  # transcript context either side of an affected span

LABELS = ("start_late", "start_early", "end_early", "end_late", "interior_cut",
          "restored", "dropped", "added", "merged", "reordered")
BOUNDARY = ("start_late", "start_early", "end_early", "end_late")
ADDED_LIKE = ("added", "restored", "merged")


# --------------------------------------------------------------------------- spans

def union(spans, tol: float = EPS) -> list[list[float]]:
    out: list[list[float]] = []
    for a, b in sorted((float(a), float(b)) for a, b in spans if b > a):
        if out and a <= out[-1][1] + tol:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def intersect(xs, ys) -> list[list[float]]:
    out = []
    for a, b in xs:
        for c, d in ys:
            lo, hi = max(a, c), min(b, d)
            if hi > lo:
                out.append([lo, hi])
    return union(out, 0.0)


def subtract(xs, ys) -> list[list[float]]:
    """xs minus ys."""
    out = []
    ys = union(ys, 0.0)
    for a, b in xs:
        cur = a
        for c, d in ys:
            if d <= cur or c >= b:
                continue
            if c > cur:
                out.append([cur, c])
            cur = max(cur, d)
        if cur < b:
            out.append([cur, b])
    return out


def total(spans) -> float:
    return sum(b - a for a, b in spans)


def _big(spans) -> list[list[float]]:
    return [s for s in spans if s[1] - s[0] >= MIN_SPAN]


# --------------------------------------------------------------------------- context

class Context:
    """Transcript + signal lookups for diff records. All inputs optional."""

    def __init__(self, words=None, loud=None, chat=None, beats=None):
        self.words = words or []
        self.starts = [w["start"] for w in self.words]
        self.loud = sorted(loud or [])
        self.chat = sorted(chat or [])
        self.beats = beats or []

    def text(self, a: float, b: float) -> str:
        lo, hi = bisect_left(self.starts, a), bisect_left(self.starts, b)
        return " ".join(w["text"] for w in self.words[lo:hi])

    @staticmethod
    def _stat(sig, a: float, b: float) -> dict | None:
        if not sig:
            return None
        ts = [t for t, _ in sig]
        half = (ts[1] - ts[0]) / 2 if len(ts) > 1 else 2.5
        zs = [z for _, z in sig[bisect_left(ts, a - half):bisect_right(ts, b + half)]]
        if not zs:
            return None
        return {"mean": round(sum(zs) / len(zs), 2), "max": round(max(zs), 2)}

    def beat(self, a: float, b: float) -> dict | None:
        best, ov = None, 0.0
        for bt in self.beats:
            segs = bt.get("segments") or [[bt.get("start", 0), bt.get("end", 0)]]
            o = total(intersect([[a, b]], segs))
            if o > ov:
                best, ov = bt, o
        if not best:
            return None
        return {k: best.get(k) for k in ("title", "role", "energy", "texture", "intent")}

    def describe(self, a: float, b: float, clip: dict | None = None) -> dict:
        lo, hi = (clip["start"], clip["end"]) if clip else (a, b)
        return {"text": self.text(a, b),
                "before": self.text(a - CONTEXT_S, a),
                "after": self.text(b, b + CONTEXT_S),
                "signals": {"loud_z": self._stat(self.loud, a, b),
                            "chat_z": self._stat(self.chat, a, b),
                            "motion": None},
                "beat": self.beat(lo, hi)}


# --------------------------------------------------------------------------- diff

def _lis(xs: list[float]) -> set[int]:
    """Indices of one longest non-decreasing subsequence of xs (O(n^2), n ~ 60 clips).
    Non-decreasing: merged clips share one final position."""
    n = len(xs)
    best, prev = [1] * n, [-1] * n
    for i in range(n):
        for j in range(i):
            if xs[j] <= xs[i] and best[j] + 1 > best[i]:
                best[i], prev[i] = best[j] + 1, j
    if not n:
        return set()
    i, out = max(range(n), key=lambda k: best[k]), set()
    while i != -1:
        out.add(i)
        i = prev[i]
    return out


def diff(prop: list[dict], final: list[dict], ctx: Context | None = None) -> dict:
    """`prop` = align.proposal(...) clips (play order); `final` = align.map_items(...)
    segments. Returns {"records", "clips", "summary"}."""
    ctx = ctx or Context()
    cover = union([[s["vod_start"], s["vod_end"]] for s in final])
    vorder = sorted(range(len(prop)), key=lambda i: prop[i]["start"])
    vis = {i: (prop[i]["keep"][0][0], prop[i]["keep"][-1][1]) for i in range(len(prop))}

    # attach each coverage piece to the clips it touches, else to the nearest clip in reach
    attached: dict[int, list[list[float]]] = {i: [] for i in range(len(prop))}
    owners: list[list[int]] = []
    added = []
    for a, b in cover:
        hit = [i for i in vorder if a < vis[i][1] + EPS and b > vis[i][0] - EPS]
        if not hit:
            near = [(min(abs(a - vis[i][1]), abs(vis[i][0] - b)), i) for i in vorder]
            d, i = min(near, default=(None, None))
            hit = [i] if d is not None and d <= REACH else []
        owners.append(hit)
        for i in hit:
            attached[i].append([a, b])
        if not hit:
            added.append([a, b])

    # VOD-adjacent clips sharing one piece were joined into one continuous span
    merged_with = set()
    records: list[dict] = []
    for hit in owners:
        for x, y in zip(hit, hit[1:]):
            gap = vis[y][0] - vis[x][1]
            if gap < -EPS:
                continue        # overlapping clips (a cold open re-uses beat footage)
            merged_with |= {(x, "end"), (y, "start")}
            if gap >= MIN_SPAN:
                records.append({"label": "merged", "clips": [x, y], "vod_start": vis[x][1],
                                "vod_end": vis[y][0], "delta_s": round(gap, 2)})

    clips_out = []
    final_pos = {}
    for i, p in enumerate(prop):
        ks, ke = vis[i]
        keep_dur = total(p["keep"])
        pieces = attached[i]
        if not pieces:
            records.append({"label": "dropped", "clips": [i], "vod_start": ks,
                            "vod_end": ke, "delta_s": round(-keep_dur, 2)})
            clips_out.append({"i": i, "kept": False, "d_start": None, "d_end": None})
            continue
        fs = ks if (i, "start") in merged_with else min(a for a, _ in pieces)
        fe = ke if (i, "end") in merged_with else max(b for _, b in pieces)
        d_start, d_end = ks - fs, fe - ke          # + = human kept more
        if d_start >= MIN_SPAN:
            records.append({"label": "start_late", "clips": [i], "vod_start": fs,
                            "vod_end": ks, "delta_s": round(d_start, 2)})
        elif d_start <= -MIN_SPAN:
            records.append({"label": "start_early", "clips": [i], "vod_start": ks,
                            "vod_end": fs, "delta_s": round(d_start, 2)})
        if d_end >= MIN_SPAN:
            records.append({"label": "end_early", "clips": [i], "vod_start": ke,
                            "vod_end": fe, "delta_s": round(d_end, 2)})
        elif d_end <= -MIN_SPAN:
            records.append({"label": "end_late", "clips": [i], "vod_start": fe,
                            "vod_end": ke, "delta_s": round(d_end, 2)})
        win = [[max(ks, fs), min(ke, fe)]]
        for a, b in _big(subtract(intersect(p["keep"], win), cover)):
            records.append({"label": "interior_cut", "clips": [i], "vod_start": a,
                            "vod_end": b, "delta_s": round(a - b, 2)})
        for a, b in _big(subtract(intersect(pieces, win), p["keep"])):
            records.append({"label": "restored", "clips": [i], "vod_start": a,
                            "vod_end": b, "delta_s": round(b - a, 2)})
        seqs = [s["seq_in"] for s in final
                if s["vod_start"] < fe and s["vod_end"] > fs]
        final_pos[i] = min(seqs) if seqs else float("inf")
        clips_out.append({"i": i, "kept": True,
                          "d_start": round(d_start, 2), "d_end": round(d_end, 2)})

    for a, b in _big(added):
        records.append({"label": "added", "clips": [], "vod_start": a, "vod_end": b,
                        "delta_s": round(b - a, 2)})

    kept = [i for i in range(len(prop)) if i in final_pos]       # play order
    lis = _lis([final_pos[i] for i in kept])
    for k, i in enumerate(kept):
        if k not in lis:
            records.append({"label": "reordered", "clips": [i], "vod_start": vis[i][0],
                            "vod_end": vis[i][1], "delta_s": 0.0})

    for r in records:
        r["vod_start"], r["vod_end"] = round(r["vod_start"], 2), round(r["vod_end"], 2)
        clip = prop[r["clips"][0]] if r["clips"] else None
        r.update(ctx.describe(r["vod_start"], r["vod_end"], clip))
    records.sort(key=lambda r: (r["vod_start"], r["label"]))
    proposed_keep = union([s for p in prop for s in p["keep"]], 0.0)
    return {"records": records, "clips": clips_out,
            "summary": summarize(records, clips_out, cover, proposed_keep)}


def fix_cost(records) -> dict:
    by = {k: 0.0 for k in LABELS}
    for r in records:
        by[r["label"]] += abs(r["delta_s"])
    parts = {"boundary": sum(by[k] for k in BOUNDARY),
             "interior_cut": by["interior_cut"], "dropped": by["dropped"],
             "added": sum(by[k] for k in ADDED_LIKE)}
    return {"total": round(sum(parts.values()), 1),
            **{k: round(v, 1) for k, v in parts.items()}}


def markers_needed(records, cluster_s: float = 3.0) -> int:
    """Distinct spots an editor has to touch: records closer than `cluster_s` are one."""
    pts = sorted(r["vod_start"] for r in records if r["label"] != "reordered")
    n, last = 0, None
    for t in pts:
        if last is None or t - last > cluster_s:
            n += 1
        last = t
    return n


def summarize(records, clips, cover, proposed_keep) -> dict:
    counts = {k: 0 for k in LABELS}
    secs = {k: 0.0 for k in LABELS}
    deltas: dict[str, list[float]] = {k: [] for k in LABELS}
    for r in records:
        counts[r["label"]] += 1
        secs[r["label"]] += abs(r["delta_s"])
        deltas[r["label"]].append(r["delta_s"])
    final_s = total(cover)
    same = total(intersect(cover, proposed_keep))
    kept = [c for c in clips if c["kept"]]
    return {
        "fix_cost": fix_cost(records),
        "counts": counts,
        "seconds": {k: round(v, 1) for k, v in secs.items()},
        "median_delta": {k: (round(median(v), 2) if v else None) for k, v in deltas.items()},
        "boundary_error": [{"i": c["i"], "start": c["d_start"], "end": c["d_end"]}
                           for c in kept],
        "clips_proposed": len(clips), "clips_kept": len(kept),
        "final_s": round(final_s, 1), "proposed_s": round(total(proposed_keep), 1),
        "unchanged_pct": round(100 * same / final_s, 1) if final_s else 0.0,
        "markers_needed": markers_needed(records),
    }
