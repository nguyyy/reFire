"""`refire eval`: score reFire's cut against the finished edits, in seconds of correction.

Replay (default): the proposal is the cut reFire actually shipped for that VOD (from
clip_diff.json, i.e. the manifest the human edited), passed through whichever post-
selection stages are switched on (`--boundary`, `--trim`, ...), then diffed against the
human's final. No Whisper (transcripts are the run's cache), no Claude unless a stage
that calls it is on. `--manifest` scores a fresh `refire make` output instead.

Leave-one-out: every lesson a stage reads while scoring VOD X is rebuilt from the other
VODs' diffs only (`lessons.load_lessons(exclude=(X,))`), so the score is never in-sample.
Writes run/eval/<timestamp>.json and prints the table against the previous eval and the
latest one tagged `baseline`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import align, diff, lessons

STAGES = ("boundary", "trim", "critics", "markers")


def rekeep(clips, words, voiced, changed) -> list[dict]:
    """Recompute `keep` (compress_silence, as build_manifest would) for changed clips."""
    from ..select import compress_silence
    out = []
    for c in clips:
        if c["i"] in changed:
            a, b = c["start"], c["end"]
            k, _r, _d = compress_silence(words, a, b, voiced=voiced)
            c = {**c, "keep": [[a + x, a + y] for x, y in k]}
        out.append(c)
    return out


def apply_stages(prop, stages, ctx, cache_dir, loo, words, voiced, log=None,
                 marker_model=None, marker_threshold=None):
    """The replay transform: reFire's recorded cut -> what the switched-on stages make of it.
    Stage modules are imported lazily so a baseline eval never loads them."""
    log = log if log is not None else {}
    prop = [dict(c) for c in prop]
    if "boundary" in stages:
        from ..boundary import refine
        prop, changed, log["boundary"] = refine(prop, ctx, loo)
        prop = rekeep(prop, words, voiced, changed)
    if "trim" in stages:
        from ..trim import trim_clips
        prop, log["trim"] = trim_clips(prop, ctx, loo, voiced=voiced)
    if "critics" in stages:
        from ..critics import replay_critics
        prop, log["critics"] = replay_critics(prop, ctx, loo, cache_dir)
    if "markers" in stages:
        from .. import markers
        log["markers"] = markers.clip_markers(
            prop, ctx, marker_model or markers.DEFAULT,
            markers.THRESHOLD if marker_threshold is None else marker_threshold)
    return prop, log


def score_vod(d: dict, stages=(), run_root="run", vods_dir="vods",
              manifest: dict | None = None, marker_threshold=None) -> dict:
    cache_dir = Path(run_root) / d["vod"]
    ctx = lessons._context(cache_dir, Path(run_root) / d["vod"] / d["run"], vods_dir)
    words = ctx.words
    loo = lessons.load_lessons(run_root, d["game"], exclude=(d["vod"],)) or {}
    prop = align.proposal(manifest) if manifest else d["proposal"]
    model = None
    if "markers" in stages:                    # trained on the OTHER VODs' edits only
        from .. import markers
        model = markers.fit(markers.training_rows(lessons.load_diffs(run_root),
                                                  exclude=(d["vod"],)))
    prop, log = apply_stages(prop, stages, ctx, cache_dir, loo, words,
                             lessons.voiced_of(cache_dir), marker_model=model,
                             marker_threshold=marker_threshold)
    res = diff.diff(prop, d["final"], ctx)
    out = {"vod": d["vod"], "proposal_from": "manifest arg" if manifest
           else d.get("proposal_from", "manifest"), **res["summary"], "stage_log": log}
    if "markers" in stages:
        from ..markers import marker_score
        out["markers"] = marker_score([m for c in prop for m in c.get("markers", [])],
                                      res["records"])
        edges = [{"vod": c["keep"][0][0]} for c in prop] + [{"vod": c["keep"][-1][1]}
                                                            for c in prop]
        out["markers"]["base_precision"] = marker_score(edges, res["records"])["precision"]
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def overall(per: list[dict]) -> dict:
    keys = ("total", "boundary", "interior_cut", "dropped", "added")
    out = {k: round(sum(p["fix_cost"][k] for p in per), 1) for k in keys}
    out["counts"] = {lab: sum(p["counts"][lab] for p in per) for lab in diff.LABELS}
    errs = [abs(e[s]) for p in per for e in p["boundary_error"] for s in ("start", "end")]
    out["mean_abs_boundary_err"] = _mean(errs)
    out["unchanged_pct"] = _mean([p["unchanged_pct"] for p in per])
    out["markers_needed"] = sum(p["markers_needed"] for p in per)
    ms = [p["markers"] for p in per if p.get("markers")]
    if ms:
        out["markers"] = {"per_video": _mean([m["n"] for m in ms]),
                          "precision": _mean([m["precision"] for m in ms]),
                          "base_precision": _mean([m["base_precision"] for m in ms]),
                          "recall_s": _mean([m["recall_s"] for m in ms])}
    return out


def _latest(eval_dir, tag=None, before=None) -> dict | None:
    files = sorted(Path(eval_dir).glob("*.json"), reverse=True)
    for f in files:
        if before and f.name >= before:
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        if tag is None or d.get("tag") == tag:
            return d
    return None


def table(cur: dict, prev: dict | None, base: dict | None) -> str:
    def fc(ev, vod):
        if not ev:
            return None
        for p in ev["per_vod"]:
            if p["vod"] == vod:
                return p["fix_cost"]["total"]
        return None

    head = f"{'vod':<24}{'fix_cost':>10}{'prev':>10}{'baseline':>10}{'dropped':>9}{'added':>7}"
    lines = [f"eval {cur['stamp']}  stages={','.join(cur['stages']) or 'none'}"
             f"  tag={cur.get('tag') or '-'}", head, "-" * len(head)]
    rows = [(p["vod"], p) for p in cur["per_vod"]] + [("OVERALL", None)]
    for vod, p in rows:
        if p:
            v, dr, ad = p["fix_cost"]["total"], p["counts"]["dropped"], p["counts"]["added"]
            pv, bv = fc(prev, vod), fc(base, vod)
        else:
            v, dr, ad = (cur["overall"]["total"], cur["overall"]["counts"]["dropped"],
                         cur["overall"]["counts"]["added"])
            pv = prev["overall"]["total"] if prev else None
            bv = base["overall"]["total"] if base else None
        lines.append(f"{vod:<24}{v:>10}{str(pv if pv is not None else '-'):>10}"
                     f"{str(bv if bv is not None else '-'):>10}{dr:>9}{ad:>7}")
    mk = cur["overall"].get("markers")
    if mk:
        lines.append(f"markers: {mk['per_video']}/video, precision {mk['precision']} "
                     f"(marking every edge: {mk['base_precision']}), "
                     f"recall {mk['recall_s']} of fix_cost seconds")
    return "\n".join(lines)


def evaluate(vod=None, stages=(), run_root="run", vods_dir="vods", tag="",
             manifest_path=None, save=True, marker_threshold=None) -> dict:
    diffs = [d for d in lessons.load_diffs(run_root)
             if vod is None or d["vod"].split("@")[0] == str(vod) or d["vod"] == str(vod)]
    if not diffs:
        raise SystemExit("no clip_diff.json to score against -- run `refire learn` first")
    manifest = (json.loads(Path(manifest_path).read_text(encoding="utf-8"))
                if manifest_path else None)
    if manifest and len(diffs) != 1:
        raise SystemExit("--manifest scores one VOD: pass its vod_id too")
    per = [score_vod(d, stages, run_root, vods_dir, manifest, marker_threshold)
           for d in diffs]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    cur = {"stamp": stamp, "tag": tag, "stages": list(stages),
           "loo": True, "per_vod": per, "overall": overall(per)}
    eval_dir = Path(run_root) / "eval"
    prev = _latest(eval_dir) if eval_dir.exists() else None
    base = _latest(eval_dir, tag="baseline") if eval_dir.exists() else None
    print(table(cur, prev, base))
    if save:
        eval_dir.mkdir(parents=True, exist_ok=True)
        p, n = eval_dir / f"{stamp}.json", 2
        while p.exists():                       # two evals in one second
            p, n = eval_dir / f"{stamp}-{n}.json", n + 1
        p.write_text(json.dumps(cur, indent=1), encoding="utf-8")
        print(f"[eval] wrote {p}")
    return cur
