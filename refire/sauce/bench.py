"""`refire sauce bench`: a stand-in for section 7 until Nguy has reviewed real sauce runs (P6 tuning).

His four finished edits each run on the reFire run they were cut from, but on HIS final V1 (rebuilt from
the project as a timeline.tsv), so the cut is exactly the one he polished. Sauce runs blind to that
video's own exemplars, and its booms and emotes are scored against his on the same cut:
- strict: same type within 3 frames (the spec's "kept")
- loose: same type within 0.5 s (the moment was right, the frame was off)
A = matched / (placed + missed). The director + critic proposals are cached per video, so sweeping tau
afterwards costs no calls. ponytail: four videos, booms + emotes only (his zooms are four in all).
"""
from __future__ import annotations

import json
from pathlib import Path

from . import mine
from .project import PROXY, Project, classify, items, pick_final

TAUS = (0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8)
LOOSE = 0.5
OUT = mine.OUT / "bench"


def his_cut(project: Path, out: Path) -> tuple[Path | None, list[dict], Path | None]:
    """(run dir, his boom/emote events, timeline.tsv path) of one finished edit: the run whose proxies
    fill his final V1, that V1 written as the panel's timeline.tsv, his polish on it."""
    proj = Project(project)
    uid, _ = pick_final(proj)
    its = items(proj, uid)
    v1 = [i for i in its if i["track"] == "V1" and PROXY.search(i["media"] or "")]
    runs: dict[str, int] = {}
    for i in v1:
        p = Path(i["media"])
        if p.parent.name == "proxies":
            runs[str(p.parent.parent.parent)] = runs.get(str(p.parent.parent.parent), 0) + 1
    if not runs:
        return None, [], 0.0
    run_dir = Path(max(runs, key=runs.get))
    out.mkdir(parents=True, exist_ok=True)
    tsv = out / "timeline.tsv"
    rows = [f"{i['media']}\t{i['src_in']:.6f}\t{i['src_out']:.6f}\t{i['t']:.6f}\t{i['end']:.6f}"
            for i in v1 if Path(i["media"]).parent.parent.parent == run_dir]
    tsv.write_text("#timebase\t4233600000\n" + "\n".join(rows) + "\n", encoding="utf-8")
    his = [{**e, "type": mine.kind(e), "asset": e["type"].split(":", 1)[1] if ":" in e["type"] else None}
           for i in its for e in classify(i) if mine.kind(e) in ("boom", "emote")]
    return run_dir, his, tsv


def score(placed: list[dict], his: list[dict], tol: float) -> dict:
    """greedy nearest same-type matching within tol."""
    free = list(his)
    hit = same_asset = 0
    for e in sorted(placed, key=lambda e: e["t"]):
        cand = [h for h in free if h["type"] == e["type"] and abs(h["t"] - e["t"]) <= tol]
        if cand:
            h = min(cand, key=lambda h: abs(h["t"] - e["t"]))
            free.remove(h)
            hit += 1
            same_asset += e["type"] == "emote" and h.get("asset") == e.get("asset")
    den = len(placed) + len(free)
    return {"placed": len(placed), "his": len(his), "matched": hit, "missed": len(free),
            "emote_same_asset": same_asset, "A": round(hit / den, 3) if den else None,
            "precision": round(hit / len(placed), 3) if placed else None, "recall": round(hit / len(his), 3) if his else None}


def run(out: Path = OUT, refresh: bool = False, model: str = "claude-opus-5", backend: str = "cli", log=print) -> Path:
    from . import evaluate, place, plan
    prof = json.loads((mine.OUT / "style_profile.json").read_text(encoding="utf-8"))
    exemplars = [json.loads(x) for x in (mine.OUT / "exemplars.jsonl").read_text(encoding="utf-8").splitlines()]
    corpus = [v for v in json.loads((mine.OUT / "corpus.json").read_text(encoding="utf-8")) if v.get("project")]
    added, negatives = evaluate.load_corrections()
    per: dict[str, dict] = {}
    for v in corpus:
        d = out / v["key"]
        run_dir, his, tsv = his_cut(Path(v["project"]), d)
        if run_dir is None:
            log(f"[sauce bench] {v['key']}: no reFire run under its V1, skipped")
            continue
        data = plan.load(run_dir, timeline=tsv, log=log)
        cache = d / "proposals.json"
        if cache.exists() and not refresh:
            evs = json.loads(cache.read_text(encoding="utf-8"))
        else:
            blind = [r for r in exemplars if r.get("video") != v["key"]] + added       # never its own examples
            lib = place.library(prof, blind)
            names = plan.sheets(run_dir, data["phrases"], plan.marked(data["phrases"]), d)
            evs = plan.direct(data, prof, lib, blind, model=model, backend=backend, sheet_names=names, cwd=d,
                              negatives=negatives, log=log)
            cache.write_text(json.dumps(evs, indent=1, ensure_ascii=False), encoding="utf-8")
        phrases = {p["id"]: p for p in data["phrases"]}
        per[v["key"]] = {"title": v["title"], "his": his, "runs": {}}
        for src in ("confidence", "director_confidence"):          # with the critic, and the director alone
            evs_s = [{**e, "confidence": e.get(src, e["confidence"])} for e in evs]
            for tau in TAUS:
                kept, _ = place.solve(evs_s, phrases, prof, place.config({"tau": tau}), total=data["total"],
                                      cuts=data["tmap"].cuts())
                kept = [e for e in kept if e["type"] in ("boom", "emote")]
                key = tau if src == "confidence" else f"director {tau}"
                per[v["key"]]["runs"][key] = {"strict": score(kept, his, 3 / 60), "loose": score(kept, his, LOOSE)}
        log(f"[sauce bench] {v['key']}: {len(evs)} proposals, his {len(his)} booms/emotes")
    L = ["# Sauce bench", "", "His 4 finished edits, each on his own final V1, sauce blind to that video's examples.",
         "A = matched / (placed + missed). strict = within 3 frames, loose = within 0.5 s.", "",
         "| tau | placed | his | matched (loose) | A loose | A strict | precision loose | recall loose |",
         "|---|---|---|---|---|---|---|---|"]
    for tau in [*TAUS, *(f"director {t}" for t in TAUS)]:
        tot = {k: sum(p["runs"][tau]["loose"][k] for p in per.values()) for k in ("placed", "his", "matched", "missed")}
        strict = sum(p["runs"][tau]["strict"]["matched"] for p in per.values())
        den = tot["placed"] + tot["missed"]
        L.append(f"| {tau} | {tot['placed']} | {tot['his']} | {tot['matched']} | {tot['matched'] / den if den else 0:.3f} | "
                 f"{strict / (tot['placed'] + tot['his'] - strict) if tot['placed'] + tot['his'] else 0:.3f} | "
                 f"{tot['matched'] / tot['placed'] if tot['placed'] else 0:.3f} | {tot['matched'] / tot['his'] if tot['his'] else 0:.3f} |")
    L += ["", "Per video at tau 0.6 (loose):", ""]
    L += [f"- {p['title']}: placed {p['runs'][0.6]['loose']['placed']}, his {p['runs'][0.6]['loose']['his']}, "
          f"matched {p['runs'][0.6]['loose']['matched']}, same emote {p['runs'][0.6]['loose']['emote_same_asset']}"
          for p in per.values()]
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench.json").write_text(json.dumps({k: {"title": p["title"], "runs": p["runs"]} for k, p in per.items()},
                                               indent=1), encoding="utf-8")
    rep = out / "bench.md"
    rep.write_text("\n".join(L) + "\n", encoding="utf-8")
    return rep
