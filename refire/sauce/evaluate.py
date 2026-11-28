"""`refire sauce eval <run_dir>` (04-sauce.md section 7): what Nguy kept of sauce's placements.

Reads his saved Premiere project (read-only, like `refire learn`; the same data as a panel dump of
V2-V5 + A3, without the panel): the exported sequence's booms and emotes. Ours carry their
`refire:sauce:<id>` clip names, so each placement is matched exactly:
- kept: still there and moved < 3 frames (spec: that still counts as kept)
- moved: still there, moved further (a fix)
- deleted: gone
- added: a boom/emote of his that isn't ours (a moment sauce missed)
A = kept / (placed + added). Deletions and additions go to run/sauce_corrections.jsonl with their
transcript window (the director sees them on the next run), the run's score to run/sauce/eval.jsonl,
and its style is compared with style_profile.json (drift is flagged even when A looks fine).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import mine
from .project import Project, classify, items, pick_final

KEEP_FRAMES = 3
SHIP_A, SHIP_RUNS = 0.95, 3
DRIFT = (0.5, 2.0)                   # a rate this far off his (x0.5 .. x2) is flagged
CORRECTIONS = Path("run/sauce_corrections.jsonl")
TYPES = ("boom", "emote")
PREFIX = "refire:sauce:"


def find_project(run_dir, dirs=None) -> Path | None:
    """the newest saved project (no autosaves) that uses this run's footage."""
    from ..learn.lessons import run_key
    from ..learn.prproj import referenced_runs
    key = run_key(run_dir)
    hits = []
    for d in dirs or mine.PROJECT_DIRS:
        for p in Path(d).rglob("*.prproj") if Path(d).exists() else []:
            if "Auto-Save" not in str(p) and key in referenced_runs(p):
                hits.append(p)
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def final_events(project, run_dir, placed: list[dict]) -> list[dict]:
    """his exported sequence's booms + emotes; ours (by clip name) take their type from sauce.json
    (our emote is a rendered .mov, not a still)."""
    from ..learn.lessons import run_key
    proj = Project(project)
    key = run_key(run_dir).replace("/", "\\")
    uid, _ = pick_final(proj, keep=lambda m: bool(m) and key in m)
    ours = {f"{PREFIX}{e['id']}": e for e in placed}
    out = []
    for it in items(proj, uid):
        if it["name"] in ours:
            e = ours[it["name"]]
            out.append({"t": round(it["t"], 3), "end": round(it["end"], 3), "type": e["type"], "name": it["name"],
                        "asset": e.get("asset")})
            continue
        for e in classify(it):
            k = mine.kind(e)
            if k in TYPES:
                out.append({**e, "type": k, "asset": e["type"].split(":", 1)[1] if ":" in e["type"] else None})
    return sorted(out, key=lambda e: e["t"])


def diff(placed: list[dict], final: list[dict], fps: float) -> dict:
    """-> {rows: [{status, ...}], kept, moved, deleted, added, placed, A}."""
    tol = KEEP_FRAMES / fps
    by_name = {e["name"]: e for e in final if e.get("name", "").startswith(PREFIX)}
    rows = []
    for e in placed:
        f = by_name.get(f"{PREFIX}{e['id']}")
        if f is None:
            rows.append({**e, "status": "deleted"})
        elif abs(f["t"] - e["t"]) < tol:
            rows.append({**e, "status": "kept"})
        else:
            rows.append({**e, "status": "moved", "moved_to": f["t"], "by": round(f["t"] - e["t"], 3)})
    rows += [{**f, "status": "added"} for f in final if not f.get("name", "").startswith(PREFIX)]
    n = {s: sum(r["status"] == s for r in rows) for s in ("kept", "moved", "deleted", "added")}
    den = len(placed) + n["added"]
    return {"rows": sorted(rows, key=lambda r: r["t"]), **n, "placed": len(placed),
            "A": round(n["kept"] / den, 3) if den else None}


def style(final: list[dict], total: float, prof: dict) -> list[str]:
    """his final edit's rates vs the profile: drift notes (empty = in range)."""
    notes = []
    body = max(1e-9, (total - 60) / 60)
    for ty in TYPES:
        got = sum(1 for e in final if e["type"] == ty and e["t"] >= 60) / body
        ref = prof["rates"][ty]["body_per_min"]
        if ref and not DRIFT[0] <= got / ref <= DRIFT[1]:
            notes.append(f"{ty} rate {got:.2f}/min vs his usual {ref:.2f}/min")
    return notes


def run(run_dir, project=None, profile_dir: Path = mine.OUT, corrections: Path = CORRECTIONS, log=print) -> dict:
    from .. import punch
    from ..srt import _run_dir
    from ..timeline_map import TimelineMap
    run_dir = Path(run_dir)
    sj = run_dir / "sauce.json"
    if not sj.exists():
        raise FileNotFoundError(f"no {sj}: run `refire sauce {run_dir}` first")
    placed = json.loads(sj.read_text(encoding="utf-8"))["events"]
    project = Path(project) if project else find_project(run_dir)
    if project is None:
        raise FileNotFoundError("no saved Premiere project uses this run's footage (pass --project)")
    mp = run_dir / "ae" / "manifest.json"
    tmap = TimelineMap.load(mp)
    fps = 254016000000 / tmap.ticks_per_frame
    final = final_events(project, run_dir, placed)
    res = diff(placed, final, fps)
    prof = json.loads((profile_dir / "style_profile.json").read_text(encoding="utf-8"))
    res["drift"] = style(final, max(s.tl_out for s in tmap.spans), prof)
    # corrections become exemplars (spec 7.4): what he deleted (negatives) and added
    words = punch.timeline_words(sorted(json.loads((_run_dir(mp) / "transcript.json").read_text(encoding="utf-8")),
                                        key=lambda w: w["start"]), tmap)
    around = lambda t: (" ".join(w["text"] for w in words if t - 8 <= w["t0"] < t) + " [*] "       # noqa: E731
                        + " ".join(w["text"] for w in words if t <= w["t0"] < t + 8))
    from ..learn.lessons import run_key
    corrections.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d")
    with corrections.open("a", encoding="utf-8") as f:
        for r in res["rows"]:
            if r["status"] in ("deleted", "added", "moved"):
                f.write(json.dumps({"run": run_key(run_dir), "date": stamp, "status": r["status"], "type": r["type"],
                                    "asset": r.get("asset"), "t": r["t"], "by": r.get("by"), "text": around(r["t"]),
                                    "reason": r.get("reason")}, ensure_ascii=False) + "\n")
    hist = profile_dir / "eval.jsonl"
    with hist.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"run": run_key(run_dir), "date": stamp, "project": str(project),
                            **{k: res[k] for k in ("A", "kept", "moved", "deleted", "added", "placed")}}) + "\n")
    last = [json.loads(x) for x in hist.read_text(encoding="utf-8").splitlines()][-SHIP_RUNS:]
    res["ship"] = len(last) == SHIP_RUNS and all((x["A"] or 0) >= SHIP_A for x in last)
    (run_dir / "sauce_eval.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"[sauce eval] A = {res['A']} ({res['kept']} kept, {res['moved']} moved, {res['deleted']} deleted of "
        f"{res['placed']}; {res['added']} added by hand)" + (f"; drift: {'; '.join(res['drift'])}" if res["drift"] else ""))
    if res["ship"]:
        log(f"[sauce eval] ship bar met: A >= {SHIP_A} on the last {SHIP_RUNS} runs")
    return res


def load_corrections(path: Path = CORRECTIONS, n: int = 30) -> tuple[list[dict], list[dict]]:
    """(his additions as exemplars, his deletions as negatives), newest last, n of each."""
    if not path.exists():
        return [], []
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    add = [{"type": r["type"], "asset": r.get("asset"), "text": r["text"], "why": "Nguy added this by hand"}
           for r in rows if r["status"] == "added"]
    neg = [r for r in rows if r["status"] == "deleted"]
    return add[-n:], neg[-n:]
