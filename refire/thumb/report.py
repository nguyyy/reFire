"""report.md for a thumbnail run (docs/specs/03-thumb.md step 8).

Ranking with score breakdowns, per candidate: the bet it makes, its title, the beats that
pay it off, warnings; and the upload order (slot 1 = top pick: an inconclusive Test &
Compare keeps the first uploaded thumbnail). Score = the vision critic's 0-100 when
critic.json exists in the candidate folder, else the share of deterministic checks passed.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import checks


def _warnings(d: Path, spec: dict) -> list[str]:
    out = []
    for meta in sorted((d / "assets").glob("*.art.json")):          # art copied in with its sidecar
        m = json.loads(meta.read_text(encoding="utf-8"))
        if m.get("source") != "yatta":
            out.append(f"art for {m.get('name')} came from a fallback source ({m.get('source')})")
    if (d / "assets" / "off_plan.txt").exists():
        out.append("left element is off-plan: " + (d / "assets" / "off_plan.txt").read_text(encoding="utf-8"))
    face = next((L for L in spec.get("layers", []) if L.get("kind") == "face"), None)
    if face and (d / face["src"]).exists():
        from PIL import Image
        up = face["w"] / Image.open(d / face["src"]).width
        if up > 2.0:
            out.append(f"face upscaled {up:.1f}x (soft at full size)")
    return out


def build(run_dir, ocr=None, root=None) -> Path:
    """score every rendered candidate under <run_dir>/thumb (or `root`) and write report.md there."""
    root = Path(root) if root else Path(run_dir) / "thumb"
    concepts = {}
    if (root / "concepts.json").exists():
        for c in json.loads((root / "concepts.json").read_text(encoding="utf-8")).get("concepts", []):
            concepts[c.get("id")] = c
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "spec.json").exists()):
        spec = json.loads((d / "spec.json").read_text(encoding="utf-8"))
        res = checks.candidate(d, ocr)
        critic = json.loads((d / "critic.json").read_text(encoding="utf-8")) if (d / "critic.json").exists() else None
        passed = sum(r["ok"] for r in res)
        score = critic["score"] if critic else round(100 * passed / len(res))
        rows.append({"id": d.name, "dir": d, "spec": spec, "checks": res, "score": score, "critic": critic,
                     "concept": concepts.get(d.name, {}), "warnings": _warnings(d, spec)})
    # failing deterministic checks rank below any candidate that passes them all
    rows.sort(key=lambda r: (all(c["ok"] for c in r["checks"]), r["score"]), reverse=True)
    across = checks.distinct([r["dir"] for r in rows],
                             [r["concept"].get("type") for r in rows] if all(r["concept"] for r in rows) else None)
    L = ["# Thumbnail candidates", "", "## Upload order", ""]
    L += [f"{i}. **{r['id']}** — {checks.hook_text(r['spec']) or '(no hook)'}" + (" ← slot 1: kept if the test is inconclusive" if i == 1 else "")
          for i, r in enumerate(rows, 1)]
    L += ["", "## Ranking", "", "| Rank | Candidate | Score | Checks passed | Type | Title |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        L.append(f"| {i} | {r['id']} | {r['score']}{'' if r['critic'] else ' (checks only)'} | "
                 f"{sum(c['ok'] for c in r['checks'])}/{len(r['checks'])} | {r['concept'].get('type', r['spec'].get('layout', ''))} | "
                 f"{r['concept'].get('title', '—')} |")
    L += ["", "Across candidates: " + "; ".join(f"{c['name']} {'ok' if c['ok'] else 'FAIL'} ({c['value']})" for c in across), ""]
    for r in rows:
        c = r["concept"]
        L += [f"## {r['id']}: {checks.hook_text(r['spec']) or '(no hook)'}", "",
              f"- **Bet:** {c.get('bet') or c.get('type') or r['spec'].get('layout', '—')}",
              *[f"- **Plan, {k}:** {v}" for k, v in (c.get("plan") or {}).items()],
              f"- **Title:** {c.get('title', '—')}",
              f"- **Paid off by:** {', '.join(map(str, c.get('payoff_beats', []))) or '— (no payoff beats on file: concept step not run)'}",
              "- **Checks:** " + "; ".join(f"{x['name']} {'ok' if x['ok'] else 'FAIL'} ({x['value']}, need {x['need']})"
                                          for x in r["checks"])]
        if r["critic"]:
            L.append("- **Critic:** " + ", ".join(f"{k} {v}" for k, v in r["critic"].get("scores", {}).items()))
            L += [f"- **Critic issue:** {i}" for i in r["critic"].get("issues", [])]
        L += [f"- **Warning:** {w}" for w in r["warnings"]] + [""]
    out = root / "report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    return out
