"""`refire learn`: pair runs with finished projects, diff them, aggregate the lessons.

Inputs are read-only: run/<cache>/<run>/ (manifest, outline) and the user's .prproj
files (decompressed in memory). Outputs are new files only:
  run/<cache>/clip_diff.json        every diff record for that VOD
  run/clip_lessons_<game>.json      per-label counts, medians, examples, per VOD
  run/clip_lessons_report.md        where reFire goes wrong most
Safe to re-run: the outputs are rebuilt from the inputs every time.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from statistics import median

from . import align, diff, prproj

# os.pathsep-separated; Premiere's own default save folder is where the August edits live
DEFAULT_PROJECTS = os.environ.get("REFIRE_PROJECTS", os.pathsep.join(
    str(Path.home() / p) for p in ("Desktop/iphone", "Documents/Adobe/Premiere Pro")))
EXAMPLES_PER_LABEL = 4
VOICED_SINCE = time.mktime((2026, 9, 13, 0, 0, 0, 0, 0, -1))   # select.compress_silence(voiced=)


def game_of(cache_dir, fallback: str = "") -> str:
    """Game slug from the glossary file the run left behind ('genshin-impact'), else the
    `--game` the caller gave (a cache whose glossary pass failed has no file), else
    'unknown'. Lessons pool by game, so a wrong 'unknown' splits the history."""
    g = sorted(Path(cache_dir).glob("glossary_*.json"))
    if g:
        return g[0].stem[len("glossary_"):]
    from ..glossary import _slug
    return _slug(fallback) if fallback.strip() else "unknown"


def projects(projects_dir) -> list[Path]:
    """Every saved .prproj under one folder or several (os.pathsep-joined), no autosaves.

    Premiere's autosave folder is 'Adobe Premiere Pro Auto-Save'; an autosave newer than
    the last real save would otherwise win the pairing on mtime."""
    roots = [Path(p) for p in str(projects_dir).split(os.pathsep) if p.strip()]
    return sorted(p for root in roots if root.exists() for p in root.rglob("*.prproj")
                  if not any("Auto-Save" in part for part in p.parts))


def project_index(projects_dir) -> dict[str, list[tuple[int, float, Path]]]:
    """run key -> [(media refs, mtime, project)] for every project in the folder."""
    out: dict[str, list] = {}
    for p in projects(projects_dir):
        try:
            refs = prproj.referenced_runs(p)
        except (OSError, EOFError):
            continue
        for key, n in refs.items():
            out.setdefault(key.lower(), []).append((n, p.stat().st_mtime, p))
    return out


def run_key(run_dir) -> str:
    run_dir = Path(run_dir)
    return f"{run_dir.parent.name}/{run_dir.name}"


def find_runs(run_root="run", vod_id=None) -> list[Path]:
    """Run folders with a manifest, optionally only one VOD's."""
    out = []
    for m in sorted(Path(run_root).glob("*/*/ae/manifest.json")):
        cache = m.parent.parent.parent.name
        if vod_id is None or cache.split("@")[0] == str(vod_id) or cache == str(vod_id):
            out.append(m.parent.parent)
    return out


def voiced_of(cache_dir):
    """Cached Silero regions (transcribe.speech_regions' file), or None -- never rescans."""
    p = Path(cache_dir) / "speech.json"
    if not p.exists():
        return None
    from ..transcribe import speech_regions
    return speech_regions(Path(cache_dir) / "audio.wav", cache_path=p) or None


def _words(cache_dir) -> list[dict]:
    p = Path(cache_dir) / "transcript.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def _context(cache_dir, run_dir, vods_dir) -> diff.Context:
    from ..perception import load_signals
    cache_dir = Path(cache_dir)
    words = _words(cache_dir)
    loud = load_signals(cache_dir / "signals.json") or []
    chat = []
    chat_json = Path(vods_dir) / f"{cache_dir.name}.chat.json"
    if chat_json.exists() and words:
        from ..chat import chat_signal
        chat = chat_signal(chat_json, words[-1]["end"],
                           offset=align.window_start(cache_dir))
    beats = []
    ol = Path(run_dir) / "outline.json" if run_dir else None
    if ol and ol.exists():
        beats = json.loads(ol.read_text(encoding="utf-8")).get("beats") or []
    return diff.Context(words, loud, chat, beats)


def add_motion(records, video) -> None:
    """Fill each record's signals.motion from the VOD (mean/max inter-frame diff, 0-255).

    Reuses ae_export.clip_intensity, which decodes every frame of the span -- a few
    minutes per VOD, hence opt-in (`refire learn --motion`)."""
    from ..ae_export import clip_intensity
    for r in records:
        a, b = r["vod_start"], max(r["vod_end"], r["vod_start"] + 0.5)
        try:
            inten, _fps = clip_intensity(video, a, min(b, a + 30.0))
        except Exception:                     # cv2 missing / unreadable: leave None
            return
        if inten[1:]:
            r["signals"]["motion"] = {"mean": round(sum(inten[1:]) / len(inten[1:]), 2),
                                      "max": round(max(inten[1:]), 2)}


def boundary_rows(prop, res, ctx) -> list[dict]:
    """Per kept clip: the features markers.py scores + which fix the human made."""
    from ..markers import features
    fixed = {}
    for r in res["records"]:
        if r["label"] in diff.BOUNDARY and r["clips"]:
            fixed[(r["clips"][0], r["label"].split("_")[0])] = r["label"]
    kept = {c["i"] for c in res["clips"] if c["kept"]}
    out = []
    for c in prop:
        if c["i"] not in kept or c.get("section") == "Cold Open":   # markers skip teasers too
            continue
        sf, ef, _info = features(ctx, c["keep"][0][0], c["keep"][-1][1])
        out.append({"i": c["i"], "start_feats": sf, "end_feats": ef,
                    "start_label": fixed.get((c["i"], "start"), "ok"),
                    "end_label": fixed.get((c["i"], "end"), "ok")})
    return out


def learn_run(run_dir, project=None, projects_dir=DEFAULT_PROJECTS, vods_dir="vods",
              index=None, quiet=False, motion=False, game: str = "") -> Path | None:
    """One run (or orphan run key) -> run/<cache>/clip_diff.json. None if unpaired."""
    run_dir = Path(run_dir)
    key = run_key(run_dir)
    cache_dir = run_dir.parent
    say = (lambda *_: None) if quiet else print
    if project is None:
        idx = index if index is not None else project_index(projects_dir)
        cands = sorted(idx.get(key.lower(), []), key=lambda t: (t[0], t[1]), reverse=True)
        if not cands:
            say(f"[learn] {key}: no project references this run -- skipped")
            return None
        project = cands[0][2]
    proj = prproj.Project(project)
    in_run = f"/run/{key.lower()}/"
    uid, why = prproj.pick_final(proj, keep=lambda m: in_run in align.norm(m))
    if uid is None:
        say(f"[learn] {key}: {Path(project).name} imports this run but no sequence "
            f"uses its footage -- no final edit, skipped")
        return None
    items = prproj.flatten(proj, uid)
    manifest = align.load_manifest(run_dir)
    caption_check = None
    mapped = align.map_items(items, manifest, run_dir, run_key=key)
    words = _words(cache_dir)
    if manifest:
        prop, prop_from = align.proposal(manifest), "manifest"
    else:
        # compress_silence only kept Silero-voiced audio from 2026-09-13 on; a cut built
        # before that never had it, so re-deriving keep with it invents interior cuts.
        # ponytail: dated by project mtime, the only date a lost run leaves behind
        voiced = (voiced_of(cache_dir)
                  if Path(project).stat().st_mtime >= VOICED_SINCE else None)
        # bin first: its document order is the play order (see proposal_from_proxies)
        media = [p for p in _bin_media(proj) if in_run in align.norm(p)]
        media += [i["media"] for s in proj.seqs.values() for i in s["items"]
                  if i["media"] and in_run in align.norm(i["media"])]
        prop = align.proposal_from_proxies(media, words, voiced)
        prop_from = ("proxy names, bin order"
                     + (" + transcript keep" if words else " (no transcript)"))
        # the build's own captions.srt survives inside the project: check against it
        cues = next((v for k, v in prproj.captions(proj).items()
                     if align.norm(k).endswith(f"/run/{key.lower()}/ae/captions.srt")), None)
        if cues and words:
            prop, caption_check = align.use_captions(prop, cues, words, voiced)
            prop_from += f", checked against captions.srt (keep: {caption_check['keep_from']})"
    ctx = _context(cache_dir, run_dir if manifest else None, vods_dir)
    res = diff.diff(prop, mapped["segments"], ctx)
    res["boundaries"] = boundary_rows(prop, res, ctx)
    video = Path(vods_dir) / f"{cache_dir.name}.mp4"
    if motion and video.exists():
        add_motion(res["records"], video)
    out = {"vod": cache_dir.name, "run": run_dir.name, "game": game_of(cache_dir, game),
           "window_start": align.window_start(cache_dir),
           "project": str(project), "sequence": proj.seqs[uid]["name"],
           "sequence_reason": why, "align_methods": mapped["methods"],
           "proposal_from": prop_from, "caption_check": caption_check,
           "flags": mapped["flags"][:50],
           "n_flags": len(mapped["flags"]),
           "proposal": prop, "final": mapped["segments"],
           "overlays": mapped["overlays"], **res,
           "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "clip_diff.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    s = res["summary"]
    say(f"[learn] {key} <- {Path(project).name} '{out['sequence']}' ({why}; "
        f"{mapped['methods']}): fix_cost {s['fix_cost']['total']}s, "
        f"{s['clips_kept']}/{s['clips_proposed']} clips kept, "
        f"{s['unchanged_pct']}% of final unchanged")
    return path


def _bin_media(proj) -> list[str]:
    """Every media path in the project, used or not (proxies imported into the bin)."""
    return [el.findtext("FilePath") for el in proj.root if el.tag == "Media"
            and el.findtext("FilePath")]


def orphan_keys(run_root="run", projects_dir=DEFAULT_PROJECTS, index=None) -> list[str]:
    """Run keys a project references whose run folder (manifest) is gone."""
    idx = index if index is not None else project_index(projects_dir)
    have = {run_key(r).lower() for r in find_runs(run_root)}
    return sorted(k for k in idx if k not in have)


# --------------------------------------------------------------------------- lessons

def load_diffs(run_root="run", exclude=()) -> list[dict]:
    out = []
    for p in sorted(Path(run_root).glob("*/clip_diff.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["vod"] not in exclude:
            out.append(d)
    return out


def _example(r: dict, vod: str) -> dict:
    return {"vod": vod, "label": r["label"], "delta_s": r["delta_s"],
            "vod_start": r["vod_start"], "vod_end": r["vod_end"],
            "text": r["text"][:300], "before": r["before"][-200:],
            "after": r["after"][:200], "signals": r["signals"], "beat": r["beat"]}


def aggregate(diffs: list[dict]) -> dict:
    """Per-label counts / medians / representative examples across VODs.

    Examples keep their `vod` so a leave-one-out caller can drop its own VOD's.
    'Representative' = closest to the label's median delta, one per VOD first.
    """
    by: dict[str, list[tuple[str, dict]]] = {k: [] for k in diff.LABELS}
    for d in diffs:
        for r in d["records"]:
            by[r["label"]].append((d["vod"], r))
    labels = {}
    for k, rs in by.items():
        ds = [r["delta_s"] for _, r in rs]
        med = median(ds) if ds else None
        ranked = sorted(rs, key=lambda vr: abs(vr[1]["delta_s"] - (med or 0)))
        ex, seen = [], set()
        for vod, r in ranked:                       # spread examples over VODs first
            if vod not in seen and r["text"].strip():
                ex.append(_example(r, vod))
                seen.add(vod)
        for vod, r in ranked:
            if len(ex) >= EXAMPLES_PER_LABEL:
                break
            if r["text"].strip() and all(e["vod_start"] != r["vod_start"] for e in ex):
                ex.append(_example(r, vod))
        labels[k] = {"count": len(rs), "seconds": round(sum(abs(x) for x in ds), 1),
                     "median_delta": round(med, 2) if med is not None else None,
                     "p75_abs": round(sorted(abs(x) for x in ds)[int(0.75 * (len(ds) - 1))], 2)
                     if ds else None,
                     "examples": ex[:EXAMPLES_PER_LABEL]}
    return {"vods": [{"vod": d["vod"], "run": d["run"], "project": Path(d["project"]).name,
                      "sequence": d["sequence"], "fix_cost": d["summary"]["fix_cost"],
                      "proposal_from": d.get("proposal_from", "manifest")}
                     for d in diffs],
            "labels": labels,
            "built": time.strftime("%Y-%m-%d %H:%M:%S")}


def lessons_path(run_root, game: str) -> Path:
    return Path(run_root) / f"clip_lessons_{game}.json"


def load_lessons(run_root="run", game: str = "", exclude=()) -> dict | None:
    """Lessons for `game`, rebuilt without `exclude` VODs when given (leave-one-out)."""
    if exclude:
        diffs = [d for d in load_diffs(run_root, exclude) if not game or d["game"] == game]
        return aggregate(diffs) if diffs else None
    p = lessons_path(run_root, game or "unknown")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def report_md(diffs: list[dict], lessons: dict) -> str:
    rows = sorted(((k, v) for k, v in lessons["labels"].items() if v["count"]),
                  key=lambda kv: -kv[1]["seconds"])
    out = ["# reFire clip lessons", "",
           f"Built {lessons['built']} from {len(diffs)} finished edit(s).", "",
           "| VOD | project / sequence | proposal from | fix_cost (s) | kept | unchanged |",
           "|---|---|---|---|---|---|"]
    for d in diffs:
        s = d["summary"]
        out.append(f"| {d['vod']} | {Path(d['project']).name} / {d['sequence']} | "
                   f"{d.get('proposal_from', 'manifest')} | {s['fix_cost']['total']} | "
                   f"{s['clips_kept']}/{s['clips_proposed']} | {s['unchanged_pct']}% |")
    out += ["", "## Where reFire goes wrong (by seconds of correction)", "",
            "| label | count | seconds | median delta | p75 abs |", "|---|---|---|---|---|"]
    for k, v in rows:
        out.append(f"| {k} | {v['count']} | {v['seconds']} | {v['median_delta']} | "
                   f"{v['p75_abs']} |")
    out += ["", "## Examples", ""]
    for k, v in rows[:6]:
        out.append(f"### {k}")
        for e in v["examples"][:2]:
            beat = e.get("beat") or {}
            out.append(f"- {e['vod']} @ {e['vod_start']:.0f}s ({e['delta_s']:+.1f}s, "
                       f"{beat.get('role') or '?'}): ...{e['before'][-80:]} "
                       f"[[{e['text'][:160]}]] {e['after'][:80]}...")
        out.append("")
    return "\n".join(out).strip() + "\n"


def write_lessons(run_root="run") -> list[Path]:
    diffs = load_diffs(run_root)
    paths = []
    for game in sorted({d["game"] for d in diffs}):
        ds = [d for d in diffs if d["game"] == game]
        les = aggregate(ds)
        p = lessons_path(run_root, game)
        p.write_text(json.dumps(les, indent=1), encoding="utf-8")
        paths.append(p)
    if diffs:
        rp = Path(run_root) / "clip_lessons_report.md"
        rp.write_text(report_md(diffs, aggregate(diffs)), encoding="utf-8")
        paths.append(rp)
    return paths


def learn(vod_id=None, project=None, run_root="run", projects_dir=DEFAULT_PROJECTS,
          vods_dir="vods", motion=False, game: str = "") -> list[Path]:
    """CLI body: diff one VOD (or every paired run, orphans included) then re-aggregate."""
    idx = None if project else project_index(projects_dir)
    runs = find_runs(run_root, vod_id)
    if idx is not None:
        runs += [Path(run_root) / k for k in orphan_keys(run_root, projects_dir, idx)
                 if vod_id is None or k.split("/")[0].split("@")[0] == str(vod_id)]
    if not runs:
        raise SystemExit(f"no reFire run found for {vod_id or 'any VOD'} under {run_root}")
    done = [p for r in runs
            if (p := learn_run(r, project, projects_dir, vods_dir, idx,
                                motion=motion, game=game)) is not None]
    if not done:
        print("[learn] nothing paired -- no lessons written")
        return []
    out = write_lessons(run_root)
    for p in out:
        print(f"[learn] wrote {p}")
    return done + out


# --------------------------------------------------------------------------- few-shot

_VERB = {"start_late": "started earlier, keeping", "start_early": "trimmed the lead-in",
         "end_early": "kept going through", "end_late": "cut the overstay",
         "interior_cut": "removed", "dropped": "dropped the whole clip",
         "restored": "put back", "added": "added", "merged": "joined across"}


def few_shot(lessons: dict | None, labels, n: int = 2, role: str | None = None,
             energy: int | None = None) -> list[str]:
    """Up to `n` "reFire proposed X -> the editor did Y" lines per label.

    Retrieval is by similarity to the moment being judged: same beat role first, then
    closest energy, then the label's own ranking (closest to its median delta). The
    caller passes leave-one-out lessons when scoring, so a VOD never sees its own edits.
    Small and fixed on purpose -- these ride in prompts."""
    out = []
    for lab in labels:
        ex = ((((lessons or {}).get("labels") or {}).get(lab) or {}).get("examples")) or []

        def rank(k_e):
            k, e = k_e
            b = e.get("beat") or {}
            same = role is not None and b.get("role") == role
            de = abs(int(b.get("energy") or 3) - energy) if energy is not None else 0
            return (not same, de, k)
        for _k, e in sorted(enumerate(ex), key=rank)[:n]:
            b = e.get("beat") or {}
            tag = f"{lab} {e['delta_s']:+.1f}s" + (f", {b['role']}" if b.get("role") else "")
            out.append(f'- ({tag}) ..."{e["before"][-70:]}" -> the editor '
                       f'{_VERB.get(lab, "changed")} [[{e["text"][:150]}]]')
    return out
