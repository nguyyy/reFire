"""`refire sauce mine` (04-sauce.md section 3): learn Nguy's polish layer from what he shipped.

Corpus = every long-form video on @fye_er: the rendered exports on disk plus channel uploads that
aren't (downloaded once to vods/sauce_refs/). A video whose Premiere project is on disk (matched
by duration) takes its events from the project, exactly; the rest get them from detect (booms,
emotes), whose accuracy is measured on the project videos and printed in the report. Every video
is transcribed from its own audio (render time = sequence time). Writes run/sauce/:
style_profile.json, exemplars.jsonl, style_report.md (+ corpus.json, cache/).
"""
from __future__ import annotations

import json
import os
import statistics as st
import subprocess
from pathlib import Path

from . import detect
from .project import mine as mine_project

HOME = Path.home()
PROJECT_DIRS = [HOME / "Desktop" / "premiere",
                *[Path(p) for p in os.environ.get("REFIRE_PROJECTS", "").split(os.pathsep) if p]]
VIDEO_DIRS = [HOME / "Desktop" / "iphone" / "sf90" / "rendered out" / "zy0x rot" / "videos", Path("vods/sauce_refs")]
CHANNEL = "https://www.youtube.com/@fye_er/videos"
OUT = Path("run/sauce")
SAME = 2.0                   # s; two renders this close in length are the same video
POLISH = ("boom", "emote")   # a project with fewer than MIN_POLISH of these isn't one of his finished edits
MIN_POLISH = 3
WINDOW = 8.0                 # s of transcript either side of an event (spec 3.5)


def duration(p) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                       capture_output=True, text=True)
    return float(r.stdout.strip() or 0)


def channel(out: Path = OUT, refresh: bool = False) -> list[dict]:
    """[{id, duration, title}] of the channel's uploads (yt-dlp, cached)."""
    p = out / "channel.json"
    if p.exists() and not refresh:
        return json.loads(p.read_text(encoding="utf-8"))
    r = subprocess.run(["yt-dlp", "--flat-playlist", "--print", "%(id)s|%(duration)s|%(title)s", CHANNEL],
                       capture_output=True, text=True, encoding="utf-8")
    rows = [dict(zip(("id", "duration", "title"), ln.split("|", 2))) for ln in r.stdout.splitlines() if ln.count("|") >= 2]
    for x in rows:
        x["duration"] = float(x["duration"] or 0)
    out.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return rows


def download(missing: list[dict], dest: Path = Path("vods/sauce_refs")) -> None:
    """channel uploads with no render on disk -> dest/<id>.mp4 (1080p)."""
    if missing:
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["yt-dlp", "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]",
                        "--merge-output-format", "mp4", "--no-progress", "-o", str(dest / "%(id)s.%(ext)s"),
                        *[m["id"] for m in missing]], check=True)


def projects(dirs=PROJECT_DIRS) -> list[dict]:
    """his finished edits: mined projects (autosaves skipped) with real polish on them."""
    out = []
    for d in dirs:
        for p in sorted(Path(d).rglob("*.prproj")) if Path(d).exists() else []:
            if "Auto-Save" in str(p):
                continue
            try:
                m = mine_project(p)
            except Exception as e:                       # empty / foreign project
                print(f"[sauce mine] skip {p.name}: {type(e).__name__}")
                continue
            if sum(e["type"].split(":")[0] in POLISH for e in m["events"]) >= MIN_POLISH:
                out.append(m)
    return out


def corpus(out: Path = OUT, video_dirs=VIDEO_DIRS, project_dirs=PROJECT_DIRS, fetch: bool = True) -> list[dict]:
    """one entry per distinct video: {key, video, duration, title, channel_id, project (mined) | None}."""
    ch = channel(out)
    files = [(p, duration(p)) for d in video_dirs for p in sorted(Path(d).glob("*.mp4")) if Path(d).exists()]
    if fetch:
        missing = [c for c in ch if not any(abs(c["duration"] - d) <= SAME for _, d in files)]
        download(missing)
        files += [(p, duration(p)) for p in sorted(Path("vods/sauce_refs").glob("*.mp4"))
                  if p not in {f for f, _ in files}]
    vids: list[tuple[Path, float]] = []
    for p, d in sorted(files, key=lambda f: (f[0].parent.name != "sauce_refs", f[0].name)):   # the upload wins
        if not any(abs(d - e) <= SAME for _, e in vids):
            vids.append((p, d))
    projs = projects(project_dirs)
    rows = []
    for p, d in vids:
        pr = next((m for m in projs if abs(m["duration"] - d) <= SAME), None)
        c = next((c for c in ch if abs(c["duration"] - d) <= SAME), {})
        rows.append({"key": c.get("id") or p.stem, "video": str(p), "duration": round(d, 2),
                     "title": c.get("title", p.stem), "channel_id": c.get("id"), "project": pr})
    return rows


def transcript(v: dict, cache: Path) -> list[dict]:
    """words of the video's own audio, cached."""
    from ..transcribe import transcribe
    cache.mkdir(parents=True, exist_ok=True)
    wav = cache / f"{v['key']}.wav"
    if not wav.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", v["video"], "-ac", "1", "-ar", "16000", str(wav)], check=True)
    return transcribe(wav, cache / f"{v['key']}.words.json")


def events(v: dict, cache: Path, lib=None) -> list[dict]:
    """the project's events, else detected booms + emotes (cached)."""
    if v["project"]:
        return [{**e, "source": "project"} for e in v["project"]["events"]]
    p = cache / f"{v['key']}.events.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    ev = [{"t": round(t, 3), "end": round(t + 1.25, 3), "type": "boom", "asset": detect.BOOM_FILE.name,
           "score": round(s, 3), "source": "detected"} for t, s in detect.booms(v["video"])]
    ev += [{"t": round(e["t"], 3), "end": round(e["end"] + 1 / detect.EMOTE_FPS, 3), "type": f"emote:{e['name']}",
            "asset": e["name"], "score": round(e["score"], 3), "source": "detected"}
           for e in detect.emotes(v["video"], lib)]
    ev.sort(key=lambda e: e["t"])
    p.write_text(json.dumps(ev, indent=1), encoding="utf-8")
    return ev


# --------------------------------------------------------------------------- statistics

def kind(e: dict) -> str:
    return e["type"].split(":")[0]


def q(xs: list[float]) -> dict:
    """median + quartiles (+ n) of a sample."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {"n": 0}
    qs = st.quantiles(xs, n=4) if len(xs) >= 4 else [xs[0], 0, xs[-1]]      # too few for quartiles: the range
    return {"n": len(xs), "median": round(st.median(xs), 3), "p25": round(qs[0], 3), "p75": round(qs[2], 3),
            "min": round(xs[0], 3), "max": round(xs[-1], 3)}


BINS = [0, 15, 30, 45, 60, 90, 120, 180, 240, 300, 420, 600, 900, 1500]


def rate_curve(vids: list[dict], types: list[str]) -> dict:
    """events/min per type in each time bin, pooled over the videos that reach that bin."""
    out = {}
    for ty in types:
        rows = []
        for a, b in zip(BINS, BINS[1:]):
            cover = sum(max(0.0, min(b, v["duration"]) - a) for v in vids) / 60
            n = sum(1 for v in vids for e in v["events"] if kind(e) == ty and a <= e["t"] < b)
            if cover > 0:
                rows.append({"t0": a, "t1": b, "per_min": round(n / cover, 3), "n": n})
        out[ty] = rows
    return out


MAX_WORD_S = 0.7     # whisper bills a pause to the word before it: clamp before measuring gaps (subtitles._gap)


def boom_timing(v: dict) -> list[dict]:
    """per boom: the last word started before it, the gap from its (clamped) end, and whether a
    word is still being spoken at the hit."""
    out = []
    end = lambda w: min(w["end"], w["start"] + MAX_WORD_S)       # noqa: E731
    for e in v["events"]:
        if kind(e) != "boom":
            continue
        t = e["t"]
        prev = [w for w in v["words"] if w["start"] < t - 0.05]
        live = any(w["start"] < t - 0.05 and end(w) > t + 0.05 for w in v["words"])
        if prev:
            w = max(prev, key=lambda w: w["start"])
            out.append({"t": t, "word": w["text"], "delta": round(t - end(w), 3), "over_speech": live})
    return out


def overlaps(v: dict, a: str, b: str, slack: float = 0.3) -> tuple[int, int]:
    """(events of type a with a b starting within slack of it, events of type a)."""
    A = [e for e in v["events"] if kind(e) == a]
    B = [e for e in v["events"] if kind(e) == b]
    return sum(any(abs(x["t"] - y["t"]) <= slack or (y["t"] <= x["t"] <= y["end"] and b in ("zoom", "punch"))
                   for y in B) for x in A), len(A)


def profile(vids: list[dict]) -> dict:
    types = ["boom", "emote", "punch", "zoom", "sfx", "fx"]
    proj = [v for v in vids if v["project"]]
    by = {ty: [e for v in vids for e in v["events"] if kind(e) == ty] for ty in types}
    body = lambda v: max(0.0, v["duration"] - 60) / 60             # noqa: E731
    rates = {}
    for ty in types:
        src = proj if ty in ("punch", "zoom", "sfx", "fx") else vids          # detect only finds booms + emotes
        cold = sum(1 for v in src for e in v["events"] if kind(e) == ty and e["t"] < 60)
        bod = sum(1 for v in src for e in v["events"] if kind(e) == ty and e["t"] >= 60)
        rates[ty] = {"videos": len(src), "cold_open_per_min": round(cold / max(1e-9, len(src)), 3),
                     "body_per_min": round(bod / max(1e-9, sum(body(v) for v in src)), 3)}
    gaps = {}
    for ty in types:
        g = []
        for v in vids:
            ts = sorted(e["t"] for e in v["events"] if kind(e) == ty)
            g += [b - a for a, b in zip(ts, ts[1:])]
        gaps[ty] = q(g)
    anyg = []
    for v in vids:
        ts = sorted(e["t"] for e in v["events"] if kind(e) in ("boom", "emote", "punch", "zoom"))
        anyg += [b - a for a, b in zip(ts, ts[1:])]
    bt = [b for v in vids for b in boom_timing(v)]
    emo: dict[str, dict] = {}
    for e in by["emote"]:
        r = emo.setdefault(e["type"][6:], {"count": 0, "scale": [], "hold": []})
        r["count"] += 1
        r["hold"].append(e["end"] - e["t"])
        if e.get("scale"):
            r["scale"].append(e["scale"])
    lib = detect.library([e.get("file") for e in by["emote"] if e.get("file")])
    emotes = sorted(({"name": k, "count": r["count"], "file": str(lib[k]) if k in lib else None,
                      "hold": q(r["hold"]).get("median"), "scale": q(r["scale"]).get("median")}
                     for k, r in emo.items()), key=lambda r: -r["count"])
    fades = []
    for e in by["emote"]:
        ks = e.get("opacity_keys") or []
        if len(ks) >= 2 and e["end"] > e["t"]:
            fades.append(max(t for t, _ in ks) / (e["end"] - e["t"]))
    pz = by["punch"] + by["zoom"]
    co = {}
    for a, b in (("boom", "punch"), ("boom", "emote"), ("emote", "boom"), ("zoom", "emote"), ("emote", "punch")):
        src = proj if "punch" in (a, b) or "zoom" in (a, b) else vids
        hit, n = map(sum, zip(*(overlaps(v, a, b) for v in src))) if src else (0, 0)
        co[f"{a}+{b}"] = {"rate": round(hit / n, 3) if n else None, "n": n}
    sfx: dict[str, int] = {}
    for e in by["sfx"]:
        sfx[e["type"][4:]] = sfx.get(e["type"][4:], 0) + 1
    return {
        "videos": len(vids), "with_project": len(proj),
        "rates": rates, "curve": rate_curve(vids, ["boom", "emote"]) | rate_curve(proj, ["punch", "zoom"]),
        "gaps": gaps, "gap_any": q(anyg),
        "boom": {"delta": q([b["delta"] for b in bt if b["delta"] <= 3]),
                 "over_speech": round(sum(b["over_speech"] for b in bt) / max(1, len(bt)), 3),
                 "db": q([e.get("db") for e in by["boom"] if e.get("db") is not None]),
                 "db_keys": [e["db_keys"] for e in by["boom"] if e.get("db_keys")][:5],
                 "clip_len": q([e["end"] - e["t"] for e in by["boom"] if e.get("source") == "project"])},
        "emote": {"library": emotes, "hold": q([e["end"] - e["t"] for e in by["emote"] if e.get("source") == "project"]),
                  "scale": q([e["scale"] for e in by["emote"] if e.get("scale")]), "fade_end": q(fades),
                  "centered": sum(1 for e in by["emote"] if e.get("position") == [0.5, 0.5]),
                  "shadow": sum(1 for e in by["emote"] if e.get("shadow"))},
        "punch": {k: q([e[k] for e in by["punch"]]) for k in ("ramp_in", "ramp_out", "scale", "peak")}
                 | {"duration": q([e["end"] - e["t"] for e in by["punch"]])},
        "zoom": {k: q([e[k] for e in by["zoom"]]) for k in ("ramp_in", "ramp_out", "scale", "peak")}
                | {"duration": q([e["end"] - e["t"] for e in by["zoom"]]),
                   "anchors": [e["anchor"] for e in by["zoom"]]},
        "punch_zoom_n": len(pz),
        "cooccur": co, "sfx": dict(sorted(sfx.items(), key=lambda x: -x[1])),
    }


# --------------------------------------------------------------------------- exemplars

def window(words: list[dict], t: float, w: float = WINDOW) -> str:
    """the transcript around t with [*] at the event."""
    before = " ".join(x["text"] for x in words if t - w <= x["start"] < t)
    after = " ".join(x["text"] for x in words if t <= x["start"] < t + w)
    return f"{before} [*] {after}".strip()


def exemplars(vids: list[dict]) -> list[dict]:
    out = []
    for v in vids:
        for e in v["events"]:
            if kind(e) in ("boom", "emote", "punch", "zoom", "sfx"):
                out.append({"video": v["key"], "title": v["title"], "t": e["t"], "type": kind(e),
                            "asset": e["type"].split(":", 1)[1] if ":" in e["type"] else e.get("asset"),
                            "source": e.get("source"), "text": window(v["words"], e["t"])})
    return out


LABEL_SYSTEM = """You label moments from zy0x's Genshin Impact stream edits (YouTube channel @fye_er). For each
moment the editor added a polish event at [*] in the transcript (a vine boom, a meme emote, a facecam
punch-zoom, a gameplay zoom, or a sound effect). Say in <= 12 words why the editor put it there:
what is funny, awkward, out of pocket, a fail, a reveal. Be concrete about this moment, not generic.
If the transcript doesn't show a reason, say "unclear" and why in a few words.
Reply with ONLY JSON: {"labels": [{"i": 0, "why": "..."}]}"""


def label(rows: list[dict], model: str = "claude-opus-5", ask=None) -> None:
    """one Claude call for every exemplar's `why` (spec 3.5: written once, Nguy spot-checks)."""
    from pydantic import BaseModel

    from .. import director

    class L(BaseModel):
        i: int
        why: str

    class Out(BaseModel):
        labels: list[L]

    todo = [i for i, r in enumerate(rows) if not r.get("why")]
    if not todo:
        return
    text = "\n".join(f"{i}. [{rows[i]['type']}{':' + rows[i]['asset'] if rows[i]['type'] in ('emote', 'sfx') else ''}] "
                     f"{rows[i]['text']}" for i in todo)
    out = ask(text, Out) if ask else director._complete_cli(model, LABEL_SYSTEM, [{"type": "text", "text": text}], Out,
                                                            tag="sauce labels", effort="low", live=False)
    for x in out.labels:
        if 0 <= x.i < len(rows):
            rows[x.i]["why"] = x.why


# --------------------------------------------------------------------------- validation + report

def validate(v: dict, cache: Path, lib=None) -> dict:
    """detectors vs the project's exact events on its own export: recall / false hits per type."""
    p = cache / f"{v['key']}.validate.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    truth = v["project"]["events"]
    tb = [e["t"] for e in truth if kind(e) == "boom"]
    gb = [t for t, _ in detect.booms(v["video"])]
    te = [(e["t"], e["end"], e["type"][6:]) for e in truth if kind(e) == "emote"]
    ge = detect.emotes(v["video"], lib)
    named = lambda n: lib is None or n in lib                     # noqa: E731
    res = {"boom": {"truth": len(tb), "found": sum(any(abs(g - t) < 0.1 for g in gb) for t in tb),
                    "false": sum(not any(abs(g - t) < 0.1 for t in tb) for g in gb)},
           "emote": {"truth": len(te), "in_library": sum(named(n) for _, _, n in te),
                     "found": sum(any(a - 0.6 <= g["t"] <= b and g["name"] == n for g in ge) for a, b, n in te),
                     "false": sum(not any(a - 0.6 <= g["t"] <= b for a, b, _ in te) for g in ge)}}
    p.write_text(json.dumps(res), encoding="utf-8")
    return res


def _row(d: dict, unit: str = "s") -> str:
    if not d or not d.get("n"):
        return "no data"
    return f"median {d['median']}{unit} (middle half {d['p25']}-{d['p75']}{unit}, range {d['min']}-{d['max']}, n={d['n']})"


def report(vids: list[dict], prof: dict, ex: list[dict], val: dict) -> str:
    L = ["# Sauce style report", "",
         "What `refire sauce mine` measured on your finished videos. Check that it saw your style right before",
         "booms and emotes get built on it (04-sauce.md section 3).", "", "## Videos", "",
         "| Video | Length | Events from | Booms | Emotes | Punches | Zooms | Other SFX |", "|---|---|---|---|---|---|---|---|"]
    for v in vids:
        c = lambda ty: sum(1 for e in v["events"] if kind(e) == ty)       # noqa: E731
        src = "project (exact)" if v["project"] else "detected"
        L.append(f"| {v['title']} | {v['duration'] / 60:.1f} min | {src} | {c('boom')} | {c('emote')} | "
                 f"{c('punch') if v['project'] else '-'} | {c('zoom') if v['project'] else '-'} | "
                 f"{c('sfx') if v['project'] else '-'} |")
    L += ["", "Detected videos only show booms and emotes; punches, zooms and other SFX come from the "
          f"{prof['with_project']} videos whose project is on disk.", "", "## How often", "",
          "| Event | Cold open (first 60 s), per min | Body, per min | From videos |", "|---|---|---|---|"]
    for ty, r in prof["rates"].items():
        L.append(f"| {ty} | {r['cold_open_per_min']} | {r['body_per_min']} | {r['videos']} |")
    L += ["", "Rate over time (events per minute, pooled):", "", "| From | To | " + " | ".join(prof["curve"]) + " |",
          "|---|---|" + "---|" * len(prof["curve"])]
    first = next(iter(prof["curve"].values()))
    for i, b in enumerate(first):
        L.append(f"| {b['t0']}s | {b['t1']}s | " + " | ".join(
            str(c[i]["per_min"]) if i < len(c) else "" for c in prof["curve"].values()) + " |")
    L += ["", "Gaps between events of the same kind:", ""]
    L += [f"- {ty}: {_row(g)}" for ty, g in prof["gaps"].items() if g.get("n")]
    L += [f"- any two (boom, emote, punch, zoom): {_row(prof['gap_any'])}", "", "## Booms", "",
          f"- After the line: {_row(prof['boom']['delta'])} from the end of the last word to the hit",
          f"- Over speech (a word still going at the hit): {100 * prof['boom']['over_speech']:.0f}%",
          f"- Clip volume: {_row(prof['boom']['db'], ' dB')}",
          f"- Volume keyframes seen: {prof['boom']['db_keys'] or 'none'}",
          f"- Clip length: {_row(prof['boom']['clip_len'])}", "", "## Emotes", "",
          f"- Hold: {_row(prof['emote']['hold'])}; scale {_row(prof['emote']['scale'], '%')}",
          f"- Fade: opacity reaches 0 at {_row(prof['emote']['fade_end'], '')} of the hold (1.0 = the end)",
          f"- Centered: {prof['emote']['centered']}, drop shadow: {prof['emote']['shadow']} (project emotes)", "",
          "Library, by use (this becomes the closed library):", "", "| Emote | Uses | Hold | Scale | File |", "|---|---|---|---|---|"]
    L += [f"| {r['name']} | {r['count']} | {r['hold']:.1f}s | {r['scale'] or '-'} | {r['file'] or 'NOT FOUND'} |"
          for r in prof["emote"]["library"]]
    for ty in ("punch", "zoom"):
        d = prof[ty]
        L += ["", f"## {ty.title()}es" if ty == "punch" else "## Gameplay zooms", "",
              f"- Duration: {_row(d['duration'])}", f"- Ease in: {_row(d['ramp_in'])}; ease out: {_row(d['ramp_out'])}",
              f"- Scale it snaps to: {_row(d['scale'], '%')}; peak (after the slow push): {_row(d['peak'], '%')}"]
        if ty == "zoom":
            L.append(f"- Anchors: {d['anchors']}")
    L += ["", "## Together", ""]
    L += [f"- {k.replace('+', ' with ')}: {100 * v['rate']:.0f}% of {v['n']}" if v["rate"] is not None else f"- {k}: no data"
          for k, v in prof["cooccur"].items()]
    L += ["", "Other sound effects (not in the spec yet): " + (", ".join(f"{k} x{n}" for k, n in prof["sfx"].items()) or "none"),
          "", "## Detector check (on the videos that have a project)", "",
          "| Video | Booms found | False booms | Emotes found | In library | False emotes |", "|---|---|---|---|---|---|"]
    for k, r in val.items():
        b, e = r["boom"], r["emote"]
        L.append(f"| {k} | {b['found']}/{b['truth']} | {b['false']} | {e['found']}/{e['truth']} | {e['in_library']} | {e['false']} |")
    L += ["", "## Examples (Claude's guess at why; spot-check these)", ""]
    for ty in ("boom", "emote", "punch", "zoom", "sfx"):
        rows = [r for r in ex if r["type"] == ty][:4]
        for r in rows:
            L.append(f"- **{ty}{':' + str(r['asset']) if ty in ('emote', 'sfx') else ''}** {r['title'][:40]} "
                     f"{int(r['t'] // 60)}:{r['t'] % 60:04.1f} - {r.get('why', '?')}  \n  _{r['text'][:220]}_")
    return "\n".join(L) + "\n"


def run(out: Path = OUT, fetch: bool = True, labels: bool = True, log=print) -> Path:
    """the whole of section 3 -> out/style_report.md."""
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    vids = corpus(out, fetch=fetch)
    lib = detect.library([e["file"] for v in vids if v["project"] for e in v["project"]["events"] if e.get("file")])
    val = {}
    for v in vids:
        v["words"] = transcript(v, cache)
        v["events"] = events(v, cache, lib)
        if v["project"]:
            val[v["title"]] = validate(v, cache, lib)
        log(f"[sauce mine] {v['key']}: {len(v['events'])} events, {len(v['words'])} words "
            f"({'project' if v['project'] else 'detected'})")
    prof = profile(vids)
    exp = out / "exemplars.jsonl"
    old = {(r["video"], r["t"], r["type"]): r.get("why") for r in map(json.loads, exp.read_text(encoding="utf-8").splitlines())} \
        if exp.exists() else {}
    ex = exemplars(vids)
    for r in ex:
        r["why"] = old.get((r["video"], r["t"], r["type"]))
    if labels:
        label(ex)
    exp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ex), encoding="utf-8")
    (out / "style_profile.json").write_text(json.dumps(prof, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "corpus.json").write_text(json.dumps([{k: v[k] for k in ("key", "video", "duration", "title", "channel_id")}
                                                 | {"project": v["project"]["project"] if v["project"] else None}
                                                 for v in vids], indent=1, ensure_ascii=False), encoding="utf-8")
    rep = out / "style_report.md"
    rep.write_text(report(vids, prof, ex, val), encoding="utf-8")
    return rep
