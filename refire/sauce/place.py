"""`refire sauce <run_dir>`: booms + emotes on the live timeline (04-sauce.md 5.3 + 6; P2).

plan.direct proposes, this decides, deterministically: drop below tau, snap booms to the end of
the punchline word + the measured offset (Nguy: on the word end, not a pause after), a paired
emote lands on its boom's frame, highest confidence first under the density ceiling (1.5x his
recent per-bin rates) and the 3-per-10-s soft cap, no two booms overlapping, one emote on screen
(a newer one cuts the older short). Emotes are pre-rendered once as ProRes 4444 with the fade,
scale and drop shadow baked in (spec 6), booms are the vine-boom file at his flat -14 dB (Nguy).
Writes sauce.json, sauce_log.md, ppro/ops_sauce.json (V5 emotes, A3 booms, tagged refire:sauce:).
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from ..learn.prproj import TICKS
from . import detect, mine
from .project import VOL_0DB

DEFAULTS = {"tau": 0.6, "density": 1.5, "cold_open": 60.0, "soft_cap": 3, "soft_window": 10.0, "earned": 0.9,
            "boom_len": 1.25, "boom_db": -14.0, "min_emote": 0.6, "size": [1920, 1080], "max_emote_h": 700}
PREFIX = "refire:sauce:"
V_EMOTE, A_BOOM = 4, 2           # V5, A3 (00-shared track contract)
MAX_WORD_S = mine.MAX_WORD_S


def config(overrides: dict | None = None) -> dict:
    return {**DEFAULTS, **(overrides or {})}


def hit_time(phrase: dict, delta: float) -> float:
    """end of the phrase's last word (whisper's padding clamped) + the measured boom offset."""
    w = phrase["words"][-1]
    return max(phrase["t0"], min(w["t1"], w["t0"] + MAX_WORD_S) + delta)


def cap(curve: list[dict], body: float, t: float, density: float) -> tuple[tuple[float, float], int]:
    """(the curve bin holding t, the most events of a type it may take)."""
    b = next((r for r in curve if r["t0"] <= t < r["t1"]), None)
    if b is None:                                  # past the measured curve: the body rate, minute by minute
        b = {"t0": t // 60 * 60, "t1": t // 60 * 60 + 60, "per_min": body}
    span = (b["t0"], b["t1"])
    rate = max(b["per_min"], body)                 # thin samples read 0 in some bins: never below the body rate
    return span, max(1, math.ceil(density * rate * (span[1] - span[0]) / 60))


def solve(evs: list[dict], phrases: dict, prof: dict, cfg: dict) -> tuple[list[dict], list[dict]]:
    """proposals -> (kept events with times, dropped with a reason)."""
    delta = prof["boom"]["delta"].get("median", 0.0)
    hold = prof["emote"]["hold"].get("median", 1.4)
    kept: list[dict] = []
    dropped: list[dict] = []
    boom_at = {}
    for e in evs:
        p = phrases[e["phrase"]]
        e = {**e, "t": round(hit_time(p, delta), 3), "text": p["text"]}
        if e["confidence"] < cfg["tau"]:
            dropped.append({**e, "why": f"confidence {e['confidence']} < tau {cfg['tau']}"})
            continue
        e["end"] = round(e["t"] + (cfg["boom_len"] if e["type"] == "boom" else hold), 3)
        kept.append(e)
    # a paired emote lands on its boom's frame (he puts half of them there)
    for e in kept:
        if e["type"] == "boom":
            boom_at[e["phrase"]] = e["t"]
    out: list[dict] = []
    for e in sorted(kept, key=lambda e: -e["confidence"]):
        same = [o for o in out if o["type"] == e["type"]]
        span, n = cap(prof["curve"].get(e["type"], []), prof["rates"][e["type"]]["body_per_min"], e["t"], cfg["density"])
        if sum(span[0] <= o["t"] < span[1] for o in same) >= n:
            dropped.append({**e, "why": f"density ceiling ({n} {e['type']}s in {span[0]:.0f}-{span[1]:.0f}s)"})
            continue
        if e["t"] >= cfg["cold_open"] and e["confidence"] < cfg["earned"] and \
                sum(abs(o["t"] - e["t"]) < cfg["soft_window"] / 2 for o in out) >= cfg["soft_cap"]:
            dropped.append({**e, "why": f"soft cap ({cfg['soft_cap']} per {cfg['soft_window']:.0f}s)"})
            continue
        if e["type"] == "boom":
            if any(abs(o["t"] - e["t"]) < cfg["boom_len"] for o in same):
                dropped.append({**e, "why": "another boom is still playing"})
                continue
        else:
            if e["phrase"] in boom_at:
                e["t"] = boom_at[e["phrase"]]
                e["end"] = round(e["t"] + hold, 3)
            clash = [o for o in same if o["t"] < e["end"] and e["t"] < o["end"]]
            ok = True
            for o in clash:                            # one on screen: the later one cuts the earlier short
                first, second = (o, e) if o["t"] <= e["t"] else (e, o)
                if second["t"] - first["t"] < cfg["min_emote"]:
                    ok = False
                    break
            if not ok:
                dropped.append({**e, "why": "another emote is on screen"})
                continue
            for o in clash:
                first, second = (o, e) if o["t"] <= e["t"] else (e, o)
                first["end"] = second["t"]
        out.append(e)
    out.sort(key=lambda e: e["t"])
    n = {"boom": 0, "emote": 0}
    for e in out:
        n[e["type"]] += 1
        e["id"] = f"{e['type'][0]}{n[e['type']]:02d}"
    return out, dropped


# --------------------------------------------------------------------------- assets

def library(prof: dict, exemplars: list[dict]) -> list[dict]:
    """the closed library (his emotes by use; one-off screenshots aren't emotes) + the moods they
    were used for, from the exemplar labels."""
    out = []
    for r in prof["emote"]["library"]:
        if not r.get("file") or r["name"].lower().startswith("screenshot"):
            continue
        whys = [x["why"] for x in exemplars if x["type"] == "emote" and x["asset"] == r["name"] and x.get("why")]
        out.append({**r, "moods": "; ".join(whys[:2])})
    return out


def display_height(r: dict, prof: dict, cfg: dict) -> int:
    """px on the 1080 frame: his own scale for this emote, else the median of his others."""
    from PIL import Image

    def h(row):
        with Image.open(row["file"]) as im:
            return im.height * row["scale"] / 100
    known = [h(x) for x in prof["emote"]["library"] if x.get("scale") and x.get("file") and Path(x["file"]).exists()]
    got = h(r) if r.get("scale") else (sorted(known)[len(known) // 2] if known else 280)
    return int(min(cfg["max_emote_h"], got) * cfg["size"][1] / 1080) // 2 * 2


def render_emote(src: Path, out: Path, height: int, hold: float, fade_end: float, fps: float, size=(1920, 1080)) -> Path:
    """a centered emote on a transparent frame, drop shadow, opacity 1 -> 0 by fade_end x hold, as a
    ProRes 4444 clip (spec 6: no keyframe easing in Premiere). Cached by name + size."""
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    src = Path(src)
    if src.suffix.lower() == ".gif":
        inp = ["-ignore_loop", "0", "-t", f"{hold}", "-i", str(src)]
    else:                                          # webp/avif/jpg/png: one still, via PIL (ffmpeg's webp is spotty)
        from PIL import Image
        png = out.with_suffix(".src.png")
        Image.open(src).convert("RGBA").save(png)
        inp = ["-loop", "1", "-t", f"{hold}", "-i", str(png)]
    W, H = size
    fade = max(0.1, fade_end * hold)
    fc = (f"[0:v]format=rgba,scale=-2:{height},split[e][s];"
          f"[s]colorchannelmixer=rr=0:gg=0:bb=0:aa=0.5,boxblur=4:1[sh];"
          f"[1:v][sh]overlay=x=(W-w)/2+8:y=(H-h)/2+8:shortest=1[b];"
          f"[b][e]overlay=x=(W-w)/2:y=(H-h)/2:shortest=1,fade=t=out:st=0:d={fade:.3f}:alpha=1,format=yuva444p10le[v]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inp, "-f", "lavfi", "-t", f"{hold}",
                    "-i", f"color=c=black@0.0:s={W}x{H}:r={fps:g},format=rgba", "-filter_complex", fc, "-map", "[v]",
                    "-t", f"{hold}", "-r", f"{fps:g}", "-c:v", "prores_ks", "-profile:v", "4444",
                    "-pix_fmt", "yuva444p10le", "-an", str(out)], check=True)
    return out


def ops(kept: list[dict], tmap, emote_files: dict, cfg: dict) -> dict:
    level = round(VOL_0DB * 10 ** (cfg["boom_db"] / 20), 6)
    out = [{"op": "sauce_clear", "prefix": PREFIX, "video": [V_EMOTE], "audio": [A_BOOM]}]
    for e in kept:
        a, b = tmap.ticks(e["t"]), tmap.ticks(e["end"])
        base = {"op": "place_media", "bin": "refire_sauce", "prefix": PREFIX, "name": f"{PREFIX}{e['id']}",
                "start": str(a), "end": str(b)}
        if e["type"] == "boom":
            out.append({**base, "file": str(detect.BOOM_FILE.resolve()), "audio": True, "track": A_BOOM, "level": level})
        else:
            out.append({**base, "file": str(Path(emote_files[e["asset"]]).resolve()), "audio": False, "track": V_EMOTE})
    return {"version": 1, "feature": "sauce", "ticks_per_frame": tmap.ticks_per_frame, "ops": out}


def log_md(kept: list[dict], dropped: list[dict], cfg: dict) -> str:
    stamp = lambda t: f"{int(t // 60):02d}:{t % 60:05.2f}"       # noqa: E731
    L = ["# sauce log", "", f"{sum(e['type'] == 'boom' for e in kept)} booms, {sum(e['type'] == 'emote' for e in kept)} emotes "
         f"(tau {cfg['tau']}, density {cfg['density']}x)", "", "## Placed", ""]
    L += [f"- {stamp(e['t'])} {e['type']}{'+' + e['asset'] if e['type'] == 'emote' else ''} ({e['confidence']:.2f}) "
          f"\"{e['text']}\" - {e['reason']}" + (f" / critic: {e['critic']}" if e.get("critic") else "") for e in kept]
    L += ["", "## Dropped", ""]
    L += [f"- {stamp(e['t'])} {e['type']}{'+' + str(e.get('asset')) if e['type'] == 'emote' else ''} ({e['confidence']:.2f}) "
          f"\"{e['text']}\" - {e['why']}" for e in sorted(dropped, key=lambda e: e["t"])]
    return "\n".join(L) + "\n"


def run(run_dir, timeline=None, overrides: dict | None = None, model: str = "claude-opus-5", backend: str = "cli",
        ask=None, dry_run: bool = False, profile_dir: Path = mine.OUT, log=print) -> dict:
    run_dir = Path(run_dir)
    cfg = config(overrides)
    pp, xp = profile_dir / "style_profile.json", profile_dir / "exemplars.jsonl"
    if not pp.exists():
        raise FileNotFoundError(f"no {pp}: run `refire sauce mine` first")
    prof = json.loads(pp.read_text(encoding="utf-8"))
    exemplars = [json.loads(x) for x in xp.read_text(encoding="utf-8").splitlines()] if xp.exists() else []
    from . import evaluate, plan
    added, negatives = evaluate.load_corrections()          # his fixes on earlier runs (spec 7.4)
    exemplars = exemplars + added
    data = plan.load(run_dir, timeline, log=log)
    lib = library(prof, exemplars)
    root = run_dir / "sauce"
    names = [] if ask else plan.sheets(run_dir, data["phrases"], plan.marked(data["phrases"]), root)
    evs = plan.direct(data, prof, lib, exemplars, model=model, backend=backend, ask=ask, sheet_names=names,
                      cwd=root, negatives=negatives, log=log)
    kept, dropped = solve(evs, {p["id"]: p for p in data["phrases"]}, prof, cfg)
    (run_dir / "sauce.json").write_text(json.dumps({"version": 1, "params": cfg, "events": kept}, indent=1,
                                                   ensure_ascii=False), encoding="utf-8")
    (run_dir / "sauce_log.md").write_text(log_md(kept, dropped, cfg), encoding="utf-8")
    if not dry_run:
        fps = TICKS / data["tmap"].ticks_per_frame
        hold = prof["emote"]["hold"].get("median", 1.4)
        fade_end = prof["emote"]["fade_end"].get("median", 0.9)
        files = {}
        for name in sorted({e["asset"] for e in kept if e["type"] == "emote"}):
            r = next(x for x in lib if x["name"] == name)
            h = display_height(r, prof, cfg)
            files[name] = render_emote(Path(r["file"]), run_dir / "sauce" / f"{name}_{h}.mov", h, hold, fade_end, fps,
                                       tuple(cfg["size"]))
        (run_dir / "ppro").mkdir(exist_ok=True)
        (run_dir / "ppro" / "ops_sauce.json").write_text(json.dumps(ops(kept, data["tmap"], files, cfg), indent=1),
                                                         encoding="utf-8")
    log(f"[sauce] {sum(e['type'] == 'boom' for e in kept)} booms, {sum(e['type'] == 'emote' for e in kept)} emotes "
        f"placed; {len(dropped)} dropped (sauce_log.md)")
    return {"kept": kept, "dropped": dropped}
