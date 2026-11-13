"""`refire auto <vod>`: a VOD number in, a finished base edit (mp4) out. Nothing else to type.

Every input `make` needs is decided here, each with the reason it was picked; the
choices land in auto.json and summary.md next to the video:

  title, game  the VOD's own Twitch metadata (the chat download carries it)
  brief        none: the director's stream-driven mode reads the whole stream and picks its
               strongest story. A pre-pass writing a brief would judge on less than the
               director sees
  duration     median runtime of your finished edits (`refire learn`), capped by the VOD
  stages       the measured defaults: boundary / trim / critics / markers stay off, each
               raised fix_cost in `refire eval`; 15 ms de-pop fades stay on
  base edit    no emotes, sfx, music, section cards or zoom punch-ins: the cut plus
               captions, at the source's 1080p
  encoder      h264_nvenc when this machine has a working NVENC, else libx264
  loudness     one two-pass loudnorm over the finished audio to -14 LUFS (YouTube's
               reference; it turns loud uploads down but never quiet ones up), video copied

Then the mp4 is checked (length, canvas, audio) before the run counts as done.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from statistics import median

DEFAULT_DURATION = 900.0     # no finished edits learned yet: your usual ~15 min target
MIN_DURATION = 180.0
MAX_SHARE = 0.5              # a cut never asks for more than half the stream
SIZE = (1920, 1080)
OUT_NAME = "final.mp4"
LUFS, TRUE_PEAK, LRA = -14.0, -1.5, 11.0


def vod_info(chat_path) -> dict:
    """Title / game / length from the chat download's `video` block ({} if absent)."""
    try:
        v = json.loads(Path(chat_path).read_text(encoding="utf-8")).get("video") or {}
    except (OSError, ValueError):
        return {}
    return {"title": v.get("title") or "", "game": v.get("game") or "",
            "length": float(v.get("length") or 0)}


def learned_duration(run_root="run", vod_len: float = 0.0) -> tuple[float, str]:
    """Target runtime = median runtime of the finished edits `refire learn` has seen."""
    ends = []
    for f in sorted(Path(run_root).glob("*/clip_diff.json")):
        try:
            final = json.loads(f.read_text(encoding="utf-8")).get("final") or []
        except (OSError, ValueError):
            continue
        if final:
            ends.append(max(s["seq_out"] for s in final))
    if ends:
        t, why = median(ends), f"median runtime of your {len(ends)} finished edits"
    else:
        t, why = DEFAULT_DURATION, "no finished edits learned yet, default ~15 min"
    if vod_len and t > MAX_SHARE * vod_len:
        t, why = MAX_SHARE * vod_len, why + f", capped at half the {vod_len / 60:.0f} min VOD"
    return round(max(MIN_DURATION, t)), why


def pick_encoder() -> str:
    """h264_nvenc if a 0.2s probe encode works (driver + card present), else libx264."""
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                            "-i", "color=c=black:s=256x256:d=0.2", "-c:v", "h264_nvenc",
                            "-f", "null", "-"], capture_output=True, timeout=60)
        return "h264_nvenc" if r.returncode == 0 else "libx264"
    except (OSError, subprocess.TimeoutExpired):
        return "libx264"


def plan(info: dict, run_root="run", duration: float | None = None,
         brief: str | None = None) -> dict:
    """make() keyword arguments for this VOD + a {setting: reason} map. Pure but for the
    encoder probe."""
    dur, dur_why = ((duration, "given on the command line") if duration
                    else learned_duration(run_root, info.get("length", 0.0)))
    enc = pick_encoder()
    kw = {"brief": brief, "duration_s": float(dur), "title": info.get("title", ""),
          "game": (info.get("game") or "").lower(), "render": True, "encoder": enc,
          "size": SIZE, "out_name": OUT_NAME,
          # base edit: the cut and its captions, nothing stylized on top
          "emotes": False, "sfx": False, "music": False, "cards": False,
          "motion_zoom": False,
          # measured stage defaults, spelled out so a later default flip shows up here
          "boundary": False, "trim": False, "critics": False, "markers": False,
          "lessons": False}
    why = {"title": "Twitch metadata", "game": "Twitch metadata",
           "brief": ("given on the command line" if brief else
                     "none: the director picks the stream's strongest story"),
           "duration_s": dur_why,
           "encoder": "working NVENC found" if enc.endswith("nvenc") else "no NVENC, CPU x264",
           "base edit": "no emotes / sfx / music / cards / zoom punch-ins; 1080p + captions",
           "stages": "boundary, trim, critics, markers off: each raised fix_cost in "
                     "refire eval; 15 ms de-pop fades on"}
    return {"kwargs": kw, "why": why}


def loudnorm_filter(measured: dict | None = None) -> str:
    """loudnorm for pass 1 (measure, json) or pass 2 (apply pass 1's numbers, linear
    when the peak allows it -- loudnorm drops to dynamic on its own when it doesn't)."""
    base = f"loudnorm=I={LUFS}:TP={TRUE_PEAK}:LRA={LRA}"
    if not measured:
        return base + ":print_format=json"
    return (base + f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
            f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
            f":offset={measured['target_offset']}:linear=true:print_format=summary")


def master_audio(mp4) -> dict:
    """Normalize the whole program's loudness in place (audio re-encoded, video copied).
    Whole-program, not per clip: per-clip loudnorm would flatten a scream to the level of
    a mumble. Returns pass 1's measurement."""
    mp4 = Path(mp4)
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(mp4), "-vn",
                        "-af", loudnorm_filter(), "-f", "null", "-"],
                       capture_output=True, text=True, errors="replace")
    err = r.stderr
    measured = json.loads(err[err.rindex("{"):err.rindex("}") + 1])
    tmp = mp4.with_name(mp4.stem + "_mastered.mp4")
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(mp4), "-map", "0:v",
                        "-map", "0:a", "-c:v", "copy", "-af", loudnorm_filter(measured),
                        "-ar", "48000", "-c:a", "aac", "-b:a", "192k", "-y", str(tmp)],
                       capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        raise RuntimeError("loudness pass failed:\n" + r.stderr[-1500:])
    tmp.replace(mp4)
    return measured


def probe(mp4) -> dict:
    """Length / canvas / audio of the rendered file (ffprobe)."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration:stream=codec_type,width,height", "-of", "json",
                        str(mp4)], capture_output=True, text=True)
    d = json.loads(r.stdout or "{}")
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), {})
    return {"duration_s": round(float(d.get("format", {}).get("duration") or 0), 1),
            "width": v.get("width"), "height": v.get("height"),
            "audio": any(s.get("codec_type") == "audio" for s in d.get("streams", []))}


def verify(mp4, target_s: float, tol: float = 0.35) -> tuple[dict, list[str], list[str]]:
    """Hard problems: no file, wrong canvas, no audio. Note: runtime outside make's own
    duration slack -- the director is told the target is a guide, so that's a heads-up,
    not a failed render. Returns (probe, problems, notes)."""
    if not Path(mp4).exists():
        return {}, [f"{mp4} was not written"], []
    p = probe(mp4)
    bad, notes = [], []
    if (p["width"], p["height"]) != SIZE:
        bad.append(f"canvas {p['width']}x{p['height']}, expected {SIZE[0]}x{SIZE[1]}")
    if not p["audio"]:
        bad.append("no audio stream")
    lo, hi = target_s * (1 - tol), target_s * (1 + tol)
    if not lo <= p["duration_s"] <= hi:
        notes.append(f"runtime {p['duration_s']:.0f}s outside {lo:.0f}-{hi:.0f}s "
                     f"(target {target_s:.0f}s)")
    return p, bad, notes


def _mmss(s: float) -> str:
    return f"{int(s) // 60}:{int(s) % 60:02d}"


def summary_md(vod_id, info: dict, pl: dict, out_dir: Path, check: dict,
               problems: list[str], notes: list[str] = ()) -> str:
    """summary.md: what the video is, the settings and why, the story, time and cost."""
    out_dir = Path(out_dir)
    lines = [f"# {info.get('title') or vod_id}", "",
             f"VOD {vod_id} | {info.get('game') or '?'} | "
             f"{(info.get('length') or 0) / 3600:.1f} h stream | run `{out_dir.name}`", "",
             f"**Video:** `{out_dir / OUT_NAME}` -- {_mmss(check.get('duration_s', 0))}, "
             f"{check.get('width')}x{check.get('height')}, "
             f"{'audio ok' if check.get('audio') else 'NO AUDIO'}", "",
             "**Check:** " + ("; ".join(problems) if problems else "passed")
             + ("".join(f" (note: {n})" for n in notes)), "",
             "## Settings", "", "| setting | value | why |", "|---|---|---|"]
    kw = pl["kwargs"]
    for k, why in pl["why"].items():
        val = kw.get(k, "")
        if k == "duration_s":
            val = f"{val:.0f}s"
        elif k in ("base edit", "stages"):
            val = "on" if k == "base edit" else "defaults"
        lines.append(f"| {k} | {str(val or '-')[:80]} | {why} |")
    plan_md = out_dir / "cut_plan.md"
    if plan_md.exists():
        head = plan_md.read_text(encoding="utf-8").split("\n## Beat")[0]
        lines += ["", "## Story", "", head.replace("# Cut Plan", "").strip()]
    ol = out_dir / "outline.json"
    if ol.exists():
        beats = json.loads(ol.read_text(encoding="utf-8")).get("beats") or []
        lines += ["", f"## Beats ({len(beats)})", ""]
        lines += [f"{n}. {b.get('title', '')} ({b.get('role', '')}) -- stream "
                  f"{_mmss(b.get('start') or 0)}-{_mmss(b.get('end') or 0)}"
                  for n, b in enumerate(beats, 1)]
    tm = out_dir / "timings.json"
    if tm.exists():
        t = json.loads(tm.read_text(encoding="utf-8"))
        lines += ["", f"## Time: {_mmss(t.get('total_s', 0))} (m:ss)", "",
                  ", ".join(f"{s['stage']} {_mmss(s['s'])}" for s in t.get("stages", []))]
    cost = out_dir.parent / "cost.json"
    if cost.exists():
        run = json.loads(cost.read_text(encoding="utf-8")).get("runs", {}).get(out_dir.name)
        if run:
            lines += ["", "## Claude calls", ""]
            lines += [f"- {st}: {b['calls']} call(s), {b['input_tokens'] + b['cache_read_input_tokens'] + b['cache_creation_input_tokens']:,} "
                      f"input / {b['output_tokens']:,} output tokens"
                      for st, b in (run.get("by_stage") or {}).items()]
    return "\n".join(lines) + "\n"


def run(vod_id, duration: float | None = None, brief: str | None = None,
        cache_dir="vods", run_root="run", progress=None) -> Path:
    """VOD number -> final.mp4 + summary.md + auto.json. Returns the video path."""
    from .ingest import ensure_vod
    from .pipeline import make
    # the chat download carries the metadata the plan needs; make re-uses both files
    _video, chat = ensure_vod(vod_id, cache_dir)
    info = vod_info(chat)
    pl = plan(info, run_root, duration, brief)
    title = info.get("title", "")[:70].encode("ascii", "replace").decode()   # cp1252 console
    print(f"[auto] {title!r} | {info.get('game') or '?'} | "
          f"target {pl['kwargs']['duration_s']:.0f}s ({pl['why']['duration_s']}) | "
          f"{pl['kwargs']['encoder']}")
    mp = make(vod_id, cache_dir=cache_dir, progress=progress, **pl["kwargs"])
    out_dir = Path(mp).parent.parent                 # <run>/ae/manifest.json
    mp4 = out_dir / OUT_NAME
    if mp4.exists():
        m = master_audio(mp4)
        pl["why"]["loudness"] = (f"{float(m['input_i']):.1f} -> {LUFS:.0f} LUFS, one pass "
                                 f"over the whole program")
    check, problems, notes = verify(mp4, pl["kwargs"]["duration_s"])
    (out_dir / "auto.json").write_text(json.dumps(
        {"vod": str(vod_id), "info": info, "plan": pl, "check": check,
         "problems": problems, "notes": notes}, indent=2, default=list), encoding="utf-8")
    (out_dir / "summary.md").write_text(
        summary_md(vod_id, info, pl, out_dir, check, problems, notes), encoding="utf-8")
    print(f"[auto] {mp4} -- {_mmss(check.get('duration_s', 0))}, "
          + ("CHECK FAILED: " + "; ".join(problems) if problems else "check passed")
          + "".join(f" (note: {n})" for n in notes))
    print(f"[auto] summary: {out_dir / 'summary.md'}")
    if problems:
        raise SystemExit(1)
    return mp4
