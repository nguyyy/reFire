"""`refire preview <run_dir>`: one watchable mp4 of the run's live timeline with every layer reFire placed
(punch-ins, sauce zooms / splits / emotes / booms, captions), for when Premiere isn't open.

V1 comes from timeline.tsv (else the manifest), cut straight from the intra-frame DNxHR proxies, so the
concat is frame-exact. Layers stack like the Premiere tracks: V2 zoom crop, V3 punch crop on top of that,
V4 split comp, V5 emotes, captions last; booms mixed at their level.
ponytail: approximates the Premiere look: hard-cut crops, plain white captions (not the MOGRT), foreign
media and speed changes on V1 are skipped. Premiere stays the real check.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .timeline_map import TICKS, TimelineMap

W, H = 1920, 1080
VOL_0DB = 0.17782794


def windows(run_dir: Path) -> list[tuple[float, float, float, float, float, float]]:
    """(t0, t1, x, y, w, h) crops in frame px: sauce zooms first (V2), then punches (V3)."""
    out = []
    sp = run_dir / "sauce.json"
    for e in json.loads(sp.read_text(encoding="utf-8"))["events"] if sp.exists() else []:
        if e["type"] == "zoom" and e.get("anchor"):
            w, h = W / e["scale"], H / e["scale"]
            x = min(max(e["anchor"][0] * W - w / 2, 0), W - w)
            y = min(max(e["anchor"][1] * H - h / 2, 0), H - h)
            out.append((e["t"], e["end"], x, y, w, h))
    pp = run_dir / "punch.json"
    for p in json.loads(pp.read_text(encoding="utf-8"))["punches"] if pp.exists() else []:
        w, h = W / p["scale"], H / p["scale"]                     # scaled about the pinned corner
        out.append((p["t_in"], p["t_out"], p["pin"][0] * (W - w), p["pin"][1] * (H - h), w, h))
    return out


def srt(run_dir: Path, out: Path) -> Path | None:
    """caption blocks (caption_review.tsv) -> plain srt."""
    tsv = run_dir / "caption_review.tsv"
    if not tsv.exists():
        return None

    def ts(t):
        ms = round(t * 1000)
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    rows = [r.split("\t") for r in tsv.read_text(encoding="utf-8").splitlines()[1:] if r.strip()]
    out.write_text("".join(f"{i}\n{ts(float(r[1]))} --> {ts(float(r[2]))}\n{r[3]}\n\n" for i, r in enumerate(rows, 1)),
                   encoding="utf-8")
    return out


def graph(wins, overlays: list[tuple[float, float]], booms: list[tuple[float, float]], subs: str | None,
          fps: float) -> str:
    """the filter_complex: input 0 = V1 concat, 1..n = overlay movs, n+1 = the boom file (if any)."""
    f = [f"[0:v]scale={W}:{H},setsar=1,fps={fps:g}[v0]"]
    v = "v0"
    for i, (a, b, x, y, w, h) in enumerate(wins):
        f.append(f"[{v}]split[{v}a][{v}b];[{v}b]trim={a:.3f}:{b:.3f},crop={w:.0f}:{h:.0f}:{x:.0f}:{y:.0f},"
                 f"scale={W}:{H},setsar=1[z{i}];[{v}a][z{i}]overlay=0:0:eof_action=pass[w{i}]")   # trim: crop only its window
        v = f"w{i}"
    for i, (a, b) in enumerate(overlays, 1):
        f.append(f"[{i}:v]setpts=PTS-STARTPTS+{a:.3f}/TB[o{i}];[{v}][o{i}]overlay=0:0:eof_action=pass[x{i}]")
        v = f"x{i}"
    if subs:
        f.append(f"[{v}]subtitles={subs}:force_style='FontName=Arial,Fontsize=22,Bold=1,Outline=2,MarginV=40'[vs]")
        v = "vs"
    f.append(f"[{v}]null[vout]")
    if booms:
        n = len(overlays) + 1
        f.append(f"[{n}:a]asplit={len(booms)}" + "".join(f"[s{i}]" for i in range(len(booms))))
        for i, (t, lvl) in enumerate(booms):
            f.append(f"[s{i}]adelay={round(t * 1000)}:all=1,volume={lvl / VOL_0DB:.4f}[b{i}]")
        f.append("[0:a]" + "".join(f"[b{i}]" for i in range(len(booms)))
                 + f"amix=inputs={len(booms) + 1}:normalize=0:duration=first[aout]")
    else:
        f.append("[0:a]anull[aout]")
    return ";".join(f)


def render(run_dir, out: Path | None = None, encoder: str | None = None, log=print) -> Path:
    from .auto import pick_encoder
    run_dir = Path(run_dir)
    mp = run_dir / "ae" / "manifest.json"
    tmap = TimelineMap.load(mp) if (mp.parent / "timeline.tsv").exists() else TimelineMap.from_manifest(mp)
    fps = TICKS / tmap.ticks_per_frame if tmap.ticks_per_frame else 60.0
    work = run_dir / "preview"
    work.mkdir(exist_ok=True)
    lst = work / "v1.txt"
    lst.write_text("".join(f"file '{Path(s.media).as_posix()}'\ninpoint {s.src_in - s.offset:.6f}\n"
                           f"outpoint {s.src_out - s.offset:.6f}\n" for s in tmap.ours()), encoding="utf-8")
    sp = run_dir / "sauce.json"
    evs = json.loads(sp.read_text(encoding="utf-8"))["events"] if sp.exists() else []
    movs = sorted([e for e in evs if e["type"] in ("split", "emote") and e.get("file")],
                  key=lambda e: (e["type"] == "emote", e["t"]))     # V4 splits under V5 emotes
    booms = [(e["t"], e.get("level", 0.035481)) for e in evs if e["type"] == "boom"]
    boom_file = json.loads((run_dir / "ppro" / "ops_sauce.json").read_text(encoding="utf-8"))
    boom_file = next((o["file"] for o in boom_file["ops"] if o.get("audio") is True), None) if booms else None
    subs = srt(run_dir, work / "captions.srt")
    fc = graph(windows(run_dir), [(e["t"], e["end"]) for e in movs], booms if boom_file else [],
               subs and "captions.srt", fps)
    (work / "graph.txt").write_text(fc, encoding="utf-8")
    out = Path(out or run_dir / "preview.mp4").resolve()
    enc = encoder or pick_encoder()
    cmd = ["ffmpeg", "-v", "error", "-stats", "-y", "-f", "concat", "-safe", "0", "-i", "v1.txt"]
    for e in movs:
        cmd += ["-i", str(Path(e["file"]).resolve())]
    if booms and boom_file:
        cmd += ["-i", boom_file]
    cmd += ["-filter_complex_script", "graph.txt", "-map", "[vout]", "-map", "[aout]", "-c:v", enc,
            *(["-preset", "p4", "-cq", "23"] if enc == "h264_nvenc" else ["-preset", "veryfast", "-crf", "21"]),
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(out)]
    log(f"[preview] {len(tmap.ours())} V1 clips, {len(windows(run_dir))} crops, {len(movs)} overlays, "
        f"{len(booms)} booms, captions {'on' if subs else 'off'} -> {out}")
    subprocess.run(cmd, cwd=work, check=True)         # cwd: the subtitles filter can't take a drive-letter path
    return out
