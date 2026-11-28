"""Polish events out of a finished Premiere edit (04-sauce.md section 3, steps 1-2). Read-only.

On top of learn.prproj's object graph: every top-level item of the exported sequence with its
effects and keyframes, classified by media and effect. Measured on Nguy's four edits (Aug 2026):
- boom: vine-boom.mp3 on A2/A3, the whole 1.25 s file, clip Volume about -14 dB
- sfx:<name>: any other short sound (lie-detector-meme, awkward-moment, rizz, whoosh, ...)
- emote:<name>: a still/gif, Motion centered at 240-420%, Opacity keyed 100 -> 0, drop shadow
- punch / zoom: an adjustment layer whose Transform scale ramps 100 -> 190 in ~0.47 s; anchored
  in a corner (the facecam's) = punch, anywhere else = gameplay zoom
- fx: an adjustment layer doing something else (camera shake + tint under an awkward-moment sting)
Keyframe times are media ticks; they count from the clip's InPoint.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from ..learn.prproj import TICKS, Project, pick_final

VOL_0DB = 0.17782794100389229        # premiere stores clip volume linear, +15 dB = 1.0 (-15 dB = 0.0316)
STILL = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif", ".bmp"}
SFX_MAX = 6.0                        # longer non-footage audio is music
PROXY = re.compile(r"p_[a-z0-9_]+?_(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\.(?:mov|mp4)$", re.I)
CORNER = 0.1                         # an anchor this close to a corner pins the facecam


def db(v: float) -> float:
    return round(20 * math.log10(max(v, 1e-9) / VOL_0DB), 1)


def _val(s: str):
    s = s.strip()
    if ":" in s:
        return [float(x) for x in s.split(":")[:2]]
    if s in ("true", "false"):
        return s == "true"
    try:
        return float(s)
    except ValueError:
        return s


def _keys(prm, t0: int) -> tuple[object, list]:
    """(static value, [(seconds from the clip's in point, value)])."""
    sk = (prm.findtext("StartKeyframe") or "").split(",")
    static = _val(sk[1]) if len(sk) > 1 else None
    keys = []
    for k in (prm.findtext("Keyframes") or "").split(";"):
        f = k.split(",")
        if len(f) > 1 and f[0].strip().lstrip("-").isdigit():
            keys.append((round((int(f[0]) - t0) / TICKS, 4), _val(f[1])))
    return static, keys


def effects(proj: Project, cti, t0: int) -> dict:
    """{match name: {param name: {"v": static, "k": keys}}} of one ClipTrackItem."""
    out = {}
    chain = proj.ref(cti.find("ComponentOwner/Components"))
    if chain is None:
        return out
    for c in chain.findall("ComponentChain/Components/Component"):
        comp = proj.ref(c)
        if comp is None:
            continue
        name = comp.findtext("Component/MatchName") or comp.findtext("MatchName") or comp.tag
        params = {}
        for p in comp.iter("Param"):
            prm = proj.ref(p)
            if prm is None:
                continue
            pn = prm.findtext("Name") or f"#{prm.findtext('ParameterID') or '?'}"
            v, k = _keys(prm, t0)
            params.setdefault(pn, {"v": v, "k": k})
        out.setdefault(name, params)
    return out


def items(proj: Project, uid: str) -> list[dict]:
    """every top-level item of a sequence: track, times, media, effects."""
    seq = proj.by_uid[uid]
    out = []
    for g in seq.findall("TrackGroups/TrackGroup"):
        grp = proj.ref(g.find("Second"))
        if grp is None or grp.tag not in ("VideoTrackGroup", "AudioTrackGroup"):
            continue
        kind = "V" if grp.tag == "VideoTrackGroup" else "A"
        for t in grp.findall("TrackGroup/Tracks/Track"):
            track = proj.ref(t)
            if track is None:
                continue
            tname = f"{kind}{int(t.attrib.get('Index', 0)) + 1}"
            for tie in track.findall("ClipTrack/ClipItems/TrackItems/TrackItem"):
                el = proj.ref(tie)
                cti = el.find("ClipTrackItem") if el is not None else None
                ti = cti.find("TrackItem") if cti is not None else None
                sub = proj.ref(cti.find("SubClip")) if cti is not None else None
                clip = proj.ref(sub.find("Clip")) if sub is not None else None
                if ti is None or clip is None:
                    continue
                c = clip.find("Clip")
                src = proj.ref(c.find("Source")) if c is not None else None
                media = proj.ref(src.find("MediaSource/Media")) if src is not None else None
                path = (media.findtext("FilePath") or media.findtext("ActualMediaFilePath") or "") if media is not None else ""
                t0 = int(c.findtext("InPoint") or 0) if c is not None else 0
                out.append({"track": tname, "t": int(ti.findtext("Start") or 0) / TICKS,
                            "end": int(ti.findtext("End") or 0) / TICKS, "name": sub.findtext("Name") or "",
                            "media": path, "src_in": t0 / TICKS,
                            "src_out": int(c.findtext("OutPoint") or 0) / TICKS if c is not None else 0.0,
                            "adjustment": (clip.findtext("AdjustmentLayer") or "") == "true",
                            "nested": src is not None and src.find("SequenceSource/Sequence") is not None,
                            "fx": effects(proj, cti, t0)})
    return sorted(out, key=lambda i: (i["t"], i["track"]))


REST = 101.0                         # scale at or under this = not zoomed


def segments(keys: list, static: float, dur: float) -> list[dict]:
    """a Transform's scale keys -> each zoomed stretch inside the visible [0, dur]:
    {t0, t1, ramp_in, ramp_out, first (the level the snap lands on), peak}. One layer can hold
    several (his ease in to 190, creep to 200, ease out to 100, and again). No keys = held at
    `static` for the whole layer (a hard cut in and out)."""
    if not keys:
        return [{"t0": 0.0, "t1": dur, "ramp_in": 0.0, "ramp_out": 0.0, "first": static, "peak": static}] \
            if static > REST else []
    out, run, start = [], [], 0.0
    pts = sorted(keys) + [(float("inf"), REST)]
    for t, v in pts:
        if v > REST:
            run.append((t, v))
            continue
        if run:
            a, b = max(0.0, start), min(dur, t)
            if b > a and run[-1][0] > 0 and run[0][0] < dur:
                out.append({"t0": round(a, 3), "t1": round(b, 3), "ramp_in": round(max(0.0, run[0][0] - start), 3),
                            "ramp_out": round(max(0.0, t - run[-1][0]) if t != float("inf") else 0.0, 3),
                            "first": run[0][1], "peak": max(v for _, v in run)})
            run = []
        start = t
    return out


def _transform(fx: dict) -> dict:
    return fx.get("AE.ADBE Geometry2") or fx.get("AE.ADBE Geometry") or {}


def classify(it: dict) -> list[dict]:
    """one item -> its polish events (an adjustment layer can hold several punches)."""
    if not it["adjustment"]:
        e = _classify(it)
        return [e] if e else []
    fx, dur = it["fx"], it["end"] - it["t"]
    tr = _transform(fx)
    p = tr.get("Scale Height") or tr.get("Scale") or {}
    anchor = (tr.get("Anchor Point") or {}).get("v")
    other = [k for k in fx if k not in ("AE.ADBE Geometry2", "AE.ADBE Geometry", "AE.ADBE Opacity", "AE.ADBE Motion")]
    segs = segments(p.get("k", []), float(p.get("v") or 100.0), dur)
    if other or not segs:
        return [{"t": round(it["t"], 3), "end": round(it["end"], 3), "track": it["track"], "type": "fx",
                 "asset": "+".join(k.replace("AE.", "") for k in fx) or "adjustment", "scale": p.get("v")}]
    corner = isinstance(anchor, list) and all(min(a, 1 - a) <= CORNER for a in anchor)
    return [{"t": round(it["t"] + s["t0"], 3), "end": round(it["t"] + s["t1"], 3), "track": it["track"],
             "type": "punch" if corner else "zoom", "asset": "transform", "anchor": anchor,
             "scale": s["first"], "peak": s["peak"], "ramp_in": s["ramp_in"], "ramp_out": s["ramp_out"]}
            for s in segs]


def _classify(it: dict) -> dict | None:
    """a non-adjustment item -> a polish event, or None (footage, music, captions)."""
    base = {"t": round(it["t"], 3), "end": round(it["end"], 3), "track": it["track"]}
    stem = Path(it["media"] or it["name"]).stem
    ext = Path(it["media"]).suffix.lower()
    if it["nested"] or PROXY.search(it["media"] or ""):
        return None
    if it["track"].startswith("A"):
        if it["end"] - it["t"] > SFX_MAX:
            return None
        vol = next((fx for fx in it["fx"].values() if "Level" in fx), {}).get("Level", {})
        lvl = {"db": db(vol["v"]) if isinstance(vol.get("v"), float) else None,
               "db_keys": [(t, db(v)) for t, v in vol.get("k", []) if isinstance(v, float)]}
        kind = "boom" if "vine-boom" in stem.lower() else f"sfx:{stem}"
        return {**base, "type": kind, "asset": Path(it["media"]).name, **lvl}
    if ext in STILL:
        mo = it["fx"].get("AE.ADBE Motion", {})
        op = it["fx"].get("AE.ADBE Opacity", {}).get("Opacity", {})
        return {**base, "type": f"emote:{stem}", "asset": Path(it["media"]).name, "file": it["media"],
                "scale": (mo.get("Scale") or {}).get("v") or 100.0,            # no Motion = the default 100
                "position": (mo.get("Position") or {}).get("v") or [0.5, 0.5],
                "opacity_keys": op.get("k", []), "shadow": "AE.ADBE Drop Shadow" in it["fx"]}
    return None


def mine(path) -> dict:
    """one finished project -> {project, sequence, duration, events, footage (V1 proxies)}."""
    proj = Project(path)
    uid, why = pick_final(proj)
    its = items(proj, uid)
    events = sorted((e for i in its for e in classify(i)), key=lambda e: (e["t"], e["track"]))
    footage = [{k: i[k] for k in ("t", "end", "media", "src_in", "src_out")}
               for i in its if i["track"] == "V1" and PROXY.search(i["media"] or "")]
    return {"project": str(path), "sequence": proj.seqs[uid]["name"], "why": why,
            "duration": round(max((i["end"] for i in its), default=0.0), 3),
            "events": events, "footage": footage}
