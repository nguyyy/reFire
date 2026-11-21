"""`refire caption`: Nguy's caption look in Premiere (docs/specs/02-caption.md).

Every caption block is its own copy of the caption MOGRT with the text baked in (R5), with
a Transform carrying captionIn's animation, all inside the `refire_captions` nest on main
V5, which carries ccStyle (shake, shadow, glow) once. Intro/outro keep the preset's
durations anchored to each block's start/end. Python decides everything here; the .jsx
(ppro/applyOps.jsx) only applies ops_caption.json. Times are timeline seconds.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets" / "premiere"
DEFAULTS = {
    "mode": "styled", "mogrt": str(ASSETS / "captMograph.mogrt"), "mogrt_text_param": "TextLayer",
    "preset_in": str(ASSETS / "captionIn.prfpset"), "preset_style": str(ASSETS / "ccStyle.prfpset"),
    "easing": "exact", "motion_blur": True, "shutter_angle": 360, "tail": 0.0, "max_tracks": 4,
    "nest_track": 5, "chunk": 50,
}
NEST = "refire_captions"
TRANSFORM = "AE.ADBE Geometry"      # what QE's "Transform" adds; same params as the preset's Geometry2 (S7)
QE_NAMES = {"AE.ADBE Geometry2": "Transform", "AE.ADBE Geometry": "Transform",
            "AE.Impact_Camera_Shake_FX": "Camera Shake", "AE.ADBE Drop Shadow": "Drop Shadow",
            "AE.Impact_Edge_Glow_FX": "Edge Glow"}
# exact easing: drop samples a straight line already reproduces within this (value units)
TOLERANCE = {"point": 0.5 / 1080, "scalar": 1.0}       # half a pixel, 1% opacity


def config(overrides: dict | None = None) -> dict:
    return {**DEFAULTS, **(overrides or {})}


# --------------------------------------------------------------------------- layering

def layer(blocks: list[dict], max_tracks: int) -> list[dict]:
    """step 4: a block overlapping the previous one goes one track up (newer on top); past
    max_tracks the previous block is trimmed to end where this one starts and this one takes
    its track. 1-based tracks. A block trimmed to nothing is dropped."""
    out: list[dict] = []
    for b in sorted(blocks, key=lambda b: b["t_in"]):
        b = dict(b)
        if out and b["t_in"] < out[-1]["t_out"]:
            tr = out[-1]["track"] + 1
            if tr > max_tracks:
                out[-1]["t_out"] = b["t_in"]
                tr = out[-1]["track"]
        else:
            tr = 1
        for o in out:                       # an older block still running on that track yields
            if o["track"] == tr and o["t_out"] > b["t_in"]:
                o["t_out"] = b["t_in"]
        b["track"] = tr
        out.append(b)
    return [b for b in out if b["t_out"] - b["t_in"] > 1e-3]


# --------------------------------------------------------------------------- timing

@dataclass
class Ramp:
    index: int                  # Transform properties[j]
    point: bool
    intro: list                 # prfpset.Key, preset-relative times
    outro: list


def _same(a, b) -> bool:
    if isinstance(a, list):
        return all(abs(x - y) < 1e-9 for x, y in zip(a, b))
    return abs(a - b) < 1e-9


def ramps(effect) -> tuple[list[Ramp], float, float]:
    """split each animated param at its plateaus: keys up to the first plateau's start are the
    intro (anchored to block start), keys from the last plateau's end are the outro (anchored
    to block end). No plateau -> all intro (Position). -> (ramps, t_intro, t_outro)."""
    out, t_in, t_out = [], 0.0, 0.0
    for p in effect.params:
        ks = p.keys
        if len(ks) < 2 or p.skip:
            continue
        flat = [i for i in range(len(ks) - 1) if _same(ks[i].value, ks[i + 1].value)]
        if flat:
            intro, outro = ks[:flat[0] + 1], ks[flat[-1] + 1:]
        else:
            intro, outro = ks, []
        out.append(Ramp(p.index, isinstance(ks[0].value, list), intro, outro))
        t_in = max(t_in, intro[-1].t)
        if outro:
            t_out = max(t_out, effect.duration - outro[0].t)
    return out, t_in, t_out


def _bezier(a, b):
    """value(t) between keys a -> b (preset time), after effects style ease: speed + influence
    on each side give a cubic bezier in (time, value). points move along the straight line."""
    d = b.t - a.t
    if a.interp == 4 or d <= 0:                                # hold
        return lambda t: a.value
    if a.interp == 0:                                          # linear
        return lambda t: _lerp(a.value, b.value, (t - a.t) / d)
    s_out, i_out = float(a.tangents[3]), float(a.tangents[4])
    s_in, i_in = float(b.tangents[1]), float(b.tangents[2])
    if isinstance(a.value, list):
        span = sum((y - x) ** 2 for x, y in zip(a.value, b.value)) ** 0.5
        y0, y3 = 0.0, span
    else:
        span, y0, y3 = None, a.value, b.value
    x = (a.t, a.t + i_out * d, b.t - i_in * d, b.t)
    y = (y0, y0 + s_out * i_out * d, y3 - s_in * i_in * d, y3)

    def cub(c, s):
        return (1 - s) ** 3 * c[0] + 3 * (1 - s) ** 2 * s * c[1] + 3 * (1 - s) * s * s * c[2] + s ** 3 * c[3]

    def at(t):
        lo, hi = 0.0, 1.0
        for _ in range(40):                                    # time is monotonic in s
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if cub(x, mid) < t else (lo, mid)
        v = cub(y, (lo + hi) / 2)
        if span is None:
            return v
        return _lerp(a.value, b.value, v / span if span else 1.0)
    return at


def _lerp(a, b, f):
    if isinstance(a, list):
        return [x + (y - x) * f for x, y in zip(a, b)]
    return a + (b - a) * f


def curve(keys):
    """value at any preset time across a run of keys (held outside)."""
    segs = [(a.t, b.t, _bezier(a, b)) for a, b in zip(keys, keys[1:])]

    def at(t):
        if t <= keys[0].t:
            return keys[0].value
        for t0, t1, f in segs:
            if t <= t1:
                return f(t)
        return keys[-1].value
    return at


def _simplify(pts: list[tuple[float, object]], tol: float) -> list[tuple[float, object]]:
    """ramer-douglas-peucker in (t, value) against linear interpolation; ends kept."""
    if len(pts) < 3:
        return pts

    def err(i, a, b):
        (ta, va), (tb, vb), (t, v) = pts[a], pts[b], pts[i]
        lin = _lerp(va, vb, (t - ta) / (tb - ta))
        return max(abs(p - q) for p, q in zip(v, lin)) if isinstance(v, list) else abs(v - lin)

    keep, stack = {0, len(pts) - 1}, [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        i = max(range(a + 1, b), key=lambda i: err(i, a, b))
        if err(i, a, b) > tol:
            keep.add(i)
            stack += [(a, i), (i, b)]
    return [pts[i] for i in sorted(keep)]


def block_keys(L: float, rs: list[Ramp], t_intro: float, t_outro: float, D: float, fps: float,
               easing: str = "exact") -> tuple[dict, int, float]:
    """keys for a block of length L: {param index: [(t from block start, value)]}, interp, k.
    k < 1 = a short block whose intro + outro were squeezed to fit (spec: log it)."""
    k = min(1.0, L / (t_intro + t_outro)) if t_intro + t_outro > 0 else 1.0
    out = {}
    for r in rs:
        pts = []
        for keys, to_block in ((r.intro, lambda t: t * k), (r.outro, lambda t: L - (D - t) * k)):
            if not keys:
                continue
            if easing != "exact":
                pts += [(to_block(x.t), x.value) for x in keys]
                continue
            f, a, b = curve(keys), to_block(keys[0].t), to_block(keys[-1].t)
            ts = [a] + [n / fps for n in range(int(a * fps) + 1, int(b * fps - 1e-9) + 1) if a < n / fps < b] + [b]
            from_block = (lambda t: t / k) if keys is r.intro else (lambda t: D - (L - t) / k)
            seg = [(t, f(from_block(t))) for t in ts]
            pts += _simplify(seg, TOLERANCE["point" if r.point else "scalar"])
        out[r.index] = [(round(t, 6), [round(v, 6) for v in val] if isinstance(val, list) else round(val, 4))
                        for t, val in pts]
    return out, (0 if easing == "exact" else 5), round(k, 4)


# --------------------------------------------------------------------------- ops

def style_ops(style_effects) -> list[dict]:
    """ccStyle's effects for the nest clip, every named param, in the preset's order."""
    out = []
    for fx in style_effects:
        params = []
        for p in fx.params:
            if p.skip or p.value is None or p.keys:
                continue
            params.append([p.index, {"color": p.value} if p.control_type == 5 else p.value])
        out.append({"effect": QE_NAMES.get(fx.match_name, fx.display_name), "match": fx.match_name,
                    "params": params})
    return out


def transform_statics(effect, cfg: dict) -> list:
    """captionIn's static Transform params, with motion blur forced on (shutter flag off)."""
    out = []
    for p in effect.params:
        if p.keys or p.skip or p.value is None:
            continue
        v = p.value
        if p.param_id == 9:                 # use composition's shutter angle
            v = not cfg["motion_blur"]
        elif p.param_id == 10:              # shutter angle
            v = cfg["shutter_angle"]
        out.append([p.index, v])
    return out


def ops(blocks: list[dict], tmap, cfg: dict, effect, style_effects, restyle: bool = False,
        anchor: str = "") -> dict:
    v_nest = int(cfg["nest_track"]) - 1
    out = [{"op": "ensure_nest", "nest": NEST, "v": v_nest, "anchor": anchor,
            "tracks": int(cfg["max_tracks"]) + 1, "fresh": restyle},
           {"op": "style_nest", "nest": NEST, "v": v_nest, "effects": style_ops(style_effects)},
           {"op": "clear_nest", "nest": NEST, "tracks": int(cfg["max_tracks"])}]
    statics = transform_statics(effect, cfg)
    for b in blocks:
        a, e = str(tmap.ticks(b["t_in"])), str(tmap.ticks(b["t_out"]))
        out += [{"op": "insert_caption", "id": b["id"], "seq": NEST, "v": b["track"] - 1, "mogrt": b["mogrt"],
                 "start": a, "end": e, "name": f"refire:caption:{b['id']}"},
                {"op": "add_effect", "seq": NEST, "v": b["track"] - 1, "start": a, "effect": "Transform",
                 "match": TRANSFORM, "params": statics, "interp": b["interp"],
                 "keys": [[j, ks] for j, ks in sorted(b["keys"].items())]}]
    return {"version": 1, "feature": "caption", "ticks_per_frame": tmap.ticks_per_frame, "ops": out}


# --------------------------------------------------------------------------- run

def run(run_dir, timeline=None, overrides: dict | None = None, words_per_line: int = 3, game: str = "",
        corrections_dir="run", dry_run: bool = False, restyle: bool = False, log=print) -> dict:
    """live timeline -> caption blocks -> baked MOGRTs + ops_caption.json + caption_review.tsv."""
    import csv
    import json

    from .ppro import mogrt, prfpset
    from .srt import timeline_cues
    from .timeline_map import TICKS, TimelineMap

    run_dir = Path(run_dir)
    mp = run_dir / "ae" / "manifest.json"
    cfg = config(overrides)
    tmap = TimelineMap.load(mp, timeline)
    fps = TICKS / tmap.ticks_per_frame
    tpl = mogrt.load(cfg["mogrt"], cfg["mogrt_text_param"])
    for w in mogrt.lint(tpl, seq_fps=fps):          # raises on a MOGRT with baked animation
        log(f"[caption] warning: {w}")
    (effect,) = [e for e in prfpset.load(cfg["preset_in"]) if "Geometry" in e.match_name]
    style = prfpset.load(cfg["preset_style"])
    rs, t_intro, t_outro = ramps(effect)

    cues = timeline_cues(mp, timeline, 0.0, words_per_line)
    raw = [c[2] for c in cues]
    fixed = raw
    if game:                                        # learned proper-noun fixes, free (no llm)
        from .caption_fix import fix_lines
        fixed, _ = fix_lines(raw, llm=False, game=game, corrections_dir=corrections_dir)
    blocks = [{"id": f"b{i + 1:03d}", "t_in": a, "t_out": b + cfg["tail"], "text": t, "raw": r}
              for i, ((a, b, _), t, r) in enumerate(zip(cues, fixed, raw))]
    blocks = layer(blocks, int(cfg["max_tracks"]))

    out_dir = run_dir / "ppro" / "captions"
    for b in blocks:
        b["mogrt"] = str(mogrt.bake(tpl, b["text"], out_dir).resolve()).replace("\\", "/")
        L = (tmap.ticks(b["t_out"]) - tmap.ticks(b["t_in"])) / TICKS     # what Premiere will play
        b["keys"], b["interp"], b["k"] = block_keys(L, rs, t_intro, t_outro, effect.duration, fps,
                                                    cfg["easing"])
    anchor = str(mogrt.bake(tpl, " ", out_dir).resolve()).replace("\\", "/")
    scaled = sum(b["k"] < 1 for b in blocks)
    if scaled:
        log(f"[caption] {scaled} short block(s) got a squeezed intro/outro (anim_scaled in the review)")

    with open(run_dir / "caption_review.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "t_in", "t_out", "text", "track", "anim_scaled", "corrected_from"])
        for b in blocks:
            w.writerow([b["id"], round(b["t_in"], 3), round(b["t_out"], 3), b["text"], b["track"],
                        b["k"] if b["k"] < 1 else "", b["raw"] if b["raw"] != b["text"] else ""])
    if not dry_run:
        (run_dir / "ppro").mkdir(exist_ok=True)
        (run_dir / "ppro" / "ops_caption.json").write_text(
            json.dumps(ops(blocks, tmap, cfg, effect, style, restyle, anchor), indent=1), encoding="utf-8")
    log(f"[caption] {len(blocks)} blocks on {max((b['track'] for b in blocks), default=0)} track(s), "
        f"easing {cfg['easing']}")
    return {"blocks": blocks}
