"""`refire caption`: Nguy's caption look in Premiere (docs/specs/02-caption.md).

The blocks go in as caption tracks (one TTML file per overlap layer, styled like the caption
MOGRT), and Premiere's own Upgrade Caption to Graphic turns them all into graphic clips in one
go (decision R6, spike S8). Each graphic then gets its text position and a Transform carrying
captionIn's animation, all inside the `refire_captions` nest on main V6, which carries ccStyle
(shake, shadow, glow) once. Intro/outro keep the preset's durations anchored to each block's
start/end. Python decides everything here; the .jsx (ppro/applyOps.jsx) only applies
ops_caption.json, and ppro/menu.py runs the one menu command scripting lacks. Times are
timeline seconds.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets" / "premiere"
DEFAULTS = {
    "mode": "styled", "mogrt": str(ASSETS / "captMograph.mogrt"), "mogrt_text_param": "TextLayer",
    "preset_in": str(ASSETS / "captionIn.prfpset"), "preset_style": str(ASSETS / "ccStyle.prfpset"),
    "easing": "exact", "motion_blur": True, "shutter_angle": 360, "tail": 0.0, "max_tracks": 4,
    "nest_track": 6, "chunk": 60,      # 04-sauce layout: V5 is emotes now
    # the caption MOGRT's look as TTML (S8: CoolveticaRg-Regular 55, stroke 3, centered in a
    # 1440 box). Premiere drops TTML regions, so every graphic lands at (0.05, 0.05) and gets
    # text_pos: the MOGRT's box measured on screen, its glyphs within 2 px.
    "font": "Coolvetica", "font_size": 55, "color": "#FFFFFF", "outline": "#000000 3px",
    "text_pos": [0.125, 0.9155],
}
NEST = "refire_captions"
SRC_BIN = "refire_caption_files"       # the imported TTML items, deleted once their tracks exist
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


def ttml(blocks: list[dict], tmap, cfg: dict) -> str:
    """one caption track's blocks (no overlaps) as TTML. Premiere honors font, size, color,
    outline and textAlign from a referenced style; regions, origin and displayAlign are ignored
    (S8). Times go through the timeline's frame grid, rounded UP to the ms: Premiere floors them
    to frames (0.633 for frame 38 at 60 fps landed on 37)."""
    import math
    from xml.sax.saxutils import escape, quoteattr

    from .timeline_map import TICKS

    def clock(t: float) -> str:
        ms = math.ceil(round(tmap.ticks(t) / TICKS * 1000, 6))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"
    style = " ".join(f"tts:{k}={quoteattr(str(v))}" for k, v in (
        ("fontFamily", cfg["font"]), ("fontSize", f"{cfg['font_size']}px"), ("color", cfg["color"]),
        ("textOutline", cfg["outline"]), ("textAlign", "center")))
    body = "\n".join(f'<p begin="{clock(b["t_in"])}" end="{clock(b["t_out"])}" style="s">{escape(b["text"])}</p>'
                     for b in blocks)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling" xml:lang="en">\n'
            f'<head><styling><style xml:id="s" {style}/></styling></head>\n'
            f'<body><div>\n{body}\n</div></body></tt>\n')


def ops(blocks: list[dict], tmap, cfg: dict, effect, style_effects, files: list[str], restyle: bool = False,
        anchor: str = "") -> dict:
    """nest + ccStyle, caption tracks from `files` (one per layer, bottom first), the menu's
    upgrade, then per block: text position + keyed Transform on its graphic. Upgrade puts layer
    k's graphics on the k-th of the top `layers` video tracks; `index` is the block's place on it."""
    v_nest = int(cfg["nest_track"]) - 1
    layers = max((b["track"] for b in blocks), default=0)
    out = [{"op": "ensure_nest", "nest": NEST, "v": v_nest, "anchor": anchor, "tracks": 1, "fresh": restyle},
           {"op": "style_nest", "nest": NEST, "v": v_nest, "effects": style_ops(style_effects)},
           {"op": "clear_nest", "nest": NEST}]
    if blocks:
        out += [{"op": "caption_tracks", "seq": NEST, "files": files, "bin": SRC_BIN},
                {"op": "menu", "do": "upgrade_captions", "seq": NEST},
                {"op": "after_upgrade", "seq": NEST, "layers": layers, "want": len(blocks)}]
    statics, seen = transform_statics(effect, cfg), {}
    for b in sorted(blocks, key=lambda b: b["t_in"]):
        idx = seen[b["track"]] = seen.get(b["track"], -1) + 1
        out.append({"op": "caption_fx", "id": b["id"], "seq": NEST, "layer": b["track"], "layers": layers,
                    "index": idx, "start": str(tmap.ticks(b["t_in"])), "name": f"refire:caption:{b['id']}",
                    "pos": cfg["text_pos"], "effect": "Transform", "match": TRANSFORM, "params": statics,
                    "interp": b["interp"], "keys": [[j, ks] for j, ks in sorted(b["keys"].items())]})
    return {"version": 2, "feature": "caption", "ticks_per_frame": tmap.ticks_per_frame, "ops": out}


# --------------------------------------------------------------------------- run

def run(run_dir, timeline=None, overrides: dict | None = None, words_per_line: int = 3, game: str = "",
        corrections_dir="run", dry_run: bool = False, restyle: bool = False, log=print) -> dict:
    """live timeline -> caption blocks -> one TTML per layer + ops_caption.json + caption_review.tsv."""
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
    if not mogrt._font_installed(cfg["font"]):          # premiere would quietly substitute
        log(f"[caption] warning: font {cfg['font']!r} isn't installed")
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
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for k in range(1, max((b["track"] for b in blocks), default=0) + 1):
        f = out_dir / f"layer_{k}.dfxp"
        f.write_text(ttml([b for b in blocks if b["track"] == k], tmap, cfg), encoding="utf-8")
        files.append(str(f.resolve()).replace("\\", "/"))
    for b in blocks:
        L = (tmap.ticks(b["t_out"]) - tmap.ticks(b["t_in"])) / TICKS     # what Premiere will play
        b["keys"], b["interp"], b["k"] = block_keys(L, rs, t_intro, t_outro, effect.duration, fps,
                                                    cfg["easing"])
    # the nest's end marker: an empty-text graphic (an empty nest can't be placed, S5)
    anchor = str(mogrt.bake(mogrt.load(cfg["mogrt"], cfg["mogrt_text_param"]), " ", out_dir).resolve()).replace("\\", "/")
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
            json.dumps(ops(blocks, tmap, cfg, effect, style, files, restyle, anchor), indent=1), encoding="utf-8")
    log(f"[caption] {len(blocks)} blocks on {max((b['track'] for b in blocks), default=0)} track(s), "
        f"easing {cfg['easing']}")
    return {"blocks": blocks}
