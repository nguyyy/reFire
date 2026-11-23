"""House-style hook text, rendered on its own to a transparent PNG and placed on the thumbnail.

Matched to Nguy's thumbnails (refire/thumb/refs): Conneqt Black caps, tight tracking, every
line the same font size and centered on the others, the block anchored to a bottom corner.
Fill = metallic shine running ACROSS each line (base -> light -> base, left to right), white
or a palette accent on the accent phrase. Heavy black stroke painted under the fill; one
continuous black slab behind the lines (a rect per line at most of the cap height, each
reaching down to the next) so the outlines read as one shape; hard drop + soft dark haze.
Glyph x positions are computed here from the font's advances (fontTools) and written per
character, so the browser draws exactly what was measured (no kerning, no css spacing).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from html import escape
from pathlib import Path

_USER_FONTS = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts"
_SYS_FONTS = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
DEFAULT_FONT = _USER_FONTS / "Conneqt-Black.ttf"
# the refs use two faces: Conneqt Black (wide, geometric) and Coolvetica (rounded, condensed:
# STRONGBOX..., DAY TWO..., the "(birthday)" aside). a hook or line picks one by name or path
def _installed(name: str) -> Path:
    """a font file from the user's or the system font folder (user wins)."""
    return next((d / name for d in (_USER_FONTS, _SYS_FONTS) if (d / name).exists()), _USER_FONTS / name)


FONTS = {"conneqt": DEFAULT_FONT, "coolvetica": _installed("coolvetica rg.otf")}


def font_path(name_or_path) -> Path:
    return Path(FONTS.get(str(name_or_path).lower(), name_or_path)) if name_or_path else DEFAULT_FONT
WHITE = ("#BDBDBD", "#FFFFFF")                      # metallic white: base, light
PALETTE = {"blue": ("#2C66F3", "#64A1FF"), "sky": ("#0884F2", "#58D3F3"), "green": ("#26CE27", "#52F753"),
           "mint": ("#26C075", "#53DA8E"), "gold": ("#D2A51B", "#F8CE4F"), "cyan": ("#1FFFF8", "#65FFFB")}


@dataclass
class Style:
    """every knob of the hook look; spec.json `text` keys override these."""
    width_frac: float = 0.62        # widest line's stroked width / canvas width
    max_size_frac: float = 0.24     # font size cap / canvas height (refs: caps ~0.16 H)
    cap_frac: float | None = None   # target cap height / canvas height (refs 0.125-0.163); the widest
                                    # line still may not pass width_frac. None = fill width_frac
    margin: tuple = (12, 25)        # x, y px from the anchored corner (refs: letters ~27 px in, bottom ~688)
    tracking: float = -0.035        # em added between glyphs (negative = tighter)
    squeeze: float = 0.95           # width of every line (measured on ARTIFACTS & STALLING: 0.94-0.95)
    font: str = "conneqt"           # face for every line (a line can override)
    end_inset: float = 0.28         # slab starts/stops this far into the end glyphs (hidden behind them)
    stroke: float = 0.14            # stroke width, em (half of it shows outside the glyph)
    leading: float = 1.09           # baseline gap / cap height (refs measure 1.07-1.12)
    rect_inset: float = 2.5         # slab = cap height minus this many px top and bottom (at 720p)
    shine_peak: float = 0.5         # where the shine is lightest across the line (refs: 0.45-0.55)
    shine_gain: float = 0.12        # accent: peak lightness above the palette's light color (HLS L)
    shine_edge: float = 0.06        # accent: edges this much darker than the palette's base
    white: tuple = (0.60, 0.90)     # metallic white: edge, peak lightness (refs 0.53-0.73 / 0.88-0.91)
    skew: float = 0.0               # deg, negative leans right (italic look)
    keystone: float = 1.0           # far-end height / anchored-end height, width kept (photoshop's
                                    # perspective transform). refs: strongbox 0.87, abyss 0.97, tier list 1.18
    shadow: tuple = (7, 9, 0.85)    # hard drop: dx, dy, alpha
    haze: tuple = (30, 1.0)         # soft dark glow: blur px, alpha (halo measured 0.59/0.81/0.93 vs refs 0.55/0.82/0.94)
    ann_size: float = 0.30          # annotation size / line size
    palette: dict = field(default_factory=dict)

    @classmethod
    def from_spec(cls, d: dict | None) -> "Style":
        d = dict(d or {})
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        for k in ("margin", "white", "shadow", "haze"):
            if k in known:
                known[k] = tuple(known[k])
        return cls(**known)


@dataclass(frozen=True)
class Metrics:
    upm: int
    cap: int
    advances: dict
    fallback: int

    def advances_px(self, text: str, size: float, tracking: float) -> list[float]:
        """advance of each glyph incl. tracking (none after the last)."""
        out = [self.advances.get(c, self.fallback) * size / self.upm for c in text]
        return [a + (tracking * size if i < len(out) - 1 else 0.0) for i, a in enumerate(out)]

    def width(self, text: str, size: float, tracking: float = 0.0) -> float:
        return sum(self.advances_px(text, size, tracking))


@lru_cache(maxsize=4)
def metrics(font_path: str) -> Metrics:
    from fontTools.ttLib import TTFont
    f = TTFont(font_path, lazy=True)
    hmtx, cmap = f["hmtx"], f.getBestCmap()
    adv = {chr(cp): hmtx[g][0] for cp, g in cmap.items() if cp < 0x2100}
    cap = getattr(f["OS/2"], "sCapHeight", 0) or int(f["head"].unitsPerEm * 0.7)
    return Metrics(f["head"].unitsPerEm, cap, adv, adv.get("M", f["head"].unitsPerEm // 2))


@dataclass
class Line:
    text: str
    size: float
    x: float                # left of the first glyph
    baseline: float
    width: float            # drawn width (after scale_x)
    sx: float = 1.0         # horizontal squeeze, like a free-transform in photoshop (stroke unscaled)
    font: str = "conneqt"

    def cap(self, m: Metrics) -> float:
        return m.cap * self.size / m.upm


# straight quotes a display font may lack -> the typographic glyph it does have (Conneqt has
# no U+0027: measured at the fallback width while the browser drew a narrow substitute = a gap)
TYPOGRAPHIC = {"'": "’", '"': "”", "`": "‘"}


def typeset(text: str, m: Metrics) -> str:
    return "".join(TYPOGRAPHIC[c] if c not in m.advances and TYPOGRAPHIC.get(c) in m.advances else c for c in text)


def _line(t, default_font: str) -> tuple[str, float, str, float]:
    """a hook line is "TEXT" or {"text", "scale_x" (squeeze), "font", "scale" (size vs the others)}."""
    if isinstance(t, str):
        return t, 1.0, default_font, 1.0
    return t["text"], float(t.get("scale_x", 1.0)), t.get("font", default_font), float(t.get("scale", 1.0))


def layout(lines: list, m, canvas: tuple[int, int], st: Style, anchor: str = "bottom-left") -> list[Line]:
    """one font size for every line (the widest drawn line fills width_frac, capped), each line
    optionally squeezed by scale_x and set in its own face, lines centered on each other,
    stacked so the strokes touch, block in a bottom corner. `m` is unused (kept for callers);
    metrics come from each line's font."""
    W, H = canvas
    items = [(t, sx * st.squeeze, f, k) for t, sx, f, k in (_line(t, st.font) for t in lines)]
    mets = [metrics(str(font_path(f))) for _, _, f, _ in items]
    items = [(typeset(t, mm), sx, f, k) for (t, sx, f, k), mm in zip(items, mets)]
    unit = max((mm.width(t, k, st.tracking) * sx + st.stroke * k) for (t, sx, _, k), mm in zip(items, mets))
    size = min(st.max_size_frac * H / max(k for *_, k in items), st.width_frac * W / unit if unit else H)
    if st.cap_frac:
        size = min(size, st.cap_frac * H * mets[0].upm / mets[0].cap / max(k for *_, k in items))
    widths = [mm.width(t, size * k, st.tracking) * sx for (t, sx, _, k), mm in zip(items, mets)]
    block_w = max(w + st.stroke * size * k for w, (*_, k) in zip(widths, items))
    left = st.margin[0] if anchor.endswith("left") else W - st.margin[0] - block_w
    out, base, below_cap = [], None, 0.0
    for (t, sx, f, k), w, mm in reversed(list(zip(items, widths, mets))):
        s_ = size * k
        # bottom-up: a line's baseline sits one (leading x) cap of the line BELOW above that line's
        base = H - st.margin[1] - st.stroke * s_ / 2 if base is None else base - below_cap * st.leading
        below_cap = mm.cap * s_ / mm.upm
        x = left + (block_w - w) / 2
        out.append(Line(t, round(s_, 2), round(x, 2), round(base, 2), round(w, 2), sx, f))
    return list(reversed(out))


def _split_accent(text: str, accent: str | None) -> list[tuple[str, bool]]:
    if not accent:
        return [(text, False)]
    i = text.upper().find(accent.upper())
    if i < 0:
        return [(text, False)]
    parts = [(text[:i], False), (text[i:i + len(accent)], True), (text[i + len(accent):], False)]
    return [p for p in parts if p[0]]


def shine_stops(acc: tuple | None, st: Style, n: int = 9) -> list[tuple[float, str]]:
    """the metallic shine across a line, measured off the refs: lightness ramps roughly
    linearly from dark edges to a light peak near the middle (ARTIFACTS: 0.40 -> 0.83 -> 0.45,
    & STALLING: 0.53 -> 0.88 -> 0.63). acc = (base, light) hex, None = metallic white."""
    import colorsys

    def hls(hx):
        return colorsys.rgb_to_hls(*(int(hx[i:i + 2], 16) / 255 for i in (1, 3, 5)))
    if acc is None:
        h = s_ = 0.0
        lo, hi = st.white
    else:
        (h, lb, s_), (_, ll, _) = hls(acc[0]), hls(acc[1])
        lo, hi = max(0.05, lb - st.shine_edge), min(0.92, max(ll, lb) + st.shine_gain)
    out = []
    for k in range(n):
        x = k / (n - 1)
        tri = 1 - abs(x - st.shine_peak) / max(st.shine_peak, 1 - st.shine_peak)
        r, g, b = colorsys.hls_to_rgb(h, lo + (hi - lo) * tri, s_)
        out.append((x, "#%02X%02X%02X" % (round(r * 255), round(g * 255), round(b * 255))))
    return out


def families(lines: list[Line]) -> dict:
    """font name -> css family (F0, F1, ...) in line order; svg() and page() agree on it."""
    out = {}
    for ln in lines:
        out.setdefault(ln.font, f"F{len(out)}")
    return out


def svg(lines: list[Line], m: Metrics, canvas: tuple[int, int], st: Style, accent_phrase: str | None = None,
        accent: str = "gold", annotation: str | None = None, family: str = "HookFont") -> str:
    """the hook as one canvas-sized <svg> (transparent)."""
    W, H = canvas
    acc = {**PALETTE, **st.palette}.get(accent, PALETTE["gold"])
    defs, strokes, fills = [], [], []
    boxes = []
    fams = families(lines)
    for i, ln in enumerate(lines):
        m = metrics(str(font_path(ln.font)))
        cap, sw = ln.cap(m), st.stroke * ln.size
        rh = max(cap * 0.5, cap - 2 * st.rect_inset * H / 720)
        adv0 = m.advances_px(ln.text, ln.size, st.tracking)
        a = st.end_inset * adv0[0] * ln.sx                     # tuck the slab ends behind the end glyphs
        # trailing dots/commas sit below the slab's band: end it inside the last tall glyph instead
        body = ln.text.rstrip(".,…")
        tail = len(ln.text) - len(body)
        last = body[-1] if body else ln.text[-1]
        z = (sum(adv0[len(ln.text) - tail:]) + (st.tracking * ln.size if tail else 0)) * ln.sx             + st.end_inset * m.advances.get(last, m.fallback) * ln.size / m.upm * ln.sx
        boxes.append([ln.x + a, ln.baseline - cap / 2 - rh / 2, ln.width - a - z, rh])
        nat = ln.width / ln.sx                      # gradient lives in the unsqueezed space
        for name, stops in (("w", shine_stops(None, st)), ("a", shine_stops(acc, st))):
            stops = "".join(f'<stop offset="{o:.3f}" stop-color="{c}"/>' for o, c in stops)
            defs.append(f'<linearGradient id="g{name}{i}" gradientUnits="userSpaceOnUse" x1="{ln.x:.2f}" y1="0" '
                        f'x2="{ln.x + nat:.2f}" y2="0">{stops}</linearGradient>')
        spans, k = [], 0
        adv = m.advances_px(ln.text, ln.size, st.tracking)
        pos = [ln.x + sum(adv[:j]) for j in range(len(ln.text))]
        for chunk, is_acc in _split_accent(ln.text, accent_phrase):
            px = " ".join(f"{p:.2f}" for p in pos[k:k + len(chunk)])
            spans.append((px, f"url(#g{'a' if is_acc else 'w'}{i})", escape(chunk)))
            k += len(chunk)
        sq = (f' transform="translate({ln.x:.2f} 0) scale({ln.sx} 1) translate({-ln.x:.2f} 0)"'
              if ln.sx != 1 else "")
        head = f'<text y="{ln.baseline:.2f}" font-family="{fams[ln.font]}" font-size="{ln.size:.2f}"{sq}'
        # two copies of every line: the outline goes in the stroke layer, the gradient fill in the
        # fill layer above it, so no line's stroke can ever cover another line's (or its own) fill
        strokes.append(f'{head} stroke-width="{sw:.2f}" vector-effect="non-scaling-stroke">'
                       + "".join(f'<tspan x="{px}">{t}</tspan>' for px, _, t in spans) + "</text>")
        fills.append(f'{head}>' + "".join(f'<tspan x="{px}" fill="{f}">{t}</tspan>' for px, f, t in spans) + "</text>")
    for i in range(len(boxes) - 1):                 # each slab reaches down to the next line's
        boxes[i][3] = boxes[i + 1][1] - boxes[i][1]
    rects = [f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="#000"/>' for x, y, w, h in boxes]
    if annotation and len(lines) > 1:
        ln = lines[0]
        s = ln.size * st.ann_size
        m0 = metrics(str(font_path(ln.font)))
        head = (f'<text class="ann" x="{ln.x + ln.width * 0.30:.2f}" y="{ln.baseline - ln.cap(m0) - s * 0.15:.2f}" '
                f'font-size="{s:.2f}"')
        strokes.append(f'{head} stroke-width="{s * 0.18:.2f}">{escape(annotation)}</text>')
        fills.append(f'{head} fill="#fff">{escape(annotation)}</text>')
    # skew about the block's bottom-left so the anchor corner stays put
    bx, by = min(b[0] for b in boxes), max(ln.baseline for ln in lines)
    tf = f' transform="translate({bx:.2f} {by:.2f}) skewX({st.skew}) translate({-bx:.2f} {-by:.2f})"' if st.skew else ""
    dx, dy, a = st.shadow
    blur, ha = st.haze
    filt = (f'<filter id="sh" x="-20%" y="-40%" width="140%" height="180%">'
            f'<feGaussianBlur in="SourceAlpha" stdDeviation="{blur / 2:.1f}" result="b"/>'
            f'<feFlood flood-color="#000" flood-opacity="{ha}"/><feComposite in2="b" operator="in" result="haze"/>'
            f'<feOffset in="SourceAlpha" dx="{dx}" dy="{dy}" result="o"/>'
            f'<feFlood flood-color="#000" flood-opacity="{a}"/><feComposite in2="o" operator="in" result="drop"/>'
            f'<feMerge><feMergeNode in="haze"/><feMergeNode in="drop"/><feMergeNode in="SourceGraphic"/></feMerge>'
            f'</filter>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
            f'<defs>{filt}{"".join(defs)}</defs><g filter="url(#sh)"{tf}>{"".join(rects)}'
            f'<g class="strokes" fill="#000" stroke="#000" stroke-linejoin="round">{"".join(strokes)}</g>'
            f'<g class="fills" stroke="none">{"".join(fills)}</g></g></svg>')


PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
{faces}
html, body {{ margin: 0; background: transparent; }}
text {{ font-kerning: none; text-rendering: geometricPrecision; }}
.ann {{ font-family: AnnFont, "Segoe UI", sans-serif; }}
</style></head><body>{svg}</body></html>"""


def _face(family: str, path: Path) -> str:
    import base64
    fmt = "opentype" if path.suffix.lower() == ".otf" else "truetype"
    mime = "font/otf" if fmt == "opentype" else "font/ttf"
    return (f'@font-face {{ font-family: {family}; src: url(data:{mime};base64,'
            f'{base64.b64encode(path.read_bytes()).decode()}) format("{fmt}"); }}')


def page(hook: dict, canvas: tuple[int, int], font: Path | None = None) -> str:
    """a standalone page holding just the hook (for the raster pass)."""
    st = Style.from_spec(hook)
    if font:
        st.font = str(font)
    lines = layout(hook["lines"], None, canvas, st, hook.get("anchor", "bottom-left"))
    body = svg(lines, None, canvas, st, hook.get("accent_phrase"), hook.get("accent", "gold"), hook.get("annotation"))
    faces = [_face(fam, font_path(f)) for f, fam in families(lines).items()]
    if hook.get("annotation") and font_path("coolvetica").exists():
        faces.append(_face("AnnFont", font_path("coolvetica")))
    if st.keystone != 1.0:
        body = f'<div style="transform-origin: 0 0; transform: {keystone_css(lines, st.keystone, hook.get("anchor", "bottom-left"))}">{body}</div>'
    return PAGE.format(faces="\n".join(faces), svg=body)


def keystone(lines: list[Line], k: float, anchor: str = "bottom-left"):
    """3x3 homography: the block's box -> the same width, the far end's height times k about
    the block's center line, the anchored end untouched (the whole hook = one plane in 3D)."""
    import numpy as np
    m0 = metrics(str(font_path(lines[0].font)))
    x0, x1 = min(ln.x for ln in lines), max(ln.x + ln.width for ln in lines)
    yt, yb = lines[0].baseline - lines[0].cap(m0), lines[-1].baseline
    yc, h = (yt + yb) / 2, (yb - yt) / 2
    near_l, near_r = (1.0, k) if anchor.endswith("left") else (k, 1.0)
    src = [(x0, yt), (x1, yt), (x1, yb), (x0, yb)]
    dst = [(x0, yc - h * near_l), (x1, yc - h * near_r), (x1, yc + h * near_r), (x0, yc + h * near_l)]
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        b += [u, v]
    return np.append(np.linalg.solve(np.array(A, float), np.array(b, float)), 1.0).reshape(3, 3)


def keystone_css(lines: list[Line], k: float, anchor: str = "bottom-left") -> str:
    """the homography as css matrix3d (column-major, z passed through), origin at the page's 0,0."""
    H = keystone(lines, k, anchor)
    cols = [(H[0, 0], H[1, 0], 0, H[2, 0]), (H[0, 1], H[1, 1], 0, H[2, 1]), (0, 0, 1, 0), (H[0, 2], H[1, 2], 0, H[2, 2])]
    return "matrix3d(" + ", ".join(f"{v:.9g}" for c in cols for v in c) + ")"


def raster(hook: dict, out_png: Path, canvas: tuple[int, int] = (1280, 720), font: Path | None = None,
           browser=None, scale: int = 1) -> dict:
    """render the hook alone to a transparent PNG cropped to its pixels -> {src, x, y, w, h}
    for placing it on the thumbnail. Same hook -> same bytes."""
    import io

    from PIL import Image
    html = page(hook, canvas, font)

    def shoot(b):
        pg = b.new_page(viewport={"width": canvas[0], "height": canvas[1]}, device_scale_factor=scale)
        try:
            pg.set_content(html, wait_until="load")
            pg.evaluate("document.fonts.ready.then(() => true)")
            return pg.screenshot(type="png", omit_background=True)
        finally:
            pg.close()

    if browser is None:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            try:
                png = shoot(b)
            finally:
                b.close()
    else:
        png = shoot(browser)
    im = Image.open(io.BytesIO(png)).convert("RGBA")
    x0, y0, x1, y1 = im.getchannel("A").getbbox() or (0, 0, 1, 1)
    im.crop((x0, y0, x1, y1)).save(out_png, "PNG")     # scale>1 keeps the sharp pixels; placed at 1x size
    return {"src": out_png.name, "x": x0 / scale, "y": y0 / scale, "w": (x1 - x0) / scale, "h": (y1 - y0) / scale}
