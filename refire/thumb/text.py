"""House-style hook text as SVG (docs/specs/03-thumb.md "House style > Text").

Metallic three-stop gradient fill, heavy black stroke painted under it, a black backing
rectangle per line (shorter than the caps, so letters rise above and below it) that fuses
the outlines into one shape, lines fitted to equal width. Widths come from the font file
(fontTools); the SVG turns kerning off so Chromium draws exactly what Python measured.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from html import escape
from pathlib import Path

DEFAULT_FONT = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts" / "Conneqt-Black.ttf"
WHITE = ("#D6D6D6", "#FFFFFF")                      # metallic white: base, light
PALETTE = {"blue": ("#2C66F3", "#64A1FF"), "sky": ("#0884F2", "#58D3F3"), "green": ("#26CE27", "#52F753"),
           "mint": ("#26C075", "#53DA8E"), "gold": ("#D2A51B", "#F8CE4F"), "cyan": ("#1FFFF8", "#65FFFB")}


@dataclass(frozen=True)
class Metrics:
    upm: int
    cap: int                    # cap height, font units
    advances: dict              # char -> advance, font units
    fallback: int

    def width(self, text: str, size: float) -> float:
        return sum(self.advances.get(c, self.fallback) for c in text) * size / self.upm


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
    size: float                 # font size px
    x: float                    # left of the glyphs
    baseline: float
    width: float                # glyph advance width, px

    def cap(self, m: Metrics) -> float:
        return m.cap * self.size / m.upm


def fit(lines: list[str], m: Metrics, width: float, max_size: float, stroke_ratio: float) -> list[float]:
    """font size per line so every line's stroked width is `width` (equal-width lines),
    capped at max_size so a one-word line doesn't explode."""
    out = []
    for t in lines:
        unit = m.width(t, 1.0) + stroke_ratio        # stroke adds ratio*size across both ends
        out.append(min(max_size, width / unit if unit else max_size))
    return out


def layout(lines: list[str], m: Metrics, canvas: tuple[int, int], anchor: str = "bottom-left",
           width_frac: float = 0.57, margin: float = 40.0, max_size_frac: float = 0.24,
           stroke_ratio: float = 0.11, leading: float = 1.06) -> list[Line]:
    """place the block: lines fitted to width_frac of the canvas, stacked tight, anchored to
    a bottom corner `margin` px in (the stroke stays inside the margin)."""
    W, H = canvas
    sizes = fit(lines, m, width_frac * W, max_size_frac * H, stroke_ratio)
    out, base = [], None
    for t, s in reversed(list(zip(lines, sizes))):           # build bottom-up
        cap = m.cap * s / m.upm
        # upper line's baseline sits a hair above the lower line's caps (leading - 1 of a cap)
        base = H - margin - stroke_ratio * s / 2 if base is None else base - prev_cap * leading
        w = m.width(t, s)
        stroked = w + stroke_ratio * s
        x = margin + stroke_ratio * s / 2 if anchor.endswith("left") else W - margin - stroked + stroke_ratio * s / 2
        out.append(Line(t, round(s, 2), round(x, 2), round(base, 2), round(w, 2)))
        prev_cap = cap
    return list(reversed(out))


def _split_accent(text: str, accent: str | None) -> list[tuple[str, bool]]:
    """line -> [(chunk, is_accent)] for the first case-insensitive match of the accent phrase."""
    if not accent:
        return [(text, False)]
    i = text.upper().find(accent.upper())
    if i < 0:
        return [(text, False)]
    parts = [(text[:i], False), (text[i:i + len(accent)], True), (text[i + len(accent):], False)]
    return [p for p in parts if p[0]]


def svg(lines: list[Line], m: Metrics, canvas: tuple[int, int], accent_phrase: str | None = None,
        accent: str = "gold", stroke_ratio: float = 0.11, rect_ratio: float = 0.62,
        annotation: str | None = None, family: str = "HookFont", palette: dict | None = None) -> str:
    """the text block as one <svg> layer (canvas-sized, transparent)."""
    W, H = canvas
    pal = {**PALETTE, **(palette or {})}
    acc = pal.get(accent, PALETTE["gold"])
    defs, rects, texts = [], [], []
    for i, ln in enumerate(lines):
        cap, sw = ln.cap(m), stroke_ratio * ln.size
        top = ln.baseline - cap
        for name, (base, light) in (("w", WHITE), ("a", acc)):
            defs.append(f'<linearGradient id="g{name}{i}" gradientUnits="userSpaceOnUse" x1="0" y1="{top:.2f}" '
                        f'x2="0" y2="{ln.baseline:.2f}"><stop offset="0" stop-color="{base}"/>'
                        f'<stop offset="0.5" stop-color="{light}"/><stop offset="1" stop-color="{base}"/>'
                        f'</linearGradient>')
        rh = rect_ratio * cap
        rects.append(f'<rect x="{ln.x - sw / 2:.2f}" y="{ln.baseline - cap / 2 - rh / 2:.2f}" '
                     f'width="{ln.width + sw:.2f}" height="{rh:.2f}" fill="#000"/>')
        spans = "".join(f'<tspan fill="url(#g{"a" if a else "w"}{i})">{escape(t)}</tspan>'
                        for t, a in _split_accent(ln.text, accent_phrase))
        texts.append(f'<text x="{ln.x:.2f}" y="{ln.baseline:.2f}" font-size="{ln.size:.2f}" '
                     f'stroke-width="{sw:.2f}">{spans}</text>')
    ann = ""
    if annotation and len(lines) > 1:
        ln = lines[-1]
        s = ln.size * 0.28
        ann = (f'<text class="ann" x="{ln.x + ln.width * 0.55:.2f}" y="{ln.baseline - ln.cap(m) - s * 0.25:.2f}" '
               f'font-size="{s:.2f}" stroke-width="{s * 0.16:.2f}">{escape(annotation)}</text>')
    return (f'<svg class="hook" xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
            f'<defs>{"".join(defs)}</defs><g class="block">{"".join(rects)}'
            f'<g class="lines" font-family="{family}" stroke="#000" stroke-linejoin="round" paint-order="stroke">'
            f'{"".join(texts)}</g>{ann}</g></svg>')
