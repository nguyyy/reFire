"""Pixel effects for the reveal treatment (docs/specs/03-thumb.md "the pixelated reveal").

6 of 7 refs hide something (a tier placement, stat values, a character) behind a mosaic,
often with a pixel-art "???" bubble or big blocky question marks. All drawn in python so
it's deterministic: mosaic = nearest-neighbor down/up at block ~= target width / 10.
"""
from __future__ import annotations

from PIL import Image, ImageDraw

# 5x7 bitmap glyphs, enough for teaser bubbles
GLYPHS = {
    "?": [".###.", "#...#", "....#", "..##.", "..#..", ".....", "..#.."],
    "!": ["..#..", "..#..", "..#..", "..#..", "..#..", ".....", "..#.."],
    ".": [".....", ".....", ".....", ".....", ".....", ".....", "..#.."],
}


def pixelate(img: Image.Image, box: tuple[float, float, float, float], block: int | None = None,
             normalized: bool = True) -> Image.Image:
    """mosaic `box` (x0, y0, x1, y1; 0-1 of the image unless normalized=False) in place-ish:
    returns a copy. block defaults to the target width / 10 (spec)."""
    im = img.copy()
    W, H = im.size
    x0, y0, x1, y1 = box
    if normalized:
        x0, y0, x1, y1 = x0 * W, y0 * H, x1 * W, y1 * H
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    b = max(2, block or round(w / 10))
    region = im.crop((x0, y0, x0 + w, y0 + h))
    small = region.resize((max(1, round(w / b)), max(1, round(h / b))), Image.BOX)
    im.paste(small.resize((w, h), Image.NEAREST), (x0, y0))
    return im


def _grid_text(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, color) -> None:
    for i, ch in enumerate(text):
        for r, row in enumerate(GLYPHS.get(ch, GLYPHS["?"])):
            for c, v in enumerate(row):
                if v == "#":
                    draw.point((x + i * 6 + c, y + r), fill=color)


def bubble(text: str = "???", cell: int = 9, tail: str = "left") -> Image.Image:
    """pixel-art speech bubble: white body, 1-cell black outline, stepped corners, a tail.
    grid rows: 0 outline, 1 pad, 2-8 glyphs, 9 pad, 10 outline, 11-13 tail."""
    n = len(text)
    bw, bh = n * 6 + 5, 11
    g = Image.new("RGBA", (bw, bh + 3), (0, 0, 0, 0))
    d = ImageDraw.Draw(g)
    d.rectangle([1, 0, bw - 2, bh - 1], fill="black")           # outline with stepped corners
    d.rectangle([0, 1, bw - 1, bh - 2], fill="black")
    d.rectangle([2, 1, bw - 3, bh - 2], fill="white")
    d.rectangle([1, 2, bw - 2, bh - 3], fill="white")
    tx = 3 if tail == "left" else bw - 7
    for k in range(3):                                           # stepped tail under the body
        d.rectangle([tx, bh - 1 + k, tx + 3 - k, bh - 1 + k], fill="black")
        if k < 2:
            d.rectangle([tx + 1, bh - 1 + k, tx + 2 - k, bh - 1 + k], fill="white")
    _grid_text(d, text, 3, 2, "black")
    return g.resize((bw * cell, (bh + 3) * cell), Image.NEAREST)


def glyph(ch: str = "?", cell: int = 12, outline: int = 1) -> Image.Image:
    """one big blocky glyph, white with a black pixel outline (the skirk ref's question marks).
    strokes are doubled first so the white reads thicker than the outline."""
    rows = GLYPHS.get(ch, GLYPHS["?"])
    on = {(c * 2 + dx + outline, r * 2 + dy + outline) for r, row in enumerate(rows)
          for c, v in enumerate(row) if v == "#" for dx in (0, 1) for dy in (0, 1)}
    gw, gh = 10 + 2 * outline, 14 + 2 * outline
    g = Image.new("RGBA", (gw, gh), (0, 0, 0, 0))
    d = ImageDraw.Draw(g)
    for x, y in on:                                              # outline = the on-cells grown by one
        d.rectangle([x - outline, y - outline, x + outline, y + outline], fill="black")
    for x, y in on:
        d.point((x, y), fill="white")
    return g.resize((gw * cell, gh * cell), Image.NEAREST)


def fade(img: Image.Image, edges: tuple[float, float, float, float]) -> Image.Image:
    """soft alpha edges so an in-game asset blends into the scene instead of ending in a hard
    border (nguy): edges = (left, top, right, bottom) ramp widths as fractions of the image's
    width/height, smoothstep from 0 at the edge to 1. multiplies any alpha already there."""
    import numpy as np
    im = img.convert("RGBA")
    W, H = im.size

    def ramp(n, a, b):
        t = np.arange(n, dtype=np.float32) + 0.5
        lo = np.clip(t / max(a * n, 1e-6), 0, 1) if a > 0 else np.ones(n, np.float32)
        hi = np.clip((n - t) / max(b * n, 1e-6), 0, 1) if b > 0 else np.ones(n, np.float32)
        s = np.minimum(lo, hi)
        return s * s * (3 - 2 * s)
    left, top, right, bottom = edges
    m = ramp(H, top, bottom)[:, None] * ramp(W, left, right)[None, :]
    arr = np.asarray(im).copy()
    arr[..., 3] = (arr[..., 3].astype(np.float32) * m).round().astype(np.uint8)
    return Image.fromarray(arr)
