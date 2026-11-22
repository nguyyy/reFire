"""Measure hook text in Nguy's thumbnails vs ours, numerically (the text one-to-one loop).

Per line: cap height (fill ink), width, width/cap, baseline gap / cap, slab height vs cap
(dark run in the gaps between letters), outline px, and the shine profile across the line
(lightness of the fill per x, normalized): how much lighter the middle is and how fast it
ramps. Ours = the same hook rendered by refire.thumb.text over black.

    uv run python tests/smoke/thumb_text_measure.py            # table: ref vs ours per line
"""
from __future__ import annotations

import colorsys
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from refire.thumb import text as T  # noqa: E402
from thumb_text_vs_refs import HOOKS, REFS  # noqa: E402

# hook region per ref (x0, y0, x1, y1) on the 1280x720 frame, read off the refs
REGIONS = {
    "artifacts+stall.png": (0, 430, 960, 720),
    "stronboxthumbnail - Copy.png": (0, 440, 820, 720),
    "abysseasyzy0xzyoxthumbnailspiral.png": (0, 440, 700, 720),
    "schz.png": (0, 420, 900, 720),
    "funnytierlistthumbnailB.png": (0, 400, 1120, 720),
    "tierlistpng - Copy.png": (420, 440, 1280, 720),
    "zajrevthumb.png": (0, 430, 1000, 720),
}


def masks(rgb: np.ndarray):
    """fill (white metallic or saturated accent) and dark (outline/slab) masks."""
    a = rgb.astype(np.float32) / 255
    mx, mn = a.max(-1), a.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    fill = mx > 0.30                      # anything not dark; the outline test below isolates letters
    dark = mx < 0.16
    return letters_only(fill, dark), dark


def letters_only(fill: np.ndarray, dark: np.ndarray, need: float = 0.7) -> np.ndarray:
    """keep fill blobs that a black outline surrounds (every hook letter has one; bright
    background patches don't)."""
    import cv2
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fill.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(fill)
    k = np.ones((7, 7), np.uint8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 25:
            continue
        x0, y0, x1, y1 = max(0, x - 4), max(0, y - 4), min(fill.shape[1], x + w + 4), min(fill.shape[0], y + h + 4)
        m = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
        ring = (cv2.dilate(m, k) > 0) & (m == 0) & ~fill[y0:y1, x0:x1]
        if ring.sum() and dark[y0:y1, x0:x1][ring].mean() >= need:
            keep[y0:y1, x0:x1] |= m.astype(bool)
    return keep


def bands(fill: np.ndarray, n: int):
    """split rows into n text lines at the deepest fill-count valleys."""
    rows = fill.sum(1).astype(float)
    k = np.convolve(rows, np.ones(5) / 5, mode="same")
    on = np.flatnonzero(k > 0.12 * k.max())
    top, bot = on[0], on[-1]
    cuts = []
    seg = k[top:bot + 1]
    for _ in range(n - 1):                                  # deepest valley away from chosen cuts
        order = np.argsort(seg)
        for i in order:
            y = top + i
            if all(abs(y - c) > 25 for c in cuts) and y - top > 25 and bot - y > 25:
                cuts.append(y)
                break
    edges = [top] + sorted(cuts) + [bot]
    return [(edges[i], edges[i + 1]) for i in range(n)]


def measure_line(rgb, fill, dark, y0, y1):
    f = fill[y0:y1]
    rows = f.sum(1)
    on = np.flatnonzero(rows > 0.15 * rows.max())
    ft, fb = y0 + on[0], y0 + on[-1]                        # fill ink top/bottom
    cols = fill[ft:fb + 1].sum(0)
    xs = np.flatnonzero(cols > 0)
    xl, xr = xs[0], xs[-1]
    cap = fb - ft + 1
    mid = (ft + fb) // 2
    # slab: columns between letters with no fill in the ink band -> dark run through mid
    gaps = [x for x in range(xl + 3, xr - 3) if cols[x] == 0]
    runs = []
    for x in gaps:
        col = dark[:, x]
        if not col[mid]:
            continue
        a = mid
        while a > 0 and col[a - 1]:
            a -= 1
        b = mid
        while b < len(col) - 1 and col[b + 1]:
            b += 1
        runs.append(b - a + 1)
    slab = float(np.median(runs)) if runs else 0.0
    # outline: dark run left of the first fill pixel on the mid row
    row = fill[mid]
    x0 = np.flatnonzero(row)[0]
    xx = x0 - 1
    while xx > 0 and not dark[mid, xx] and x0 - xx < 4:   # step over the anti-aliased edge
        xx -= 1
    o = 0
    while xx - o >= 0 and dark[mid, xx - o]:
        o += 1
    # shine: lightness of fill per column across the line, normalized x
    prof = []
    for x in range(xl, xr + 1):
        m = fill[ft:fb + 1, x]
        if m.sum() >= 3:
            c = rgb[ft:fb + 1, x][m].mean(0) / 255
            prof.append(((x - xl) / max(1, xr - xl), colorsys.rgb_to_hls(*c)[1]))
    prof = np.array(prof)
    bins = np.clip((prof[:, 0] * 10).astype(int), 0, 9)
    curve = [round(float(prof[bins == i, 1].mean()), 3) if (bins == i).any() else None for i in range(10)]
    vals = [c for c in curve if c is not None]
    return {"cap": int(cap), "top": int(ft), "bottom": int(fb), "width": int(xr - xl + 1),
            "w_per_cap": round((xr - xl + 1) / cap, 2), "slab": slab, "slab_minus_cap": round(slab - cap, 1),
            "outline": o, "shine": curve, "shine_gain": round(max(vals) - (vals[0] + vals[-1]) / 2, 3) if vals else 0,
            "shine_peak_at": round(float(np.argmax([c if c is not None else -1 for c in curve]) / 10 + 0.05), 2)}


def measure(img: Image.Image, n: int, debug: Path | None = None) -> list[dict]:
    rgb = np.asarray(img.convert("RGB"))
    fill, dark = masks(rgb)
    if debug:
        vis = rgb.copy()
        vis[fill] = (vis[fill] * 0.3 + np.array([255, 0, 255]) * 0.7).astype(np.uint8)
        for a, b in bands(fill, n):
            vis[a] = (0, 255, 0); vis[b] = (0, 255, 0)
        Image.fromarray(vis).save(debug)
    out = [measure_line(rgb, fill, dark, a, b) for a, b in bands(fill, n)]
    for i in range(1, len(out)):
        out[i]["gap_per_cap"] = round((out[i]["bottom"] - out[i - 1]["bottom"]) / out[i]["cap"], 3)
    return out


def ours(hook: dict) -> Image.Image:
    tmp = Path(tempfile.gettempdir()) / "refire_measure_hook.png"
    info = T.raster(hook, tmp)
    canvas = Image.new("RGB", (1280, 720), "black")
    h = Image.open(tmp)
    canvas.paste(h, (round(info["x"]), round(info["y"])), h)
    return canvas


def main(argv) -> int:
    rows = {}
    for name, hook in HOOKS.items():
        if argv and not any(a.lower() in name.lower() for a in argv):
            continue
        r = REGIONS[name]
        n = len(hook["lines"])
        ref = measure(Image.open(REFS / name).crop(r), n, Path(tempfile.gettempdir()) / f"measure_{Path(name).stem}.png")
        mine_full = ours(hook)
        # our block sits wherever the layout puts it: measure the bottom 300 px band it lives in
        mine = measure(mine_full.crop((0, 380, 1280, 720)), n)
        rows[name] = {"ref": ref, "ours": mine}
        print(f"\n== {name}")
        for i, (a, b) in enumerate(zip(ref, mine)):
            print(f"  line {i}: cap {a['cap']:>4}/{b['cap']:<4} w/cap {a['w_per_cap']:>5}/{b['w_per_cap']:<5} "
                  f"gap/cap {a.get('gap_per_cap', '-')!s:>6}/{b.get('gap_per_cap', '-')!s:<6} "
                  f"slab-cap {a['slab_minus_cap']:>6}/{b['slab_minus_cap']:<6} outline {a['outline']:>3}/{b['outline']:<3} "
                  f"gain {a['shine_gain']:>6}/{b['shine_gain']:<6} peak {a['shine_peak_at']}/{b['shine_peak_at']}")
            print(f"          shine ref  {a['shine']}\n          shine ours {b['shine']}")
    (Path(tempfile.gettempdir()) / "refire_text_measure.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
