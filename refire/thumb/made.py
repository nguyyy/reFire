"""Left-side graphics built in code when the stream didn't show a clean panel (Nguy: "you have
the freedom to create an asset/panel yourself"). Rendered by the pinned Chromium to a
transparent PNG, then used as a card in the 3D scene like any panel.

  tierlist  {"rows": [{"label": "GOD TIER", "color": "#FF7F7F", "chars": ["Furina", ...]}]}
            (the refs: GOD TIER / OKAY / RUN AWAY, T0 / T0.5 / T67), official character icons
  stat_card {"title": "Bloom of the Mind's Desire", "subtitle": "Flower of Life", "main": ["HP", "717"],
             "stars": 5, "level": "+0", "rows": [["CRIT DMG", "+23.3%"], ...], "hidden": [0, 1]}
            (the in-game artifact card; hidden rows get pixel-mosaic values, the reveal treatment)
"""
from __future__ import annotations

import base64
import io
import json
import random
from html import escape
from pathlib import Path

from . import art

TIER_COLORS = ["#FF7F7F", "#FFBF7F", "#FFDF7F", "#BFFF7F", "#7FFFFF", "#7FBFFF"]


def icon(name: str, f: art.Fetcher | None = None, cache: Path = art.CACHE) -> Path | None:
    """official avatar icon (enka's UI_AvatarIcon_X.png, ~256 px with alpha), cached."""
    f = f or art.Fetcher()
    roster = art.characters(f, cache)
    canon = art.resolve(name, roster)
    if not canon:
        return None
    out = cache / f"{art.slug(canon)}_icon.png"
    if out.exists():
        return out
    for url in (f"https://enka.network/ui/{roster[canon]}.png", f"{art.YATTA}/assets/UI/{roster[canon]}.png"):
        try:
            body, _, final = f.get(url, accept=art.IMAGES)
        except art.ArtError:
            continue
        cache.mkdir(parents=True, exist_ok=True)
        out.write_bytes(body)
        out.with_suffix(".json").write_text(json.dumps({"name": canon, "url": final}), encoding="utf-8")
        return out
    return None


def _pixelated(p: Path) -> Path:
    """an icon reduced to an 11x11 mosaic (nearest-neighbor), cached beside it."""
    from PIL import Image
    out = p.with_name(p.stem + "_px.png")
    if not out.exists():
        im = Image.open(p).convert("RGBA")
        im.resize((11, 11), Image.BOX).resize(im.size, Image.NEAREST).save(out)
    return out


def _data_uri(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def _mosaic(seed: int, cols: int = 7, rows: int = 2, cell: int = 9) -> str:
    """a pixel block standing in for a hidden value (deterministic per seed)."""
    rnd = random.Random(seed)
    cells = "".join(f'<i style="background:rgb({g},{g},{g + 8})"></i>'
                    for g in (rnd.randint(70, 200) for _ in range(cols * rows)))
    return (f'<span class="mz" style="grid-template-columns:repeat({cols},{cell}px);'
            f'grid-auto-rows:{cell}px">{cells}</span>')


PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;background:transparent;font-family:"Segoe UI",Arial,sans-serif}}
{css}</style></head><body>{body}</body></html>"""

TIER_CSS = """
.tl{display:inline-grid;grid-template-columns:150px auto;gap:4px;background:#1b1b1f;padding:4px;border-radius:6px}
.lab{display:flex;align-items:center;justify-content:center;font-weight:800;font-size:30px;color:#111;
     text-align:center;line-height:1.05;padding:0 8px;min-height:120px}
.row{display:flex;gap:4px;background:#2a2a30;padding:4px;min-width:500px}
.row img{width:112px;height:112px;object-fit:cover;background:radial-gradient(circle at 50% 35%,#5b4b7a,#2b2338);
         border-radius:4px}
"""

CARD_CSS = """
.card{display:inline-block;width:520px;border-radius:10px;overflow:hidden;background:#ece5d8;
      box-shadow:0 0 0 3px #d8cdb8;font-weight:600}
.hd{background:linear-gradient(180deg,#b97a3e,#d39a54 60%,#c3884a);padding:14px 20px 18px;color:#fff}
.t{font-size:30px;font-weight:800;text-shadow:0 2px 0 rgba(0,0,0,.25)}
.st{font-size:19px;opacity:.9;margin-top:4px}
.ms{font-size:20px;opacity:.85;margin-top:22px}.mv{font-size:52px;font-weight:800;line-height:1}
.stars{color:#ffcc32;font-size:30px;letter-spacing:2px;margin-top:6px;text-shadow:0 1px 0 #8a5b16}
.bd{padding:14px 22px 20px;color:#495366;font-size:24px}
.lv{display:inline-block;background:#3a4256;color:#fff;font-size:20px;padding:2px 10px;border-radius:4px;margin-bottom:10px}
.r{margin:6px 0;display:flex;align-items:center;gap:8px}.r b{color:#495366}
.mz{display:inline-grid;gap:0;vertical-align:middle}.mz i{display:block}
"""


def tierlist_html(rows: list[dict], icon_fn=icon, hidden=()) -> str:
    """hidden = [[row, col], ...]: those icons drawn as a coarse pixel mosaic (the reveal)."""
    out, hide = [], {tuple(h) for h in hidden or []}
    for i, r in enumerate(rows[:5]):
        color = r.get("color") or TIER_COLORS[i % len(TIER_COLORS)]
        cells = []
        for j, p in enumerate(icon_fn(n) for n in r.get("chars", [])[:5]):
            if not p:
                continue
            if (i, j) in hide:
                cells.append(f'<img class="px" src="{_data_uri(_pixelated(p))}">')
            else:
                cells.append(f'<img src="{_data_uri(p)}">')
        imgs = "".join(cells)
        out.append(f'<div class="lab" style="background:{escape(color)}">{escape(r.get("label", "?"))}</div>'
                   f'<div class="row">{imgs}</div>')
    return PAGE.format(css=TIER_CSS, body=f'<div class="tl">{"".join(out)}</div>')


def stat_card_html(c: dict) -> str:
    hidden = set(c.get("hidden") or [])
    rows = "".join(
        f'<div class="r">• <b>{escape(lbl)}</b>' + (_mosaic(i + 1) + " ???" if i in hidden else f" {escape(val)}")
        + "</div>" for i, (lbl, val) in enumerate(c.get("rows", [])[:6]))
    main = c.get("main") or ["", ""]
    stars = "★" * int(c.get("stars", 5))
    body = (f'<div class="card"><div class="hd"><div class="t">{escape(c.get("title", ""))}</div>'
            f'<div class="st">{escape(c.get("subtitle", ""))}</div><div class="ms">{escape(main[0])}</div>'
            f'<div class="mv">{escape(main[1])}</div><div class="stars">{stars}</div></div>'
            f'<div class="bd"><span class="lv">{escape(c.get("level", "+0"))}</span>{rows}</div></div>')
    return PAGE.format(css=CARD_CSS, body=body)


def render(kind: str, data: dict, out: Path, browser=None, icon_fn=icon) -> Path:
    """made graphic -> transparent PNG cropped to its pixels (2x for crisp edges)."""
    from PIL import Image
    html = (tierlist_html(data.get("rows", []), icon_fn, data.get("hidden")) if kind == "tierlist"
            else stat_card_html(data))

    def shoot(b):
        pg = b.new_page(viewport={"width": 1400, "height": 1000}, device_scale_factor=2)
        try:
            pg.set_content(html, wait_until="load")
            return pg.screenshot(type="png", omit_background=True, full_page=True)
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
    im.crop(im.getchannel("A").getbbox() or (0, 0, 1, 1)).save(out)
    return out
