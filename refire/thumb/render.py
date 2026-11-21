"""spec.json -> index.html -> thumb.png (Playwright's pinned headless Chromium).

One stage template + house.css for every layout: a layout is just where the spec puts its
layers, so the spec is the whole design and editing it + re-rendering is the tweak loop.
The same spec always renders the same bytes (pinned Chromium, no timestamps, no network).
One-time setup: `uv run playwright install chromium` (the version pinned by uv.lock).
"""
from __future__ import annotations

import base64
import json
from html import escape
from pathlib import Path
from string import Template

from . import text as T

TEMPLATES = Path(__file__).with_name("templates")
MAX_BYTES = 2 * 1024 * 1024          # youtube custom thumbnail limit (historically 2 MB)
# width/leading tuned against the refs: ~62% wide, strokes of stacked lines touching
DEFAULT_TEXT = {"anchor": "bottom-left", "width_frac": 0.62, "margin": 40, "stroke_ratio": 0.11,
                "rect_ratio": 0.62, "max_size_frac": 0.24, "leading": 1.0, "accent": "gold",
                "accent_phrase": None, "annotation": None}
TEXT_SHADOW = "drop-shadow(6px 8px 0 rgba(0,0,0,0.55)) drop-shadow(0 0 18px rgba(0,0,0,0.7))"


def _style(d: dict) -> str:
    return "; ".join(f"{k}: {v}" for k, v in d.items())


def _layer(L: dict) -> str:
    """one <img> layer. kind face: rim glow (tight + wide); subject: drop shadow + rotation."""
    st = {"left": f"{L.get('x', 0)}px", "top": f"{L.get('y', 0)}px"}
    if "w" in L:
        st["width"] = f"{L['w']}px"
    if "h" in L:
        st["height"] = f"{L['h']}px"
    tf = []
    if L.get("rotate"):
        tf.append(f"rotate({L['rotate']}deg)")
    if L.get("perspective"):
        tf.append(f"perspective(1200px) rotateY({L['perspective']}deg)")
    if tf:
        st["transform"] = " ".join(tf)
    fx = []
    if L.get("glow"):
        g = L["glow"]
        fx += [f"drop-shadow(0 0 3px {g})", f"drop-shadow(0 0 22px {g})"]
    if L.get("shadow"):
        fx.append(f"drop-shadow(10px 14px 16px rgba(0,0,0,{L['shadow']}))")
    if fx:
        st["filter"] = " ".join(fx)
    return f'<img class="layer {escape(L.get("kind", ""))}" src="{escape(L["src"])}" style="{_style(st)}">'


def _background(bg: dict) -> str:
    if not bg:
        return ""
    st = {"background-image": f"url('{bg['src']}')",
          "filter": f"blur({bg.get('blur', 12)}px) brightness({bg.get('brightness', 0.62)}) "
                    f"saturate({bg.get('saturate', 1.15)})",
          "transform": f"scale({bg.get('scale', 1.08)})"}
    v = bg.get("vignette", 0.55)
    vig = (f'<div class="vignette" style="background: radial-gradient(ellipse at center, rgba(0,0,0,0) 45%, '
           f'rgba(0,0,0,{v}) 100%)"></div>') if v else ""
    return f'<div class="bg" style="{_style(st)}"></div>{vig}'


def html(spec: dict) -> str:
    W, H = spec.get("canvas", [1280, 720])
    font = Path(spec.get("font") or T.DEFAULT_FONT)
    m = T.metrics(str(font))
    t = {**DEFAULT_TEXT, **(spec.get("text") or {})}
    hook = ""
    if t.get("lines"):
        lines = T.layout(t["lines"], m, (W, H), t["anchor"], t["width_frac"], t["margin"], t["max_size_frac"],
                         t["stroke_ratio"], t["leading"])
        hook = T.svg(lines, m, (W, H), t["accent_phrase"], t["accent"], t["stroke_ratio"], t["rect_ratio"],
                     t["annotation"])
    font_url = "data:font/ttf;base64," + base64.b64encode(font.read_bytes()).decode()
    return Template((TEMPLATES / "stage.html").read_text(encoding="utf-8")).substitute(
        font_url=font_url, house_css=(TEMPLATES / "house.css").read_text(encoding="utf-8"), w=W, h=H,
        text_shadow=TEXT_SHADOW, background=_background(spec.get("background")),
        layers="\n".join(_layer(L) for L in spec.get("layers", [])), text=hook)


def render(spec_path, hires: bool | None = None, browser=None) -> Path:
    """render <dir>/spec.json -> <dir>/index.html + <dir>/thumb.png (jpeg q92 if over the
    size limit). `browser` = an open playwright Browser to reuse across candidates."""
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    d = spec_path.parent
    W, H = spec.get("canvas", [1280, 720])
    hires = spec.get("hires", False) if hires is None else hires
    page_path = d / "index.html"
    page_path.write_text(html(spec), encoding="utf-8")
    out = d / "thumb.png"
    if browser is None:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            try:
                return _shoot(b, page_path, out, W, H, hires)
            finally:
                b.close()
    return _shoot(browser, page_path, out, W, H, hires)


def _shoot(browser, page_path: Path, out: Path, W: int, H: int, hires: bool) -> Path:
    page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=2 if hires else 1)
    try:
        page.goto(page_path.resolve().as_uri(), wait_until="load")
        page.evaluate("document.fonts.ready.then(() => true)")
        png = page.screenshot(type="png", clip={"x": 0, "y": 0, "width": W, "height": H})
    finally:
        page.close()
    if hires:                                   # 2x render, lanczos down: crisper text edges
        import io

        from PIL import Image
        im = Image.open(io.BytesIO(png)).convert("RGB").resize((W, H), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "PNG")
        png = buf.getvalue()
    out.write_bytes(png)
    if len(png) > MAX_BYTES:
        import io

        from PIL import Image
        jpg = out.with_suffix(".jpg")
        Image.open(io.BytesIO(png)).convert("RGB").save(jpg, "JPEG", quality=92)
        return jpg
    return out
