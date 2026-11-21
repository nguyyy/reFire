"""spec.json -> index.html -> thumb.png (Playwright's pinned headless Chromium).

One stage template + house.css for every layout: a layout is just where the spec puts its
layers, so the spec is the whole design and editing it + re-rendering is the tweak loop.
The same spec always renders the same bytes (pinned Chromium, no timestamps, no network).
One-time setup: `uv run playwright install chromium` (the version pinned by uv.lock).
"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path
from string import Template

from . import text as T

TEMPLATES = Path(__file__).with_name("templates")
MAX_BYTES = 2 * 1024 * 1024          # youtube custom thumbnail limit (historically 2 MB)


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


def html(spec: dict, layers: list | None = None) -> str:
    W, H = spec.get("canvas", [1280, 720])
    return Template((TEMPLATES / "stage.html").read_text(encoding="utf-8")).substitute(
        house_css=(TEMPLATES / "house.css").read_text(encoding="utf-8"), w=W, h=H,
        background=_background(spec.get("background")),
        layers="\n".join(_layer(L) for L in (spec.get("layers", []) if layers is None else layers)))


def render(spec_path, hires: bool | None = None, browser=None) -> Path:
    """render <dir>/spec.json -> <dir>/index.html + <dir>/thumb.png (jpeg q92 if over the
    size limit). `browser` = an open playwright Browser to reuse across candidates."""
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    d = spec_path.parent
    W, H = spec.get("canvas", [1280, 720])
    hires = spec.get("hires", False) if hires is None else hires
    out = d / "thumb.png"
    if browser is None:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            try:
                return _render(b, spec, d, out, W, H, hires)
            finally:
                b.close()
    return _render(browser, spec, d, out, W, H, hires)


def _render(browser, spec: dict, d: Path, out: Path, W: int, H: int, hires: bool) -> Path:
    """hook text first (its own raster, refire.thumb.text), then the page with it on top."""
    layers = list(spec.get("layers", []))
    t = spec.get("text") or {}
    if t.get("lines"):
        (d / "assets").mkdir(exist_ok=True)
        info = T.raster(t, d / "assets" / "hook.png", (W, H), font=spec.get("font"), browser=browser,
                        scale=2 if hires else 1)
        layers.append({"kind": "hook", "src": "assets/hook.png", "x": info["x"], "y": info["y"], "w": info["w"]})
    page_path = d / "index.html"
    page_path.write_text(html(spec, layers), encoding="utf-8")
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
