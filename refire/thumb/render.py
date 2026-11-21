"""spec.json -> index.html -> thumb.png (Playwright's pinned headless Chromium).

One stage template + house.css for every layout: a layout is just where the spec puts its
layers, so the spec is the whole design and editing it + re-rendering is the tweak loop.
`spec.scene` adds the 3D world (templates/scene.js, vendored three.js: background plate,
cards in perspective, rays, sparkles, bloom) under the flat layers (zy0x, the hook).
The same spec always renders the same bytes: pinned Chromium, CPU WebGL (SwiftShader),
seeded randomness, no network (the page is served from a fake http://refire.local origin).
One-time setup: `uv run playwright install chromium` (the version pinned by uv.lock).
"""
from __future__ import annotations

import json
import mimetypes
from html import escape
from pathlib import Path
from string import Template

from . import text as T

HERE = Path(__file__).parent
TEMPLATES = HERE / "templates"
VENDOR = HERE / "vendor"
ORIGIN = "http://refire.local/"
MAX_BYTES = 2 * 1024 * 1024          # youtube custom thumbnail limit (historically 2 MB)
CHROMIUM_ARGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]


def launch(p):
    """a chromium that can do (deterministic, CPU) webgl."""
    return p.chromium.launch(args=CHROMIUM_ARGS)


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
    return f'<div class="bg" style="{_style(st)}"></div>'


def _vignette(v: float) -> str:
    return (f'<div class="vignette" style="background: radial-gradient(ellipse at center, rgba(0,0,0,0) 45%, '
            f'rgba(0,0,0,{v}) 100%)"></div>') if v else ""


def _scene(spec: dict, W: int, H: int, dpr: int) -> str:
    sc = dict(spec["scene"], w=W, h=H, dpr=dpr)
    imports = {"imports": {"three": "/vendor/three/three.module.min.js", "three/addons/": "/vendor/three/addons/"}}
    return (f'<canvas id="scene" class="layer" style="left:0;top:0;width:{W}px;height:{H}px"></canvas>'
            f'<script type="importmap">{json.dumps(imports)}</script>'
            f'<script id="scene-spec" type="application/json">{json.dumps(sc)}</script>'
            '<script type="module" src="/scene.js"></script>')


def html(spec: dict, layers: list | None = None, dpr: int = 1) -> str:
    W, H = spec.get("canvas", [1280, 720])
    bg = spec.get("background") or {}
    world = _scene(spec, W, H, dpr) if spec.get("scene") else _background(bg)
    return Template((TEMPLATES / "stage.html").read_text(encoding="utf-8")).substitute(
        house_css=(TEMPLATES / "house.css").read_text(encoding="utf-8"), w=W, h=H,
        background=world + _vignette(bg.get("vignette", 0.55)),
        layers="\n".join(_layer(L) for L in (spec.get("layers", []) if layers is None else layers)))


def bake_plate(src: Path, out: Path, blur: float = 12, brightness: float = 0.62, saturate: float = 1.15) -> Path:
    """the scene's background plate with the css-style grade baked in (gaussian sigma = blur)."""
    from PIL import Image, ImageEnhance, ImageFilter
    im = Image.open(src).convert("RGB").filter(ImageFilter.GaussianBlur(blur))
    im = ImageEnhance.Color(ImageEnhance.Brightness(im).enhance(brightness)).enhance(saturate)
    im.save(out, "JPEG", quality=92)
    return out


def render(spec_path, hires: bool | None = None, browser=None) -> Path:
    """render <dir>/spec.json -> <dir>/index.html + <dir>/thumb.png (jpeg q92 if over the
    size limit). `browser` = an open playwright Browser (from launch()) to reuse."""
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    d = spec_path.parent
    hires = spec.get("hires", False) if hires is None else hires
    if browser is None:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = launch(p)
            try:
                return _render(b, spec, d, hires)
            finally:
                b.close()
    return _render(browser, spec, d, hires)


def _render(browser, spec: dict, d: Path, hires: bool) -> Path:
    """hook text first (its own raster, refire.thumb.text), then the page with it on top."""
    W, H = spec.get("canvas", [1280, 720])
    dpr = 2 if hires else 1
    layers = list(spec.get("layers", []))
    t = spec.get("text") or {}
    (d / "assets").mkdir(exist_ok=True)
    if t.get("lines"):
        info = T.raster(t, d / "assets" / "hook.png", (W, H), font=spec.get("font"), browser=browser, scale=dpr)
        layers.append({"kind": "hook", "src": "assets/hook.png", "x": info["x"], "y": info["y"], "w": info["w"]})
    sc = spec.get("scene")
    if sc and sc.get("bg", {}).get("raw"):          # grade the plate once, deterministically, in python
        b = sc["bg"]
        bake_plate(d / b["raw"], d / "assets" / "plate.jpg", b.get("blur", 12), b.get("brightness", 0.62),
                   b.get("saturate", 1.15))
        sc["bg"] = {**b, "src": "assets/plate.jpg"}
    page_path = d / "index.html"
    page_path.write_text(html(spec, layers, dpr), encoding="utf-8")
    return _shoot(browser, d, page_path, d / "thumb.png", W, H, hires, bool(sc))


def _serve(route, d: Path):
    """http://refire.local/<path> -> the candidate folder, /vendor/ + /scene.js -> the package."""
    rel = route.request.url[len(ORIGIN):].split("?")[0]
    if rel.startswith("vendor/"):
        f = VENDOR / rel[len("vendor/"):]
    elif rel == "scene.js":
        f = TEMPLATES / "scene.js"
    else:
        f = d / rel
    if not f.is_file():
        return route.fulfill(status=404, body="")
    ctype = "text/javascript" if f.suffix == ".js" else (mimetypes.guess_type(f.name)[0] or "application/octet-stream")
    route.fulfill(status=200, body=f.read_bytes(), headers={"content-type": ctype})


def _shoot(browser, d: Path, page_path: Path, out: Path, W: int, H: int, hires: bool, scene: bool) -> Path:
    page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=2 if hires else 1)
    try:
        page.route(ORIGIN + "**", lambda r: _serve(r, d))
        page.goto(ORIGIN + page_path.name, wait_until="load")
        if scene:
            page.wait_for_function("window.__ready === true", timeout=60000)
            err = page.evaluate("window.__error || null")
            if err:
                raise RuntimeError(f"3d scene failed: {err}")
        page.evaluate("document.fonts.ready.then(() => true)")
        png = page.screenshot(type="png", clip={"x": 0, "y": 0, "width": W, "height": H})
    finally:
        page.close()
    import io

    from PIL import Image
    if hires:                                   # 2x render, lanczos down: crisper edges
        im = Image.open(io.BytesIO(png)).convert("RGB").resize((W, H), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "PNG")
        png = buf.getvalue()
    out.write_bytes(png)
    if len(png) > MAX_BYTES:
        jpg = out.with_suffix(".jpg")
        Image.open(io.BytesIO(png)).convert("RGB").save(jpg, "JPEG", quality=92)
        return jpg
    return out
