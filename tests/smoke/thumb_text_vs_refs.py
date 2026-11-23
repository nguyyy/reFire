"""Text one-to-one loop: render each ref thumbnail's hook with refire.thumb.text and stack the
ref's text crop (top) over ours (bottom), over the ref's own frame blurred so the text is
the only thing that differs. Writes %TEMP%/refire_thumb_text/<name>.png + an overview.

    uv run python tests/smoke/thumb_text_vs_refs.py [name-substring ...]
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from refire.thumb import text as T  # noqa: E402

REFS = ROOT / "refire" / "thumb" / "refs"
OUT = Path(tempfile.gettempdir()) / "refire_thumb_text"
# what each ref's hook says + how it sits (anchor, width) -- read off the refs by eye
HOOKS = {
    "artifacts+stall.png": {"lines": ["ARTIFACTS", "& STALLING..."], "accent_phrase": "ARTIFACTS",
                            "accent": "gold", "width_frac": 0.71},
    "stronboxthumbnail - Copy.png": {"lines": ["STRONGBOX...", "GONE RIGHT?"], "font": "coolvetica",
                                     "accent_phrase": "RIGHT?", "accent": "green", "annotation": "(birthday)",
                                     "width_frac": 0.60, "squeeze": 1.0},
    "tierlistpng - Copy.png": {"lines": ["THE BEST", "TIER LIST?"], "accent_phrase": "TIER LIST?", "accent": "mint",
                               "anchor": "bottom-right", "width_frac": 0.56, "skew": -12},
    "abysseasyzy0xzyoxthumbnailspiral.png": {"lines": ["EASIEST", "ABYSS?"], "accent_phrase": "EASIEST",
                                             "accent": "green", "width_frac": 0.55},
    "schz.png": {"lines": ["SNEZHNAYA", {"text": "DAY TWO...", "scale_x": 0.92, "scale": 1.15}], "accent_phrase": "SNEZHNAYA",
                 "accent": "sky", "width_frac": 0.66},
    "funnytierlistthumbnailB.png": {"lines": ["MORE", "TIER LISTS"], "accent_phrase": "MORE", "accent": "blue",
                                    "width_frac": 0.65},
    "skirkThumbnail_1.png": {"lines": ["NUTTY", "SKIRK PULLS"], "width_frac": 0.56, "skew": -10},
    "zajrevthumb.png": {"lines": ["ZAJEF ACCOUNT", "REVIEW"], "font": "coolvetica", "squeeze": 1.0,
                        "accent_phrase": "ZAJEF ACCOUNT", "accent": "cyan", "width_frac": 0.66, "skew": -8},
}


# the pipeline's fitted hook plane (refire.thumb.text.PLANE) on every ref's words
for _h in HOOKS.values():
    _h.setdefault("plane", T.PLANE)


def main(argv) -> int:
    OUT.mkdir(exist_ok=True)
    from playwright.sync_api import sync_playwright
    tiles = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        for name, hook in HOOKS.items():
            if argv and not any(a.lower() in name.lower() for a in argv):
                continue
            ref = Image.open(REFS / name).convert("RGB")
            info = T.raster(hook, OUT / f"_{name}.hook.png", browser=b)
            hook_im = Image.open(OUT / f"_{name}.hook.png")
            ours = ref.filter(ImageFilter.GaussianBlur(14))
            ours.paste(hook_im, (round(info["x"]), round(info["y"])), hook_im)
            y0 = int(max(0, min(info["y"] - 20, 720 - 300)))
            box = (0, y0, 1280, 720)
            pair = Image.new("RGB", (1280, (720 - y0) * 2 + 6), "#ff00ff")
            pair.paste(ref.crop(box), (0, 0))
            pair.paste(ours.crop(box), (0, 720 - y0 + 6))
            pair.save(OUT / f"{Path(name).stem}.png")
            tiles.append(pair.resize((640, pair.height // 2)))
        b.close()
    if tiles:
        sheet = Image.new("RGB", (1280, sum(t.height for t in tiles[::2]) + 10), "black")
        y = 0
        for i in range(0, len(tiles), 2):
            sheet.paste(tiles[i], (0, y))
            if i + 1 < len(tiles):
                sheet.paste(tiles[i + 1], (640, y))
            y += max(t.height for t in tiles[i:i + 2])
        sheet.save(OUT / "overview.jpg", quality=88)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
