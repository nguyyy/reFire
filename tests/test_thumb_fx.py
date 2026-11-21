"""pixel reveal effects (refire.thumb.fx) + compose.reveal wiring."""
import json

import numpy as np
from PIL import Image

from refire.thumb import compose, fx


def test_pixelate_blocks_inside_and_leaves_the_rest():
    rng = np.random.default_rng(0)
    im = Image.fromarray(rng.integers(0, 255, (100, 200, 3), dtype=np.uint8))
    out = np.asarray(fx.pixelate(im, (0.25, 0.2, 0.75, 0.8)))
    src = np.asarray(im)
    assert (out[:, :50] == src[:, :50]).all() and (out[:15] == src[:15]).all()   # outside untouched
    block = out[20:30, 50:60]                                   # block = 100 px wide / 10
    assert (block == block[0, 0]).all()


def test_bubble_and_glyph_are_crisp_pixel_art():
    b = np.asarray(fx.bubble("???", cell=9))
    assert b.shape == (14 * 9, 23 * 9, 4)
    vals = {tuple(v) for v in b.reshape(-1, 4)[::97]}
    assert vals <= {(0, 0, 0, 0), (0, 0, 0, 255), (255, 255, 255, 255)}            # no blur, 3 colors
    g = np.asarray(fx.glyph("?", cell=4))
    assert g.shape == (16 * 4, 12 * 4, 4) and (g[..., 3] == 255).sum() > 0


def test_reveal_mosaics_the_card_and_adds_layers(tmp_path):
    a = tmp_path / "assets"
    a.mkdir()
    Image.new("RGB", (300, 200), "red").save(a / "subject.png")
    spec = {"scene": {"cards": [{"src": "assets/subject.png", "x": 0, "y": 0, "w": 300}]}, "layers": []}
    compose.reveal(spec, tmp_path, boxes=[(0.1, 0.1, 0.5, 0.5)], bubble_at=(500, 40), glyphs=[("?", 800, 60, 10)])
    assert spec["scene"]["cards"][0]["src"] == "assets/subject_px.png" and (a / "subject_px.png").exists()
    assert [L["kind"] for L in spec["layers"]] == ["bubble", "glyph"]
    json.dumps(spec)
