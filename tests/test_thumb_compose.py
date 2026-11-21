"""reaction layout placement (refire.thumb.compose), pure math."""
import pytest

from refire.thumb import compose


def test_figure_fills_88_percent_from_head_to_frame_bottom():
    # cutout 480x420, head starts at row 30, face box 120x150 at (180, 90)
    f = compose.place_face((480, 420), [180, 90, 120, 150], top=30, side="right")
    s = f["w"] / 480
    assert f["y"] + 420 * s == pytest.approx(720, abs=0.5)                  # cut edge on the frame bottom
    assert 720 - (f["y"] + 30 * s) == pytest.approx(0.88 * 720, abs=0.5)    # head top -> bottom = 88%
    assert f["face"][0] + f["face"][2] / 2 == pytest.approx(0.74 * 1280, abs=0.5)


def test_tight_crop_caps_the_face():
    f = compose.place_face((480, 200), [180, 20, 120, 150], top=10)
    assert f["face"][3] <= compose.FACE_MAX * 720 + 0.5


def test_left_side_mirrors():
    f = compose.place_face((480, 420), [180, 90, 120, 150], top=30, side="left")
    assert f["face"][0] + f["face"][2] / 2 == pytest.approx(0.26 * 1280, abs=0.5)


def test_head_top_reads_the_alpha():
    import numpy as np
    a = np.zeros((100, 50, 4), np.uint8)
    a[37:, 10:40, 3] = 255
    assert compose.head_top(a) == 37


def test_subject_fits_its_box():
    s = compose.place_subject((800, 400))
    assert s["w"] == pytest.approx(0.36 * 1280) and s["x"] == pytest.approx(64)
    tall = compose.place_subject((300, 900))
    assert tall["w"] / 300 * 900 == pytest.approx(0.60 * 720)


def test_badge_zone():
    assert compose.badge_clear([800, 300, 200, 300])                    # face above the badge row
    assert not compose.badge_clear([1150, 650, 80, 60])                 # inside bottom-right 16% x 12%
    assert compose.badge_clear([100, 650, 300, 60])                     # bottom-left is fine


def test_reaction_spec_shape(tmp_path):
    from PIL import Image
    a = tmp_path / "assets"
    a.mkdir()
    Image.new("RGBA", (480, 420), (0, 0, 0, 255)).save(a / "face.png")
    Image.new("RGB", (400, 600)).save(a / "subject.png")
    Image.new("RGB", (1920, 1080)).save(a / "bg.jpg")
    s = compose.reaction_spec(a / "face.png", [180, 90, 120, 150], a / "subject.png", a / "bg.jpg",
                              {"lines": ["MAX WIN", "IN ONE ROLL?"], "accent_phrase": "MAX WIN", "accent": "gold"})
    assert [L["kind"] for L in s["layers"]] == ["subject", "face"]       # face drawn over the subject
    assert s["text"]["anchor"] == "bottom-left" and s["background"]["src"] == "assets/bg.jpg"


def test_render_page_puts_the_hook_raster_on_top():
    from refire.thumb import render
    spec = {"canvas": [1280, 720], "background": {"src": "assets/bg.jpg"},
            "layers": [{"kind": "face", "src": "assets/face.png", "x": 700, "y": 100, "w": 500, "glow": "#fff"}]}
    h = render.html(spec, spec["layers"] + [{"kind": "hook", "src": "assets/hook.png", "x": 30, "y": 500, "w": 800}])
    assert "blur(12px) brightness(0.62) saturate(1.15)" in h and "drop-shadow(0 0 3px #fff)" in h
    assert h.index('class="layer face"') < h.index('class="layer hook"')
