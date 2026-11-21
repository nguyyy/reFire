"""reaction layout placement (refire.thumb.compose), pure math."""
import pytest

from refire.thumb import compose


def test_face_hits_target_height_on_the_right_third():
    # cam crop 480x420, face box 120x150 at (180, 90); plenty of body below -> no growth needed
    f = compose.place_face((480, 1200), [180, 90, 120, 150], side="right", target=0.38)
    assert f["face"][3] == pytest.approx(0.38 * 720, abs=0.5)
    assert f["face"][0] + f["face"][2] / 2 == pytest.approx(0.74 * 1280, abs=0.5)
    assert f["y"] + f["w"] / 480 * 1200 >= 720                          # body leaves the frame


def test_short_crop_grows_then_drops_so_no_bottom_edge_shows():
    f = compose.place_face((480, 420), [180, 90, 120, 150], target=0.38)
    s = f["w"] / 480
    assert f["face"][3] / 720 <= compose.FACE_H[1] + 1e-6               # never past 45%
    assert f["y"] + 420 * s >= 720 - 0.5                                # cut edge at/below the frame


def test_left_side_mirrors():
    f = compose.place_face((480, 1200), [180, 90, 120, 150], side="left")
    assert f["face"][0] + f["face"][2] / 2 == pytest.approx(0.26 * 1280, abs=0.5)


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
    Image.new("RGBA", (480, 420)).save(a / "face.png")
    Image.new("RGB", (400, 600)).save(a / "subject.png")
    Image.new("RGB", (1920, 1080)).save(a / "bg.jpg")
    s = compose.reaction_spec(a / "face.png", [180, 90, 120, 150], a / "subject.png", a / "bg.jpg",
                              {"lines": ["MAX WIN", "IN ONE ROLL?"], "accent_phrase": "MAX WIN", "accent": "gold"})
    assert [L["kind"] for L in s["layers"]] == ["subject", "face"]       # face drawn over the subject
    assert s["text"]["anchor"] == "bottom-left" and s["background"]["src"] == "assets/bg.jpg"
