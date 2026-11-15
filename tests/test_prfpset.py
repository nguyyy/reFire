"""prfpset parser against Nguy's two presets; expected values from docs/specs/02-caption.md."""
from pathlib import Path

import pytest

from refire.ppro import prfpset

ASSETS = Path(__file__).resolve().parents[1] / "refire" / "assets" / "premiere"


@pytest.fixture(scope="module")
def cap():
    (fx,) = prfpset.load(ASSETS / "captionIn.prfpset")
    return fx


@pytest.fixture(scope="module")
def cc():
    return prfpset.load(ASSETS / "ccStyle.prfpset")


def test_caption_in_effect(cap):
    assert cap.match_name == "AE.ADBE Geometry2" and cap.display_name == "Transform"
    assert cap.type == 0
    assert cap.duration == pytest.approx(0.45)


def test_caption_in_statics(cap):
    by_id = {p.param_id: p for p in cap.params}
    assert cap.param("Anchor Point").value == [0.5, 0.5]
    assert by_id[11].value is True and by_id[11].name is None and by_id[11].index == 2   # uniform scale
    assert by_id[9].value is True and by_id[9].index == 9                                  # use comp shutter
    assert cap.param("Scale Height").value == cap.param("Scale Width").value == 100.0
    for n in ("Skew", "Skew Axis", "Rotation"):
        assert cap.param(n).value == 0.0
    assert cap.param("Shutter Angle").value == 360.0
    assert cap.param("Sampling").value == 0
    # params index order = Premiere's properties[j] order (spike S7)
    assert [p.index for p in cap.params] == list(range(12))
    assert cap.param("Opacity").index == 8 and cap.param("Position").index == 1


def test_caption_in_keys(cap):
    pos = cap.param("Position").keys
    assert [round(k.t, 3) for k in pos] == [0.0, 0.247]
    assert pos[0].value == pytest.approx([0.5, 0.5324], abs=1e-4) and pos[1].value == [0.5, 0.5]
    op = cap.param("Opacity").keys
    assert [round(k.t, 3) for k in op] == [0.0, 0.058, 0.267, 0.435]
    assert [k.value for k in op] == [0.0, 100.0, 100.0, 0.0]
    assert {k.interp for k in pos + op} == {5}
    assert op[0].tangents[:3] == ["0", "0", "0.16666666666666666"]     # kept raw


def test_cc_style_order_and_skips(cc):
    assert [e.match_name for e in cc] == ["AE.Impact_Camera_Shake_FX", "AE.ADBE Drop Shadow",
                                          "AE.Impact_Edge_Glow_FX"]
    shake, _, glow = cc
    assert len(shake.params) == 47 and len(glow.params) == 28          # = DOM property counts (S5)
    skipped = {p.name for p in shake.params if p.skip}
    assert {"_ Applied Version", "Motion Controls", "Pre Transform", None} <= skipped
    assert not shake.param("Seed").skip and shake.param("Seed").value == 0.0


def test_camera_shake(cc):
    s = cc[0]
    want = {"Camera Mode": 1, "Speed": 25.0, "Master": 100.0, "Variation": 20.0, "Stabilize": 25.0,
            "Lean (deg)": 0.0, "Auto Scale": True, "Enable Motion Blur": True, "Motion Blur": 20.0,
            "Edge Behavior": 0, "Seed": 0.0, "Apply Prescale": False, "Scale": 50.0}
    assert {n: s.param(n).value for n in want} == want
    triplets = [p.value for p in s.params if p.name in ("Strafe", "Stride", "Roll")]
    assert triplets == pytest.approx([5.87, 10.34, 3.95, 8.48, 9.45, 0, 71.49, 56.45, 40.84,
                                      48.24, 55.83, 57.46, 64, 76, 22.22], abs=0.01)


def test_drop_shadow(cc):
    d = cc[1]
    assert d.param("Shadow Color").value == [255, 0, 0, 0]
    assert {n: d.param(n).value for n in ("Opacity", "Direction", "Distance", "Softness", "Shadow Only")} == \
        {"Opacity": 165.75, "Direction": 135.0, "Distance": 5.0, "Softness": 0.0, "Shadow Only": False}


def test_edge_glow(cc):
    g = cc[2]
    assert g.param("Color").value == [255, 0xFF, 0xDF, 0xB3] and g.param("Color").index == 11
    want = {"Intensity": 40.0, "Highlights Only": 40.0, "Size": 15.0, "Width": 15.0, "Height": 15.0,
            "Uniform Size": True, "Angle": 0.0, "Colorize": 0.0, "Vibrance": 10.0, "Desaturate": 0.0,
            "Chromatic Aberration": 10.0, "Blend Mode": 0, "Source Opacity": 100.0}
    assert {n: g.param(n).value for n in want} == want
    assert g.param("Intensity").index == 4


def test_color_decode_uses_high_byte():
    v = (0xFFFF << 48) | (0x12AB << 32) | (0x00FF << 16) | 0x8001
    assert prfpset._value(str(v), prfpset.COLOR) == [0xFF, 0x12, 0x00, 0x80]
