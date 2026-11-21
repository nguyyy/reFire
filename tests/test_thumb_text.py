"""thumbnail hook text v2 (refire.thumb.text): matched to the refs."""
import re

import pytest

from refire.thumb import text as T

pytestmark = pytest.mark.skipif(not T.DEFAULT_FONT.exists(), reason="Conneqt font not installed")


@pytest.fixture(scope="module")
def m():
    return T.metrics(str(T.DEFAULT_FONT))


def test_metrics_and_tracking(m):
    assert m.upm == 1000 and m.cap == 699
    assert m.width("WW", 100, -0.05) == pytest.approx(2 * m.width("W", 100) - 5)   # one gap, not two
    assert m.width("", 50) == 0


def test_lines_share_a_size_and_center(m):
    st = T.Style(width_frac=0.7)
    a, b = T.layout(["ARTIFACTS", "& STALLING..."], m, (1280, 720), st)
    assert a.size == b.size                                         # same font size, not equal width
    assert a.x + a.width / 2 == pytest.approx(b.x + b.width / 2, abs=0.02)   # centered on each other
    assert b.width + st.stroke * b.size == pytest.approx(0.7 * 1280, abs=0.05)
    assert b.width == pytest.approx(m.width("& STALLING...", b.size, st.tracking) * st.squeeze, abs=0.05)


def test_block_corner_margins_and_stacking(m):
    st = T.Style()
    a, b = T.layout(["EASIEST", "ABYSS?"], m, (1280, 720), st)
    assert b.baseline + st.stroke * b.size / 2 == pytest.approx(720 - st.margin[1], abs=0.05)
    assert min(a.x, b.x) - st.stroke * a.size / 2 == pytest.approx(st.margin[0], abs=0.05)
    assert b.baseline - a.baseline == pytest.approx(b.cap(m) * st.leading, abs=0.05)
    r = T.layout(["EASIEST", "ABYSS?"], m, (1280, 720), st, anchor="bottom-right")
    assert max(ln.x + ln.width for ln in r) + st.stroke * r[0].size / 2 == pytest.approx(1280 - st.margin[0], abs=0.05)


def test_line_options_squeeze_scale_font(m):
    st = T.Style()
    a, b = T.layout(["SNEZHNAYA", {"text": "DAY TWO...", "scale_x": 0.9, "scale": 1.15}], m, (1280, 720), st)
    assert b.size == pytest.approx(a.size * 1.15, abs=0.02) and b.sx == pytest.approx(0.9 * st.squeeze)
    assert b.baseline - a.baseline == pytest.approx(b.cap(m) * st.leading, abs=0.05)   # spaced by the lower line
    if T.font_path("coolvetica").exists():
        (c,) = T.layout([{"text": "GONE RIGHT?", "font": "coolvetica"}], m, (1280, 720), st)
        assert c.font == "coolvetica"


def test_svg_shine_runs_across_and_slab_connects(m):
    st = T.Style()
    lines = T.layout(["MORE", "TIER LISTS"], m, (1280, 720), st)
    s = T.svg(lines, m, (1280, 720), st, accent_phrase="more", accent="blue")
    g = re.search(r'<linearGradient id="ga0"[^>]*x1="([\d.]+)" y1="0" x2="([\d.]+)" y2="0"', s)
    assert g and float(g.group(2)) > float(g.group(1))              # horizontal: left -> right
    assert 'stop-color="#2C66F3"' in s and 'stop-color="#64A1FF"' in s
    rects = [tuple(map(float, r)) for r in re.findall(r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"', s)]
    assert len(rects) == 2 and rects[0][1] + rects[0][3] == pytest.approx(rects[1][1], abs=0.02)   # slabs touch
    assert rects[1][3] == pytest.approx(st.rect * lines[1].cap(m), abs=0.02)
    assert rects[1][0] > lines[1].x                                  # tucked behind the first glyph
    assert '<tspan x="' in s and 'paint-order="stroke"' in s and 'vector-effect="non-scaling-stroke"' in s


def test_skew_and_page_faces(m):
    hook = {"lines": ["THE BEST", "TIER LIST?"], "accent_phrase": "TIER LIST?", "accent": "mint", "skew": -12,
            "annotation": "(birthday)"}
    html = T.page(hook, (1280, 720))
    assert "skewX(-12" in html and "@font-face { font-family: F0" in html
    assert "http" not in re.sub(r'xmlns="http[^"]+"', "", html)
