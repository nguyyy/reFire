"""thumbnail hook text: metrics, equal-width fit, layout, svg (refire.thumb.text / render)."""
import re

import pytest

from refire.thumb import render, text as T

pytestmark = pytest.mark.skipif(not T.DEFAULT_FONT.exists(), reason="Conneqt font not installed")


@pytest.fixture(scope="module")
def m():
    return T.metrics(str(T.DEFAULT_FONT))


def test_metrics_read_from_the_font(m):
    assert m.upm == 1000 and m.cap == 699
    assert m.width("WW", 100) == pytest.approx(2 * m.width("W", 100))
    assert m.width("", 50) == 0


def test_lines_fit_to_equal_stroked_width(m):
    lines = T.layout(["ARTIFACTS", "& STALLING..."], m, (1280, 720), width_frac=0.62, stroke_ratio=0.11)
    stroked = [ln.width + 0.11 * ln.size for ln in lines]
    assert stroked == pytest.approx([0.62 * 1280] * 2, rel=1e-3)
    assert lines[0].size > lines[1].size                       # shorter line gets the bigger font


def test_block_sits_in_the_bottom_corner_margins(m):
    lines = T.layout(["EASIEST", "ABYSS?"], m, (1280, 720), margin=40, stroke_ratio=0.11)
    last = lines[-1]
    assert last.baseline + 0.11 * last.size / 2 == pytest.approx(720 - 40, abs=0.05)
    assert min(ln.x - 0.11 * ln.size / 2 for ln in lines) == pytest.approx(40, abs=0.05)
    assert lines[0].baseline < lines[1].baseline - lines[1].cap(m) + 1      # stacked, upper above
    right = T.layout(["EASIEST", "ABYSS?"], m, (1280, 720), anchor="bottom-right", margin=40)
    assert max(ln.x + ln.width + 0.11 * ln.size / 2 for ln in right) == pytest.approx(1240, abs=0.05)


def test_one_word_line_is_capped(m):
    (ln,) = T.layout(["HI"], m, (1280, 720), max_size_frac=0.24)
    assert ln.size == pytest.approx(0.24 * 720)


def test_svg_has_gradients_backing_rects_and_accent(m):
    lines = T.layout(["MORE", "TIER LISTS"], m, (1280, 720))
    s = T.svg(lines, m, (1280, 720), accent_phrase="more", accent="blue", rect_ratio=0.62)
    assert s.count("<rect") == 2 and s.count("<linearGradient") == 4
    assert 'stop-color="#2C66F3"' in s and 'stop-color="#64A1FF"' in s       # blue accent ramp
    assert 'paint-order="stroke"' in s and 'stroke-linejoin="round"' in s
    assert re.search(r'<tspan fill="url\(#ga0\)">MORE</tspan>', s)          # accent only on MORE
    assert '<tspan fill="url(#gw1)">TIER LISTS</tspan>' in s
    h = [float(x) for x in re.findall(r'<rect [^>]*height="([\d.]+)"', s)]
    assert h[1] == pytest.approx(0.62 * lines[1].cap(m), abs=0.01)           # shorter than the caps


def test_accent_split_and_escaping():
    assert T._split_accent("GONE RIGHT?", "right?") == [("GONE ", False), ("RIGHT?", True)]
    assert T._split_accent("A & B", None) == [("A & B", False)]
    assert "&amp;" in T.svg([T.Line("A & B", 50, 40, 600, 100)], T.Metrics(1000, 700, {}, 500), (1280, 720))


def test_html_is_self_contained_and_spec_driven(tmp_path):
    spec = {"canvas": [1280, 720], "background": {"src": "assets/bg.png"},
            "layers": [{"kind": "face", "src": "assets/f.png", "x": 700, "y": 100, "w": 500, "glow": "#fff"},
                       {"kind": "subject", "src": "assets/s.png", "x": 60, "y": 40, "w": 400, "rotate": -3,
                        "shadow": 0.6}],
            "text": {"lines": ["EASIEST", "ABYSS?"], "accent_phrase": "EASIEST", "accent": "green"}}
    h = render.html(spec)
    assert "data:font/ttf;base64," in h and not re.search(r"""(src="|url\(')https?:""", h)   # nothing fetched
    assert "blur(12px) brightness(0.62) saturate(1.15)" in h and "scale(1.08)" in h
    assert "rotate(-3deg)" in h and "drop-shadow(0 0 3px #fff) drop-shadow(0 0 22px #fff)" in h
    assert h.index('class="layer face"') < h.index('class="hook"')            # text on top
