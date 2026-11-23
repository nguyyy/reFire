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


def test_cap_frac_sets_the_size_unless_the_line_is_too_wide(m):
    st = T.Style(cap_frac=0.152, width_frac=0.74)
    a, b = T.layout(["MORE", "TIER LISTS"], m, (1280, 720), st)
    assert b.cap(m) == pytest.approx(0.152 * 720, abs=0.05)         # short lines: the refs' cap height
    wide = T.layout(["GENSHIN OWED ME", "THIS PIECE"], m, (1280, 720), st)
    assert wide[0].cap(m) < 0.152 * 720                              # a long line shrinks to fit width_frac
    assert wide[0].width + st.stroke * wide[0].size == pytest.approx(0.74 * 1280, abs=0.05)


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
    assert s.count("<stop", s.index('id="ga0"'), s.index('id="gw1"')) == 9          # sampled ramp, not 3 stops
    rects = [tuple(map(float, r)) for r in re.findall(r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"', s)]
    assert len(rects) == 2 and rects[0][1] + rects[0][3] == pytest.approx(rects[1][1], abs=0.02)   # slabs touch
    assert rects[1][3] == pytest.approx(lines[1].cap(m) - 2 * st.rect_inset, abs=0.02)   # ~5 px under the caps
    assert rects[1][0] > lines[1].x                                  # tucked behind the first glyph
    assert '<tspan x="' in s and 'vector-effect="non-scaling-stroke"' in s
    # every stroke sits under every fill: one stroke layer for all lines, one fill layer above it
    st_i, fi_i = s.index('<g class="strokes"'), s.index('<g class="fills" stroke="none">')
    assert st_i < fi_i and s.count("<text", st_i, fi_i) == 2 and s.count("<text", fi_i) == 2
    assert "url(#g" not in s[st_i:fi_i] and "stroke-width" not in s[fi_i:]


def test_skew_and_page_faces(m):
    hook = {"lines": ["THE BEST", "TIER LIST?"], "accent_phrase": "TIER LIST?", "accent": "mint", "skew": -12,
            "annotation": "(birthday)"}
    html = T.page(hook, (1280, 720))
    assert "skewX(-12" in html and "@font-face { font-family: F0" in html
    assert "http" not in re.sub(r'xmlns="http[^"]+"', "", html)


def test_slab_stops_before_trailing_dots(m):
    st = T.Style()
    (ln,) = T.layout(["STALLING..."], m, (1280, 720), st)
    s = T.svg([ln], m, (1280, 720), st)
    x, w = map(float, re.search(r'<rect x="([\d.]+)" y="[\d.]+" width="([\d.]+)"', s).groups())
    adv = m.advances_px("STALLING...", ln.size, st.tracking)
    g_end = ln.x + sum(adv[:8]) * ln.sx                               # right edge of the G
    assert x + w < g_end                                               # tucked inside the G, not over the dots


def test_straight_apostrophe_uses_the_fonts_curly_glyph(m):
    assert "'" not in m.advances and "’" in m.advances                 # conneqt: curly only
    (ln,) = T.layout(["I'M NOT"], m, (1280, 720), T.Style())
    assert ln.text == "I’M NOT"
    assert ln.width == pytest.approx(m.width("I’M NOT", ln.size, T.Style().tracking) * ln.sx, abs=0.05)


def test_shine_ramps_from_dark_edges_to_a_light_middle():
    import colorsys
    st = T.Style()
    for acc in (None, T.PALETTE["gold"]):
        stops = T.shine_stops(acc, st)
        L = [colorsys.rgb_to_hls(*(int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)))[1] for _, c in stops]
        peak = L.index(max(L))
        assert stops[peak][0] == pytest.approx(st.shine_peak, abs=0.07)
        assert all(a <= b + 1e-3 for a, b in zip(L[:peak], L[1:peak + 1]))                  # rises to the peak
        assert max(L) - (L[0] + L[-1]) / 2 > 0.25                                           # a real shine


def test_keystone_keeps_width_and_anchor_and_scales_the_far_end(m):
    import numpy as np
    lines = T.layout(["MORE", "TIER LISTS"], m, (1280, 720), T.Style())
    x0, x1 = min(ln.x for ln in lines), max(ln.x + ln.width for ln in lines)
    yt, yb = lines[0].baseline - lines[0].cap(m), lines[-1].baseline

    def at(H, x, y):
        u, v, w = H @ np.array([x, y, 1.0])
        return u / w, v / w
    for anchor, near, far in (("bottom-left", x0, x1), ("bottom-right", x1, x0)):
        H = T.keystone(lines, 1.2, anchor)
        for y in (yt, yb):
            assert np.allclose(at(H, near, y), (near, y))                  # anchored edge untouched
        (fx, ft), (_, fb) = at(H, far, yt), at(H, far, yb)
        assert abs(fx - far) < 1e-6 and abs((fb - ft) - 1.2 * (yb - yt)) < 1e-6   # width kept, far end x1.2
    assert "matrix3d(" in T.page({"lines": ["MORE", "TIER LISTS"], "keystone": 0.88}, (1280, 720))
    assert "matrix3d(" not in T.page({"lines": ["MORE", "TIER LISTS"]}, (1280, 720))
