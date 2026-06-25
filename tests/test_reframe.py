import numpy as np

from refire.reframe import zoom_track

FPS = 30


def test_calm_stays_at_100_percent():
    z = zoom_track([0.0] * 60, fps=FPS)
    assert np.allclose(z, 1.0)


def test_tapered_punch_then_slow_creep():
    # one 7s interesting moment
    inten = [0.0] * 15 + [10.0] * (7 * FPS) + [0.0] * 60
    z = zoom_track(inten, fps=FPS)
    s = 15
    # ~190% reached by the end of the 0.5s punch
    assert 1.85 <= z[s + int(0.5 * FPS)] <= 1.95
    # then keeps creeping up toward 200% over the rest of the moment
    assert z[s + 6 * FPS] > z[s + 1 * FPS]
    assert z[s + 7 * FPS - 1] > 1.97
    # punch slope (0.9 over 0.5s) is far steeper than creep slope (0.1 over 6.5s)
    punch_slope = (z[s + int(0.5 * FPS)] - z[s]) / (0.5 * FPS)
    creep_slope = (z[s + 7 * FPS - 1] - z[s + int(0.5 * FPS)]) / (6.5 * FPS)
    assert punch_slope > creep_slope * 5


def test_zooms_back_out_after_moment():
    inten = [0.0] * 15 + [10.0] * (3 * FPS) + [0.0] * 90
    z = zoom_track(inten, fps=FPS)
    assert z[-1] < 1.05


def test_stacked_zooms_merge_through_short_gap():
    # burst, short calm gap (1s < BRIDGE_GAP_S 1.5s), burst again
    inten = [0.0] * 15 + [10.0] * 60 + [0.0] * 30 + [10.0] * 60 + [0.0] * 60
    z = zoom_track(inten, fps=FPS)
    # in the middle of the gap the zoom should NOT have dropped back to 100%
    gap_mid = 15 + 60 + 15
    assert z[gap_mid] > 1.5


def test_far_apart_zooms_do_not_merge():
    # two bursts separated by a long calm gap (3s > BRIDGE_GAP_S)
    inten = [0.0] * 15 + [10.0] * 45 + [0.0] * 90 + [10.0] * 45 + [0.0] * 60
    z = zoom_track(inten, fps=FPS)
    gap_mid = 15 + 45 + 45
    assert z[gap_mid] < 1.05  # fully zoomed out between them


def test_empty_input():
    assert zoom_track([]).size == 0
