import numpy as np

from refire.reframe import zoom_track


def test_calm_stays_at_100_percent_no_shake():
    z = zoom_track([0.0] * 60, fps=30)
    assert np.allclose(z, 1.0)  # dead calm -> no zoom at all, perfectly flat


def test_burst_zooms_in_then_back_out():
    inten = [0.0] * 20 + [10.0] * 30 + [0.0] * 60
    z = zoom_track(inten, fps=30, zmax=2.0)
    assert z[0] == 1.0
    assert z[48] > 1.8          # punched in ~200% during the burst
    assert z[-1] < 1.1          # quick-zoomed back to ~100% after


def test_stable_during_sustained_burst_no_oscillation():
    z = zoom_track([10.0] * 90, fps=30)
    tail = z[40:]               # after the ramp settles
    assert tail.std() < 0.01    # flat hold near 200%, not shaking
    assert tail.mean() > 1.9


def test_hysteresis_holds_through_a_brief_dip():
    # enter, then a short dip that's below ENTER but above EXIT -> stays zoomed
    inten = [0.0] * 10 + [10.0] * 15 + [2.0] * 5 + [10.0] * 15 + [0.0] * 30
    z = zoom_track(inten, fps=30, enter=0.45, exit=0.18, min_hold_s=0.8)
    # during the dip (around frame 27) it should still be zoomed in
    assert z[28] > 1.5


def test_empty_input():
    assert zoom_track([]).size == 0
