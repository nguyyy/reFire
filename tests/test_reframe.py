import numpy as np

from refire.reframe import smooth_path, zoom_envelope


def test_zoom_rises_on_action_and_releases():
    # calm, then a burst of motion, then calm again
    inten = [0.0] * 20 + [10.0] * 20 + [0.0] * 40
    z = zoom_envelope(inten, zmax=1.5, attack=0.35, release=0.06)
    assert z[0] == 1.0
    assert z[39] > 1.3                 # zoomed in during the burst
    assert z[-1] < z[39]               # releases back out afterwards
    assert z[-1] < 1.1                 # nearly fully zoomed out by the end
    assert np.all(z >= 1.0) and np.all(z <= 1.5)


def test_attack_faster_than_release():
    inten = [0.0, 10.0, 0.0]
    fast = zoom_envelope(inten, attack=0.9, release=0.01)
    # big jump up on the spike, tiny drop after
    rise = fast[1] - fast[0]
    fall = fast[1] - fast[2]
    assert rise > fall


def test_smooth_path_lags_toward_target():
    cents = [(0.5, 0.5)] + [(1.0, 0.0)] * 10
    p = smooth_path(cents, coeff=0.15)
    assert p[0][0] == 0.5
    assert 0.5 < p[5][0] < 1.0         # eased, not snapped
    assert p[-1][0] > p[5][0]          # still approaching target


def test_empty_inputs():
    assert zoom_envelope([]).size == 0
    assert smooth_path([]).size == 0
