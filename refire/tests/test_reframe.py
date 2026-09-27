import numpy as np

from refire.reframe import hold_through_speech, motion_intervals, zoom_track

FPS = 30


def test_motion_intervals_calm_is_empty():
    assert motion_intervals([0.0] * 60, fps=FPS) == []
    assert motion_intervals([], fps=FPS) == []


def test_motion_intervals_one_burst_in_seconds():
    inten = [0.0] * 15 + [10.0] * (7 * FPS) + [0.0] * 60
    ivs = motion_intervals(inten, fps=FPS)
    assert len(ivs) == 1
    s, e = ivs[0]
    assert 0.3 <= s <= 0.7 and 7.0 <= e <= 8.0   # ~0.5s .. ~7.5s


def test_motion_intervals_merge_and_split():
    near = [0.0] * 15 + [10.0] * 60 + [0.0] * 30 + [10.0] * 60 + [0.0] * 60
    assert len(motion_intervals(near, fps=FPS)) == 1        # 1s gap < bridge -> merged
    far = [0.0] * 15 + [10.0] * 45 + [0.0] * 90 + [10.0] * 45 + [0.0] * 60
    assert len(motion_intervals(far, fps=FPS)) == 2         # 3s gap -> two episodes


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


def test_hold_through_speech_extends_end_to_phrase_end():
    # episode ends at 2.0 mid-phrase (speech 1.0-3.0) -> end pushed to 3.0
    assert hold_through_speech([(0.5, 2.0)], [(1.0, 3.0)], limit=10.0) == [(0.5, 3.0)]


def test_hold_through_speech_no_speech_unchanged():
    assert hold_through_speech([(0.5, 2.0)], [], 10.0) == [(0.5, 2.0)]


def test_hold_capped_by_next_episode():
    out = hold_through_speech([(0.0, 2.0), (5.0, 6.0)], [(1.0, 9.0)], 10.0)
    assert out[0][1] == 5.0          # can't slide past the next episode's start


def test_motion_intervals_holds_zoom_through_speech():
    inten = [0.0] * 15 + [10.0] * 45 + [0.0] * 120     # burst ~0.5-2.0s, then calm
    base = motion_intervals(inten, fps=FPS)[0][1]
    held = motion_intervals(inten, fps=FPS, speech_intervals=[(1.0, 3.0)])[0][1]
    assert held > base and held >= 3.0 - 1e-6          # released at phrase end, not 2.0
