"""YAMNet frontend + laughter runs (refire.laughter). The forward pass runs only when the
official weights are already cached (tests never download)."""
import numpy as np
import pytest

from refire import laughter


def test_mel_matrix_is_tf_shaped():
    m = laughter._MEL
    assert m.shape == (257, 64) and not m[0].any()            # dc bin zeroed, like tf
    assert (m >= 0).all() and (m.max(axis=0) > 0.5).all()      # every band has a peak


def test_patches_follow_yamnet_padding():
    assert laughter.patches(np.zeros(100, np.float32)).shape == (1, 96, 64)   # short -> one patch
    assert laughter.patches(np.zeros(15600, np.float32)).shape == (1, 96, 64)
    assert laughter.patches(np.zeros(15601, np.float32)).shape == (2, 96, 64)  # pads to a full hop
    assert laughter.patches(np.zeros(16000 * 10, np.float32)).shape[0] == 20   # (10 - 0.975) / 0.48 -> 19 + 1


def test_same_pad_matches_tf():
    x = np.zeros((1, 96, 64, 1), np.float32)
    assert laughter.YAMNet._same_pad(x, 2).shape == (1, 97, 65, 1)             # stride 2: one extra, bottom/right
    assert laughter.YAMNet._same_pad(x, 1).shape == (1, 98, 66, 1)


def test_one_patch_is_never_sustained():
    t = laughter.HOP_S * np.arange(8)
    s = np.array([0.1, 0.9, 0.2, 0.5, 0.6, 0.1, 0.4, 0.05])
    segs = laughter.segments(t, s, thr=0.3)
    assert segs == [(0.48, 1.44, 0.9), (1.44, 2.88, 0.6), (2.88, 3.84, 0.4)]
    durs = [b - a for a, b, _ in segs]
    assert durs[0] < 1.0 and durs[1] == pytest.approx(1.44)                   # 0.96 s chuckle vs 1.44 s laugh


def test_merge_pads_and_joins_ranges():
    assert laughter._merge([(10, 12), (12.5, 20), (40, 41)], 1.0) == [[9, 21], [39, 42]]


@pytest.mark.skipif(not (laughter.MODELS / laughter.WEIGHTS[1]).exists(), reason="yamnet weights not cached")
def test_forward_pass_shape_and_silence():
    net = laughter.YAMNet()
    sc = net.scores(np.zeros(16000 * 2, np.float32))
    assert sc.shape == (4, 521) and 0 <= sc.min() and sc.max() <= 1
    assert net.names[int(sc.mean(0).argmax())] == "Silence"
