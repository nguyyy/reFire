from refire.ae_export import decimate_zoom
from refire.reframe import zoom_track
from refire.subtitles import group_words

FPS = 30


def _words(n, base=100.0, step=0.5):
    return [{"text": f"w{i}", "start": base + i * step, "end": base + i * step + 0.4}
            for i in range(n)]


def test_group_words_grouping_and_zero_based():
    groups = group_words(_words(7), 100.0, 110.0, words_per_line=3)
    assert [len(g) for g in groups] == [3, 3, 1]  # 7 words / 3 per line
    assert groups[0][0]["start"] == 0.0           # zero-based to clip
    assert groups[0][0]["text"] == "w0"
    assert groups[-1][-1]["text"] == "w6"


def test_decimate_zoom_calm_is_flat_100():
    kf = decimate_zoom([1.0] * 60, FPS)
    assert len(kf) <= 3
    assert all(abs(p["scale"] - 100.0) < 1e-6 for p in kf)


def test_decimate_zoom_burst_rises_and_returns():
    inten = [0.0] * 15 + [10.0] * (7 * FPS) + [0.0] * 60
    z = zoom_track(inten, fps=FPS)
    kf = decimate_zoom(z, FPS)
    scales = [p["scale"] for p in kf]
    assert max(scales) > 195            # punches toward 200%
    assert kf[-1]["scale"] < 105        # returns to ~100%
    assert len(kf) < len(z) / 4         # decimated, not one-per-frame
    ts = [p["t"] for p in kf]
    assert ts == sorted(ts)             # monotonic, in-clip times


def test_decimate_zoom_empty():
    assert decimate_zoom([], FPS) == []
