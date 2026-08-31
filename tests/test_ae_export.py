import json

from refire.ae_export import build_manifest, decimate_zoom
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

def test_build_manifest_can_skip_motion_scan(monkeypatch, tmp_path):
    """motion_zoom=False writes static-fit clips without touching OpenCV/video frames."""
    monkeypatch.setattr("refire.ae_export.clip_intensity",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("scan called")))
    words = [
        {"text": "Hello", "start": 0.0, "end": 0.4, "emph": False},
        {"text": "there.", "start": 0.5, "end": 0.9, "emph": False},
    ]
    sections = [{"title": "Open", "clips": [{"start": 0.0, "end": 1.0}]}]

    mp = build_manifest("missing-video.mp4", tmp_path, words, sections, motion_zoom=False)
    data = json.loads(mp.read_text(encoding="utf-8"))

    assert data["clips"][0]["zoom_episodes"] == []


def _deadspace_manifest(tmp_path, **kw):
    """One 20s clip with 15s of dead air in the middle -> manifest dict."""
    words = [{"text": "Hello", "start": 0.0, "end": 0.5, "emph": False},
             {"text": "there.", "start": 0.6, "end": 1.0, "emph": False},
             {"text": "Back.", "start": 16.0, "end": 16.5, "emph": False}]
    sections = [{"title": "Open", "clips": [{"start": 0.0, "end": 20.0}]}]
    overlays = [[{"start": 16.0, "duration": 3.0, "asset": "e.png", "sfx": None}]]
    mp = build_manifest("missing-video.mp4", tmp_path, words, sections,
                        motion_zoom=False, overlays_by_clip=overlays, **kw)
    return json.loads(mp.read_text(encoding="utf-8"))["clips"][0]


def test_build_manifest_cuts_dead_space(tmp_path):
    clip = _deadspace_manifest(tmp_path)
    assert len(clip["keep"]) == 2                      # the 15s gap was cut out
    assert clip["dur"] < 6.0, clip["dur"]              # ~2 phrases + breath, not 20s
    assert abs(clip["dur"] - sum(b - a for a, b in clip["keep"])) < 1e-3
    assert clip["start"] == 0.0 and clip["end"] == 20.0   # source times stay absolute
    # captions + the overlay ride the tightened timeline, not the source one
    assert clip["captions"][-1]["end"] <= clip["dur"] + 1e-6
    assert clip["overlays"][0]["start"] < clip["dur"]


def test_build_manifest_deadspace_off_keeps_the_gap(tmp_path):
    clip = _deadspace_manifest(tmp_path, deadspace=False)
    assert "keep" not in clip and "dur" not in clip
    assert clip["overlays"][0]["start"] == 16.0        # untouched source-relative time


def test_build_manifest_cards_toggle(tmp_path):
    words = [{"text": "Hi.", "start": 0.0, "end": 0.4, "emph": False}]
    sections = [{"title": "Open", "clips": [{"start": 0.0, "end": 1.0}]}]

    def secs(**kw):
        mp = build_manifest("missing-video.mp4", tmp_path, words, sections,
                            motion_zoom=False, **kw)
        return json.loads(mp.read_text(encoding="utf-8"))["sections"]

    assert "card" not in secs()[0]                  # titled section keeps its card
    off = secs(cards=False)[0]
    assert off["card"] is False and off["title"] == "Open"   # title survives for debugging


def test_build_manifest_points_at_intra_proxies(monkeypatch, tmp_path):
    """proxy=True -> one all-intra proxy per clip, each clip's `src`/offset pointing at it."""
    cuts = []

    def fake_proxy(video, proxy_dir, clips, pad=2.0):
        cuts.extend(clips)
        return [(tmp_path / f"p{i}.mov", max(0.0, c["start"] - pad))
                for i, c in enumerate(clips)]

    monkeypatch.setattr("refire.ae_export.proxy_clips", fake_proxy)
    words = [{"text": "Hi.", "start": 0.0, "end": 0.4, "emph": False},
             {"text": "Yo.", "start": 100.0, "end": 100.4, "emph": False}]
    sections = [{"title": "Open", "clips": [{"start": 0.0, "end": 1.0},
                                            {"start": 100.0, "end": 101.0}]}]

    mp = build_manifest("missing-video.mp4", tmp_path, words, sections, motion_zoom=False)
    data = json.loads(mp.read_text(encoding="utf-8"))

    assert len(cuts) == 2                                   # only the used spans transcoded
    assert [c["src"] for c in data["clips"]] == [0, 1]
    assert data["sources"][1]["offset"] == 98.0             # padded head
    # AE subtracts the offset, so the 2nd clip starts 2s into its own proxy
    assert data["clips"][1]["start"] - data["sources"][1]["offset"] == 2.0


# --- long-VOD splitting: AE cannot address past 3h of one file ---------------


def test_split_points_never_tear_a_clip():
    from refire.ae_export import _split_points

    seg = 9000.0
    clips = [{"start": 8990.0, "end": 9050.0},    # sits right on the 1st cut
             {"start": 17900.0, "end": 17960.0}]  # nowhere near the 2nd
    pts = _split_points(21600.0, clips, seg)
    assert len(pts) == 2, pts
    assert pts[0] < 8990.0, pts                   # walked back out of the clip
    assert pts[1] == 18000.0, pts                 # untouched
    for t in pts:
        assert not any(c["start"] <= t <= c["end"] for c in clips), (t, pts)


def test_assign_parts_tags_the_right_part():
    from refire.ae_export import _assign_parts

    parts = [("a.mp4", 0.0), ("b.mp4", 8985.0), ("c.mp4", 18000.0)]
    clips = [{"start": 10.0, "end": 40.0},
             {"start": 8990.0, "end": 9050.0},
             {"start": 19000.0, "end": 19060.0}]
    _assign_parts(clips, parts)
    assert [c["src"] for c in clips] == [0, 1, 2]
    # single part -> no `src` key at all (manifest stays byte-identical)
    solo = [{"start": 10.0, "end": 40.0}]
    _assign_parts(solo, [("a.mp4", 0.0)])
    assert "src" not in solo[0]


def test_captions_emph_keeps_only_the_lines_that_need_text(tmp_path):
    """A full subtitle track competes with the cuts in a fast style; sparse text just
    prevents the confusion the cutting can't."""
    words = [
        {"text": "just", "start": 0.0, "end": 0.4, "emph": False},
        {"text": "talking.", "start": 0.5, "end": 0.9, "emph": False},
        {"text": "WHAT", "start": 1.0, "end": 1.4, "emph": True},
        {"text": "no.", "start": 1.5, "end": 1.9, "emph": False},
    ]
    sections = [{"title": "Open", "clips": [{"start": 0.0, "end": 2.0}]}]
    kw = dict(motion_zoom=False, deadspace=False, words_per_line=2)

    every = json.loads(build_manifest("v.mp4", tmp_path / "a", words, sections,
                                      **kw).read_text(encoding="utf-8"))
    sparse = json.loads(build_manifest("v.mp4", tmp_path / "b", words, sections,
                                       captions="emph", **kw).read_text(encoding="utf-8"))
    assert len(every["clips"][0]["captions"]) == 2          # both pairs captioned
    kept = sparse["clips"][0]["captions"]
    assert len(kept) == 1 and "WHAT" in kept[0]["text"]     # only the yelled line
    # the surviving line keeps its original timing -- filtering must not re-time anything
    assert kept[0] == every["clips"][0]["captions"][1]


def test_captions_default_is_the_full_track(tmp_path):
    words = [{"text": "a", "start": 0.0, "end": 0.4, "emph": False},
             {"text": "b.", "start": 0.5, "end": 0.9, "emph": False}]
    sections = [{"title": "Open", "clips": [{"start": 0.0, "end": 1.0}]}]
    kw = dict(motion_zoom=False, deadspace=False)
    a = json.loads(build_manifest("v.mp4", tmp_path / "a", words, sections,
                                  **kw).read_text(encoding="utf-8"))
    b = json.loads(build_manifest("v.mp4", tmp_path / "b", words, sections,
                                  captions="all", **kw).read_text(encoding="utf-8"))
    assert a["clips"][0]["captions"] == b["clips"][0]["captions"]
    assert a["clips"][0]["captions"]                        # not silently emptied
