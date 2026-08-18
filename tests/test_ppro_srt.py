import json

from refire.srt import build_srt, write_srt

_MANIFEST = {
    "clips": [
        {"start": 100.0, "end": 110.0,
         "captions": [{"text": "hello", "start": 1.0, "end": 2.0}]},
        # dead air cut: a 30s span that plays as 8s
        {"start": 200.0, "end": 230.0, "dur": 8.0, "keep": [[0, 4], [20, 24]],
         "captions": [{"text": "world", "start": 0.5, "end": 1.5}]},
    ],
    "sections": [{"title": "one", "clip_indices": [0]},
                 {"title": "two", "clip_indices": [1]}],
}


def test_cues_accumulate_by_tightened_duration():
    out = build_srt(_MANIFEST)
    # clip 0 is 10s raw; clip 1's cue must start at 10 + 0.5, not at clip 1's raw span
    assert "00:00:01,000 --> 00:00:02,000\nhello" in out
    assert "00:00:10,500 --> 00:00:11,500\nworld" in out
    assert out.count(" --> ") == 2


def test_section_order_wins_over_clip_order():
    m = dict(_MANIFEST, sections=[{"title": "two", "clip_indices": [1]},
                                  {"title": "one", "clip_indices": [0]}])
    out = build_srt(m)
    # clip 1 (dur 8) now lays down first, so "hello" is pushed to 8 + 1.0
    assert out.startswith("1\n00:00:00,500 --> 00:00:01,500\nworld\n")
    assert "00:00:09,000 --> 00:00:10,000\nhello" in out


def test_offset_shifts_and_clamps():
    assert "00:00:03,000 --> 00:00:04,000\nhello" in build_srt(_MANIFEST, offset=2.0)
    # a negative nudge past zero clamps instead of emitting a negative timestamp
    assert build_srt(_MANIFEST, offset=-5.0).startswith(
        "1\n00:00:00,000 --> 00:00:00,000\nhello")


def test_caption_overrunning_the_cut_is_clamped():
    m = {"clips": [{"start": 0.0, "end": 5.0,
                    "captions": [{"text": "long", "start": 4.0, "end": 99.0},
                                 {"text": "past", "start": 60.0, "end": 61.0},
                                 {"text": "  ", "start": 1.0, "end": 2.0}]}]}
    out = build_srt(m)
    # clamped to the clip end; the fully-past and blank cues are dropped entirely
    assert out == "1\n00:00:04,000 --> 00:00:05,000\nlong\n\n"


def test_write_srt_lands_beside_the_manifest(tmp_path):
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(_MANIFEST), encoding="utf-8")
    out = write_srt(mp, offset=0.0)
    assert out == tmp_path / "captions.srt"
    assert "hello" in out.read_text(encoding="utf-8")
