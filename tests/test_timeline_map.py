"""timeline.tsv <-> source time (refire.timeline_map)."""
import json

import pytest

from refire.timeline_map import TICKS, VOD_BASE, Span, TimelineMap


def _load(tmp_path, rows, header="#timebase\t4233600000", sources=None, fps=30):
    ae = tmp_path / "ae"
    ae.mkdir()
    sources = sources or [{"path": "C:/p/A.mov", "offset": 100.0}]
    (ae / "manifest.json").write_text(json.dumps({"fps": fps, "sources": sources}), encoding="utf-8")
    lines = ([header] if header else []) + ["\t".join(str(c) for c in r) for r in rows]
    (ae / "timeline.tsv").write_text("\n".join(lines), encoding="utf-8")
    return TimelineMap.load(ae / "manifest.json")


def test_maps_through_offsets_and_speed(tmp_path):
    # clip 1: proxy 5-8 at timeline 0-3; clip 2: proxy 0-4 played 2x at 3-5
    m = _load(tmp_path, [("C:\\p\\a.mov", 5, 8, 0, 3), ("C:/p/A.mov", 0, 4, 3, 5)])
    assert m.timeline_to_source(1.0) == (0, 106.0)          # path match ignores case + slashes
    assert m.timeline_to_source(4.0) == (0, 102.0)          # 1 s into a 2x clip = 2 s of source
    assert m.timeline_to_source(3.0) == (0, 100.0)          # a cut belongs to the clip starting there
    assert m.timeline_to_source(5.0) is None                # past the end


def test_source_moment_shown_twice(tmp_path):
    m = _load(tmp_path, [("C:/p/A.mov", 0, 4, 0, 4), ("C:/p/A.mov", 2, 6, 10, 14)])
    assert m.source_to_timeline(0, 103.0) == [3.0, 11.0]
    assert m.source_to_timeline(0, 99.0) == []
    assert m.source_to_timeline(1, 103.0) == []             # other vod


def test_cuts_gaps_and_foreign_media(tmp_path):
    m = _load(tmp_path, [("C:/p/A.mov", 0, 2, 0, 2), ("C:/broll/x.mov", 0, 3, 2, 5),
                         ("C:/p/A.mov", 4, 6, 7, 9)])
    assert m.cuts() == [2.0, 5.0, 7.0]                      # b-roll edges are still edits
    assert m.gaps() == [(5.0, 7.0)]
    assert m.timeline_to_source(3.0) is None                # b-roll maps to no source
    assert m.timeline_to_source(6.0) is None                # gap
    assert [s.media for s in m.ours()] == ["C:/p/A.mov", "C:/p/A.mov"]


def test_loud_runs_split_sources_by_vod_base(tmp_path):
    srcs = [{"path": "C:/w/0.mov", "offset": 50.0}, {"path": "C:/w/1.mov", "offset": VOD_BASE + 70.0}]
    m = _load(tmp_path, [("C:/w/0.mov", 0, 5, 0, 5), ("C:/w/1.mov", 1, 3, 5, 7)], sources=srcs)
    assert m.timeline_to_source(2.0) == (0, 52.0)
    sid, t = m.timeline_to_source(6.0)
    assert sid == 1 and t == pytest.approx(VOD_BASE + 72.0)
    assert m.source_to_timeline(1, VOD_BASE + 72.0) == [pytest.approx(6.0)]


def test_ticks_snap_to_the_sequence_frame_grid(tmp_path):
    m = _load(tmp_path, [("C:/p/A.mov", 0, 4, 0, 4)])       # 60 fps sequence, manifest says 30
    tpf = TICKS // 60
    assert m.ticks_per_frame == tpf
    assert m.ticks(1.0) == TICKS
    assert m.ticks(1.0 / 60 * 7 + 0.004) == 7 * tpf          # nearest frame


def test_old_dump_without_header_falls_back_to_manifest_fps(tmp_path):
    m = _load(tmp_path, [("C:/p/A.mov", 0, 4, 0, 4)], header=None, fps=30)
    assert m.ticks_per_frame == TICKS // 30
    with pytest.raises(ValueError):
        TimelineMap([Span(0, 1, "x")]).ticks(0.5)


def test_bad_rows_are_skipped(tmp_path):
    m = _load(tmp_path, [("C:/p/A.mov", "nan?", 4, 0, 4), ("C:/p/A.mov", 0, 4, 4, 4),
                         ("C:/p/A.mov", 0, 4, 4, 8)])
    assert [(s.tl_in, s.tl_out) for s in m.spans] == [(4.0, 8.0)]
