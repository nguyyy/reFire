"""Re-caption a hand-recut Premiere timeline (refire.srt.recaption)."""
import json

import pytest

from refire.srt import recaption

# transcript words, absolute VOD time: w0 at 100.0, w1 at 101.0, ... w9 at 109.0
WORDS = [{"text": f"w{i}", "start": 100.0 + i, "end": 100.4 + i} for i in range(10)]


def _run(tmp_path, rows, **kw):
    """Lay out a run tree, write `rows` as timeline.tsv, recaption -> cue list."""
    vod = tmp_path / "run" / "2854020361"
    ae = vod / "cut" / "ae"
    ae.mkdir(parents=True)
    (vod / "transcript.json").write_text(json.dumps(WORDS), encoding="utf-8")

    proxy = ae / "p.mov"
    proxy.write_bytes(b"")
    # the manifest's own clips cover 100-103 ONLY: anything the timeline pulls in past
    # that has to come from the transcript, which is the whole point of the transcript path
    (ae / "manifest.json").write_text(json.dumps({
        "source": str(proxy), "sources": [{"path": str(proxy), "offset": 100.0}],
        "clips": [{"start": 100.0, "end": 103.0, "src": 0,
                   "captions": [{"text": "stale", "start": 0.0, "end": 1.0}]}],
        "sections": [{"title": "", "clip_indices": [0]}],
    }), encoding="utf-8")

    tsv = "\n".join("\t".join(str(c) for c in ([str(proxy)] + list(r))[-5:])
                    for r in rows)
    (ae / "timeline.tsv").write_text(tsv, encoding="utf-8")

    out = recaption(ae / "manifest.json", words_per_line=1, **kw)
    return out, _cues(out.read_text(encoding="utf-8"))


def _cues(srt):
    """SRT text -> [(start_seconds, text)], so assertions read as timeline positions."""
    out = []
    for block in srt.strip().split("\n\n"):
        _, times, *text = block.split("\n")
        h, m, s = times.split(" --> ")[0].split(":")
        out.append((int(h) * 3600 + int(m) * 60 + float(s.replace(",", ".")),
                    " ".join(text)))
    return out


# rows are (in, out, at, end) in seconds -- source in/out are PROXY time (offset 100)

def test_reordered_timeline_recaptions_in_timeline_order(tmp_path):
    # the back half of the source is cut to the front: cues must follow the timeline
    _, cues = _run(tmp_path, [(5, 8, 0, 3), (0, 3, 3, 6)])
    assert [t for _, t in cues] == ["w5", "w6", "w7", "w0", "w1", "w2"]
    assert [round(a, 2) for a, _ in cues] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_widened_handle_picks_up_words_the_manifest_never_had(tmp_path):
    # the manifest's clips stop at 103; the timeline asks for 105-108 anyway
    _, cues = _run(tmp_path, [(5, 8, 0, 3)])
    assert [t for _, t in cues] == ["w5", "w6", "w7"]
    assert "stale" not in [t for _, t in cues]


def test_trim_clips_cues_to_the_trackitem(tmp_path):
    _, cues = _run(tmp_path, [(1.5, 3.5, 0, 2)])
    assert [t for _, t in cues] == ["w2", "w3"]      # w1 starts at 101.0, before the in


def test_speed_change_scales_cue_times(tmp_path):
    # 4s of source played over 2s of timeline = 2x, so w2 lands halfway, not 2s in
    _, cues = _run(tmp_path, [(0, 4, 10, 12)])
    assert [round(a, 2) for a, _ in cues] == [10.0, 10.5, 11.0, 11.5]


def test_offset_shifts_every_cue(tmp_path):
    _, cues = _run(tmp_path, [(0, 3, 0, 3)], offset=0.25)
    assert [round(a, 2) for a, _ in cues] == [0.25, 1.25, 2.25]


def test_foreign_media_is_skipped(tmp_path):
    # b-roll the user dragged in themselves is not in sources[] -- no garbage cues
    out, cues = _run(tmp_path, [("C:/broll/x.mov", 0, 3, 0, 3), (0, 3, 3, 6)])
    assert [t for _, t in cues] == ["w0", "w1", "w2"]
    assert [round(a, 2) for a, _ in cues] == [3.0, 4.0, 5.0]


def test_each_press_writes_a_new_file(tmp_path):
    # Premiere's importFiles skips a path already in the project, so overwriting
    # captions.srt would leave the user dragging in a stale caption track
    first, _ = _run(tmp_path, [(0, 3, 0, 3)])
    assert first.name == "captions.recut1.srt"
    second = recaption(first.parent / "manifest.json", words_per_line=1)
    assert second.name == "captions.recut2.srt"
    assert first.exists()


def test_empty_timeline_is_an_error_not_an_empty_srt(tmp_path):
    with pytest.raises(ValueError):
        _run(tmp_path, [])
