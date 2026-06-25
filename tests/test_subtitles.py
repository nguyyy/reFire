from refire.subtitles import _ts, build_ass


def _words(n, base=100.0, step=0.5):
    return [{"text": f"w{i}", "start": base + i * step, "end": base + i * step + 0.4}
            for i in range(n)]


def test_ts_format():
    assert _ts(0) == "0:00:00.00"
    assert _ts(65.25) == "0:01:05.25"
    assert _ts(-5) == "0:00:00.00"  # clamped


def test_events_and_zero_based_timing():
    words = _words(10)  # span 100.0 .. ~104.9
    ass = build_ass(words, seg_start=100.0, seg_end=110.0, words_per_line=4)
    body = ass.split("[Events]")[1]
    dialogues = [l for l in body.splitlines() if l.startswith("Dialogue:")]
    assert len(dialogues) == 3  # 10 words / 4 per line -> 4+4+2

    # all timestamps are 0-based: first dialogue starts at 0:00:00.00
    assert dialogues[0].split(",")[1] == "0:00:00.00"
    # words preserved and in order
    assert "w0" in dialogues[0] and "w9" in dialogues[-1]


def test_words_outside_segment_excluded():
    words = _words(4, base=50.0)  # all before the segment window
    ass = build_ass(words, seg_start=100.0, seg_end=110.0)
    assert "Dialogue:" not in ass
