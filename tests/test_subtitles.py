import re

from refire.select import snap_to_sentences
from refire.subtitles import _ts, build_ass, group_words


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


def test_group_breaks_on_sentence_end():
    words = [{"text": "hello", "start": 100.0, "end": 100.4},
             {"text": "world.", "start": 100.5, "end": 100.9},
             {"text": "next", "start": 101.0, "end": 101.4}]
    groups = group_words(words, 100.0, 110.0, words_per_line=5)
    assert [[w["text"] for w in g] for g in groups] == [["hello", "world."], ["next"]]


def test_group_breaks_on_long_pause():
    words = [{"text": "a", "start": 100.0, "end": 100.4},
             {"text": "b", "start": 100.5, "end": 100.9},   # 0.8s gap before "c"
             {"text": "c", "start": 101.7, "end": 102.0}]
    groups = group_words(words, 100.0, 110.0, words_per_line=5)
    assert [len(g) for g in groups] == [2, 1]               # caption can't hang over silence


def test_caption_lowercase_by_default_caps_on_emphasis():
    words = [{"text": "Hello", "start": 0.0, "end": 0.4, "emph": False},
             {"text": "LOL", "start": 0.5, "end": 0.9, "emph": True}]
    ass = build_ass(words, 0.0, 10.0)          # 0.1s gap -> one line
    assert "hello" in ass and "LOL" in ass
    assert "Hello" not in ass                   # default-cased text is gone


def test_karaoke_hold_clamped_over_gap():
    # 0.3s gap (< PAUSE_GAP) keeps both on one line; the first word's highlight
    # must not stretch the whole 0.5s across the silence -> clamped to end+carry.
    words = [{"text": "a", "start": 0.0, "end": 0.2},
             {"text": "b", "start": 0.5, "end": 0.7}]
    ass = build_ass(words, 0.0, 10.0)
    ks = [int(m) for m in re.findall(r"\\k(\d+)", ass)]
    assert ks[0] <= 35                          # 0.35s hold, not the full 0.5s


def test_snap_extends_to_sentence_boundaries():
    ws = [{"text": t, "start": i, "end": i + 0.8}
          for i, t in enumerate("Today we review builds. Lets go now.".split())]
    s, e = snap_to_sentences(ws, 2.5, 3.0)   # cut lands mid-"review builds."
    assert s == 0.0 and e == 3.8             # pulled to sentence start + end
