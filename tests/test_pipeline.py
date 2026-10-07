import json

from refire.chat import chat_signal
from refire.ingest import window_tag
from refire.pipeline import _run_name


def test_run_name_includes_vod_id_and_varies():
    name = _run_name(2824313956)
    assert name.startswith("2824313956-")
    parts = name.split("-")
    assert len(parts) == 3   # vod_id, adjective, noun

    # not deterministic, 30 samples should hit more than one phrase
    assert len({_run_name(123) for _ in range(30)}) > 1


def test_window_tag_separates_caches():
    assert window_tag() == "" and window_tag(0, None) == ""   # whole VOD: unchanged paths
    assert window_tag(14400, 18000) == "@14400-18000"
    assert window_tag(14400, None) == "@14400-end"
    assert window_tag(14400, 18000) != window_tag(18000, 21600)


def test_chat_signal_offsets_to_window(tmp_path):
    """Chat offsets are absolute stream time; a --start window is zero-based."""
    p = tmp_path / "chat.json"
    p.write_text(json.dumps({"comments": [
        {"content_offset_seconds": 100.0},     # before the window -> dropped
        {"content_offset_seconds": 14401.0},   # 1s into it
        {"content_offset_seconds": 14402.0},
    ]}), encoding="utf-8")
    sig = chat_signal(p, duration=20.0, window=5.0, offset=14400.0)
    counts = [z for _, z in sig]
    assert counts[0] == max(counts)            # both messages land in the first bucket
    assert counts[1] == counts[2] == counts[3]  # and nothing leaked into the rest


def test_stopwatch_logs_each_stage_and_saves_ascii_timings(tmp_path, capsys):
    from refire.pipeline import _Stopwatch

    clock = _Stopwatch()
    for name in ("download", "cast 1", "review 1", "cast 2"):
        clock.lap(name)
    clock.save(tmp_path / "timings.json")

    data = json.loads((tmp_path / "timings.json").read_text(encoding="utf-8"))
    assert [s["stage"] for s in data["stages"]] == ["download", "cast 1", "review 1", "cast 2"]
    assert data["total_s"] >= 0
    out = capsys.readouterr().out
    assert "[time] total" in out
    out.encode("cp1252")   # the console is cp1252, a stray glyph would kill the run
