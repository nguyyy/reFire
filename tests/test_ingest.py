import sys
from pathlib import Path

import pytest

from refire.ingest import _done, _run, _status, window_tag


def test_empty_download_is_not_cached(tmp_path: Path):
    stub = tmp_path / "vod.mp4"           # what an interrupted run leaves behind
    stub.touch()
    assert not _done(stub)
    stub.write_bytes(b"\x00")
    assert _done(stub)
    assert not _done(tmp_path / "nope.mp4")


def test_window_tag():
    assert window_tag() == ""
    assert window_tag(60, 120) == "@60-120"
    assert window_tag(60, None) == "@60-end"


def test_status_lines_carry_the_phase_and_its_place_in_the_run():
    assert _status("[STATUS] - Fetching Video Info [1/4]") == (0.0, "fetching video info")
    frac, msg = _status("[STATUS] - Downloading 47% [2/4]")
    assert msg == "downloading 45%"          # 5s, or the log fills with 400 lines
    assert round(frac, 4) == 0.3675          # one of four phases, 47% through it
    assert _status("[STATUS] - Finalizing Video 100% [4/4]") == (1.0, "finalizing video 100%")
    assert _status("some other output") is None


def test_progress_arrives_before_the_phase_ends():
    """The downloader separates updates with a BARE \\r and only ends a line when a phase
    finishes -- so a line reader shows nothing for the whole download. That is the bug."""
    src = ("import sys; sys.stdout.write('[STATUS] - Downloading 0% [2/4]\\r"
           "[STATUS] - Downloading 50% [2/4]\\r')")
    seen = []
    _run([sys.executable, "-c", src], lambda f, m: seen.append(m))
    assert seen == ["downloading 0%", "downloading 50%"]


def test_a_failed_download_raises_with_the_downloader_s_own_reason():
    """Capturing stdout for progress also swallows the CLI's error text -- and a download
    that fails 30 minutes in with 'exit code 1' and nothing else is unfixable."""
    src = "import sys; print('[ERROR] - Video not found'); raise SystemExit(2)"
    with pytest.raises(RuntimeError, match="Video not found"):
        _run([sys.executable, "-c", src], lambda f, m: None)
