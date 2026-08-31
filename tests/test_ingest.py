from pathlib import Path

from reFire.ingest import _done, window_tag


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
