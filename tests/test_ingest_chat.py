from pathlib import Path

from refire import ingest


def test_a_failed_chat_download_does_not_stop_a_cached_vod(tmp_path, monkeypatch):
    (tmp_path / "123.mp4").write_bytes(b"x" * 2048)
    monkeypatch.setattr(ingest, "CLI", Path(__file__))                  # "exists"

    def boom(cmd, progress=None):
        raise RuntimeError(f"{cmd[1]} exited 1: vod deleted")
    monkeypatch.setattr(ingest, "_run", boom)
    video, chat = ingest.ensure_vod("123", cache_dir=tmp_path)
    assert video == tmp_path / "123.mp4" and not chat.exists()
