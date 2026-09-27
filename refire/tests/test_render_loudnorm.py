"""`--style loudnorm`: every clip lands at the same perceived loudness.

A cut that jumps between moments recorded minutes apart also jumps in level. Once the
shots are short that level jump lands on every cut, and it reads as fatigue rather than
as energy -- which is why a dense style calls normalization load-bearing.
"""
from refire import render


def _args(monkeypatch, **kw):
    """Capture every ffmpeg argv render_clip would run, without running ffmpeg."""
    calls = []
    monkeypatch.setattr(render, "_run", lambda cmd: calls.append(cmd))
    monkeypatch.setattr(render, "reframe_clip", lambda *a, **k: None)
    render.render_clip("v.mp4", {"start": 0.0, "end": 5.0}, "c.ass", "out.mp4",
                       motion_zoom=False, **kw)
    return calls


def _af(calls):
    for cmd in calls:
        if "-af" in cmd:
            return cmd[cmd.index("-af") + 1]
    return None


def test_loudnorm_off_by_default(monkeypatch):
    assert _af(_args(monkeypatch)) is None


def test_loudnorm_reaches_the_audio_chain(monkeypatch):
    assert _af(_args(monkeypatch, loudnorm=True)) == render.LOUDNORM


def test_loudnorm_composes_with_the_dead_air_cut(monkeypatch):
    """Both filters have to survive: `aselect` drops the silence, loudnorm levels what
    is left. Order matters -- normalizing the silence first would pump the gain."""
    af = _af(_args(monkeypatch, loudnorm=True, keep=[(0.0, 2.0), (3.0, 5.0)]))
    assert af.startswith("aselect=") and af.endswith(render.LOUDNORM)
    assert "asetpts=N/SR/TB" in af


def test_dead_air_cut_alone_is_unchanged(monkeypatch):
    af = _af(_args(monkeypatch, keep=[(0.0, 2.0), (3.0, 5.0)]))
    assert af == "aselect='between(t,0.000,2.000)+between(t,3.000,5.000)',asetpts=N/SR/TB"
