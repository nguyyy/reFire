"""`--style loudnorm`: every clip lands at the same perceived loudness.

A cut that jumps between moments recorded minutes apart also jumps in level. Once the
shots are short that level jump lands on every cut, and it reads as fatigue rather than
as energy -- which is why a dense style calls normalization load-bearing.
"""
from refire import render


def _args(monkeypatch, **kw):
    """Capture every ffmpeg argv render_clip would run, without running ffmpeg.
    fade=0 by default here: these tests pin the hard-cut chain; the de-pop graph that is
    the default now has its own tests below."""
    calls = []
    kw.setdefault("fade", 0.0)
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


def _fc(calls):
    for cmd in calls:
        if "-filter_complex" in cmd:
            return cmd[cmd.index("-filter_complex") + 1]
    return None


def test_default_fades_every_span_and_keeps_loudnorm_last(monkeypatch):
    fc = _fc(_args(monkeypatch, fade=render.FADE_S, loudnorm=True,
                   keep=[(0.0, 2.0), (3.0, 5.0)]))
    assert "select='between(t,0.000,2.000)+between(t,3.000,5.000)'" in fc
    assert fc.count("afade=t=in") == 2 and fc.count("afade=t=out") == 2
    assert "concat=n=2:v=0:a=1," + render.LOUDNORM + "[a]" in fc


def test_audio_graph_lcut_mixes_the_tail_under_the_next_span():
    g = render.audio_graph([(0.0, 2.0), (3.0, 5.0)], 5.0, tails=[0.3, 0.0])
    assert "atrim=start=0.0000:end=2.3000" in g            # span 0 runs 0.3s long
    assert "afade=t=out:st=2.0000:d=0.3000" in g           # and fades over that tail
    assert "adelay=2000.0:all=1" in g and "amix=inputs=2:normalize=0" in g
    assert g.endswith("atrim=end=4.0000[a]")               # total = video length
    one = render.audio_graph(None, 5.0)
    assert one.startswith("[0:a]atrim=start=0.0000:end=5.0000") and one.endswith("anull[a]")


def test_lcut_tails_only_after_a_shout_and_never_past_the_gap():
    from refire.assemble import LCUT_S, lcut_tails
    words = [{"text": "NO", "start": 101.0, "end": 101.5, "emph": True},
             {"text": "ok", "start": 103.2, "end": 103.6},
             {"text": "fine", "start": 106.0, "end": 106.4}]
    keep = [(0.7, 1.8), (3.0, 3.9), (5.8, 6.6)]
    assert lcut_tails(words, 100.0, keep) == [LCUT_S, 0.0, 0.0]
    assert lcut_tails(words, 100.0, [(0.7, 1.8), (1.9, 3.9)])[0] == 0.1
