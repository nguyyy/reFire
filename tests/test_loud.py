"""Cross-VOD loudest-moments cut: the spike metric, and the multi-source manifest."""
import json
import wave
from pathlib import Path

import numpy as np

from refire import perception
from refire.ae_export import build_manifest
from refire.loud import VOD_BASE, spike_signal

SR = 16000
# 300 one-second buckets: a long stretch that's loud the whole time (80s of game audio) and
# one short scream in quiet. the sustained part wins on raw level on purpose, that's the
# failure this module avoids
QUIET, SUSTAINED, SCREAM = 300, 9000, 22000
SUS_A, SUS_B = 100, 180
SCR_A, SCR_B = 280, 283


def _wav(path):
    amp = np.full(300, QUIET, dtype=np.int16)
    amp[SUS_A:SUS_B] = SUSTAINED
    amp[SCR_A:SCR_B] = SCREAM
    frames = np.repeat(amp, SR)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(frames.tobytes())
    return path


def _top(sig):
    (a, b), = perception.top_windows(sig, k=1, span_s=10.0)
    return a, b


def test_raw_level_picks_the_merely_loud_stretch(tmp_path):
    # the baseline we're fixing: a window sum wins on duration, so 80s of steady game audio
    # beats a 3s scream that's twice as loud
    a, b = _top(perception.loudness_signal(_wav(tmp_path / "a.wav"), window=1.0))
    assert SUS_A <= a and b <= SUS_B, (a, b)


def test_spike_picks_the_scream(tmp_path):
    # against a rolling median the sustained stretch is its own baseline (~0), the scream stands out
    a, b = _top(spike_signal(_wav(tmp_path / "a.wav")))
    assert a <= SCR_A and b >= SCR_B, (a, b)


def test_spike_missing_wav_is_empty():
    assert spike_signal("nope.wav") == []


def test_spike_on_a_clip_shorter_than_the_baseline(tmp_path):
    # too short for a local baseline -> raw signal, don't crash
    sig = spike_signal(_wav(tmp_path / "a.wav"), baseline_s=600.0)
    assert len(sig) == len(perception.loudness_signal(tmp_path / "a.wav", window=1.0))


def test_multi_vod_sources_survive_the_manifest(tmp_path):
    # two moments from two vods on the virtual timeline. the panel only computes
    # clip.start - sources[src].offset, so that has to land inside each clip's own window file
    words = [{"text": "what", "start": t, "end": t + 0.4, "emph": True}
             for t in (10.5, VOD_BASE + 10.5)]
    clips = [{"start": 10.0, "end": 14.0, "src": 0},
             {"start": VOD_BASE + 10.0, "end": VOD_BASE + 14.0, "src": 1}]
    srcs = [(Path("w000.mp4"), 5.0), (Path("w001.mp4"), VOD_BASE + 5.0)]
    mp = build_manifest("unused.mp4", tmp_path, words, [{"title": "", "clips": clips}],
                        motion_zoom=False, proxy=False, deadspace=False, cards=False,
                        sources=srcs)
    m = json.loads(mp.read_text(encoding="utf-8"))
    assert [c["src"] for c in m["clips"]] == [0, 1]
    assert [s["offset"] for s in m["sources"]] == [5.0, VOD_BASE + 5.0]
    # both clips start 5s into their own file, however far apart they are globally
    assert [c["start"] - m["sources"][c["src"]]["offset"] for c in m["clips"]] == [5.0, 5.0]
    assert m["clips"][1]["captions"], "the second VOD's words landed on its own clip"


def test_single_source_map_is_still_written(tmp_path):
    # one-clip cut still needs its offset, without sources the panel uses M.source at offset 0
    # and misplaces the clip
    clips = [{"start": 10.0, "end": 14.0, "src": 0}]
    mp = build_manifest("unused.mp4", tmp_path, [], [{"title": "", "clips": clips}],
                        motion_zoom=False, proxy=False, deadspace=False, cards=False,
                        sources=[(Path("w000.mp4"), 5.0)])
    m = json.loads(mp.read_text(encoding="utf-8"))
    assert m["sources"] == [{"path": str(Path("w000.mp4").resolve()).replace("\\", "/"),
                             "offset": 5.0}]
