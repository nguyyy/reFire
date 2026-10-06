"""Perception fusion: moment map v2, loudness buckets, top windows, mode selection."""
import struct
import wave

from refire import director, perception
from refire.rank import select_by_signal


def _w(text, start, end):
    return {"text": text, "start": start, "end": end}


WORDS = [_w("Hello", 0.0, 0.4), _w("world.", 0.5, 0.9),
         _w("Next", 65.2, 65.5), _w("one.", 65.6, 66.0)]


def test_sentences_split_and_span():
    assert perception._sentences(WORDS) == [
        (0, 0.9, "Hello world."), (65, 66.0, "Next one.")]


def test_moment_map_degrades_to_stream_map():
    # no signals or captions -> same as the v1 transcript map
    assert perception.moment_map(WORDS) == director.stream_map(WORDS)


def test_moment_map_marks_thresholds():
    audio = [(0.5, 3.4), (65.5, 1.8)]   # (LOUD) on the first sentence, (loud) on the second
    out = perception.moment_map(WORDS, audio_z=audio).splitlines()
    assert out[0] == "[0s] (LOUD) Hello world."
    assert out[1] == "[65s] (loud) Next one."


def test_moment_map_interleaves_scene_lines():
    out = perception.moment_map(WORDS, captions={30.0: "boss fight starts"}).splitlines()
    assert out == ["[0s] Hello world.",
                   "[30s] [scene: boss fight starts]",
                   "[65s] Next one."]


def test_loudness_signal_spikes_on_the_loud_window(tmp_path):
    sr = 16000
    quiet = [200] * sr * 5
    loud = [12000] * sr * 5
    frames = quiet * 3 + loud + quiet * 3     # 35s, spike in bucket 3
    p = tmp_path / "a.wav"
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{len(frames)}h", *frames))
    sig = perception.loudness_signal(p, window=5.0)
    assert len(sig) == 7
    peak = max(sig, key=lambda tz: tz[1])
    assert peak[0] == 17.5 and peak[1] > 2.0


def test_loudness_signal_missing_wav_is_empty():
    assert perception.loudness_signal("nope.wav") == []


def test_top_windows_nonoverlap_and_order():
    audio = [(t + 2.5, 0.0) for t in range(0, 600, 5)]
    audio[20] = (102.5, 5.0)     # spike ~100s
    audio[80] = (402.5, 4.0)     # spike ~400s
    wins = perception.top_windows(audio, k=2, span_s=60.0)
    assert len(wins) == 2
    assert wins == sorted(wins)
    (a0, b0), (a1, b1) = wins
    assert b0 <= a1                        # disjoint
    assert a0 <= 102.5 <= b0 and a1 <= 402.5 <= b1


def test_top_windows_empty_signals():
    assert perception.top_windows(None) == []


def test_signals_roundtrip(tmp_path):
    p = tmp_path / "signals.json"
    perception.save_signals(p, [(2.5, -0.3)])
    assert perception.load_signals(p) == [(2.5, -0.3)]
    assert perception.load_signals(tmp_path / "absent.json") is None


def test_pick_system_modes():
    briefed = director.pick_system("roast his builds")
    free = director.pick_system(None)
    assert "editor's brief" in briefed.lower()
    assert "NO editorial brief" in free
    assert "find the STORY" in free            # shared arc rules retained
    assert "{n}" in briefed and "{n}" in free  # still format-able
    assert "[scene:" in briefed and "[scene:" in free


def test_header_without_brief():
    h = director._header(None, "title", 60)
    assert "find the stream's own best story" in h


def test_select_by_signal_ranks_by_hype():
    chunks = [{"start": 0, "end": 30, "chat_z": 0.1},
              {"start": 30, "end": 60, "chat_z": 2.0},
              {"start": 60, "end": 90, "chat_z": -1.0}]
    audio = [(75.0, 4.0)]        # third chunk is quiet in chat but loud
    got = select_by_signal(chunks, 2, audio_z=audio)
    assert [c["start"] for c in got] == [60, 30]
