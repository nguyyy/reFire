import wave

import numpy as np

from refire.emphasis import annotate_emphasis, loud_word_flags, style_text


def test_keyword_and_allcaps_flagged_without_audio():
    words = [{"text": "lol", "start": 0.0, "end": 0.2},
             {"text": "WTF", "start": 0.3, "end": 0.6},
             {"text": "character", "start": 0.7, "end": 1.1}]
    annotate_emphasis(words, "no_such_file.wav")     # loudness degrades to off
    assert [w["emph"] for w in words] == [True, True, False]


def test_style_text_casing():
    assert style_text("Insane", True) == "INSANE"
    assert style_text("Character", False) == "character"


def test_loud_word_detected_from_wav(tmp_path):
    sr = 16000
    quiet = (np.ones(sr) * 500).astype(np.int16)      # 1s soft
    yell = (np.ones(sr) * 12000).astype(np.int16)     # 1s loud
    wav = tmp_path / "audio.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(quiet.tobytes() + yell.tobytes())
    words = [{"text": "soft", "start": 0.1, "end": 0.9},
             {"text": "loud", "start": 1.1, "end": 1.9}]
    assert loud_word_flags(wav, words) == [False, True]
