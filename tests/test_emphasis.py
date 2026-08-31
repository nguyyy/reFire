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


def test_annotate_stamps_the_rms_the_transient_snap_reads(tmp_path):
    """`select.snap_to_transient` cuts on this number; without it a transient run would
    silently degrade to phrase snapping with no error anywhere."""
    from refire.emphasis import annotate_emphasis

    ws = [{"text": "hi", "start": 0.0, "end": 0.2}]
    annotate_emphasis(ws, tmp_path / "missing.wav")     # no wav -> zeros, not a crash
    assert ws[0]["rms"] == 0.0 and ws[0]["emph"] is False


def test_word_rms_and_loud_flags_agree(tmp_path):
    """loud_word_flags is now word_rms + a threshold; they must not drift apart."""
    import wave

    import numpy as np

    from refire.emphasis import LOUD_K, loud_word_flags, word_rms

    sr = 16000
    sig = np.concatenate([np.full(sr, 500, np.int16), np.full(sr, 12000, np.int16),
                          np.full(sr, 500, np.int16)])
    wav = tmp_path / "a.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1), wf.setsampwidth(2), wf.setframerate(sr)
        wf.writeframes(sig.tobytes())
    ws = [{"text": "a", "start": 0.0, "end": 1.0},
          {"text": "b", "start": 1.0, "end": 2.0},
          {"text": "c", "start": 2.0, "end": 3.0}]
    rms = word_rms(wav, ws)
    assert rms[1] > rms[0] * LOUD_K
    assert loud_word_flags(wav, ws) == [False, True, False]
