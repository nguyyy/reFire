import json

from refire import preview


def test_punch_crops_pin_to_their_corner_and_zooms_center_on_the_anchor(tmp_path):
    (tmp_path / "punch.json").write_text(json.dumps({"punches": [{"t_in": 1, "t_out": 2, "pin": [0.0, 1.0], "scale": 2.0}]}))
    (tmp_path / "sauce.json").write_text(json.dumps({"events": [
        {"type": "zoom", "t": 5, "end": 7, "anchor": [0.99, 0.5], "scale": 2.0}, {"type": "boom", "t": 3}]}))
    zoom, punch = preview.windows(tmp_path)
    assert zoom == (5, 7, 960, 270, 960, 540)                    # clamped inside the frame, V2 before V3
    assert punch == (1, 2, 0.0, 540.0, 960, 540)                 # bottom-left corner


def test_captions_become_an_srt_and_booms_mix_at_their_level(tmp_path):
    (tmp_path / "caption_review.tsv").write_text("id\tt_in\tt_out\ttext\n" "b1\t61.5\t62.25\tLET'S go\n")
    preview.srt(tmp_path, tmp_path / "c.srt")
    assert (tmp_path / "c.srt").read_text() == "1\n00:01:01,500 --> 00:01:02,250\nLET'S go\n\n"
    g = preview.graph([(1, 2, 0, 540, 960, 540)], [(4.0, 5.0)], [(3.0, 0.035481)], "c.srt", 60)
    assert "trim=1.000:2.000" in g and "PTS-STARTPTS+4.000/TB" in g and "subtitles=c.srt" in g
    assert "adelay=3000" in g and "volume=0.1995" in g and "amix=inputs=2" in g    # -14 dB
