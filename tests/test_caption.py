"""refire caption: timing, easing curves, layering (docs/specs/02-caption.md acceptance)."""
import json
from pathlib import Path

import pytest

from refire import caption
from refire.ppro import prfpset

ROOT = Path(__file__).resolve().parents[1]
FIX = json.loads((ROOT / "tests" / "fixtures" / "captionIn_curve.json").read_text())


@pytest.fixture(scope="module")
def cap():
    (fx,) = prfpset.load(caption.ASSETS / "captionIn.prfpset")
    return fx


def test_real_preset_ramps(cap):
    rs, t_in, t_out = caption.ramps(cap)
    assert t_in == pytest.approx(0.247, abs=1e-3) and t_out == pytest.approx(0.183, abs=1e-3)
    by = {r.index: r for r in rs}
    assert set(by) == {1, 8} and by[1].point and not by[1].outro      # position: intro only
    keys, interp, k = caption.block_keys(2.0, rs, t_in, t_out, cap.duration, 60, "approx")
    assert interp == 5 and k == 1.0
    assert [round(t, 3) for t, _ in keys[1]] == [0.0, 0.247]
    assert [round(t, 3) for t, _ in keys[8]] == [0.0, 0.058, round(2.0 - 0.183, 3), round(2.0 - 0.015, 3)]
    assert sorted({round(t, 3) for ks in keys.values() for t, _ in ks if t < 1}) == [0.0, 0.058, 0.247]


def test_decoded_curve_matches_premiere(cap):
    """the speed/influence decode reproduces what Premiere rendered for the hand-applied preset
    (fixture keys are stretched by media/preset duration; undo that first)."""
    rs, *_ = caption.ramps(cap)
    by = {r.index: r for r in rs}
    pos_t = [k.t for k in by[1].intro]
    stretch = FIX["key_times_position"][1] / pos_t[1]
    pos = caption.curve(by[1].intro)
    op_in, op_out = caption.curve(by[8].intro), caption.curve(by[8].outro)
    for t, y, o in zip(FIX["t"], FIX["position_y"], FIX["opacity"]):
        tp = t / stretch
        if tp <= pos_t[1]:
            assert pos(tp)[1] == pytest.approx(y, abs=3e-4), t
        if tp <= by[8].intro[-1].t:
            assert op_in(tp) == pytest.approx(o, abs=2.5), t
        elif tp >= by[8].outro[0].t:
            assert op_out(tp) == pytest.approx(o, abs=2.5), t


def test_exact_keys_are_linear_frames_that_track_the_curve(cap):
    rs, t_in, t_out = caption.ramps(cap)
    keys, interp, _ = caption.block_keys(1.5, rs, t_in, t_out, cap.duration, 60, "exact")
    assert interp == 0
    p = keys[1]
    assert p[0][0] == 0.0 and p[-1][0] == pytest.approx(0.2468, abs=1e-3)
    assert p[0][1] == pytest.approx([0.5, 0.5324], abs=1e-4) and p[-1][1] == pytest.approx([0.5, 0.5])
    assert 3 <= len(p) <= 16                                         # simplified, not every frame
    # hard ease-out: most of the slide is done in the first 0.08 s
    early = next(v for t, v in p if t >= 0.08)
    assert early[1] < 0.5324 - 0.6 * 0.0324
    o = keys[8]
    assert o[0] == (0.0, 0.0) and o[-1][0] == pytest.approx(1.5 - 0.015, abs=1e-3) and o[-1][1] == 0.0


def _synthetic(fps=30, intro_f=10, outro_f=10, dur_f=30):
    """a linear opacity preset with a 10-frame intro and 10-frame outro."""
    f = 1 / fps
    ks = [prfpset.Key(0, 0.0, 0, []), prfpset.Key(intro_f * f, 100.0, 0, []),
          prfpset.Key((dur_f - outro_f) * f, 100.0, 0, []), prfpset.Key(dur_f * f, 0.0, 0, [])]
    return prfpset.Effect("AE.ADBE Geometry2", "Transform", 0, 0.0, dur_f * f,
                          [prfpset.Param(8, "Opacity", 8, 2, 100.0, ks)])


@pytest.mark.parametrize("frames, want, k", [
    (50, [0, 10, 40, 50], 1.0),
    (100, [0, 10, 90, 100], 1.0),
    (12, [0, 6, 6, 12], 0.6),                       # too short: squeezed by k = 12 / 20
])
def test_fixed_intro_outro_anchored(frames, want, k):
    fx = _synthetic()
    rs, t_in, t_out = caption.ramps(fx)
    keys, _, got_k = caption.block_keys(frames / 30, rs, t_in, t_out, fx.duration, 30, "approx")
    assert [round(t * 30, 3) for t, _ in keys[8]] == want and got_k == k


def test_exact_linear_preset_needs_only_its_corners():
    fx = _synthetic()
    rs, t_in, t_out = caption.ramps(fx)
    keys, interp, _ = caption.block_keys(2.0, rs, t_in, t_out, fx.duration, 30, "exact")
    assert [(round(t * 30), v) for t, v in keys[8]] == [(0, 0.0), (10, 100.0), (50, 100.0), (60, 0.0)]


def B(a, b, text="x"):
    return {"t_in": a, "t_out": b, "text": text}


def test_layering_newer_on_top_and_max_tracks():
    out = caption.layer([B(0, 2), B(1, 3), B(1.5, 4), B(5, 6)], max_tracks=4)
    assert [b["track"] for b in out] == [1, 2, 3, 1]
    out = caption.layer([B(0, 9), B(1, 9), B(2, 9), B(3, 9), B(4, 9)], max_tracks=4)
    assert max(b["track"] for b in out) == 4
    assert out[3]["t_out"] == 4 and out[4]["track"] == out[3]["track"]      # 5th trims the 4th
    for i, a in enumerate(out):                                           # during any overlap, newer is higher
        for b in out[i + 1:]:
            if b["t_in"] < a["t_out"]:
                assert b["track"] > a["track"]


def test_layering_drops_blocks_trimmed_to_nothing():
    out = caption.layer([B(0, 5), B(0, 1), B(0, 1), B(0, 1), B(0, 1)], max_tracks=4)
    assert all(b["t_out"] > b["t_in"] for b in out)


def test_run_writes_ttml_caption_tracks_and_ops(tmp_path):
    import xml.etree.ElementTree as ET
    vod = tmp_path / "run" / "123"
    ae = vod / "cut" / "ae"
    ae.mkdir(parents=True)
    words = [{"text": t, "start": 1000 + s, "end": 1000 + s + 0.3}
             for t, s in (("farzan", 0.0), ("is", 0.4), ("cracked", 0.8), ("no", 3.0), ("way", 3.4))]
    (vod / "transcript.json").write_text(json.dumps(words), encoding="utf-8")
    (ae / "manifest.json").write_text(json.dumps({"fps": 30, "sources": [{"path": "C:/p/A.mov", "offset": 1000.0}]}))
    (ae / "timeline.tsv").write_text("#timebase\t4233600000\nC:/p/A.mov\t0\t10\t0\t10\n", encoding="utf-8")
    (tmp_path / "corrections_genshin-impact.json").write_text(json.dumps({"farzan": "faruzan"}))

    res = caption.run(vod / "cut", words_per_line=3, game="genshin impact", corrections_dir=tmp_path,
                      log=lambda *a: None)
    blocks = res["blocks"]
    assert [b["text"] for b in blocks] == ["faruzan is cracked", "NO way"]     # caps-on-excitement kept
    assert [b["track"] for b in blocks] == [1, 1] and all(b["k"] == 1.0 for b in blocks)
    # one caption file per layer, styled like the MOGRT, times on the frame grid
    tt = ET.parse(vod / "cut" / "ppro" / "captions" / "layer_1.dfxp").getroot()
    ns = {"t": "http://www.w3.org/ns/ttml"}
    ps = tt.findall(".//t:p", ns)
    assert [x.text for x in ps] == ["faruzan is cracked", "NO way"] and ps[0].get("begin") == "00:00:00.000"
    style = tt.find(".//t:style", ns).attrib
    sty = {k.split("}")[1]: v for k, v in style.items() if "}" in k and "styling" in k}
    assert sty == {"fontFamily": "Coolvetica", "fontSize": "55px", "color": "#FFFFFF",
                   "textOutline": "#000000 3px", "textAlign": "center"}
    o = json.loads((vod / "cut" / "ppro" / "ops_caption.json").read_text())["ops"]
    assert [x["op"] for x in o[:6]] == ["ensure_nest", "style_nest", "clear_nest", "caption_tracks", "menu",
                                        "after_upgrade"]
    assert o[0]["v"] == 4 and o[0]["tracks"] == 1 and o[0]["anchor"].endswith(".mogrt")
    fx = o[1]["effects"]
    assert [e["match"] for e in fx] == ["AE.Impact_Camera_Shake_FX", "AE.ADBE Drop Shadow", "AE.Impact_Edge_Glow_FX"]
    assert [11, {"color": [255, 255, 223, 179]}] in fx[2]["params"] and [1, 165.75] in fx[1]["params"]
    assert not any(i == 42 for i, _ in fx[0]["params"])         # _ Applied Version skipped
    assert o[3]["files"][0].endswith("captions/layer_1.dfxp") and o[4]["do"] == "upgrade_captions"
    assert o[5]["layers"] == 1 and o[5]["want"] == 2
    tr, tr2 = o[6], o[7]
    assert tr["op"] == "caption_fx" and tr["seq"] == "refire_captions" and (tr["layer"], tr["index"]) == (1, 0)
    assert tr2["index"] == 1 and tr["pos"] == [0.125, 0.9155] and tr["name"] == "refire:caption:b001"
    assert int(tr["start"]) % 4233600000 == 0
    assert tr["match"] == "AE.ADBE Geometry" and tr["interp"] == 0
    statics = dict((i, v) for i, v in tr["params"])
    assert statics[9] is False and statics[10] == 360 and statics[2] is True and statics[0] == [0.5, 0.5]
    assert [j for j, _ in tr["keys"]] == [1, 8]
    review = (vod / "cut" / "caption_review.tsv").read_text(encoding="utf-8").splitlines()
    assert review[1].split("\t")[-1] == "farzan is cracked"     # corrected_from
    # twice -> identical ops (idempotent nest rebuild), dry run -> no ops file
    first = (vod / "cut" / "ppro" / "ops_caption.json").read_text()
    caption.run(vod / "cut", words_per_line=3, game="genshin impact", corrections_dir=tmp_path, log=lambda *a: None)
    assert (vod / "cut" / "ppro" / "ops_caption.json").read_text() == first
    (vod / "cut" / "ppro" / "ops_caption.json").unlink()
    caption.run(vod / "cut", dry_run=True, log=lambda *a: None)
    assert not (vod / "cut" / "ppro" / "ops_caption.json").exists()


def test_overlaps_get_their_own_caption_track_and_index():
    class TM:
        ticks_per_frame = 4233600000

        def ticks(self, t):
            return round(t * 60) * 4233600000
    blocks = caption.layer([B(0, 2), B(1, 3), B(4, 5)], max_tracks=4)
    for b, t in zip(blocks, ("a", "b & <c>", "d")):
        b.update(id=t, text=t, keys={}, interp=0)
    assert [b["track"] for b in blocks] == [1, 2, 1]
    assert "b &amp; &lt;c&gt;" in caption.ttml([blocks[1]], TM(), caption.config())
    fx = type("E", (), {"params": []})()
    o = caption.ops(blocks, TM(), caption.config(), fx, [], ["l1", "l2"])["ops"]
    got = [(x["layer"], x["index"], x["layers"]) for x in o if x["op"] == "caption_fx"]
    assert got == [(1, 0, 2), (2, 0, 2), (1, 1, 2)]
    assert [x["op"] for x in caption.ops([], TM(), caption.config(), fx, [], [])["ops"]] ==         ["ensure_nest", "style_nest", "clear_nest"]           # nothing to upgrade


def test_ttml_times_round_up_so_premiere_floors_onto_the_right_frame():
    class TM:
        ticks_per_frame = 4233600000

        def ticks(self, t):
            return round(t * 60) * 4233600000
    x = caption.ttml([{"t_in": 38 / 60, "t_out": 3661.5, "text": "a"}], TM(), caption.config())
    assert 'begin="00:00:00.634" end="01:01:01.500"' in x          # 0.633 landed on frame 37 (S8)
