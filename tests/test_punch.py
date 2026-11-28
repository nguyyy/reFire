"""refire punch analysis (docs/specs/01-punch.md acceptance criteria), fake judge, no models."""
import pytest

from refire import punch
from refire.facecam import Facecam
from refire.timeline_map import Span, TimelineMap

CAM = Facecam([0.0, 0.62, 0.27, 1.0], [0.12, 0.8], "bottom-left", 0.95, 60)
OFF = 1000.0                                      # transcript clock = timeline + 1000


def say(text, t, per=0.25, emph=None):
    """words at timeline t (transcript clock t + OFF); ALLCAPS words count as emphasized."""
    out = []
    for i, w in enumerate(text.split()):
        out.append({"text": w, "start": OFF + t + i * per, "end": OFF + t + i * per + per * 0.9,
                    "emph": w.isupper() if emph is None else emph})
    return out


VERDICT = {"okay i see": ("filler", 0.05), "OH SHIT": ("exclamation", 0.95),
           "[laughter]": ("laughter", 0.85)}


def judge(cands):
    out = {}
    for c in cands:
        cat, w = VERDICT.get(c["text"], ("reading", 0.3) if "quest" in c["text"] else ("exclamation", 0.9))
        out[c["id"]] = {"category": cat, "worthy": w, "reason": "test"}
    return out


def run(words, laughs=(), loud=(), spans=None, face=lambda p: 1.0, cams=None, cfg=None, total=300.0):
    tm = TimelineMap(spans or [Span(0, total, "A.mov", OFF, OFF + total, OFF)], 4233600000)
    words = sorted(words, key=lambda w: w["start"])
    phr = punch.phrases(punch.timeline_words(words, tm), tm)
    phr += punch.laugh_candidates(laughs, tm, phr, len(phr))
    sig = punch.Signal(loud)
    feats = {p["id"]: punch.features(p, sig, punch.Signal([]), punch.load_lexicon(), laughs) for p in phr}
    return punch.analyze(phr, feats, judge, face, cams or {0: CAM}, tm.cuts(), tm, total,
                         punch.config(cfg), sig)


def loud_at(t, z=3.0, width=1.0):
    """spike signal (transcript clock, 0.5 s buckets) with z around timeline t."""
    return [(OFF + k * 0.5, z if abs(k * 0.5 - t) <= width else 0.0) for k in range(600)]


def by_reason(review):
    return {r["text"]: r["drop_reason"] or r["status"] for r in review}


# --- Nguy's examples ------------------------------------------------------------

def test_flat_okay_i_see_never_punches():
    punches, review = run(say("okay i see", 10, emph=False))
    assert punches == [] and by_reason(review)["okay i see"] == "no_signal"


def test_okay_i_see_even_when_loud_is_filler():
    punches, review = run(say("okay i see", 10, emph=False), loud=loud_at(10.3))
    assert punches == [] and by_reason(review)["okay i see"] == "filler"


def test_loud_oh_shit_punches():
    punches, _ = run(say("OH SHIT", 20), loud=loud_at(20.2))
    assert len(punches) == 1
    p = punches[0]
    assert p["text"] == "OH SHIT" and p["category"] == "exclamation"
    assert p["t_in"] == pytest.approx(19.9) and p["t_out"] == pytest.approx(20.475 + 0.1, abs=1e-3)
    assert p["pin"] == [0.0, 1.0] and p["scale"] == 2.1


def test_flat_quest_read_does_not_punch():
    # a bit loud (passes the prefilter) but the judge hears quest dialogue
    words = say("the quest says bring three crystal cores to the old man", 30, emph=False)
    punches, review = run(words, loud=loud_at(31, z=1.8))
    assert punches == [] and review[0]["drop_reason"] == "reading" and review[0]["prefilter"] == "loud"


def test_dramatic_read_needs_real_loudness():
    global VERDICT
    words = say("and so the quest ends here forever", 30, emph=False)
    VERDICT = {**VERDICT, words and " ".join(w["text"] for w in words): ("dramatic_read", 0.95)}
    quiet, _ = run(words + say("what", 40, emph=False), loud=loud_at(31, z=1.6))
    loud, _ = run(words, loud=loud_at(31, z=2.5))
    assert quiet == [] and len(loud) == 1


def test_sustained_laugh_punches_but_a_chuckle_does_not():
    laughs = [(OFF + 50.0, OFF + 51.4, 0.9)]
    punches, _ = run([], laughs=laughs)
    assert len(punches) == 1 and punches[0]["category"] == "laughter"
    assert punches[0]["t_out"] - punches[0]["t_in"] == pytest.approx(1.6)
    chuckle, review = run([], laughs=[(OFF + 50.0, OFF + 50.6, 0.9)])
    assert chuckle == [] and review == []


# --- spacing ---------------------------------------------------------------------

def test_close_punches_merge_and_mid_gaps_keep_the_better():
    words = say("OH SHIT", 10) + say("NO WAY", 11.5) + say("LETS GO", 14.5) + say("HOLY SHIT", 40)
    VERDICT.update({"NO WAY": ("exclamation", 0.9), "LETS GO": ("exclamation", 1.0)})
    punches, review = run(words)
    # 10 and 11.5 merge (gap < 2 s); the merged one then sits 2-4 s before LETS GO -> better wins
    assert [p["text"] for p in punches] == ["LETS GO", "HOLY SHIT"]
    reasons = by_reason(review)
    assert reasons["OH SHIT"] == reasons["NO WAY"] == "spacing_lost_to:p01"
    gaps = [b["t_in"] - a["t_out"] for a, b in zip(punches, punches[1:])]
    assert all(g >= 4.0 for g in gaps)


def test_merge_keeps_zoom_through():
    punches, _ = run(say("OH SHIT", 10) + say("HOLY SHIT", 11.5))
    assert len(punches) == 1 and punches[0]["merged_from"] == ["c0", "c1"]
    assert punches[0]["t_in"] == pytest.approx(9.9) and punches[0]["t_out"] > 12


def test_coverage_cap_drops_lowest():
    # 2.2 s punches every 6.5 s (gaps clear min_gap): ~33% of the minute is punched
    words = [w for k in range(9) for w in say("OH SHIT " * 4, 2 + k * 6.5)]
    punches, review = run(words, total=60.0)
    assert sum(p["t_out"] - p["t_in"] for p in punches) <= 0.20 * 60 + 1e-6
    assert "coverage_cap" in {r["drop_reason"] for r in review}


# --- face, cuts --------------------------------------------------------------------

def test_no_face_no_punch():
    punches, review = run(say("OH SHIT", 20), face=lambda p: 0.33)
    assert punches == [] and by_reason(review)["OH SHIT"] == "no_face"


def test_punch_starts_on_a_cut_just_before_it():
    spans = [Span(0, 20.0, "A.mov", OFF, OFF + 20, OFF), Span(20.0, 300, "A.mov", OFF + 50, OFF + 330, OFF)]
    words = say("OH SHIT", 50.25 - 30)            # timeline 20.25 -> preroll 20.15, cut at 20.0
    punches, _ = run([{**w, "start": w["start"] + 30, "end": w["end"] + 30} for w in words], spans=spans)
    assert punches[0]["t_in"] == 20.0 and punches[0]["snapped_to_cut"] == "in"


def test_phrase_across_a_jump_cut_is_dropped():
    spans = [Span(0, 20.0, "A.mov", OFF, OFF + 20, OFF), Span(20.0, 300, "A.mov", OFF + 50, OFF + 330, OFF)]
    words = say("OH", 19.7) + [{"text": "SHIT", "start": OFF + 50.05, "end": OFF + 50.2, "emph": True}]
    punches, review = run(words, spans=spans)
    assert punches == [] and by_reason(review)["OH SHIT"] == "straddles_cut"


# --- pin + scale -------------------------------------------------------------------

@pytest.mark.parametrize("box, want_pin, want_s", [
    ([0.0, 0.62, 0.27, 1.0], [0.0, 1.0], 2.1),          # bottom-left
    ([0.0, 0.35, 0.25, 0.65], [0.0, 0.5], 2.1),         # middle-left: pin (0, cy)
    ([0.75, 0.0, 1.0, 0.3], [1.0, 0.0], 2.1),           # top-right
    ([0.4, 0.4, 0.6, 0.6], [0.5, 0.5], 2.1),            # center
    ([0.0, 0.3, 0.6, 1.0], [0.0, 0.65], 1.0),           # too large: middle-left pin, tall box
])
def test_pin_and_scale(box, want_pin, want_s):
    p = punch.pin(box)
    assert p == want_pin
    assert punch.max_scale(box, p, 2.1) == pytest.approx(want_s, abs=1e-3)


def test_too_large_facecam_skips_the_source():
    big = Facecam([0.0, 0.3, 0.6, 1.0], [0.3, 0.65], "bottom-left", 0.9, 60)
    punches, review = run(say("OH SHIT", 20), cams={0: big})
    assert punches == [] and by_reason(review)["OH SHIT"] == "facecam_too_large"


def test_scaled_box_stays_on_screen():
    for box in ([0.05, 0.6, 0.3, 0.95], [0.7, 0.05, 0.98, 0.4], [0.35, 0.2, 0.65, 0.5]):
        p = punch.pin(box)
        s = punch.max_scale(box, p, 2.1)
        for e, q in ((box[0], p[0]), (box[2], p[0]), (box[1], p[1]), (box[3], p[1])):
            assert -1e-6 <= q + s * (e - q) <= 1 + 1e-6


def test_config_merges_partial_maps():
    c = punch.config({"threshold": 0.7, "weights": {"judge": 0.6}})
    assert c["threshold"] == 0.7 and c["weights"] == {"judge": 0.6, "loud": 0.2, "chat": 0.15, "flags": 0.15}


def test_judge_one_call_cached_with_context(tmp_path, monkeypatch):
    from refire import director
    calls = []

    def fake_cli(model, system, content, cls, **kw):
        calls.append(content[0]["text"])
        return cls.model_validate({"verdicts": [
            {"id": "c0", "category": "exclamation", "worthy": 1.4, "reason": "loud"},
            {"id": "c1", "category": "banter", "worthy": 0.2}]})

    monkeypatch.setattr(director, "_complete_cli", fake_cli)
    words = say("so we go in here and OH SHIT the boss", 10)
    j = punch.make_judge(tmp_path, words, glossary=["Paimon"])
    cands = [{"id": "c0", "text": "OH SHIT", "src_t0": OFF + 11.25, "src_t1": OFF + 11.7,
              "features": {"z_loud_peak": 3.0}}, {"id": "c1", "text": "x", "src_t0": OFF, "src_t1": OFF,
                                                    "features": {}}]
    got = j(cands)
    assert got["c0"] == {"category": "exclamation", "worthy": 1.0, "reason": "loud"}   # clamped
    assert got["c1"]["category"] == "other"                                           # unknown -> other
    assert "so we go in here" in calls[0] and "Paimon" in calls[0]                     # +-8 s context
    assert j(cands) == got and len(calls) == 1                                        # cache hit
    assert punch.make_judge(tmp_path, words, glossary=["Paimon"])(cands) == got and len(calls) == 1
    assert j([]) == {}


def test_run_writes_outputs_and_tick_exact_ops(tmp_path, monkeypatch):
    import json

    from refire import facecam
    vod = tmp_path / "run" / "123"
    ae = vod / "cut" / "ae"
    ae.mkdir(parents=True)
    words = say("so yeah", 3, emph=False) + say("OH SHIT", 10) + say("okay i see", 20, emph=False)
    (vod / "transcript.json").write_text(json.dumps(words), encoding="utf-8")
    (ae / "manifest.json").write_text(json.dumps({"fps": 30, "sources": [{"path": "C:/p/A.mov", "offset": OFF}],
                                                  "clips": [{"start": OFF, "end": OFF + 30, "src": 0}]}))
    (ae / "timeline.tsv").write_text("#timebase\t4233600000\nC:/p/A.mov\t0\t30\t0\t30\n", encoding="utf-8")
    monkeypatch.setattr(facecam.Facecams, "face_present", lambda self, *a, **k: 1.0)
    res = punch.run(vod / "cut", facecam_override={"box": CAM.box}, judge=judge, log=lambda *a: None)
    assert [p["text"] for p in res["punches"]] == ["OH SHIT"]
    assert json.loads((vod / "cut" / "punch.json").read_text())["punches"][0]["id"] == "p01"
    rows = (vod / "cut" / "punch_review.tsv").read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("id\tt0\tt1\tstatus\tdrop_reason") and len(rows) == 4
    o = json.loads((vod / "cut" / "ppro" / "ops_punch.json").read_text())
    assert [x["op"] for x in o["ops"]] == ["clear_track", "clear_track", "insert_adjustment", "add_effect", "add_effect"]
    assert [x["v"] for x in o["ops"][:2]] == [2, 1]          # V3 now; the legacy V2 punches cleared too
    ins = o["ops"][2]
    assert ins["v"] == 2 and int(ins["start"]) % 4233600000 == 0 and int(ins["end"]) % 4233600000 == 0
    assert int(ins["start"]) == round(9.9 * 60) * 4233600000                     # 60 fps frame grid
    tr = o["ops"][3]
    assert tr["match"] == "AE.ADBE Geometry" and [0, [0.0, 1.0]] in tr["params"] and [3, 210.0] in tr["params"]
    assert o["ops"][4]["params"] == [[1, 50], [2, 1.0], [3, 0]]
    # dry run: no ops file
    (vod / "cut" / "ppro" / "ops_punch.json").unlink()
    punch.run(vod / "cut", facecam_override={"box": CAM.box}, judge=judge, dry_run=True, log=lambda *a: None)
    assert not (vod / "cut" / "ppro" / "ops_punch.json").exists()


def test_match_ratio_is_word_order_aware():
    assert punch.match_ratio("bring three crystal cores", "Quest: Bring three Crystal Cores to Old Man") == 1.0
    assert punch.match_ratio("oh my god what", "Bring three crystal cores") == 0.0
    assert punch.match_ratio("", "x") == 0.0


def test_ocr_marks_reads_and_respects_budget():
    import numpy as np
    seen = {"words": "bring three crystal cores to the old man"}
    frame = np.zeros((1080, 1920, 3), np.uint8)
    calls = []

    def engine(img, use_cls=True):
        calls.append(img.shape)
        return [[None, seen["words"], 0.9]], None

    def ph(i, text):
        return {"id": i, "text": text, "words": text.split(), "src_t0": 10.0, "src_t1": 12.0, "source": 0}

    phr = [ph("c0", "bring three crystal cores to the old man"), ph("c1", "OH SHIT"),
           ph("c2", "i love this character so much honestly")]
    r = punch.ocr_reader(lambda sid, ts: [frame] * len(ts), "on", engine=engine)(phr)
    assert r == {"c0": True, "c2": False}                       # c1: too short to be a read
    assert len(calls) == 2 * 2 * 2                               # 2 phrases x 2 frames x 2 regions
    assert calls[0] == (int(0.97 * 1080) - int(0.70 * 1080), int(0.85 * 1920) - int(0.15 * 1920), 3)
    assert punch.ocr_reader(lambda *a: [frame], "auto", budget=-1, engine=engine, log=lambda *a: None)(phr) == {}
    assert punch.ocr_reader(lambda *a: [frame], "off", engine=engine)(phr) == {}


def test_ocr_reading_overrides_an_eager_judge():
    words = say("BRING THREE CRYSTAL CORES TO THE OLD MAN", 30)     # shouted, judge says exclamation
    VERDICT[" ".join(w["text"] for w in words)] = ("exclamation", 0.9)
    tm = TimelineMap([Span(0, 300, "A.mov", OFF, OFF + 300, OFF)], 4233600000)
    phr = punch.phrases(punch.timeline_words(words, tm), tm)
    sig = punch.Signal([])
    feats = {p["id"]: punch.features(p, sig, sig, punch.load_lexicon(), []) for p in phr}
    punches, review = punch.analyze(phr, feats, judge, lambda p: 1.0, {0: CAM}, [], tm, 300.0,
                                    punch.config(), sig, reading=lambda ps: {ps[0]["id"]: True})
    assert punches == [] and review[0]["reading"] is True and review[0]["drop_reason"] == "reading"
