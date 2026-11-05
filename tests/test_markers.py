"""Review markers: features, learned risk, placement, scoring, manifest + csv."""
import json

from refire import markers
from refire.learn.diff import Context


def _words(spec):
    return [{"text": t, "start": a, "end": b} for t, a, b in spec]


CTX = Context(_words([("so", 9.8, 10.0), ("he", 10.1, 10.4), ("did", 10.5, 11.0),
                      ("it.", 11.1, 12.0), ("what", 12.1, 12.5), ("ok", 30.0, 30.5)]),
              loud=[(12.5, 2.0)], chat=[(17.5, 2.5)])


def test_features_read_speech_gaps_punctuation_and_signals():
    sf, ef, info = markers.features(CTX, 10.1, 12.0)
    assert sf["speech_before_03"] and sf["referent"]        # 'so' ends 0.1s before 'he'
    assert ef["speech_after_03"] and not ef["no_terminal"]  # 'what' starts 0.1s after
    assert ef["loud_end"] and ef["chat_after_2"]
    sf2, ef2, _ = markers.features(CTX, 30.0, 30.5)
    assert not sf2["speech_before_1"] and ef2["no_terminal"] and not ef2["loud_end"]


def _rows(n_fix, n_ok, on_fix, on_ok, kind="end_early", feat="loud_end"):
    side = kind.split("_")[0]
    other = {"end_early": "end_late", "end_late": "end_early",
             "start_late": "start_early", "start_early": "start_late"}[kind]
    rows = []
    for i in range(n_fix):
        rows.append({f"{side}_feats": {feat: i < on_fix}, f"{side}_label": kind})
    for i in range(n_ok):
        rows.append({f"{side}_feats": {feat: i < on_ok}, f"{side}_label": "ok"})
    rows.append({f"{side}_feats": {}, f"{side}_label": other})
    return rows


def test_fit_learns_direction_and_falls_back_when_thin():
    b = _rows(40, 160, 30, 20) + [{"start_feats": {}, "start_label": "ok"}] * 200
    rows = markers.training_rows([{"vod": "A", "boundaries": [
        {**r, "start_feats": r.get("start_feats", {}), "start_label": r.get("start_label", "ok"),
         "end_feats": r.get("end_feats", {}), "end_label": r.get("end_label", "ok")}
        for r in b]}])
    m = markers.fit(rows)
    on, _ = markers.risk(m, "end_early", {"loud_end": True})
    off, _ = markers.risk(m, "end_early", {"loud_end": False})
    assert on > 0.4 > off
    assert markers.fit([]) is markers.DEFAULT                     # no history -> priors
    assert markers.training_rows([{"vod": "A", "boundaries": []}], exclude=("A",)) == []


def test_clip_markers_one_per_edge_with_ascii_reason():
    clips = [{"start": 10.1, "end": 12.0}, {"start": 30.0, "end": 30.5}]
    n = markers.clip_markers(clips, CTX, markers.DEFAULT, threshold=0.0)
    assert n == 4 and all(len(c["markers"]) == 2 for c in clips)
    m = clips[0]["markers"][1]
    assert m["side"] == "end" and m["kind"] in ("end_early", "end_late")
    assert all(ch < "\x7f" for c in clips for x in c["markers"] for ch in x["comment"])
    assert markers.clip_markers([{"start": 30.0, "end": 30.5}], CTX, markers.DEFAULT, 1.0) == 0


def test_marker_score_precision_recall():
    recs = [{"label": "end_early", "vod_start": 12.0, "vod_end": 15.0, "delta_s": 3.0},
            {"label": "interior_cut", "vod_start": 100.0, "vod_end": 101.0, "delta_s": -1.0}]
    s = markers.marker_score([{"vod": 12.0}, {"vod": 50.0}], recs)
    assert s == {"n": 2, "precision": 0.5, "recall_s": 0.75}


def test_annotate_manifest_places_markers_and_writes_csv(tmp_path):
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps({"clips": [
        {"start": 10.1, "end": 12.0, "captions": []},
        {"start": 30.0, "end": 31.0, "keep": [[0.0, 0.5]], "dur": 0.5, "captions": []}],
        "sections": [{"title": "", "clip_indices": [1, 0]}]}))
    n = markers.annotate_manifest(mp, CTX, markers.DEFAULT, threshold=0.0)
    man = json.loads(mp.read_text())
    assert n == 4
    ends = [m for c in man["clips"] for m in c["markers"] if m["side"] == "end"]
    assert sorted(m["at"] for m in ends) == [0.0, 1.4]           # dur-0.5, clamped at 0
    rows = (tmp_path / "markers.csv").read_text().splitlines()
    assert rows[0].startswith("timecode,master_s") and len(rows) == 5
    assert rows[3].split(",")[1] == "0.5"                         # clip 0 plays second
