import json
import shutil

import numpy as np
from PIL import Image

from refire.thumb import feedback, pipeline


def noise(p, seed):
    p.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (72, 128, 3), dtype=np.uint8)).resize((1280, 720)).save(p)


def a_set(runs, name="2900/2900-x", seed=0, order=("B", "A", "C")):
    s = runs / name / "thumb" / "v1"
    types = {"A": "reaction", "B": "subject", "C": "reveal"}
    for i, k in enumerate("ABC"):
        noise(s / k / "thumb.png", seed + i)
        (s / k / "spec.json").write_text(json.dumps({"layout": types[k], "text": {"lines": [f"HOOK {k}?"], "accent": "gold"}}),
                                         encoding="utf-8")
    (s / "concepts.json").write_text(json.dumps({"concepts": [
        {"id": k, "type": types[k], "face": {"expression": "shock" if k == "B" else "deadpan"},
         "hook": {"accent_color": "gold"}, "subject": {"source": "made", "made": {"kind": "stat_card", "hidden": [0]}}
         if k == "C" else {"source": "art"}} for k in "ABC"]}), encoding="utf-8")
    (s / "upload_order.json").write_text(json.dumps(list(order)), encoding="utf-8")
    return s


class FakeYT:
    def __init__(self, live: dict, published="2026-01-01T00:00:00Z"):
        self.live, self.published = live, published          # video id -> the png that is live

    def uploads(self, n=50):
        return list(self.live)

    def videos(self, ids):
        return [{"id": i, "title": f"t {i}", "published": self.published, "thumb": str(self.live[i])} for i in ids]

    def ctr(self, vid, since):
        return 1200, 9.1

    def fetch(self, url, out):                               # youtube re-encodes: a jpeg of the candidate
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.open(url).convert("RGB").save(out, "JPEG", quality=80)
        return out


def test_sync_links_the_live_thumbnail_to_its_set_and_result_records_the_winner(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "LIVE", tmp_path / "live")
    runs = tmp_path / "run"
    s = a_set(runs)
    a_set(runs, "2901/2901-y", seed=10)
    noise(tmp_path / "other.png", 99)                        # someone else's thumbnail: no match
    con = feedback.connect(tmp_path / "t.sqlite")
    yt = FakeYT({"vid1": s / "B" / "thumb.png", "vid2": tmp_path / "other.png"})
    got = feedback.sync(yt, runs, con, log=lambda *a: None)
    assert [(g["video_id"], g["live"]) for g in got] == [("vid1", "B")] and got[0]["set_dir"] == s
    assert "test over" in got[0]["state"]
    v = con.execute("select * from videos").fetchone()
    assert v["ctr"] == 9.1 and v["impressions"] == 1200
    c = {r["cand"]: r for r in con.execute("select * from candidates")}
    assert c["B"]["slot"] == 1 and c["C"]["pixelate"] == 1 and c["C"]["left"] == "stat_card" and c["B"]["expression"] == "shock"
    assert feedback.result(yt, "vid1", "winner", runs, con, log=lambda *a: None)["winner"] == "B"
    hist = feedback.history(con)
    assert len(hist) == 1 and "B* slot 1 subject" in hist[0] and "CTR 9.1" in hist[0]


def test_result_needs_a_match_and_a_known_outcome(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(feedback, "LIVE", tmp_path / "live")
    runs = tmp_path / "run"
    a_set(runs)
    noise(tmp_path / "other.png", 99)
    con = feedback.connect(tmp_path / "t.sqlite")
    with pytest.raises(ValueError, match="one of"):
        feedback.result(FakeYT({}), "v", "won", runs, con)
    with pytest.raises(ValueError, match="matches none"):
        feedback.result(FakeYT({"v": tmp_path / "other.png"}), "v", "inconclusive", runs, con)


def test_a_feature_that_keeps_winning_is_favored_after_ten_tests_and_enforced(tmp_path):
    con = feedback.connect(tmp_path / "t.sqlite")
    for i in range(10):
        vid = f"v{i}"
        win = "B" if i < 6 else "A"                         # the shocked face wins 6 of 10
        con.execute("insert into videos (video_id, published, result, winner) values (?, ?, 'winner', ?)",
                    (vid, f"2026-01-{i + 1:02d}", win))
        for k, expr in (("A", "deadpan"), ("B", "shock"), ("C", "skeptical")):
            con.execute('insert into candidates (video_id, cand, slot, type, expression, accent, "left", quote, pixelate) '
                        "values (?, ?, 1, 'x', ?, 'gold', 'art', 0, 0)", (vid, k, expr))
    con.commit()
    assert feedback.favored(con)["expression"] == "shock"
    con.close()
    text, fav = feedback.context(tmp_path / "t.sqlite")
    assert "expression = shock" in text and text.count("\n- ") == 10
    Out = pipeline._models()
    out = Out.model_validate({"brief": {}, "concepts": [
        {"type": t, "hook": {"lines": ["BIG", "HOOK?"]}, "face": {"t": 1, "expression": "deadpan"},
         "subject": {"source": "art"}, "payoff_beats": [0]} for t in pipeline.TYPES]})
    assert any("expression = shock" in e for e in pipeline.validate(out, 3, {"expression": "shock"}))
    out.concepts[0].face.expression = "shock"
    assert pipeline.validate(out, 3, {"expression": "shock"}) == []


def test_no_history_means_no_prompt_and_no_db_file(tmp_path):
    assert feedback.context(tmp_path / "none.sqlite") == ("", {})
    assert not (tmp_path / "none.sqlite").exists()


def test_upload_order_falls_back_to_the_report(tmp_path):
    s = tmp_path / "v1"
    s.mkdir()
    (s / "report.md").write_text("# x\n\n## Upload order\n\n1. **C** - A\n2. **A** - B\n3. **B** - C\n", encoding="utf-8")
    assert feedback.upload_order(s) == ["C", "A", "B"]
    shutil.rmtree(s)
    s.mkdir()
    assert feedback.upload_order(s) == ["A", "B", "C"]
