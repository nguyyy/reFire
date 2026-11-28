"""sauce eval (04-sauce.md section 7): kept / moved / deleted / added, A, drift, corrections."""
import json

from refire.sauce import evaluate


def placed(i, t, ty="boom"):
    return {"id": f"{ty[0]}{i:02d}", "type": ty, "t": t, "end": t + 1.25, "asset": None, "reason": f"r{i}"}


def final(name, t, ty="boom"):
    return {"name": name, "t": t, "end": t + 1.25, "type": ty, "asset": None}


def test_a_counts_kept_under_three_frames_and_charges_moves_deletes_and_misses():
    ps = [placed(1, 10.0), placed(2, 20.0), placed(3, 30.0), placed(1, 40.0, "emote")]
    fs = [final("refire:sauce:b01", 10.03),            # 1.8 frames at 60 fps: kept
          final("refire:sauce:b02", 20.5),             # moved half a second: a fix
          final("refire:sauce:e01", 40.0, "emote"),    # kept
          final("vine-boom.mp3", 55.0)]                # his own: a miss
    r = evaluate.diff(ps, fs, 60.0)
    assert (r["kept"], r["moved"], r["deleted"], r["added"], r["placed"]) == (2, 1, 1, 1, 4)
    assert r["A"] == round(2 / 5, 3)
    moved = next(x for x in r["rows"] if x["status"] == "moved")
    assert moved["by"] == 0.5 and moved["moved_to"] == 20.5
    assert evaluate.diff([], [], 60.0)["A"] is None


def test_style_flags_a_rate_far_from_his():
    prof = {"rates": {"boom": {"body_per_min": 1.0}, "emote": {"body_per_min": 0.2}}}
    fs = [final("x", 60.0 + 6 * i) for i in range(40)]        # 40 booms in 4 min of body = 10/min
    notes = evaluate.style(fs, 300.0, prof)
    assert any(n.startswith("boom rate 10.00/min") for n in notes) and any(n.startswith("emote rate 0.00") for n in notes)


def test_corrections_feed_back_as_exemplars_and_negatives(tmp_path):
    p = tmp_path / "c.jsonl"
    rows = [{"status": "added", "type": "boom", "asset": None, "text": "a [*] b"},
            {"status": "deleted", "type": "emote", "asset": "sajj", "text": "c [*] d", "reason": "r"},
            {"status": "moved", "type": "boom", "asset": None, "text": "e [*] f"}]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    add, neg = evaluate.load_corrections(p)
    assert add == [{"type": "boom", "asset": None, "text": "a [*] b", "why": "Nguy added this by hand"}]
    assert [n["asset"] for n in neg] == ["sajj"]
    assert evaluate.load_corrections(tmp_path / "none.jsonl") == ([], [])


def test_bench_scores_same_type_nearest_within_tolerance():
    from refire.sauce import bench
    placed = [{"type": "boom", "t": 10.0}, {"type": "boom", "t": 20.0}, {"type": "emote", "t": 30.0, "asset": "sajj"}]
    his = [{"type": "boom", "t": 10.3}, {"type": "emote", "t": 30.1, "asset": "sajj"}, {"type": "boom", "t": 50.0}]
    loose = bench.score(placed, his, 0.5)
    assert (loose["matched"], loose["missed"], loose["emote_same_asset"]) == (2, 1, 1) and loose["A"] == 0.5
    strict = bench.score(placed, his, 3 / 60)
    assert strict["matched"] == 0 and strict["precision"] == 0.0
