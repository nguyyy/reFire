from refire.select import budget_select, parse_duration


def test_parse_duration_forms():
    assert parse_duration("1200") == 1200.0
    assert parse_duration("20m") == 1200.0
    assert parse_duration("90s") == 90.0
    assert parse_duration("1.5h") == 5400.0
    assert parse_duration("20:00") == 1200.0
    assert parse_duration("1:00:00") == 3600.0


def _clip(start, end, score):
    return {"start": start, "end": end, "score": score}


def test_budget_fills_to_target_highest_score_first_chrono_out():
    clips = [
        _clip(0, 60, 5.0),     # 60s
        _clip(300, 360, 9.0),  # 60s, best
        _clip(120, 180, 7.0),  # 60s
        _clip(600, 660, 3.0),  # 60s, weakest
    ]
    picked, warning = budget_select(clips, target_s=120, tol=0.25)
    assert warning is None
    # greedy picks the two highest scores (9,7) = 120s, then stops
    assert {c["score"] for c in picked} == {9.0, 7.0}
    # output is chronological
    assert [c["start"] for c in picked] == [120, 300]


def test_budget_warns_on_shortfall_without_padding():
    clips = [_clip(0, 30, 8.0), _clip(100, 130, 6.0)]   # only 60s total
    picked, warning = budget_select(clips, target_s=600, tol=0.25)
    assert len(picked) == 2          # keeps everything it has
    assert warning is not None and "not padding" in warning


def test_budget_order_score():
    clips = [_clip(0, 60, 5.0), _clip(300, 360, 9.0)]
    picked, _ = budget_select(clips, target_s=120, tol=0.25, order="score")
    assert [c["score"] for c in picked] == [9.0, 5.0]
