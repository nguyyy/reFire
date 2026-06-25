from refire.rank import _chat_to_10, rank_select


def _item(start, end, llm, chat_z, reason="r"):
    return {"start": start, "end": end, "llm_score": llm,
            "chat_z": chat_z, "reason": reason}


def test_sorted_descending_and_blended():
    scored = [
        _item(0, 30, llm=2, chat_z=0),    # low
        _item(30, 60, llm=9, chat_z=3),   # high on both
        _item(60, 90, llm=5, chat_z=1),   # mid
    ]
    segs = rank_select(scored, w_llm=0.6, w_chat=0.4)
    scores = [s["score"] for s in segs]
    assert scores == sorted(scores, reverse=True)
    assert segs[0]["start"] == 30  # highest blended
    for s in segs:
        assert s["start"] < s["end"]


def test_top_n_and_threshold():
    scored = [_item(i, i + 10, llm=i, chat_z=0) for i in range(1, 6)]
    assert len(rank_select(scored, top_n=2)) == 2
    high = rank_select(scored, threshold=4.0)
    assert all(s["score"] >= 4.0 for s in high)


def test_chat_norm_clamped():
    assert _chat_to_10(-5) == 0.0
    assert _chat_to_10(10) == 10.0
    assert _chat_to_10(0) == 4.0
