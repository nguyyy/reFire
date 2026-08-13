from refire.pipeline import _run_name


def test_run_name_includes_vod_id_and_varies():
    name = _run_name(2824313956)
    assert name.startswith("2824313956-")
    parts = name.split("-")
    assert len(parts) == 3   # vod_id, adjective, noun

    # not deterministic -- a run of samples should hit more than one phrase
    assert len({_run_name(123) for _ in range(30)}) > 1
