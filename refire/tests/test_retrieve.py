from refire.retrieve import _cosine, retrieve_with_vec


def test_cosine_basics():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0   # zero vector -> 0, no div error


def test_cosine_leads_chat_no_longer_dominates():
    chunks = [
        {"text": "a", "embedding": [1.0, 0.0], "chat_z": 0.0},   # most on-topic
        {"text": "b", "embedding": [0.0, 1.0], "chat_z": 9.0},   # off-topic but loud
        {"text": "c", "embedding": [0.9, 0.1], "chat_z": 0.0},
        {"text": "d", "embedding": [0.8, 0.2], "chat_z": 0.0},
    ]
    top = retrieve_with_vec([1.0, 0.0], chunks, 3)
    # the on-topic chunk leads (a loud off-topic chunk can't outrank it)
    assert top[0]["text"] == "a"
    assert all("sim" in c for c in top)


def test_chat_safety_net_includes_lively_chunk():
    # the reserved chat slice surfaces the loud chunk even though cosine ranks it last
    chunks = [
        {"text": "a", "embedding": [1.0, 0.0], "chat_z": 0.0},
        {"text": "c", "embedding": [0.9, 0.1], "chat_z": 0.0},
        {"text": "d", "embedding": [0.8, 0.2], "chat_z": 0.0},
        {"text": "b", "embedding": [0.0, 1.0], "chat_z": 9.0},   # loud, low cosine
    ]
    top = [c["text"] for c in retrieve_with_vec([1.0, 0.0], chunks, 3)]
    assert "b" in top                # liveliness net keeps the loud chunk as a candidate
    assert "a" in top                # and the most on-topic chunk is still there


def test_empty_and_clamped_k():
    assert retrieve_with_vec([1.0], [], 5) == []
    chunks = [{"text": "a", "embedding": [1.0], "chat_z": 0.0}]
    assert len(retrieve_with_vec([1.0], chunks, 99)) == 1   # k clamped to pool size
