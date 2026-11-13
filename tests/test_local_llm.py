"""local_llm: a busy GPU stalls or kills ollama mid-call; the call recovers or gives up cleanly."""
import pytest

from refire import local_llm


class _Client:
    """Each chat() pops the next behaviour: an exception to raise, or chunks to stream."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def chat(self, **kw):
        self.calls.append(kw)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return iter({"message": {"content": c}} for c in step)


@pytest.fixture
def unloads(monkeypatch):
    seen = []
    monkeypatch.setattr(local_llm, "unload", seen.append)
    return seen


def test_streams_and_caps_output(unloads):
    c = _Client([['{"a"', ': 1}']])
    assert local_llm.chat("m", [], num_predict=99, _client=c) == '{"a": 1}'
    assert c.calls[0]["stream"] is True and c.calls[0]["options"]["num_predict"] == 99
    assert unloads == []


def test_a_stall_reloads_the_model_then_retries(unloads, capsys):
    c = _Client([TimeoutError("read timed out"), ['ok']])     # game grabbed the vram
    assert local_llm.chat("qwen", [], tag="scout", _client=c) == "ok"
    assert unloads == ["qwen"] and len(c.calls) == 2
    assert "[scout] local model stalled or dropped" in capsys.readouterr().out


def test_gives_up_with_one_clear_error(unloads):
    c = _Client([ConnectionError("runner terminated"), ConnectionError("runner terminated")])
    with pytest.raises(local_llm.LocalLLMError, match="unavailable after 2 tries"):
        local_llm.chat("qwen", [], tag="glossary", _client=c)
    assert unloads == ["qwen"]                                   # once, between the tries


def test_a_runaway_generation_hits_the_total_ceiling(unloads, monkeypatch):
    t = iter(range(0, 10_000, 100))                              # each chunk "takes" 100s
    monkeypatch.setattr(local_llm.time, "monotonic", lambda: next(t))
    c = _Client([["x"] * 50, ["x"] * 50])
    with pytest.raises(local_llm.LocalLLMError, match="still generating"):
        local_llm.chat("m", [], total_s=250, _client=c)
