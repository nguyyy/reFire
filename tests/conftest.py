import pytest


@pytest.fixture(autouse=True)
def _no_local_llm_retry_pause(monkeypatch):
    """local_llm waits RETRY_PAUSE_S before reloading a stalled model; tests don't."""
    from refire import local_llm
    monkeypatch.setattr(local_llm, "RETRY_PAUSE_S", 0.0)
