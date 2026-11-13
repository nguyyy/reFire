"""Local-model (ollama) calls that survive a busy GPU.

When a game or a browser takes VRAM mid-call, ollama's runner either dies (the request
errors) or keeps going with its layers paged out over PCIe and slows to a crawl. From the
outside both look like a hang. Three rules:

- stream, and time the gap between tokens (`stall_s`), not the whole call: a model that is
  slow because the card is shared keeps going, one that stopped producing is caught.
  `total_s` is still a ceiling.
- cap the output (`num_predict`): a runaway generation can't run for ten minutes. The
  glossary call did exactly that on an idle card: 597s, nothing usable back.
- on a stall or a dead runner, unload the model and retry: ollama sizes how many layers sit
  on the GPU when it loads, so the reload fits the VRAM that is free now (a game running ->
  more on the CPU, slower, but moving). Still failing -> LocalLLMError, which every caller
  turns into its existing fallback (score 0, stub chapter, empty glossary, keyword emote).
"""
from __future__ import annotations

import time

RETRY_PAUSE_S = 5.0    # let the runner exit and the game's allocation settle before reloading


class LocalLLMError(RuntimeError):
    pass


def unload(model: str) -> None:
    """Drop `model` from VRAM so its next load is re-planned. Best effort."""
    try:
        import ollama
        client = ollama.Client(timeout=30)
        if any(m.model == model for m in client.ps().models):
            client.generate(model=model, keep_alive=0)
    except Exception:
        pass


def chat(model: str, messages: list[dict], *, fmt: str | None = "json",
         options: dict | None = None, num_predict: int = 1024, stall_s: float = 120.0,
         total_s: float = 600.0, keep_alive=None, retries: int = 1, tag: str = "llm",
         _client=None) -> str:
    """One chat call -> the reply text. Raises LocalLLMError once `retries` are spent.

    `stall_s` must cover a cold load plus prompt processing before the first token (a
    16k-token scout window on a part-offloaded 30B takes a minute or two)."""
    opts = {"num_predict": num_predict, **(options or {})}
    extra = {"keep_alive": keep_alive} if keep_alive is not None else {}
    last: Exception | None = None
    for attempt in range(retries + 1):
        t0 = time.monotonic()
        try:
            if _client is not None:
                client = _client
            else:
                import ollama
                client = ollama.Client(timeout=stall_s)   # httpx: max wait between chunks
            parts = []
            resp = client.chat(model=model, messages=messages, format=fmt or "",
                               options=opts, stream=True, **extra)
            for chunk in ([resp] if isinstance(resp, dict) else resp):   # dict = a stub
                parts.append(chunk["message"]["content"])
                if time.monotonic() - t0 > total_s:
                    raise TimeoutError(f"still generating after {total_s:.0f}s")
            return "".join(parts)
        except Exception as e:   # read timeout (stall), runner died, refused, total ceiling
            last = e
            if attempt < retries:
                print(f"[{tag}] local model stalled or dropped ({type(e).__name__}) -- GPU "
                      f"busy? reloading it to fit the VRAM free now, then retrying",
                      flush=True)
                unload(model)
                time.sleep(RETRY_PAUSE_S)
    raise LocalLLMError(f"{tag}: local model unavailable after {retries + 1} tries "
                        f"({type(last).__name__}: {last})")
