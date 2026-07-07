"""Local vision model (Ollama) captions for sampled frames -- the free bulk pass.

A 3B VLM sees thousands of frames for $0 so Claude only has to look at contact sheets of
the finalists. Captions are one short phrase and treated as WEAK hints downstream (small
VLMs hallucinate on gameplay); the moment map marks them `[scene: ...]` and the director
prompt says they're approximate.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_VLM = "qwen2.5vl:3b"
_FLUSH_EVERY = 10   # cache flush cadence, so a crash resumes instead of restarting


def _degenerate(cap: str) -> bool:
    """A repeated-token collapse (@@@@, !!!!, GGGG) -- not a real caption.

    Small VLMs occasionally fall into emitting one character forever (VRAM pressure,
    a bad load). Such output is truthy, so without this guard it gets cached and, since
    the cache is skipped on reload, poisons every future run's scene map permanently.

    ponytail: unique-char count is a cheap entropy proxy; swap for a real perplexity /
    n-gram check only if a genuine 1-2-char caption ever needs to survive.
    """
    t = cap.strip()
    return len(t) >= 8 and len(set(t)) <= 2


def vlm_available(model: str = DEFAULT_VLM) -> bool:
    """Is `model` (exact or same base family) installed in Ollama?"""
    import ollama
    try:
        names = [m.model for m in ollama.list().models]
    except Exception:
        return False
    base = model.split(":")[0]
    return any(n == model or n.split(":")[0] == base for n in names)


def caption_frames(frames: list[dict], cache_path: str | Path,
                   model: str = DEFAULT_VLM, game: str = "",
                   progress=None) -> dict[float, str]:
    """{t: caption} for each sampled frame, via one local VLM call per frame.

    Incrementally cached to `cache_path` (keyed per model by the caller's filename), so
    re-runs and crash-resumes only pay for new frames. Missing model / Ollama down ->
    whatever the cache already has (possibly {}), with a printed skip -- vision degrades,
    the run never dies.
    """
    cache_path = Path(cache_path)
    caps: dict[float, str] = {}
    if cache_path.exists():
        try:
            caps = {float(k): v for k, v in
                    json.loads(cache_path.read_text(encoding="utf-8")).items()
                    if not _degenerate(v)}   # drop poison so those frames re-attempt
        except (ValueError, TypeError):
            caps = {}
    todo = [f for f in frames if float(f["t"]) not in caps]
    if not todo:
        return caps
    if not vlm_available(model):
        print(f"[vlm] '{model}' not installed (ollama pull {model}); "
              f"skipping scene captions")
        return caps
    import ollama
    where = f" from a {game} Twitch stream" if game else " from a Twitch gaming stream"
    prompt = ("One short phrase: what is happening on screen in this frame"
              + where + "? No preamble.")

    def flush():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({str(k): v for k, v in caps.items()}), encoding="utf-8")

    done = 0
    bad = 0
    try:
        for f in todo:
            resp = ollama.chat(model=model, messages=[
                {"role": "user", "content": prompt, "images": [str(f["path"])]}])
            raw = (resp["message"]["content"] or "").strip().splitlines()
            cap = raw[0][:120] if raw else ""
            if cap and not _degenerate(cap):
                caps[float(f["t"])] = cap
            elif cap:
                bad += 1   # model collapsed on this frame -- don't cache it
            done += 1
            if progress:
                progress(done / len(todo), f"captioning frame {done}/{len(todo)}")
            if done % _FLUSH_EVERY == 0:
                flush()
    except Exception as e:
        print(f"[vlm] captioning stopped after {done}/{len(todo)} ({e}); "
              f"keeping what we have")
    if bad:
        print(f"[vlm] {bad} frames returned degenerate output; left uncaptioned")
    flush()
    return caps
