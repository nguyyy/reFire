"""Auto-derive a game-specific vocabulary to bias transcription.

Games full of proper nouns ("Hu Tao", "Liyue") get mistranscribed ("who tao").
One local-LLM pass turns a game name into a term list that faster-whisper takes
as `hotwords`/`initial_prompt`. Cached per game so it's generated once.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .score import DEFAULT_MODEL

_SYSTEM = (
    "You list proper nouns for speech-to-text biasing. Given a video game name, "
    "reply with its commonly-spoken proper nouns: character names, locations, "
    "items, abilities, and game-specific jargon a streamer says aloud. Prefer "
    "names that sound like ordinary words and get mistranscribed. Reply ONLY with "
    'JSON: {"terms": ["<term>", ...]} (20-60 terms, no duplicates).'
)


def _slug(game: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", game.strip().lower()).strip("-") or "game"


def parse_terms(spec: str) -> list[str]:
    """"Kinich, Arlecchino" -- or a path to a file with one term per line -- -> list."""
    spec = (spec or "").strip()
    if not spec:
        return []
    try:
        p = Path(spec)
        if p.is_file():
            spec = p.read_text(encoding="utf-8")
    except OSError:
        pass
    return [t.strip() for t in re.split(r"[,\n]", spec)
            if t.strip() and not t.startswith("#")]


def _llm_terms(game: str, model: str) -> list[str]:
    """One local-LLM pass: game name -> proper nouns. [] on any failure."""
    try:
        import ollama  # local import: optional heavy dep
        resp = ollama.chat(
            model=model, format="json",
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": f"Game: {game}"}],
        )
        data = json.loads(resp["message"]["content"])
        return [str(t).strip() for t in data.get("terms", []) if str(t).strip()]
    except Exception:
        # ponytail: any failure -> no glossary, transcription proceeds as before
        return []


def game_glossary(game: str, model: str = DEFAULT_MODEL,
                  cache_dir: str | Path = "run", extra: str = "") -> list[str]:
    """Game name -> proper-noun term list. Cached to cache_dir/glossary_<slug>.json.

    `extra` is a user-supplied term list (see parse_terms) merged ahead of the LLM's.
    It is the only lever for names the local model can't know -- llama3.1's training
    predates every character released since, so "Kinich" is simply absent from what
    it can list. Empty list for no game / no ollama / bad JSON, so transcription is
    unchanged.
    """
    user = parse_terms(extra)
    game = (game or "").strip()
    if not game:
        return user

    cache = Path(cache_dir) / f"glossary_{_slug(game)}.json"
    terms: list[str] = []
    if cache.exists():
        try:
            terms = json.loads(cache.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass  # regenerate on a corrupt cache
    if not terms:
        terms = _llm_terms(game, model)
        if terms:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(terms, indent=2), encoding="utf-8")
            except OSError:
                pass
    return list(dict.fromkeys(user + terms))  # user terms win ties, dedupe, keep order


def _demo() -> None:
    assert game_glossary("") == []
    assert game_glossary("", extra="Kinich, Arlecchino") == ["Kinich", "Arlecchino"]
    assert parse_terms("a,\n# note\nb ") == ["a", "b"]
    assert _slug("Genshin Impact") == "genshin-impact"
    print("glossary ok")


if __name__ == "__main__":
    _demo()
