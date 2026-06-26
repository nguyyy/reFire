import json

from refire.glossary import _slug, game_glossary


def test_no_game_no_glossary():
    assert game_glossary("") == []
    assert game_glossary("   ") == []


def test_slug():
    assert _slug("Genshin Impact!") == "genshin-impact"
    assert _slug("") == "game"


def test_cache_is_reused_without_ollama(tmp_path):
    # a pre-existing cache short-circuits the LLM call entirely
    (tmp_path / "glossary_genshin-impact.json").write_text(
        json.dumps(["Hu Tao", "Liyue"]), encoding="utf-8")
    assert game_glossary("Genshin Impact", cache_dir=tmp_path) == ["Hu Tao", "Liyue"]
