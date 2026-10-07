import json

from refire.caption_fix import (Fix, Fixes, apply_corrections, fix_manifest,
                                learned_pairs, vet)
from refire.chat import chat_lines, chat_names, chat_terms

# the real failure this is for: "zajef" is the streamer's friend so no game glossary has
# him, only chat and learned corrections can
_MANIFEST = {
    "clips": [
        {"start": 100.0, "end": 110.0, "captions": [
            {"text": "dude, zhegef", "start": 0.0, "end": 0.96},
            {"text": "is here", "start": 0.96, "end": 1.3}]},
        {"start": 200.0, "end": 230.0, "dur": 8.0, "captions": [
            {"text": "WHO TAO PULL", "start": 0.5, "end": 1.5}]},
    ],
    "sections": [{"title": "one", "clip_indices": [0, 1]}],
}


def _write(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(_MANIFEST), encoding="utf-8")
    return p


def _reply(*fixes):
    return lambda *_a, **_k: Fixes(fixes=list(fixes))


# --------------------------------------------------------------------------- guards


def test_stale_index_is_dropped_not_applied():
    # was doesn't match the line it points at -> fix would land on a good caption, drop it
    assert vet("is here", Fix(i=1, was="dude, zhegef", text="dude, zajef")) is None


def test_paraphrase_is_rejected():
    assert vet("is here", Fix(i=0, was="is here", text="has left the building")) is None


def test_respelling_is_accepted():
    assert vet("dude, zhegef", Fix(i=0, was="dude, zhegef",
                                   text="dude, zajef")) == "dude, zajef"


def test_model_cannot_override_styler_casing():
    # style_text owns casing: lowercase by default, caps on excitement
    assert vet("who tao pull", Fix(i=0, text="Hu Tao pull")) == "hu tao pull"
    assert vet("dude, zhegef", Fix(i=0, text="Dude, Zajef")) == "dude, zajef"
    assert vet("WHO TAO PULL", Fix(i=0, text="hu tao pull")) == "HU TAO PULL"


def test_per_word_shouting_survives_a_fix():
    # style_text shouts per word, flattening the line would kill emphasis
    assert vet("DPS and mawika", Fix(i=0, text="dps and mavuika")) == "DPS and mavuika"
    assert vet("UH, pyroin does", Fix(i=0, text="uh, prydwen does")) == "UH, prydwen does"
    assert vet("keneech IS", Fix(i=0, text="kinich is")) == "kinich IS"


def test_noop_fix_is_dropped():
    assert vet("is here", Fix(i=0, text="is here")) is None
    assert vet("is here", Fix(i=0, text="   ")) is None


# ---------------------------------------------------------------------- manifest I/O


def test_timestamps_and_line_count_survive(tmp_path):
    mp = _write(tmp_path)
    fix_manifest(mp, game="genshin impact", corrections_dir=tmp_path,
                 ask=_reply(Fix(i=0, was="dude, zhegef", text="dude, zajef"),
                            Fix(i=2, was="WHO TAO PULL", text="hu tao pull")))
    out = json.loads(mp.read_text(encoding="utf-8"))
    caps = [c for clip in out["clips"] for c in clip["captions"]]
    assert [c["text"] for c in caps] == ["dude, zajef", "is here", "HU TAO PULL"]
    assert [(c["start"], c["end"]) for c in caps] == [(0.0, 0.96), (0.96, 1.3), (0.5, 1.5)]
    # everything else in the manifest is untouched
    assert out["sections"] == _MANIFEST["sections"]
    assert out["clips"][1]["dur"] == 8.0


def test_writes_auditable_diff(tmp_path):
    mp = _write(tmp_path)
    diff = fix_manifest(mp, game="genshin impact", corrections_dir=tmp_path,
                        ask=_reply(Fix(i=0, was="dude, zhegef", text="dude, zajef")))
    assert diff == {"0": {"before": "dude, zhegef", "after": "dude, zajef"}}
    assert json.loads((mp.parent / "caption_fixes.json").read_text()) == diff


def test_llm_failure_leaves_captions_untouched(tmp_path):
    mp = _write(tmp_path)

    def boom(*_a, **_k):
        raise RuntimeError("no claude here")

    assert fix_manifest(mp, corrections_dir=tmp_path, ask=boom) == {}
    assert json.loads(mp.read_text()) == _MANIFEST


# ------------------------------------------------------------------------- learning


def test_learned_pairs_stores_the_word_not_the_line():
    # whole line would never match again, the substitution inside it will
    assert learned_pairs("dude, zhegef", "dude, zajef") == {"zhegef": "zajef"}
    # an even span splits into per-word rules that generalize
    assert learned_pairs("mwalani, sandroni yes", "mualani, sandrone yes") == {
        "mwalani": "mualani", "sandroni": "sandrone"}


def test_line_split_fragments_are_not_learned_as_rules():
    # "prydwen" across a line break gets fixed in halves, as rules those are junk, and "doin"
    # is a real word
    assert learned_pairs("they use pre", "they use pry") == {}
    assert learned_pairs("-doin as like", "-dwen as like") == {}
    # the fix still applies, it just isn't memorized
    assert vet("they use pre", Fix(i=0, text="they use pry")) == "they use pry"


def test_correction_is_learned_then_applied_with_no_llm(tmp_path):
    mp = _write(tmp_path)
    fix_manifest(mp, game="genshin impact", corrections_dir=tmp_path,
                 ask=_reply(Fix(i=0, was="dude, zhegef", text="dude, zajef")))
    assert json.loads(
        (tmp_path / "corrections_genshin-impact.json").read_text()) == {"zhegef": "zajef"}

    # second run on a fresh manifest: deterministic pass catches it before any call
    mp.write_text(json.dumps(_MANIFEST), encoding="utf-8")
    diff = fix_manifest(mp, game="genshin impact", corrections_dir=tmp_path,
                        ask=_reply())          # llm returns nothing
    assert diff["0"]["after"] == "dude, zajef"


def test_apply_corrections_respects_word_boundaries():
    corr = {"zhegef": "zajef"}
    assert apply_corrections("Dude, ZHEGEF!", corr) == "Dude, zajef!"
    assert apply_corrections("zhegefs", corr) == "zhegefs"


def test_learned_pass_also_keeps_shouting(tmp_path):
    # corrections are stored lowercase, shouted captions stay shouted
    (tmp_path / "corrections_genshin-impact.json").write_text(
        json.dumps({"mawika": "mavuika"}), encoding="utf-8")
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps({"clips": [{"captions": [
        {"text": "DPS and mawika", "start": 0.0, "end": 1.0}]}]}), encoding="utf-8")
    fix_manifest(mp, game="genshin impact", corrections_dir=tmp_path, ask=_reply())
    assert json.loads(mp.read_text())["clips"][0]["captions"][0]["text"] == "DPS and mavuika"


# ----------------------------------------------------------------------- chat mining


def _chat(tmp_path, bodies, emote_bodies=()):
    comments = [{"content_offset_seconds": i,
                 "commenter": {"display_name": "Zajef"},
                 "message": {"body": b, "fragments": [{"text": b, "emoticon": None}]}}
                for i, b in enumerate(bodies)]
    comments += [{"content_offset_seconds": 900 + i,
                  "commenter": {"display_name": "bot"},
                  "message": {"body": b, "fragments": [{"text": b, "emoticon": {"id": "1"}}]}}
                 for i, b in enumerate(emote_bodies)]
    p = tmp_path / "chat.json"
    p.write_text(json.dumps({"comments": comments}), encoding="utf-8")
    return p


def test_chat_terms_finds_names_and_skips_slang(tmp_path):
    p = _chat(tmp_path, ["Kinich pull", "Kinich!", "omg Kinich", "that that", "that",
                         "KEKW KEKW KEKW"],
              emote_bodies=["PogChamp", "PogChamp", "PogChamp"])
    terms = chat_terms(p, min_count=3)
    assert "Kinich" in terms
    assert "That" not in terms and "that" not in terms   # ordinary word, low cap ratio
    assert "KEKW" not in terms                           # all caps chat slang
    assert "PogChamp" not in terms                       # tagged emote fragment


def test_chat_names_are_the_people_a_glossary_cannot_hold(tmp_path):
    p = _chat(tmp_path, ["hi", "hello", "hey"])
    assert chat_names(p) == ["Zajef"]


def test_chat_lines_window_and_offset(tmp_path):
    p = _chat(tmp_path, ["a", "b", "c", "d"])
    assert chat_lines(p, 1, 3) == ["b", "c"]
    assert chat_lines(p, 0, 2, offset=2.0) == ["c", "d"]


def test_missing_chat_is_empty_not_fatal(tmp_path):
    missing = tmp_path / "nope.json"
    assert chat_terms(missing) == [] and chat_names(missing) == []
    assert chat_lines(missing, 0, 100) == []


def test_batches_run_in_parallel_and_one_failure_spares_the_rest(monkeypatch, tmp_path):
    """Each batch is its own call; a failed batch leaves only its own lines as built."""
    import threading

    from refire import caption_fix

    monkeypatch.setattr(caption_fix, "BATCH", 2)
    lines = ["the zhegef", "a", "the zhegef", "b", "the zhegef", "c"]
    tags, gate = [], threading.Barrier(3, timeout=5)   # all 3 batches in flight at once

    def ask(system, user, model, backend, trace=None, tag=""):
        tags.append(tag)
        gate.wait()
        first = int(user.split("CAPTION LINES:\n")[1].split(":")[0])
        if first == 2:
            raise RuntimeError("batch 2 down")
        return Fixes(fixes=[Fix(i=first, was="the zhegef", text="the zajef")])

    out, _learned = caption_fix.fix_lines(lines, corrections_dir=tmp_path, ask=ask)
    assert out == ["the zajef", "a", "the zhegef", "b", "the zajef", "c"]
    assert sorted(tags) == ["captions1", "captions2", "captions3"]   # distinct trace names
