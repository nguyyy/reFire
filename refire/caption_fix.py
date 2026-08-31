"""Fix mistranscribed proper nouns in a built manifest's captions (one Claude pass).

The transcriber has no idea what game is on screen or who the streamer's friends are, so
proper nouns land mangled and a human fixes them by hand before editing. A real shipped
caption: `"dude, zhegef"` -- the streamer's friend Zajef. `glossary.game_glossary` only
*biases* the decoder, and `transcribe.snap_to_glossary` needs a matching first letter and
0.72 similarity, which "zhegef" -> "zajef" never clears.

This pass reads the captions WITH the context the transcriber never had -- game, glossary,
stream title, and the VOD's own chat, where viewers spell names correctly at the exact
second they are said -- and rewrites only what is wrong.

It edits `manifest.json` -> `clips[].captions[].text` in place, LINE FOR LINE: the line
count never changes and no timestamp is ever touched, so both consumers (After Effects'
`buildCaption`, and `srt.build_srt` -> Premiere) pick the fix up for free.

ponytail: the ffmpeg rough cut burns karaoke ASS straight from `words` and keeps the raw
text. It's a preview, not a deliverable. Fixing there means splicing N:M word spans and
redistributing timestamps -- do it only if the rough cut becomes something you ship.
"""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

from .director import CLAUDE_MODEL, _complete, _complete_cli, _JsonModel
from .glossary import _slug, parse_terms

# Caption lines per call. Measured: a 20-min cut is ~880 lines (captions are ~3 words), so
# this is 2-3 calls. Batching at all is only a guard against a long reply drifting off the
# line numbering -- the `was` echo in `vet` catches that anyway, so raise it if you'd rather
# pay for one call and give the model the whole cut as context.
BATCH = 400
MIN_RATIO = 0.5      # similarity floor: a fix is a re-spelling, not a rewrite
MAX_WORD_DELTA = 2   # a name may merge ("a bad oh" -> "albedo") but a line may not vanish

_SYSTEM = """\
You correct speech-to-text errors in short video captions from a live game stream.

The transcriber heard audio with no idea what game was on screen or who the streamer's \
friends are, so it mangles proper nouns: character names, places, items, abilities, \
in-game jargon, and the names of real people (other streamers, friends, chatters).

FIX ONLY:
- misheard proper nouns ("who tao" -> "hu tao", "kinech" -> "kinich", "zhegef" -> "zajef")
- homophones and near-homophones that are clearly wrong given the game and the chat context

NEVER:
- rephrase, reorder, summarize, translate or censor anything
- fix grammar, slang, stutters, filler words or informal speech -- streamers talk like that \
on purpose and the caption must match what was said
- add or remove punctuation, or change the number of lines
- guess. If you are not confident a word is wrong, leave the line alone.

WRITE EVERYTHING IN LOWERCASE, including names. Capitalization is applied later by the \
subtitle styler and yours would fight it.

Reply with ONLY this JSON object, containing just the lines you changed:
{"fixes": [{"i": <line number>, "was": "<the line exactly as it was given to you>", \
"text": "<the corrected line>"}]}
If nothing is wrong, reply {"fixes": []}."""


class Fix(_JsonModel):
    i: int
    was: str = ""
    text: str = ""


class Fixes(_JsonModel):
    fixes: list[Fix] = []


# --------------------------------------------------------------------------- learned


def corrections_path(corrections_dir, game: str) -> Path:
    return Path(corrections_dir) / f"corrections_{_slug(game)}.json"


def load_corrections(corrections_dir, game: str) -> dict[str, str]:
    """Phrase -> replacement, accumulated from previous runs. Lowercase keys."""
    p = corrections_path(corrections_dir, game)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(k).lower(): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def save_corrections(corrections_dir, game: str, new: dict[str, str]) -> None:
    """Merge `new` into the file. It only ever grows -- a bad entry is one line to delete."""
    if not new:
        return
    p = corrections_path(corrections_dir, game)
    merged = {**load_corrections(corrections_dir, game), **new}
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(sorted(merged.items())), indent=2), encoding="utf-8")
    except OSError:
        pass          # ponytail: learning is a bonus, never a reason to fail a run


def apply_corrections(text: str, corrections: dict[str, str]) -> str:
    """Whole-word, case-insensitive phrase replacement. Longest phrase wins."""
    if not corrections or not text:
        return text
    for key in sorted(corrections, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(key)}\b", corrections[key], text, flags=re.I)
    return text


MIN_LEARN_SRC = 4    # "pre" -> "pry" is a name split across two caption lines, not a rule
MIN_LEARN_DST = 5    # ...and neither is "doin" -> "dwen". Same >=5 bar snap_to_glossary uses


def learned_pairs(before: str, after: str) -> dict[str, str]:
    """Word-level diff of one accepted fix -> {"zhegef": "zajef"}.

    Storing the whole line would never match again; the substitution inside it will.

    These become BLIND rules on later runs, so the bar is higher than for a one-off fix the
    model made with the whole line in view. Captions are ~3 words, so a name straddling a
    line break gets corrected in halves ("they use pre" / "-doin as like" for prydwen) and
    those halves are junk as standalone rules -- and actively harmful, since "doin" is a
    word people say. The length floors are what separate a name from a fragment; anything
    they reject is still fixed this run and still fixable next run, just by the model
    rather than by rote.
    """
    a, b = before.lower().split(), after.lower().split()
    out = {}
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if op != "replace":
            continue
        # equal-length spans split into per-word rules, which generalize; an uneven span
        # (a name merged or split) has to stay one phrase
        pairs = (zip(a[i1:i2], b[j1:j2]) if i2 - i1 == j2 - j1
                 else [(" ".join(a[i1:i2]), " ".join(b[j1:j2]))])
        for src, dst in pairs:
            src, dst = src.strip(_STRIP), dst.strip(_STRIP)
            if (src != dst and len(src) >= MIN_LEARN_SRC and len(dst) >= MIN_LEARN_DST):
                out[src] = dst
    return out


# ------------------------------------------------------------------------- validation

_STRIP = " \t\"'.,!?:;()[]…-"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _ratio(a: str, b: str) -> float:
    keep = str.maketrans("", "", _STRIP)
    return difflib.SequenceMatcher(None, a.lower().translate(keep),
                                   b.lower().translate(keep)).ratio()


def _shout(word: str) -> bool:
    return word.isupper() and any(c.isalpha() for c in word)


def _recase(original: str, fixed: str) -> str:
    """Restore the source line's PER-WORD casing onto the correction.

    `emphasis.style_text` decides case one word at a time -- lowercase by default, UPPERCASE
    on excitement -- so a line is routinely mixed ("DPS and mawika", "UH, pyroin does").
    Re-casing the whole line would flatten that emphasis, which is a worse regression than
    the misspelling being fixed. Words the model inserted with no counterpart go lowercase.

    This also makes it impossible for the model to sneak capitals past the styler.
    """
    src, dst = original.split(), fixed.split()
    if not dst:
        return fixed
    up = [False] * len(dst)
    sm = difflib.SequenceMatcher(None, [w.lower() for w in src], [w.lower() for w in dst])
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":                      # spans are the same length here
            for j in range(j1, j2):
                up[j] = _shout(src[i1 + (j - j1)])
        elif op == "replace":
            # a name may be respelled across a different number of words; if any source
            # word in the span was shouted, the whole replacement is shouted
            shout = any(_shout(w) for w in src[i1:i2])
            for j in range(j1, j2):
                up[j] = shout
    return " ".join(w.upper() if u else w.lower() for w, u in zip(dst, up))


def vet(original: str, fix: Fix) -> str | None:
    """Accepted replacement text, or None if the fix fails any guard.

    Three guards, in order of what they catch:
      `was` echo   -- a stale or hallucinated line index, which would otherwise write a
                      correction onto a perfectly good and unrelated caption
      word delta   -- a line deleted, doubled, or rewritten into a different sentence
      similarity   -- a paraphrase wearing a correction's clothes

    ponytail: MIN_RATIO is a knob, not a truth. "zhegef"->"zajef" scores 0.55, so there is
    not much headroom above it; raise it only alongside a real miss.
    """
    new = (fix.text or "").strip()
    if not new or _norm(new) == _norm(original):
        return None
    if fix.was and _norm(fix.was) != _norm(original):
        return None
    if abs(len(new.split()) - len(original.split())) > MAX_WORD_DELTA:
        return None
    if _ratio(original, new) < MIN_RATIO:
        return None
    return _recase(original, new)


# ------------------------------------------------------------------------------ prompt


def _context(game: str, title: str, glossary: list[str], chat_terms_: list[str],
             names: list[str], corrections: dict[str, str]) -> str:
    out = ["The captions below come from a live stream."]
    if game:
        out.append(f"GAME: {game}")
    if title:
        out.append(f"STREAM TITLE: {title}")
    if glossary:
        out.append("KNOWN GAME TERMS (correct spellings):\n" + ", ".join(glossary))
    if chat_terms_:
        out.append("WORDS VIEWERS TYPED IN THIS STREAM'S CHAT (correctly spelled, but the "
                   "list also contains emotes and ordinary words -- use judgement):\n"
                   + ", ".join(chat_terms_))
    if names:
        out.append("PEOPLE IN THIS STREAM'S CHAT (the streamer says these aloud):\n"
                   + ", ".join(names))
    if corrections:
        out.append("CORRECTIONS ACCEPTED ON EARLIER EPISODES OF THIS STREAM:\n"
                   + "\n".join(f'  "{k}" -> "{v}"' for k, v in list(corrections.items())[:40]))
    return "\n\n".join(out)


def _ask(system: str, user: str, model: str, backend: str, trace=None) -> Fixes:
    """One completion. CLI (subscription) first, API key as the fallback -- the same
    arrangement `director.outline` uses."""
    content = [{"type": "text", "text": user}]
    if backend == "cli":
        try:
            return _complete_cli(model, system, content, Fixes, tag="captions", trace=trace)
        except (RuntimeError, OSError) as e:
            print(f"[captions] claude CLI unavailable ({e}); using API key")
    import anthropic          # optional heavy dep; no key -> raises, caller degrades

    return _complete(anthropic.Anthropic(), model, system, content, Fixes,
                     tag="captions", trace=trace)


# -------------------------------------------------------------------------------- main


def fix_lines(lines: list[str], llm: bool = True, game: str = "", title: str = "",
              terms: str = "", chat_json=None, model: str = CLAUDE_MODEL,
              backend: str = "cli", corrections_dir="run", trace=None,
              ask=_ask) -> tuple[list[str], dict[str, str]]:
    """Correct mistranscribed proper nouns in caption lines -> (fixed lines, learned).

    Line for line: the caller re-attaches timestamps by position, so the line count
    never changes. Two passes -- everything this stream has already taught us (free,
    instant, out of `corrections_<game>.json`), then one LLM pass over what is left.

    `llm=False` stops after the free pass, which is what the panel's Recaption does by
    default: a hand-recut timeline is usually the same words in a new order, already
    corrected on the run that built them, so a call per press is waste.
    """
    corrections = load_corrections(corrections_dir, game)
    # 1. deterministic pass: everything this stream has already taught us, for free
    out = []
    for t in lines:
        fixed = apply_corrections(t, corrections)
        # corrections are stored lowercase; put the styler's shouting back on top
        out.append(_recase(t, fixed) if fixed != t else t)

    learned: dict[str, str] = {}
    if not llm or not out:
        return out, learned

    # 2. one LLM pass over what's left
    ctx = _context(game, title, parse_terms(terms),
                   _chat_terms(chat_json), _chat_names(chat_json), corrections)
    for lo in range(0, len(out), BATCH):
        chunk = out[lo:lo + BATCH]
        user = (ctx + "\n\nCAPTION LINES:\n"
                + "\n".join(f"{lo + i}: {t}" for i, t in enumerate(chunk)))
        try:
            reply = ask(_SYSTEM, user, model, backend, trace)
        except Exception as e:
            # ponytail: same degradation as the director block -- a failed pass leaves the
            # captions exactly as built, it never fails the run.
            print(f"[captions] fix pass failed ({type(e).__name__}: {e}); captions unchanged")
            break
        for fix in reply.fixes:
            if not lo <= fix.i < lo + len(chunk):
                continue
            ok = vet(out[fix.i], fix)
            if ok:
                learned.update(learned_pairs(out[fix.i], ok))
                out[fix.i] = ok
    return out, learned


def fix_manifest(manifest_path, game: str = "", title: str = "", terms: str = "",
                 chat_json=None, model: str = CLAUDE_MODEL, backend: str = "cli",
                 corrections_dir="run", trace=None, ask=_ask) -> dict:
    """Correct `manifest.json`'s caption text in place. Returns the applied diff.

    Every timestamp and the line count are left exactly as they were. `ask` is injected so
    the self-check runs without an LLM. Writes `caption_fixes.json` beside the manifest so
    the change is auditable rather than silent, and merges what it learned into
    `<corrections_dir>/corrections_<game>.json` for next time.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    caps = [cap for clip in manifest.get("clips") or []
            for cap in clip.get("captions") or [] if (cap.get("text") or "").strip()]
    if not caps:
        return {}

    before = [c["text"] for c in caps]
    lines, learned = fix_lines(before, game=game, title=title, terms=terms,
                               chat_json=chat_json, model=model, backend=backend,
                               corrections_dir=corrections_dir, trace=trace, ask=ask)

    # write back
    diff = {str(i): {"before": b, "after": a}
            for i, (b, a) in enumerate(zip(before, lines)) if b != a}
    if diff:
        for cap, text in zip(caps, lines):
            cap["text"] = text
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        (manifest_path.parent / "caption_fixes.json").write_text(
            json.dumps(diff, indent=2), encoding="utf-8")
        save_corrections(corrections_dir, game, learned)
    print(f"[captions] {len(diff)}/{len(caps)} lines corrected"
          + (f" -> {manifest_path.parent / 'caption_fixes.json'}" if diff else ""))
    for d in list(diff.values())[:10]:
        print(f"  {d['before']!r} -> {d['after']!r}")
    return diff


def _chat_terms(chat_json) -> list[str]:
    if not chat_json:
        return []
    from .chat import chat_terms
    return chat_terms(chat_json)


def _chat_names(chat_json) -> list[str]:
    if not chat_json:
        return []
    from .chat import chat_names
    return chat_names(chat_json)


def _demo() -> None:
    import tempfile

    assert vet("dude, zhegef", Fix(i=0, was="dude, zhegef", text="dude, zajef"))
    # `was` doesn't match this line -> a stale index, drop it rather than corrupt a good line
    assert vet("dude, zhegef", Fix(i=0, was="something else", text="dude, zajef")) is None
    # a paraphrase is not a correction
    assert vet("dude, zhegef", Fix(i=0, was="", text="my friend arrived")) is None
    # the model cannot override the styler's casing decision
    assert vet("dude, zhegef", Fix(i=0, was="", text="Dude, Zajef")) == "dude, zajef"
    assert vet("DUDE, ZHEGEF", Fix(i=0, was="", text="dude, zajef")) == "DUDE, ZAJEF"
    assert vet("dude, zhegef", Fix(i=0, was="", text="dude, zhegef")) is None   # no-op
    assert learned_pairs("dude, zhegef", "dude, zajef") == {"zhegef": "zajef"}
    assert apply_corrections("Dude, ZHEGEF!", {"zhegef": "zajef"}) == "Dude, zajef!"
    assert apply_corrections("zhegefs", {"zhegef": "zajef"}) == "zhegefs"   # word-boundary

    m = {"clips": [{"start": 1.0, "end": 2.0, "captions": [
        {"text": "dude, zhegef", "start": 0.0, "end": 0.96},
        {"text": "is here", "start": 0.96, "end": 1.3}]}]}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        mp = d / "manifest.json"
        mp.write_text(json.dumps(m), encoding="utf-8")
        fake = lambda *_a, **_k: Fixes(fixes=[  # noqa: E731
            Fix(i=0, was="dude, zhegef", text="dude, zajef"),
            Fix(i=1, was="is here", text="has completely left the building"),  # rejected
        ])
        diff = fix_manifest(mp, game="genshin impact", corrections_dir=d, ask=fake)
        out = json.loads(mp.read_text(encoding="utf-8"))
        caps = out["clips"][0]["captions"]
        assert [c["text"] for c in caps] == ["dude, zajef", "is here"], caps
        assert [(c["start"], c["end"]) for c in caps] == [(0.0, 0.96), (0.96, 1.3)]
        assert list(diff) == ["0"]
        assert json.loads((d / "corrections_genshin-impact.json").read_text()) == {
            "zhegef": "zajef"}
        assert (d / "caption_fixes.json").exists()

        # second run: the learned file fixes it with no LLM call at all
        mp.write_text(json.dumps(m), encoding="utf-8")
        boom = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("called"))  # noqa: E731
        fix_manifest(mp, game="genshin impact", corrections_dir=d,
                     ask=lambda s, u, *a, **k: Fixes(fixes=[]))
        assert json.loads(mp.read_text())["clips"][0]["captions"][0]["text"] == "dude, zajef"

        # a failing LLM leaves the captions exactly as built
        mp.write_text(json.dumps(m), encoding="utf-8")
        assert fix_manifest(mp, corrections_dir=d, ask=boom) == {}
        assert json.loads(mp.read_text())["clips"][0]["captions"][0]["text"] == "dude, zhegef"
    print("caption_fix ok")


if __name__ == "__main__":
    _demo()
