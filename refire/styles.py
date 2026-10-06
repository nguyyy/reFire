"""`--style`: free text, or a path to a style document whose frontmatter drives the cut.

`--brief` says WHAT to cut; `--style` says HOW. Free text reaches the director's
judgment (it can even replace the arc mandate -- see `director._STYLE_RULE`), but prose
has no leverage over the numbers the machinery actually runs on: the chronological sort
in `narrative.cast`, the sentence snap in `select`, the single-teaser cold open. A real
editing style ("cut on the transient, 8-moment montage stack, never chronological") lives
almost entirely in those.

So a style may be a .md FILE: a frontmatter block of flat `key: value` lines sets the
machinery, and everything below it is the direction prose fed to the director. A style
with no frontmatter, or given as literal text, is prose-only -- exactly the old behavior.

Pure: no I/O beyond reading the file the user named, so the whole contract is unit-
testable and a malformed document fails loudly here instead of deep inside a paid run.
"""
from __future__ import annotations

from pathlib import Path

# frontmatter key -> coercion. closed set on purpose, an unknown key is a typo and a style
# doc that silently does nothing is worse than one that complains. cli flags still win
KEYS = {
    "pace": float,          # cut speed, scales ROLE_BUDGET + beat count together
    "stack": int,           # montage-stack cold open: how many moments (0 = one teaser)
    "truncate": float,      # transient snap: seconds to cut before the peak resolves
    "order": str,           # chrono | director | whiplash
    "snap": str,            # sentence | phrase | transient
    "cards": bool,          # section title cards
    "motion_zoom": bool,    # opencv motion scan + keyframed punch-ins
    "loudnorm": bool,       # ffmpeg loudness norm in the rough cut
    "deadspace": bool,      # strip each clip's internal silence
    "words_per_line": int,  # caption pacing
    "captions": str,        # all | emph (emph = only lines with an emphasized word)
    "keep_build": bool,     # enforce the skipped-build / slow-payoff flags
}

_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}

# keys with a fixed set of values. these get checked since a typo like order: whiplashh
# would pass as a string and quietly hit the else branch, ignoring the style with no error
_ENUMS = {
    "order": {"chrono", "director", "whiplash"},
    "snap": {"sentence", "phrase", "transient"},
    "captions": {"all", "emph"},
}

FENCE = "---"


def _strip_comment(raw: str) -> str:
    """Drop a trailing ` # ...` comment. A style document is a document -- annotating why
    each number is what it is, next to the number, is the whole reason it beats flags."""
    for i, ch in enumerate(raw):
        if ch == "#" and (i == 0 or raw[i - 1].isspace()):
            return raw[:i]
    return raw


def _coerce(key: str, raw: str):
    """One frontmatter value -> its typed form. Raises ValueError on a bad literal."""
    val = _strip_comment(raw).strip()
    kind = KEYS[key]
    if kind is bool:
        low = val.lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ValueError(f"{key}: expected true/false, got {val!r}")
    if key in _ENUMS:
        if val not in _ENUMS[key]:
            raise ValueError(f"{key}: expected one of "
                             f"{', '.join(sorted(_ENUMS[key]))}, got {val!r}")
        return val
    return kind(val)


def parse_frontmatter(text: str) -> tuple[str, dict]:
    """`text` -> (body, knobs). No leading `---` fence -> (text, {}) unchanged.

    Flat `key: value` lines only -- no nesting, no lists, no YAML dependency. That is the
    whole format, and it is the same shape the project's memory files already use. Blank
    lines and `#` comments are skipped; an unknown or unparseable key WARNS and is
    dropped rather than raising, because this is read at the top of a run that goes on to
    spend real director calls and a typo must not be what kills it.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != FENCE:
        return text, {}
    try:
        end = next(i for i, ln in enumerate(lines[1:], 1) if ln.strip() == FENCE)
    except StopIteration:
        print("[style] opening --- with no closing ---; treating the whole file as prose")
        return text, {}

    knobs: dict = {}
    for ln in lines[1:end]:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        key, sep, raw = s.partition(":")
        key = key.strip().lower().replace("-", "_")
        if not sep:
            print(f"[style] ignoring frontmatter line (no colon): {s!r}")
            continue
        if key not in KEYS:
            print(f"[style] ignoring unknown frontmatter key {key!r} "
                  f"(known: {', '.join(sorted(KEYS))})")
            continue
        try:
            knobs[key] = _coerce(key, raw)
        except ValueError as e:
            print(f"[style] ignoring bad frontmatter value -- {e}")
    return "\n".join(lines[end + 1:]).strip(), knobs


def resolve(spec: str | None) -> tuple[str | None, dict]:
    """`--style` -> (direction prose, machinery knobs).

    Literal text -> (text, {}). A readable file -> its body + frontmatter. Nothing ->
    (None, {}), the identity: no direction, no overrides, a byte-identical run.

    The file test mirrors `glossary.parse_terms` (the project's existing "literal or a
    path" idiom) rather than inventing a second one: `Path.is_file()` guarded against the
    OSError a long or illegal path raises on Windows, so a paragraph of prose that
    happens to look like a path can never blow up the run.
    """
    spec = (spec or "").strip()
    if not spec:
        return None, {}
    try:
        p = Path(spec)
        if p.is_file():
            body, knobs = parse_frontmatter(p.read_text(encoding="utf-8"))
            print(f"[style] {p.name}: {len(body.splitlines())} lines of direction"
                  + (f", knobs {knobs}" if knobs else ", no frontmatter"))
            return (body or None), knobs
    except OSError:
        pass                                   # too long / illegal / unreadable -> treat as prose
    return spec, {}


def _demo() -> None:
    import tempfile

    assert resolve(None) == (None, {}) and resolve("  ") == (None, {})
    assert resolve("snappy reel") == ("snappy reel", {})          # literal passthrough

    body, knobs = parse_frontmatter("---\npace: 0.35\nstack: 8\norder: whiplash\n"
                                    "cards: false\n---\n\n# Doc\nprose here")
    assert knobs == {"pace": 0.35, "stack": 8, "order": "whiplash", "cards": False}, knobs
    assert body == "# Doc\nprose here", repr(body)

    # no fence -> all prose, no knobs
    assert parse_frontmatter("# Doc\nprose") == ("# Doc\nprose", {})
    # unopened/unclosed fence is prose, not a crash
    assert parse_frontmatter("---\npace: 1\nno closing fence")[1] == {}
    # a bad value and an unknown key are both dropped, the rest survives
    _b, k = parse_frontmatter("---\npace: fast\nnope: 1\nstack: 4\n---\nx")
    assert k == {"stack": 4}, k
    # bool spellings
    _b, k = parse_frontmatter("---\ncards: NO\nloudnorm: on\n---\nx")
    assert k == {"cards": False, "loudnorm": True}, k

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                     encoding="utf-8") as f:
        f.write("---\nsnap: transient\n---\nbe fast")
        path = f.name
    assert resolve(path) == ("be fast", {"snap": "transient"}), resolve(path)
    Path(path).unlink()
    print("styles ok")


if __name__ == "__main__":
    _demo()
