"""`--style` as free text or a style document, and the flag-beats-frontmatter rule.

This is a trust boundary -- the file comes from the user -- and every later phase reads
its knobs, so a silent misparse here ships the wrong cut with no error anywhere.
"""
import pytest

from refire import cli, styles


# --- resolve: literal text vs a document ---------------------------------
def test_empty_style_is_identity():
    assert styles.resolve(None) == (None, {})
    assert styles.resolve("   ") == (None, {})


def test_literal_text_passes_through():
    assert styles.resolve("snappy highlight reel") == ("snappy highlight reel", {})


def test_prose_that_looks_like_a_path_is_still_prose():
    # a missing path, and a paragraph long enough to make Path.is_file() raise on windows, both
    # come back as direction text, never an error
    assert styles.resolve("notes/nope.md")[0] == "notes/nope.md"
    long = "cut fast " * 400
    assert styles.resolve(long) == (long.strip(), {})


def test_document_splits_into_body_and_knobs(tmp_path):
    doc = tmp_path / "s.md"
    doc.write_text("---\npace: 0.35\nstack: 8\norder: whiplash\ncards: false\n---\n\n"
                   "# Style\nnever let a moment resolve", encoding="utf-8")
    body, knobs = styles.resolve(str(doc))
    assert knobs == {"pace": 0.35, "stack": 8, "order": "whiplash", "cards": False}
    assert body == "# Style\nnever let a moment resolve"


def test_document_without_frontmatter_is_all_prose(tmp_path):
    doc = tmp_path / "s.md"
    doc.write_text("# Style\njust prose", encoding="utf-8")
    assert styles.resolve(str(doc)) == ("# Style\njust prose", {})


# --- frontmatter parsing is forgiving, never fatal ---
@pytest.mark.parametrize("block, want", [
    ("---\nnope: 1\nstack: 4\n---\nx", {"stack": 4}),          # unknown key dropped
    ("---\npace: fast\nstack: 4\n---\nx", {"stack": 4}),        # bad value dropped
    ("---\nstack 4\nstack: 4\n---\nx", {"stack": 4}),           # no colon dropped
    ("---\n# a comment\n\nstack: 4\n---\nx", {"stack": 4}),     # comments + blanks
    ("---\nmotion-zoom: false\n---\nx", {"motion_zoom": False}),  # dashes normalize
    ("---\ncards: NO\nloudnorm: on\n---\nx", {"cards": False, "loudnorm": True}),
])
def test_bad_frontmatter_never_kills_the_run(block, want):
    body, knobs = styles.parse_frontmatter(block)
    assert knobs == want and body == "x"


@pytest.mark.parametrize("line, key, want", [
    ("pace: 0.35   # 1.5-4s shots", "pace", 0.35),
    ("order: whiplash  # no chronology", "order", "whiplash"),
    ("cards: false   # the pacing carries it", "cards", False),
    ("stack: 8 # best moment last", "stack", 8),
])
def test_trailing_comments_are_stripped(line, key, want):
    """A style document annotates why each number is what it is, next to the number --
    that is the whole reason it beats a wall of flags."""
    _b, knobs = styles.parse_frontmatter("---" + chr(10) + line + chr(10) + "---" + chr(10) + "x")
    assert knobs == {key: want}


@pytest.mark.parametrize("line", ["order: whiplashh", "snap: transiant", "captions: some"])
def test_a_misspelled_enum_is_rejected_not_silently_accepted(line):
    """The dangerous case: a bad float raises and gets reported, but an unchecked string
    would sail through and then take the else-branch -- no reordering, no error, a cut
    that silently ignored the document it was given."""
    _b, knobs = styles.parse_frontmatter("---" + chr(10) + line + chr(10) + "---" + chr(10) + "x")
    assert knobs == {}


def test_the_real_style_document_parses():
    """ryukStyl.md is checked in and is the reference case; a knob that stops reaching
    the machinery is exactly the silent failure this module exists to prevent."""
    from pathlib import Path

    doc = Path(__file__).resolve().parent.parent / "ryukStyl.md"
    if not doc.exists():
        pytest.skip("ryukStyl.md not present")
    body, knobs = styles.resolve(str(doc))
    assert set(knobs) <= set(styles.KEYS)
    assert knobs["stack"] >= 5 and knobs["pace"] < 1.0
    # chrono not whiplash: whiplash moved the stream's own sign-off into the middle (the video
    # ended twice)
    assert knobs["order"] == "chrono" and knobs["snap"] == "transient"
    assert knobs["keep_build"] is False
    assert body and body.lstrip().startswith("#") and "---" not in body.splitlines()[0]


def test_unclosed_fence_falls_back_to_prose():
    text = "---\npace: 1\nno closing fence"
    assert styles.parse_frontmatter(text) == (text, {})


# --- precedence: explicit flag beats the doc ---
def _make_kwargs(monkeypatch, argv):
    """Run `cli.main` far enough to capture what it would pass to `pipeline.make`."""
    seen = {}

    def fake_make(*a, **kw):
        seen.update(kw)
        raise SystemExit(0)          # stop before the run does any real work

    monkeypatch.setattr(cli, "make", fake_make)
    with pytest.raises(SystemExit):
        cli.main(argv)
    return seen


def test_frontmatter_supplies_defaults(monkeypatch, tmp_path):
    doc = tmp_path / "s.md"
    doc.write_text("---\npace: 0.35\norder: whiplash\nsnap: transient\nstack: 8\n"
                   "cards: false\nmotion_zoom: false\nwords_per_line: 2\n---\nbe fast",
                   encoding="utf-8")
    kw = _make_kwargs(monkeypatch, ["make", "1", "--duration", "5m", "--style", str(doc)])
    assert kw["style"] == "be fast"          # the body, not the raw path
    assert (kw["pace"], kw["order"], kw["snap"], kw["stack"]) == (0.35, "whiplash",
                                                                 "transient", 8)
    assert kw["cards"] is False and kw["motion_zoom"] is False
    assert kw["words_per_line"] == 2


def test_explicit_flag_beats_frontmatter(monkeypatch, tmp_path):
    doc = tmp_path / "s.md"
    doc.write_text("---\npace: 0.35\norder: whiplash\n---\nbe fast", encoding="utf-8")
    kw = _make_kwargs(monkeypatch, ["make", "1", "--duration", "5m", "--style", str(doc),
                                    "--pace", "1.5", "--order", "chrono"])
    assert kw["pace"] == 1.5 and kw["order"] == "chrono"


def test_no_style_is_the_identity_run(monkeypatch):
    kw = _make_kwargs(monkeypatch, ["make", "1", "--duration", "5m"])
    assert kw["style"] is None and kw["pace"] == 1.0
    assert (kw["order"], kw["snap"], kw["stack"]) == ("chrono", "sentence", 0)
    assert kw["keep_build"] is True and kw["loudnorm"] is False
    assert kw["cards"] is True and kw["motion_zoom"] is True
