"""Drift guard for the Premiere panel -- the AE one is tests/test_ae_panel.py.

The panel (refire/ppro/index.html) shells `python -m refire make ...`,
`python -m refire srt ...` and `python -m refire recap ...`, so a flag renamed in
cli.py silently breaks a button with no Python-side failure. This compares the two
sources directly.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "refire" / "ppro" / "index.html"
JSX = ROOT / "refire" / "ppro" / "reFirePpro.jsx"
CLI = ROOT / "refire" / "cli.py"

# the panel emits flags as bare "--flag" strings, css vars (--bg: ...) aren't quoted so they don't match
PANEL_FLAGS = set(re.findall(r'"(--[a-z][a-z-]+)"', PANEL.read_text(encoding="utf-8")))
# mk = make subparser in cli.main(), sr = srt, rc = recap
CLI_FLAGS = set(re.findall(r'(?:mk|sr|rc)\.add_argument\(\s*"(--[a-z][a-z-]+)"',
                           CLI.read_text(encoding="utf-8")))


def test_panel_argv_builder_found():
    assert len(PANEL_FLAGS) > 10, "no flags parsed out of the panel -- regex or file moved?"
    assert len(CLI_FLAGS) > 10, "no flags parsed out of the make/srt/recap subparsers"


def test_panel_flags_still_accepted():
    missing = sorted(PANEL_FLAGS - CLI_FLAGS)
    assert not missing, f"panel emits flags the CLI no longer accepts: {missing}"


def test_ae_only_features_are_forced_off():
    # zoom/emotes/sfx/cards never reach premiere, leaving them on wastes minutes of opencv + asset work
    for flag in ("--no-motion-zoom", "--no-emotes", "--no-sfx", "--no-cards"):
        assert flag in PANEL_FLAGS, f"{flag} no longer forced by the Premiere panel"


# --- reFirePpro.jsx must never let an exception escape ---
# CEP turns any uncaught extendscript throw into a blank "EvalScript error." so every entry
# point catches and returns the real text. run the file with no premiere globals (every call
# throws) and check we still get a string back
PROBE = """
const vm = require('vm'), fs = require('fs');
const ctx = {}; vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), ctx);
const out = {};
for (const fn of ['build', 'timeline', 'attach', 'probe']) { out[fn] = ctx.reFirePpro[fn]('x.json'); }
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run ExtendScript")
def test_jsx_api_never_throws(tmp_path):
    js = tmp_path / "refire_ppro.js"
    js.write_text(JSX.read_text(encoding="utf-8"), encoding="utf-8")
    r = subprocess.run(["node", "-e", PROBE, str(js)], capture_output=True, text=True)
    assert r.returncode == 0, f"reFirePpro.jsx let an exception escape:\n{r.stderr}"
    for name, val in json.loads(r.stdout).items():
        assert isinstance(val, str) and val, f"reFirePpro.{name} returned {val!r}"


def test_jsx_and_srt_agree_on_clip_order():
    """Both sides lay clips out by section, then by clip_indices. If they diverge,
    every caption after the first drifts against the picture."""
    jsx = JSX.read_text(encoding="utf-8")
    py = (ROOT / "refire" / "srt.py").read_text(encoding="utf-8")
    for token in ("clip_indices", "sections"):
        assert token in jsx and token in py
    # both must fall back to raw clip order when the manifest has no sections
    assert "if (!sections.length) { return clips; }" in jsx
    assert "if not sections:" in py


def test_recaption_button_is_wired():
    """The Recaption path is panel -> jsx timeline() -> `refire recap` -> jsx attach().
    Any one of those four renamed on its own leaves a button that does nothing."""
    panel = PANEL.read_text(encoding="utf-8")
    jsx = JSX.read_text(encoding="utf-8")
    cli = CLI.read_text(encoding="utf-8")
    assert 'id="recap"' in panel and '$("#recap").onclick' in panel
    assert "reFirePpro.timeline(" in panel and "timeline: guard(doTimeline)" in jsx
    assert 'ppro("attach"' in panel and "attach: guard(doAttach)" in jsx
    assert '"recap", manifest' in panel and 'sub.add_parser("recap"' in cli
    # the jsx writes what the python side reads
    assert "timeline.tsv" in jsx and "timeline.tsv" in (ROOT / "refire" / "srt.py").read_text(
        encoding="utf-8")


def test_every_pipeline_stage_is_on_the_bar():
    """The progress bar finds its segment by message prefix. A report() message no
    STAGES entry claims leaves the bar parked on the previous stage for the rest of
    the run, with nothing on the Python side to notice."""
    table = re.search(r"var STAGES = \[(.*?)\];", PANEL.read_text(encoding="utf-8"), re.S)
    assert table, "STAGES table moved -- re-point this test"
    # entries close with ]} since a bare ] would stop inside the "[review]" prefix
    prefixes = [p for m in re.findall(r"match: \[(.*?)\]\}", table.group(1), re.S)
                for p in re.findall(r'"([^"]+)"', m)]
    msgs = re.findall(r'report\([^,]+,\s*f?"([^"{]+)',
                      (ROOT / "refire" / "pipeline.py").read_text(encoding="utf-8"))
    assert len(msgs) > 10, "no report() messages parsed out of pipeline.py"
    loose = [m for m in msgs if m != "done" and not any(m.startswith(p) for p in prefixes)]
    assert not loose, f"report() messages no panel stage claims: {loose}"


def test_preview_shim_never_ships():
    """preview.py injects the fake Premiere into index.html in flight. The file on disk
    must never load it, or the real panel would run against a mock."""
    panel = PANEL.read_text(encoding="utf-8")
    assert "preview.js" not in panel
    assert panel.count("<head>") == 1, "preview.py injects right after <head>"


def test_chk_label_is_a_containing_block():
    """`.chk input` is position:absolute. Without a positioned ancestor its containing
    block is the INITIAL one, so the hidden input stops scrolling with `main`, pushes
    documentElement.scrollHeight past the viewport, and clicking a checkbox focuses an
    input the browser scrolls the whole panel off the top of the window to reach.
    Measured before the fix: html.scrollHeight 1460 vs innerHeight 767."""
    for panel in (PANEL, ROOT / "refire" / "ae" / "index.html"):
        css = panel.read_text(encoding="utf-8")
        assert ".chk input{position:absolute" in css, f"{panel.name}: rule moved, re-check"
        rule = re.search(r"^\.chk\{([^}]*)\}", css, re.M)
        assert rule, f"{panel.name}: no .chk rule"
        assert "position:relative" in rule.group(1), (
            f"{panel.name}: .chk must be positioned or the checkboxes scroll the panel away")
