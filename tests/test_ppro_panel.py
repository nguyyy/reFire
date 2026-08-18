"""Drift guard for the Premiere panel -- the AE one is tests/test_ae_panel.py.

The panel (refire/ppro/index.html) shells `python -m refire make ...` and
`python -m refire srt ...`, so a flag renamed in cli.py silently breaks a button
with no Python-side failure. This compares the two sources directly.
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

# the panel's argv builders emit flags as bare "--flag" string literals; CSS custom
# properties (--bg: ...) are never double-quoted, so they don't match.
PANEL_FLAGS = set(re.findall(r'"(--[a-z][a-z-]+)"', PANEL.read_text(encoding="utf-8")))
# `mk` is the `make` subparser in cli.main(), `sr` is `srt`
CLI_FLAGS = set(re.findall(r'(?:mk|sr)\.add_argument\(\s*"(--[a-z][a-z-]+)"',
                           CLI.read_text(encoding="utf-8")))


def test_panel_argv_builder_found():
    assert len(PANEL_FLAGS) > 10, "no flags parsed out of the panel -- regex or file moved?"
    assert len(CLI_FLAGS) > 10, "no flags parsed out of the make/srt subparsers"


def test_panel_flags_still_accepted():
    missing = sorted(PANEL_FLAGS - CLI_FLAGS)
    assert not missing, f"panel emits flags the CLI no longer accepts: {missing}"


def test_ae_only_features_are_forced_off():
    # zoom/emotes/sfx/cards never reach Premiere; leaving them on would burn minutes
    # of OpenCV scan and asset work for output this panel throws away.
    for flag in ("--no-motion-zoom", "--no-emotes", "--no-sfx", "--no-cards"):
        assert flag in PANEL_FLAGS, f"{flag} no longer forced by the Premiere panel"


# --- reFirePpro.jsx must never let an exception escape --------------------
# CEP turns any uncaught ExtendScript throw into the opaque "EvalScript error."
# with no message, so every public entry point is wrapped in a catch that returns
# the real text. Run the real file with NO Premiere globals defined -- every host
# call throws -- and assert we still get a string back.
PROBE = """
const vm = require('vm'), fs = require('fs');
const ctx = {}; vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), ctx);
const out = {};
for (const fn of ['build', 'probe']) { out[fn] = ctx.reFirePpro[fn]('x.json'); }
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
