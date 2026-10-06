"""Drift guard: every flag the AE panel emits must still exist in the CLI.

The panel (refire/ae/index.html) shells `python -m refire make ...`, so a flag
renamed in cli.py silently breaks the Make button with no Python-side failure.
This compares the two sources directly -- no argparse, no subprocess.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "refire" / "ae" / "index.html"
JSX = ROOT / "refire" / "ae" / "reFire.jsx"
CLI = ROOT / "refire" / "cli.py"

# the panel emits flags as bare "--flag" strings, css vars (--bg: ...) aren't quoted so they don't match
PANEL_FLAGS = set(re.findall(r'"(--[a-z][a-z-]+)"', PANEL.read_text(encoding="utf-8")))
# mk is the make subparser in cli.main()
CLI_FLAGS = set(re.findall(r'mk\.add_argument\(\s*"(--[a-z][a-z-]+)"',
                           CLI.read_text(encoding="utf-8")))


def test_panel_argv_builder_found():
    assert len(PANEL_FLAGS) > 10, "no flags parsed out of the panel -- regex or file moved?"
    assert len(CLI_FLAGS) > 10, "no flags parsed out of the make subparser"


def test_panel_flags_still_accepted():
    missing = sorted(PANEL_FLAGS - CLI_FLAGS)
    assert not missing, f"panel emits flags `refire make` no longer accepts: {missing}"


# --- reFire.jsx must never let an exception escape ---
# CEP turns any uncaught extendscript throw into a blank "EvalScript error." so every entry
# point catches and returns the real text. run the file with no AE globals (every call
# throws) and check we still get a string back
PROBE = """
const vm = require('vm'), fs = require('fs');
const ctx = {}; vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), ctx);
const out = {};
for (const fn of ['build', 'update', 'probe']) { out[fn] = ctx.reFire[fn]('x.json'); }
out.shift = ctx.reFire.shift('x.json', -0.15);
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run ExtendScript")
def test_jsx_api_never_throws(tmp_path):
    js = tmp_path / "refire_ae.js"
    js.write_text(JSX.read_text(encoding="utf-8"), encoding="utf-8")
    r = subprocess.run(["node", "-e", PROBE, str(js)], capture_output=True, text=True)
    assert r.returncode == 0, f"reFire.jsx let an exception escape:\n{r.stderr}"
    for name, val in json.loads(r.stdout).items():
        assert isinstance(val, str) and val, f"reFire.{name} returned {val!r}, not a message"
