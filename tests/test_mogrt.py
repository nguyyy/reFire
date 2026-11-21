"""caption MOGRT lint + per-caption bake (refire.ppro.mogrt, decision R5)."""
import base64
import re
import struct
from pathlib import Path

import pytest

from refire.ppro import mogrt

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "refire" / "assets" / "premiere" / "captMograph.mogrt"
BAKED_FX = ROOT / "tests" / "fixtures" / "captionMogrt.mogrt"


def _blob(t):
    m = re.search(r'<StartKeyframeValue Encoding="base64"[^>]*>([^<]+)<', t.xml)
    return base64.b64decode(m.group(1).strip())


def _text_at_ref(blob, old_len_hint=None):
    """follow every forward uoffset that lands on a length-prefixed utf-8 run; return the strings."""
    out = []
    for u in range(12, len(blob) - 4, 4):
        tgt = u + struct.unpack_from("<I", blob, u)[0]
        if tgt + 4 < len(blob):
            n = struct.unpack_from("<I", blob, tgt)[0]
            if 0 < n < 200 and tgt + 4 + n < len(blob) and blob[tgt + 4 + n] == 0:
                try:
                    out.append(blob[tgt + 4:tgt + 4 + n].decode("utf-8"))
                except UnicodeDecodeError:
                    pass
    return out


def test_lint_passes_clean_and_fails_the_baked_transform_export():
    t = mogrt.load(CLEAN)
    assert t.default_text == "list, BRO." and t.fps == 60 and t.fonts == ["CoolveticaRg-Regular"]
    assert mogrt.lint(t, seq_fps=60) in ([], ["font CoolveticaRg-Regular not installed"])
    assert any("fps" in w for w in mogrt.lint(t, seq_fps=30))
    with pytest.raises(mogrt.LintError, match="remove the Transform"):
        mogrt.lint(mogrt.load(BAKED_FX))


def test_bake_writes_the_text_into_the_blob_and_control(tmp_path):
    t = mogrt.load(CLEAN)
    p = mogrt.bake(t, "caption number 49 here", tmp_path)
    b = mogrt.load(p)
    assert b.default_text == "caption number 49 here"                 # definition.json control value
    blob = _blob(b)
    assert struct.unpack_from("<Q", blob)[0] == len(blob) - 12 and blob[8:12] == b"\x44\x33\x22\x11"
    strings = _text_at_ref(blob)
    assert "caption number 49 here" in strings and "list, BRO." not in strings   # old string orphaned
    assert "CoolveticaRg-Regular" in strings                          # style untouched
    mogrt.lint(b)                                                     # a baked copy still lints clean
    assert mogrt.bake(t, "caption number 49 here", tmp_path) == p     # same text -> same file
    assert mogrt.bake(t, "another", tmp_path) != p


def test_bake_handles_unicode_and_quotes(tmp_path):
    t = mogrt.load(CLEAN)
    b = mogrt.load(mogrt.bake(t, 'he said "NO WAY" — café', tmp_path))
    assert 'he said "NO WAY" — café' in _text_at_ref(_blob(b))


def test_patch_blob_refuses_unknown_text():
    with pytest.raises(mogrt.LintError):
        mogrt.patch_blob(b"\0" * 64, "nope", "x")
