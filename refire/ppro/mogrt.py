"""The caption MOGRT: lint it before every apply, bake one copy per caption (decision R5).

A Premiere-made MOGRT is a zip: definition.json (controls, fps, fonts) + project.prgraphic
(a zip holding a gzipped .prproj). The text sits in the `Source Text` param as a base64
blob: u64 payload length, magic 44 33 22 11, then a FlatBuffer whose strings are u32
length + utf-8 + NUL. Premiere won't take a new blob through setValue (spike S3b), so we
write the text into a copy of the file and import that.
"""
from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import io
import json
import os
import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

LINT_FAIL = "Caption MOGRT contains animation/effects — remove the Transform from the graphic and re-export."
TICKS = 254_016_000_000


class LintError(ValueError):
    pass


@dataclass
class Template:
    path: Path
    files: dict                 # outer zip name -> bytes
    definition: dict
    prproj_name: str
    xml: str
    control: int                # index of the text control in clientControls
    default_text: str

    @property
    def fps(self) -> float:
        tpf = self.definition["sourceInfoLocalized"]["en_US"]["framerate"]["ticksperframe"]
        return TICKS / tpf

    @property
    def fonts(self) -> list[str]:
        return self.definition.get("usedFontsLocalized", {}).get("en_US", [])


def load(path, text_param: str = "TextLayer") -> Template:
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        files = {i.filename: z.read(i.filename) for i in z.infolist()}
    d = json.loads(files["definition.json"])
    with zipfile.ZipFile(io.BytesIO(files["project.prgraphic"])) as inner:
        (name,) = [n for n in inner.namelist() if n.endswith(".prproj")]
        xml = gzip.decompress(inner.read(name)).decode("utf-8")
    for i, c in enumerate(d.get("clientControls", [])):
        if any(s.get("str") == text_param for s in c.get("uiName", {}).get("strDB", [])):
            return Template(path, files, d, name, xml, i, c["value"]["strDB"][0]["str"])
    raise LintError(f"{path.name}: no text control named {text_param!r}")


def _font_installed(ps_name: str) -> bool:
    """loose: the postscript family (CoolveticaRg-Regular -> coolveticarg) prefixes a font file name."""
    fam = re.sub(r"[^a-z0-9]", "", ps_name.split("-")[0].lower())
    dirs = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts"]
    return any(re.sub(r"[^a-z0-9]", "", f.stem.lower()).startswith(fam)
               for d in dirs if d.is_dir() for f in d.iterdir())


def lint(t: Template, seq_fps: float | None = None) -> list[str]:
    """raise LintError on baked animation/effects; return warnings (fps, fonts)."""
    if re.search(r"<IsTimeVarying>\s*true\s*</IsTimeVarying>", t.xml) or \
            any(m != "AE.ADBE Text" for m in re.findall(r"<MatchName>([^<]+)</MatchName>", t.xml)):
        raise LintError(LINT_FAIL)
    warn = []
    if seq_fps and abs(t.fps - seq_fps) > 0.01:
        warn.append(f"caption MOGRT is {t.fps:g} fps, the sequence {seq_fps:g} fps")
    warn += [f"font {f} not installed" for f in t.fonts if not _font_installed(f)]
    return warn


# --------------------------------------------------------------------------- bake

def patch_blob(blob: bytes, old: str, new: str) -> bytes:
    """swap FlatBuffer string `old` for `new`: append it and repoint the one uoffset that
    targeted the old one (offsets only point forward, so nothing else moves)."""
    raw = old.encode("utf-8")
    s = blob.find(struct.pack("<I", len(raw)) + raw)
    if s < 0:
        raise LintError(f"text {old!r} not found in the Source Text blob")
    refs = [u for u in range(12, s, 4) if u + struct.unpack_from("<I", blob, u)[0] == s]
    if len(refs) != 1:
        raise LintError(f"Source Text blob layout not understood ({len(refs)} refs to the text)")
    out = bytearray(blob)
    out += b"\0" * ((-len(out)) % 4)
    pos = len(out)
    enc = new.encode("utf-8")
    out += struct.pack("<I", len(enc)) + enc + b"\0"
    out += b"\0" * ((-len(out)) % 4)
    struct.pack_into("<I", out, refs[0], pos - refs[0])
    struct.pack_into("<Q", out, 0, len(out) - 12)
    return bytes(out)


def _with_text(t: Template, text: str) -> bytes:
    xml = t.xml
    for m in re.finditer(r'(<StartKeyframeValue Encoding="base64"[^>]*>)([^<]+)(<)', xml):
        blob = base64.b64decode(m.group(2).strip())
        if t.default_text.encode("utf-8") in blob:
            xml = xml[:m.start(2)] + base64.b64encode(patch_blob(blob, t.default_text, text)).decode() + xml[m.end(2):]
            break
    else:
        raise LintError(f"no Source Text blob holds the template text {t.default_text!r}")
    with zipfile.ZipFile(io.BytesIO(t.files["project.prgraphic"])) as inner:
        info = copy.copy(inner.getinfo(t.prproj_name))
    ib = io.BytesIO()
    with zipfile.ZipFile(ib, "w", zipfile.ZIP_DEFLATED) as zi:
        zi.writestr(info, gzip.compress(xml.encode("utf-8"), mtime=0))
    d = json.loads(json.dumps(t.definition))
    d["clientControls"][t.control]["value"]["strDB"][0]["str"] = text
    ob = io.BytesIO()
    with zipfile.ZipFile(t.path) as src, zipfile.ZipFile(ob, "w") as zo:
        for info in src.infolist():
            data = {"project.prgraphic": ib.getvalue(),
                    "definition.json": json.dumps(d).encode("utf-8")}.get(info.filename, t.files[info.filename])
            zo.writestr(copy.copy(info), data)
    return ob.getvalue()


def bake(t: Template, text: str, out_dir) -> Path:
    """<out_dir>/<hash>.mogrt holding `text` (reused when the same text comes back)."""
    key = hashlib.sha1(f"{t.path.name}\0{len(t.files['project.prgraphic'])}\0{text}".encode("utf-8")).hexdigest()[:12]
    out = Path(out_dir) / f"{key}.mogrt"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(_with_text(t, text))
    return out

