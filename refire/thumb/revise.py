"""`refire thumb revise <run_dir> <A|B|C> "<instruction>"` (03-thumb.md, Commands): one Claude call
turns a plain-language tweak into spec.json changes, then re-render + the deterministic checks.

The model sees the current thumb.png and spec.json and answers with screen-px moves for zy0x and
the left element (the critic's move_face / move_left: world units and the cam crop's bottom edge are
handled there) plus JSON-pointer edits for everything else. Edits may not point at assets that
aren't in the candidate folder. The old spec is kept as spec.prev.json; revise.jsonl logs each one.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from . import critic as crit

SYSTEM = """You tweak one YouTube thumbnail by editing its spec.json, which fully describes it (1280x720).
Read thumb.png first to see it. Do exactly what the instruction asks, nothing else.

Moves (screen px on the 1280x720 frame; 0 / 1.0 = leave it):
- face: {"dx", "dy", "scale"} moves/resizes zy0x about his face (his cut edge stays on the frame bottom;
  his anger mark and quote bubble follow him)
- left: {"dx", "dy", "scale"} moves/resizes the left element (scene.cards[0]: art or a designed card)

Edits: JSON pointers into spec.json, {"op": "set", "path": "/text/lines", "value": [...]} or
{"op": "remove", "path": "/layers/3"}; "/layers/-" appends. What the keys do:
- text.lines: the hook, ALL CAPS, usually 2 lines of ~10 characters (13 max); text.accent_phrase: the
  words in the accent color (must appear in the lines, or null); text.accent: blue|sky|green|mint|gold|cyan;
  text.annotation: small lowercase note or null; text.anchor: bottom-left|bottom-right; text.cap_frac:
  letter height / frame height (~0.14; bigger = bigger text); text.width_frac: widest line's width /
  frame width (cap); text.plane: the 3D tilt (leave it unless asked).
- layers: flat images over the scene, in order (later = on top): kind face (zy0x, glow = rim color),
  emote, quote (speech bubble), bubble, glyph (the pixel ? marks); x, y, w in screen px.
- scene: the 3D world behind: bg (blur, brightness, saturate of the backdrop), overlays (speed lines:
  opacity), cards (the left element; rot = [x, y, z] tilt degrees), sparkles (count), bloom (strength).
- background.vignette: edge darkening 0-1; shade: the dark band under the hook (alpha, height).
Never point a src at a file that isn't already in the folder. Use moves for position/size of zy0x and
the left element, not edits.

Reply with ONLY JSON:
{"note": "what you changed", "face": {"dx": 0, "dy": 0, "scale": 1.0}, "left": {"dx": 0, "dy": 0, "scale": 1.0},
 "edits": [{"op": "set", "path": "/text/lines", "value": ["LINE ONE", "LINE TWO"]}]}"""


def _models():
    from typing import Any

    from pydantic import BaseModel

    class Move(BaseModel):
        dx: float = 0.0
        dy: float = 0.0
        scale: float = 1.0

    class Edit(BaseModel):
        op: str = "set"
        path: str
        value: Any = None

    class Out(BaseModel):
        note: str = ""
        face: Move = Move()
        left: Move = Move()
        edits: list[Edit] = []

    return Out


def _walk(doc, parts: list[str]):
    for p in parts:
        doc = doc[int(p)] if isinstance(doc, list) else doc[p]
    return doc


def apply_edits(spec: dict, edits: list[dict]) -> list[str]:
    """JSON-pointer set/remove in place -> problems (empty = all applied)."""
    errs = []
    for e in edits:
        parts = [p.replace("~1", "/").replace("~0", "~") for p in e["path"].lstrip("/").split("/")]
        try:
            parent, key = _walk(spec, parts[:-1]), parts[-1]
            if e.get("op", "set") == "remove":
                del parent[int(key) if isinstance(parent, list) else key]
            elif isinstance(parent, list):
                if key == "-":
                    parent.append(e.get("value"))
                else:
                    parent[int(key)] = e.get("value")
            elif isinstance(parent, dict):
                parent[key] = e.get("value")
            else:
                raise TypeError(f"not a container: {type(parent).__name__}")
        except (KeyError, IndexError, ValueError, TypeError) as x:
            errs.append(f"{e.get('op', 'set')} {e['path']}: {type(x).__name__} {x}")
    return errs


def missing_assets(spec, d: Path) -> list[str]:
    """every src / raw string in the spec must be a file in the candidate folder."""
    out = []

    def visit(v):
        if isinstance(v, dict):
            for k, x in v.items():
                if k in ("src", "raw") and isinstance(x, str) and not (d / x).is_file():
                    out.append(x)
                visit(x)
        elif isinstance(v, list):
            for x in v:
                visit(x)
    visit(spec)
    return out


def edit(spec: dict, d: Path, out: dict) -> list[str]:
    """apply one model answer (moves, then pointer edits) to `spec` in place -> problems."""
    lo, hi = 0.3, 3.0                                  # sanity only: the instruction may ask for big moves
    for m, move in ((out["face"], crit.move_face), (out["left"], crit.move_left)):
        m = {"dx": float(m.get("dx", 0)), "dy": float(m.get("dy", 0)),
             "scale": max(lo, min(hi, float(m.get("scale", 1.0))))}
        if abs(m["dx"]) > 0.5 or abs(m["dy"]) > 0.5 or abs(m["scale"] - 1) > 1e-3:   # a no-op never re-snaps
            move(spec, d, m)
    errs = apply_edits(spec, out["edits"])
    errs += [f"no such asset: {a}" for a in missing_assets(spec, d)]
    t = spec.get("text") or {}
    ap = t.get("accent_phrase")
    if ap and ap.upper() not in " ".join(t.get("lines", [])).upper():
        errs.append(f"accent_phrase {ap!r} is not in the hook lines {t.get('lines')}")
    return errs


def revise(d, instruction: str, model: str = "claude-opus-5", backend: str = "cli", ask=None,
           ocr=None, log=print) -> dict:
    """one candidate folder + instruction -> spec edited, re-rendered, checked. ask(text, Out) stubs the call."""
    from .. import director
    from . import checks, render, report
    d = Path(d)
    spec = json.loads((d / "spec.json").read_text(encoding="utf-8"))
    Out = _models()
    user = (f"Instruction: {instruction}\n\nRead thumb.png in this folder. spec.json:\n"
            + json.dumps(spec, ensure_ascii=False))

    def call(text):
        if ask:
            return ask(text, Out)
        content = [{"type": "text", "text": text}]
        if backend == "cli":
            try:
                return director._complete_cli(model, SYSTEM, content, Out, tag="thumb revise", effort="low",
                                              live=False, tools="Read", cwd=d)
            except (RuntimeError, OSError) as e:
                director._api_fallback("thumb revise", e)
        import anthropic                                 # ponytail: the api path goes without the image
        return director._complete(anthropic.Anthropic(), model, SYSTEM, content, Out, tag="thumb revise", effort="low")

    t0 = time.monotonic()
    for attempt in range(2):                             # one corrective retry, like the director
        res = call(user).model_dump()
        new = json.loads(json.dumps(spec))
        errs = edit(new, d, res)
        if not errs:
            break
        user += "\n\nYour last answer broke these rules, fix them: " + "; ".join(errs)
    else:
        raise ValueError("thumb revise: " + "; ".join(errs))
    t1 = time.monotonic()
    shutil.copy(d / "spec.json", d / "spec.prev.json")
    (d / "spec.json").write_text(json.dumps(new, indent=1, ensure_ascii=False), encoding="utf-8")
    render.render(d / "spec.json")
    res_checks = checks.candidate(d, ocr)
    report.build(d.parent, ocr=ocr, root=d.parent)
    with (d / "revise.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%d %H:%M:%S"), "instruction": instruction, **res},
                           ensure_ascii=False) + "\n")
    note = res["note"].encode("ascii", "replace").decode()          # the console is cp1252
    log(f"[thumb revise] {d.name}: {note}  (render + checks {time.monotonic() - t1:.1f}s, "
        f"llm {t1 - t0:.0f}s)")
    for r in res_checks:
        log(f"  {r['name']} {'ok' if r['ok'] else 'FAIL'} ({r['value']}, need {r['need']})")
    return {"png": d / "thumb.png", "checks": res_checks, **res}
