"""Premiere smoke test for `refire sauce` ops (docs/specs/04-sauce.md section 6).

Needs Premiere open on a scratch sequence (>= 30 s on V1, >= 3 audio tracks, nothing of the user's on
V5 / A3 in 4-12 s) with the reFire panel. Renders one emote, builds ops for two booms + one emote on
the active sequence's frame grid, applies them twice (idempotency), asserts the A3 booms (names,
ticks, -14 dB) and the V5 emote (name, ticks), then that a foreign clip on V5 stops the apply.

    uv run python tests/smoke/sauce.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from refire.learn.prproj import TICKS  # noqa: E402
from refire.ppro import remote  # noqa: E402
from refire.sauce import detect, place  # noqa: E402
from refire.timeline_map import Span, TimelineMap  # noqa: E402

KEPT = [{"id": "b01", "type": "boom", "t": 5.0, "end": 6.25},
        {"id": "b02", "type": "boom", "t": 9.0, "end": 10.25},
        {"id": "e01", "type": "emote", "asset": "huhcat", "t": 9.0, "end": 10.4}]

DUMP = r"""
function J(v) { if (v === null || v === undefined) { return "null"; } var t = typeof v, a = [], k;
  if (t === "number" || t === "boolean") { return String(v); }
  if (t === "string") { return '"' + v.replace(/\\/g, "\\\\").replace(/"/g, '\\"') + '"'; }
  if (v.length !== undefined) { for (k = 0; k < v.length; k++) { a.push(J(v[k])); } return "[" + a.join(",") + "]"; }
  for (k in v) { a.push(J(k) + ":" + J(v[k])); } return "{" + a.join(",") + "}"; }
var s = app.project.activeSequence, out = {a3: [], v5: []};
function grab(tr, list, audio) {
  for (var i = 0; tr && i < tr.clips.numItems; i++) {
    var c = tr.clips[i], lvl = null;
    if (String(c.name).indexOf("refire:sauce:") !== 0) { continue; }
    for (var k = 0; audio && k < c.components.numItems; k++) {
      if (c.components[k].displayName !== "Volume") { continue; }
      for (var j = 0; j < c.components[k].properties.numItems; j++) {
        if (c.components[k].properties[j].displayName === "Level") { lvl = c.components[k].properties[j].getValue(); }
      }
    }
    list.push({name: c.name, start: c.start.ticks, end: c.end.ticks, level: lvl});
  }
}
grab(s.audioTracks.numTracks > 2 ? s.audioTracks[2] : null, out.a3, true);
grab(s.videoTracks.numTracks > 4 ? s.videoTracks[4] : null, out.v5, false);
return J(out);
"""


def main() -> int:
    tpf = int(remote.call("app.project.activeSequence.timebase"))
    tm = TimelineMap([Span(0, 60, "x")], tpf)
    cfg = place.config()
    lib = detect.library()
    mov = place.render_emote(lib["huhcat"], Path(tempfile.gettempdir()) / "refire_smoke_huhcat.mov", 306, 1.4, 0.9,
                             TICKS / tpf)
    ops = place.ops(KEPT, tm, {"huhcat": mov}, cfg)
    path = Path(tempfile.gettempdir()) / "refire_smoke_ops_sauce.json"
    path.write_text(json.dumps(ops), encoding="utf-8")
    level = ops["ops"][1]["level"]
    fails = 0

    def check(label, ok, got=""):
        nonlocal fails
        fails += not ok
        print(("PASS " if ok else "FAIL ") + label + (f": {got}" if got else ""))

    for rnd in (1, 2):
        print(remote.apply_ops(path))
        d = json.loads(remote.call(DUMP))
        check(f"round {rnd}: two booms on A3", len(d["a3"]) == 2, d["a3"])
        check(f"round {rnd}: one emote on V5", len(d["v5"]) == 1, d["v5"])
        for e, x in zip([k for k in KEPT if k["type"] == "boom"], d["a3"]):
            check(f"{e['id']} named, frame-exact", x["name"] == f"refire:sauce:{e['id']}" and int(x["start"]) == tm.ticks(e["t"])
                  and int(x["end"]) == tm.ticks(e["end"]), x)
            check(f"{e['id']} at -14 dB", x["level"] is not None and abs(float(x["level"]) - level) < 1e-3, x["level"])
        if d["v5"]:
            x, e = d["v5"][0], KEPT[2]
            check("e01 named, frame-exact", x["name"] == "refire:sauce:e01" and int(x["start"]) == tm.ticks(e["t"])
                  and int(x["end"]) == tm.ticks(e["end"]), x)
    # a foreign clip on V5 where an emote goes: the apply must stop, not overwrite it
    remote.call("var s = app.project.activeSequence, c = s.videoTracks[4].clips[0]; c.name = 'not ours'; 'ok'")
    try:
        res = remote.apply_ops(path)
        check("foreign V5 clip stops the apply", "where sauce goes" in str(res), res)
    except Exception as e:                                  # the remote raises on a jsx error
        check("foreign V5 clip stops the apply", "where sauce goes" in str(e), str(e)[:200])
    remote.call("var c = app.project.activeSequence.videoTracks[4].clips[0]; c.remove(false, false); 'ok'")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
