"""Premiere smoke test for `refire punch` ops (docs/specs/01-punch.md "Applying in Premiere").

Needs Premiere open on a scratch sequence (>= 30 s of footage on V1) with the reFire panel
and a project item named refire_punch_adj. Builds ops for two fixed punches on the active
sequence's frame grid, applies them twice (idempotency), asserts V3 items, ticks, effect
params, and that a mid-punch frame has no black edges.

    uv run python tests/smoke/punch.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from refire import punch  # noqa: E402
from refire.ppro import remote  # noqa: E402
from refire.timeline_map import Span, TimelineMap  # noqa: E402
import spikes  # noqa: E402  (frame export + black-edge check)

PUNCHES = [{"id": "p01", "t_in": 5.0, "t_out": 7.5, "pin": [0.0, 1.0], "scale": 2.1},
           {"id": "p02", "t_in": 20.0, "t_out": 23.333, "pin": [0.0, 1.0], "scale": 1.8}]


# es3 has no JSON: tiny writer, then V3's refire items with their Transform/Unsharp values
DUMP = r"""
function J(v) { if (v === null || v === undefined) { return "null"; } var t = typeof v, a = [], k;
  if (t === "number" || t === "boolean") { return String(v); }
  if (t === "string") { return '"' + v.replace(/\\/g, "\\\\").replace(/"/g, '\\"') + '"'; }
  if (v.length !== undefined) { for (k = 0; k < v.length; k++) { a.push(J(v[k])); } return "[" + a.join(",") + "]"; }
  for (k in v) { a.push(J(k) + ":" + J(v[k])); } return "{" + a.join(",") + "}"; }
var tr = app.project.activeSequence.videoTracks[2], out = [];
for (var i = 0; i < tr.clips.numItems; i++) {
  var c = tr.clips[i];
  if (!c.projectItem || c.projectItem.name.replace(/^\s+|\s+$/g, "") !== "refire_punch_adj") { continue; }
  var r = {name: c.name, start: c.start.ticks, end: c.end.ticks, fx: []};
  for (var k = 0; k < c.components.numItems; k++) {
    var m = c.components[k], ps = [];
    if (m.matchName !== "AE.ADBE Geometry" && m.matchName !== "AE.ADBE Unsharp Mask") { continue; }
    for (var j = 0; j < m.properties.numItems; j++) { ps.push(m.properties[j].getValue()); }
    r.fx.push({m: m.matchName, p: ps});
  }
  out.push(r);
}
return J(out);
"""


def main() -> int:
    tpf = int(remote.call("app.project.activeSequence.timebase"))
    tm = TimelineMap([Span(0, 60, "x")], tpf)
    ops = punch.ops(PUNCHES, tm, punch.config())
    path = Path(tempfile.gettempdir()) / "refire_smoke_ops_punch.json"
    path.write_text(json.dumps(ops), encoding="utf-8")
    fails = 0

    def check(label, ok, got=""):
        nonlocal fails
        fails += not ok
        print(("PASS " if ok else "FAIL ") + label + (f": {got}" if got else ""))

    for rnd in (1, 2):
        print(remote.apply_ops(path))
        dump = json.loads(remote.call(DUMP))
        check(f"round {rnd}: exactly two reFire punch items on V3", len(dump) == 2, len(dump))
        for p, d in zip(PUNCHES, dump):
            a, b = tm.ticks(p["t_in"]), tm.ticks(p["t_out"])
            check(f"{p['id']} frame-exact", int(d["start"]) == a and int(d["end"]) == b, (d["start"], d["end"]))
            check(f"{p['id']} named", d["name"] == f"refire:punch:{p['id']}", d["name"])
            fx = {f["m"]: f["p"] for f in d["fx"]}
            g = fx.get("AE.ADBE Geometry", [])
            check(f"{p['id']} transform", len(g) == 12 and g[0] == p["pin"] and g[1] == p["pin"] and g[2] is True
                  and abs(g[3] - p["scale"] * 100) < 1e-6 and abs(g[4] - p["scale"] * 100) < 1e-6, g[:5])
            u = fx.get("AE.ADBE Unsharp Mask", [])
            check(f"{p['id']} unsharp", u[1:] == [50, 1, 0], u)
            check(f"{p['id']} one instance each", len(d["fx"]) == 2, [f["m"] for f in d["fx"]])
    spikes.OUT.mkdir(exist_ok=True)
    out = str(spikes.OUT).replace("\\", "/")
    for p in PUNCHES:
        mid = (p["t_in"] + p["t_out"]) / 2
        spikes.run(f"spike.frame({mid}, {json.dumps(out + '/smoke_' + p['id'] + '.png')})")
        edges = spikes.black_borders(spikes.OUT / f"smoke_{p['id']}.png")
        check(f"{p['id']} no black edges", not edges.get("black") and "error" not in edges, edges)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
