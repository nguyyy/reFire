"""Premiere smoke test for `refire caption` ops (docs/specs/02-caption.md acceptance).

Needs Premiere open on a scratch MAIN sequence (>= 20 s on V1, >= 6 video tracks) with the
reFire panel. Builds ops for three fixed blocks (one overlap) on the active sequence's frame
grid, applies them twice (caption tracks + Upgrade Caption to Graphic, decision R6), and checks:
the refire_captions nest exists with exactly the blocks (names, times, one graphics track per
layer above the anchor on V1), one nest clip on V6 spanning main with ccStyle applied once, each
block's Transform keyed with the shutter flag off, the nest rebuilt identically on re-apply,
and frames mid-slide vs at rest (motion blur) exported for a look.

    uv run python tests/smoke/caption.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from refire import caption  # noqa: E402
from refire.ppro import mogrt, prfpset, remote  # noqa: E402
from refire.timeline_map import TICKS, Span, TimelineMap  # noqa: E402
import spikes  # noqa: E402  (frame export)
from punch import DUMP as _J  # noqa: E402  (reuse the es3 json writer)

BLOCKS = [("b001", 2.0, 4.0, "smoke caption one"), ("b002", 3.5, 6.0, "OVERLAP on top"),
          ("b003", 8.0, 8.3, "short")]
J = _J[:_J.index("var tr =")]
DUMP = J + r"""
var nest = null, i, v, c, out = {blocks: [], v5: [], style: 0};
for (i = 0; i < app.project.sequences.numSequences; i++) if (app.project.sequences[i].name === "refire_captions") nest = app.project.sequences[i];
if (!nest) return J({error: "no nest"});
for (v = 0; v < nest.videoTracks.numTracks; v++) for (c = 0; c < nest.videoTracks[v].clips.numItems; c++) {
  var ti = nest.videoTracks[v].clips[c], keys = 0, flag = null;
  for (var k = 0; k < ti.components.numItems; k++) if (ti.components[k].matchName === "AE.ADBE Geometry") {
    var g = ti.components[k]; flag = g.properties[9].getValue();
    try { keys = g.properties[1].getKeys().length + g.properties[8].getKeys().length; } catch (e) {} }
  out.blocks.push({name: ti.name, v: v, start: ti.start.ticks, end: ti.end.ticks, keys: keys, shutter_flag: flag});
}
var m = app.project.activeSequence, tr = m.videoTracks[5];
for (c = 0; c < tr.clips.numItems; c++) if (tr.clips[c].projectItem && tr.clips[c].projectItem.nodeId === nest.projectItem.nodeId) {
  var cl = tr.clips[c], n = 0;
  for (var k2 = 0; k2 < cl.components.numItems; k2++) if (/Impact|Drop Shadow/.test(cl.components[k2].matchName)) n++;
  out.v5.push({start: cl.start.ticks, end: cl.end.ticks, fx: n});
}
// main's clips, not m.end: a caption track on main (invisible to the DOM) can run past them
var me = 0, G = [m.videoTracks, m.audioTracks];
for (var g = 0; g < 2; g++) for (var t = 0; t < G[g].numTracks; t++) for (c = 0; c < G[g][t].clips.numItems; c++) {
  var cc = G[g][t].clips[c]; if (!(cc.projectItem && cc.projectItem.nodeId === nest.projectItem.nodeId)) me = Math.max(me, Number(cc.end.ticks)); }
out.main_end = String(me);
return J(out);
"""


def main() -> int:
    tpf = int(remote.call("app.project.activeSequence.timebase"))
    tm = TimelineMap([Span(0, 60, "x")], tpf)
    cfg = caption.config()
    (fx,) = [e for e in prfpset.load(cfg["preset_in"]) if "Geometry" in e.match_name]
    rs, ti, to = caption.ramps(fx)
    out_dir = Path(tempfile.gettempdir()) / "refire_smoke_captions"
    blocks = caption.layer([{"id": i, "t_in": a, "t_out": b, "text": t} for i, a, b, t in BLOCKS], 4)
    out_dir.mkdir(exist_ok=True)
    files = []
    for k in range(1, max(b["track"] for b in blocks) + 1):
        f = out_dir / f"layer_{k}.dfxp"
        f.write_text(caption.ttml([b for b in blocks if b["track"] == k], tm, cfg), encoding="utf-8")
        files.append(str(f).replace("\\", "/"))
    for b in blocks:
        L = (tm.ticks(b["t_out"]) - tm.ticks(b["t_in"])) / TICKS
        b["keys"], b["interp"], b["k"] = caption.block_keys(L, rs, ti, to, fx.duration, TICKS / tpf)
    anchor = str(mogrt.bake(mogrt.load(cfg["mogrt"]), " ", out_dir)).replace("\\", "/")
    path = out_dir / "ops_caption.json"
    path.write_text(json.dumps(caption.ops(blocks, tm, cfg, fx, prfpset.load(cfg["preset_style"]), files, False, anchor)))
    fails = 0

    def check(label, ok, got=""):
        nonlocal fails
        fails += not ok
        print(("PASS " if ok else "FAIL ") + label + (f": {got}" if got else ""))

    dumps = []
    for rnd in (1, 2):
        print(remote.apply_ops(path, chunk=60))
        d = json.loads(remote.call(DUMP))
        dumps.append(d)
        got = sorted((x for x in d["blocks"] if x["name"].startswith("refire:caption:b")), key=lambda x: x["name"])
        check(f"round {rnd}: {len(blocks)} blocks in the nest", len(got) == len(blocks), [x["name"] for x in got])
        for b, x in zip(blocks, got):
            check(f"{b['id']} frame-exact on V{b['track']}", int(x["start"]) == tm.ticks(b["t_in"])
                  and int(x["end"]) == tm.ticks(b["t_out"]) and x["v"] == b["track"], x)
            check(f"{b['id']} keyed, motion blur on", x["keys"] == sum(len(v) for v in b["keys"].values())
                  and x["shutter_flag"] is False, (x["keys"], x["shutter_flag"]))
        check(f"round {rnd}: one nest clip on V6 spanning main", len(d["v5"]) == 1 and int(d["v5"][0]["start"]) == 0
              and int(d["v5"][0]["end"]) == int(d["main_end"]), d["v5"])
        check(f"round {rnd}: ccStyle applied once", d["v5"] and d["v5"][0]["fx"] == 3, d["v5"])
    strip = lambda d: [(x["name"], x["start"], x["end"], x["v"]) for x in d["blocks"]]   # noqa: E731
    check("re-apply rebuilds an identical nest", strip(dumps[0]) == strip(dumps[1]))
    spikes.OUT.mkdir(exist_ok=True)
    out = str(spikes.OUT).replace("\\", "/")
    for tag, t in (("mid_slide", 2.0 + 0.1), ("at_rest", 2.0 + 0.6)):
        spikes.run(f"spike.frame({t}, {json.dumps(out + '/caption_' + tag + '.png')})")
    print("frames for a look (blur mid-slide, sharp at rest, shadow/glow/shake):", spikes.OUT)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
