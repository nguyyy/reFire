"""Phase 0 API spikes (docs/specs/00-shared.md S1-S7) against a scratch Premiere project.

Needs Premiere open with the reFire panel loaded (CDP on :8089). Findings go to
docs/spikes.md by hand; this only drives the probes and checks frames.

    uv run python tests/smoke/spikes.py setup <proxy.mov>     # scratch sequence, >= 5 V tracks
    uv run python tests/smoke/spikes.py S1                    # adjustment layer on V2
    uv run python tests/smoke/spikes.py S2                    # Transform zoom + black-bar check
    uv run python tests/smoke/spikes.py toggle <v> <sec> <match> <j> <json value>   # S7
    uv run python tests/smoke/spikes.py S3 | S4 | S5 | S6 [n] | effects
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from refire.ppro import remote  # noqa: E402

JSX = Path(__file__).with_name("spikes.jsx")
MOGRT = ROOT / "refire" / "assets" / "premiere" / "captMograph.mogrt"
OUT = Path(tempfile.gettempdir()) / "refire_spikes"
TPS = 254_016_000_000


def js(v) -> str:
    """python value -> extendscript literal (json is valid ES3 for these shapes)."""
    return json.dumps(v)


def run(expr: str):
    out = remote.call(expr, jsx=JSX)
    try:
        return json.loads(out)
    except (TypeError, ValueError):
        return out


def show(label, res):
    print(f"== {label}")
    print(json.dumps(res, indent=1) if not isinstance(res, str) else res)


def black_borders(png: Path, strip: int = 4, thr: float = 8.0) -> dict:
    """Mean luma of each frame edge strip. A black bar reads ~0; footage reads well above."""
    import cv2
    for _ in range(20):                      # exportFramePNG is async
        if png.exists() and png.stat().st_size:
            break
        time.sleep(0.5)
    img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"error": f"no frame at {png}"}
    edges = {"top": img[:strip], "bottom": img[-strip:], "left": img[:, :strip], "right": img[:, -strip:]}
    means = {k: round(float(v.mean()), 1) for k, v in edges.items()}
    return {"means": means, "black": [k for k, m in means.items() if m < thr]}


def caption_keys(length: float) -> dict:
    """captionIn's keys as decoded in docs/specs/02-caption.md, anchored to a block of `length`."""
    return {"position": [[0.0, [0.5, 0.5324]], [0.247, [0.5, 0.5]]],
            "opacity": [[0.0, 0.0], [0.058, 100.0],
                        [length - 0.183, 100.0], [length - 0.015, 0.0]]}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    OUT.mkdir(exist_ok=True)
    out = str(OUT).replace("\\", "/")
    cmd, args = argv[0].upper(), argv[1:]

    if cmd == "SETUP":
        show("setup", run(f"spike.setup({js(str(Path(args[0]).resolve()))})"))
    elif cmd == "EFFECTS":
        show("effects", run("spike.effects()"))
    elif cmd == "S1":
        name = args[0] if args else "refire_punch_adj"
        res = run(f"spike.s1({js(name)}, 10, [0.5, 3.333, 60])")
        show("S1", res)
        if isinstance(res, dict):
            for p in res.get("placed", []):
                print("  frame-exact:", p["got"] == p["want"], p["want"], p["got"])
    elif cmd == "S2":
        res = run(f"spike.s2(10, 210, [0, 1], {js(out)})")
        show("S2", res)
        print("  zoomed :", black_borders(OUT / "s2_zoom.png"))
        print("  plain  :", black_borders(OUT / "s2_plain.png"))
    elif cmd == "TOGGLE":
        v, sec, match, j, val = args
        show("toggle", run(f"spike.toggle({v}, {sec}, {js(match)}, {j}, {val})"))
    elif cmd == "S3":
        show("S3", run(f"spike.s3({js(MOGRT.as_posix())}, 20, 2, 2.0, "
                       f"{js(js('spike text'))}, {js(out)})"))
    elif cmd == "PLACE":                      # place <sec> <v index> <len>: a fresh caption clip
        at, v, ln = args
        show("place", run(f"spike.place({js(MOGRT.as_posix())}, {at}, {v}, {ln})"))
    elif cmd == "S4":                         # S4 [<sec> <v index>]: keys onto a placed clip
        at, v = (args + ["40", "2"])[:2] if args else ("40", "2")
        L = 2.0
        frames = [0.03, 0.12, 0.30, L - 0.10]
        show("S4", run(f"spike.s4({v}, {at}, {js(caption_keys(L))}, 5, true, {js(out)}, {js(frames)})"))
        print("  frames in", OUT, "(mid-slide = index 1; compare with the hand-applied clip)")
    elif cmd == "S5":
        show("S5", run("spike.s5('refire_captions')"))
    elif cmd == "S6":
        n, chunk, L = int(args[0]) if args else 300, 50, 1.5
        blocks = [[200 + i * 1.2, 200 + i * 1.2 + L, 1 + (i % 2)] for i in range(n)]
        total = 0
        for c in range(0, n, chunk):
            t0 = time.time()
            res = run(f"spike.s6({js(MOGRT.as_posix())}, {js(blocks[c:c + chunk])}, {js(caption_keys(L))})")
            dt = time.time() - t0
            total += dt
            print(f"  chunk {c // chunk + 1}: {res}  wall {dt:.1f}s")
        print(f"S6 total {total:.1f}s for {n} blocks (budget 300s)")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
