"""Phase 0 API spikes (docs/specs/00-shared.md S1-S7) against a scratch Premiere project.

Needs Premiere open with the reFire panel loaded (CDP on :8089). Findings go to
docs/spikes.md by hand; this only drives the probes and checks frames.

    uv run python tests/smoke/spikes.py setup <proxy.mov>     # scratch sequence, >= 5 V tracks
    uv run python tests/smoke/spikes.py S1                    # adjustment layer on V2
    uv run python tests/smoke/spikes.py S2                    # Transform zoom + black-bar check
    uv run python tests/smoke/spikes.py toggle <v> <sec> <match> <j> <json value>   # S7
    uv run python tests/smoke/spikes.py S3 | S4 | S5 | S6 [n] | effects
    uv run python tests/smoke/spikes.py BLOB <sec> <v> | TEXT <sec> <v> <old> <new> | FRAME <sec> <name>
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


def blob_bytes(hexs: str) -> bytes:
    """jsx hex (one utf-16 code unit per 4 hex chars) -> the raw little-endian blob."""
    return b"".join(int(hexs[i:i + 4], 16).to_bytes(2, "little") for i in range(0, len(hexs), 4))


def blob_hex(b: bytes) -> str:
    b += b"\0" * (len(b) % 2)
    return "".join(f"{int.from_bytes(b[i:i + 2], 'little'):04x}" for i in range(0, len(b), 2))


def patch_text(b: bytes, old: str, new: str, runs: bool) -> bytes:
    """swap the flatbuffer string `old` for `new`: append the new string at the end and repoint
    the one uoffset that targeted the old one (offsets only point forward, so nothing else moves).
    `runs`: also rewrite u32 slots equal to len(old) after the string (style run lengths?)."""
    import struct
    raw = old.encode("utf-8")
    s = b.index(struct.pack("<I", len(raw)) + raw)
    refs = [u for u in range(12, s, 4) if u + struct.unpack_from("<I", b, u)[0] == s]
    assert len(refs) == 1, refs
    out = bytearray(b)
    if runs:
        for u in range(s + 4 + len(raw), len(b) - 3, 4):
            if struct.unpack_from("<I", b, u)[0] == len(old):
                struct.pack_into("<I", out, u, len(new))
    pos = len(out) + (-len(out)) % 4
    out += b"\0" * (pos - len(out))
    enc = new.encode("utf-8")
    out += struct.pack("<I", len(enc)) + enc + b"\0"
    out += b"\0" * ((-len(out)) % 4)
    struct.pack_into("<I", out, refs[0], pos - refs[0])
    struct.pack_into("<Q", out, 0, len(out) - 12)
    return bytes(out)


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
    elif cmd == "BLOB":                       # blob <sec> <v>: dump Source Text to OUT/blob_<v>_<sec>.bin
        at, v = args
        res = run(f"spike.blob({v}, {at})")
        if not isinstance(res, dict):
            show("blob", res)
            return 1
        b = blob_bytes(res["hex"])
        (OUT / f"blob_{v}_{at}.bin").write_bytes(b)
        print(f"type {res['type']} chars {res['len']} bytes {len(b)} prefix "
              f"{int.from_bytes(b[:8], 'little')} magic {b[8:12].hex()}")
    elif cmd == "TEXT":                       # text <sec> <v> <old> <new> [runs]: patched blob back in
        at, v, old, new = args[:4]
        b = patch_text((OUT / f"blob_{v}_{at}.bin").read_bytes(), old, new, "runs" in args[4:])
        tag = f"text_{v}_{at}_{len(new)}{'_runs' if 'runs' in args[4:] else ''}"
        show("text", run(f"spike.setBlob({v}, {at}, {js(blob_hex(b))}, {js(out + '/' + tag + '.png')})"))
    elif cmd == "FRAME":                      # frame <sec> <name>
        show("frame", run(f"spike.frame({args[0]}, {js(out + '/' + args[1] + '.png')})"))
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
