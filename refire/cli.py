"""CLI: `detect` clip-worthy segments, then `edit` them into a compilation."""
from __future__ import annotations

import argparse

from .assemble import edit
from .pipeline import run
from .score import DEFAULT_MODEL


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="refire", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="find clip-worthy segments -> segments.json")
    d.add_argument("video", help="path to the stream video file")
    d.add_argument("chat", help="path to the TwitchDownloader chat JSON")
    d.add_argument("--run-dir", default="run", help="artifact/output directory")
    d.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    d.add_argument("--top-n", type=int, default=None, help="keep top N segments")
    d.add_argument("--threshold", type=float, default=None, help="min final score")
    d.add_argument("--w-llm", type=float, default=0.6, help="LLM score weight")
    d.add_argument("--w-chat", type=float, default=0.4, help="chat score weight")

    e = sub.add_parser("edit", help="assemble selected segments -> final.mp4")
    e.add_argument("video", help="path to the stream video file")
    e.add_argument("--run-dir", default="run", help="dir with segments/transcript")
    e.add_argument("--music", default=None, help="royalty-free music track")
    e.add_argument("--count", type=int, default=None, help="how many segments")
    e.add_argument("--min-score", type=float, default=None, help="min segment score")
    e.add_argument("--order", choices=["chrono", "score"], default="chrono")
    e.add_argument("--encoder", default="libx264", help="e.g. h264_nvenc for GPU")

    args = p.parse_args(argv)

    if args.cmd == "detect":
        out = run(args.video, args.chat, args.run_dir, model=args.model,
                  w_llm=args.w_llm, w_chat=args.w_chat,
                  top_n=args.top_n, threshold=args.threshold)
    else:
        out = edit(args.video, run_dir=args.run_dir, music=args.music,
                   count=args.count, min_score=args.min_score,
                   order=args.order, encoder=args.encoder)
    print(out)


if __name__ == "__main__":
    main()
