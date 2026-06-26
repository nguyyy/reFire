"""CLI: `detect` clip-worthy segments, then `edit` them into a compilation."""
from __future__ import annotations

import argparse
from pathlib import Path

from .ae_export import export_ae
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
    d.add_argument("--game", default="", help="game name; LLM-builds a glossary to "
                   "fix proper-noun mistranscriptions (e.g. \"Genshin Impact\")")

    e = sub.add_parser("edit", help="assemble selected segments -> final.mp4")
    e.add_argument("video", help="path to the stream video file")
    e.add_argument("--run-dir", default="run", help="dir with segments/transcript")
    e.add_argument("--music", default=None, help="royalty-free music track")
    e.add_argument("--count", type=int, default=None, help="how many segments")
    e.add_argument("--min-score", type=float, default=None, help="min segment score")
    e.add_argument("--order", choices=["chrono", "score"], default="chrono")
    e.add_argument("--encoder", default="libx264", help="e.g. h264_nvenc for GPU")
    e.add_argument("--topic", default="", help="stream subject; groups/orders clips for flow")
    e.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model for --topic grouping")

    a = sub.add_parser("ae", help="export an After Effects build manifest (for reFire.jsx)")
    a.add_argument("video", help="path to the stream video file")
    a.add_argument("--run-dir", default="run", help="dir with segments/transcript")
    a.add_argument("--count", type=int, default=None, help="how many segments")
    a.add_argument("--min-score", type=float, default=None, help="min segment score")
    a.add_argument("--order", choices=["chrono", "score"], default="chrono")
    a.add_argument("--words-per-line", type=int, default=3, help="caption grouping")
    a.add_argument("--topic", default="", help="stream subject; groups/orders clips into sections")
    a.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model for --topic grouping")
    a.add_argument("--zoom-sens", type=float, default=1.0,
                   help="zoom amount: >1 more/earlier punches, <1 fewer (default 1.0)")

    args = p.parse_args(argv)

    if args.cmd == "detect":
        out = run(args.video, args.chat, args.run_dir, model=args.model,
                  w_llm=args.w_llm, w_chat=args.w_chat,
                  top_n=args.top_n, threshold=args.threshold, game=args.game)
    elif args.cmd == "ae":
        out = export_ae(args.video, run_dir=args.run_dir, count=args.count,
                        min_score=args.min_score, order=args.order,
                        words_per_line=args.words_per_line, topic=args.topic,
                        model=args.model, zoom_sens=args.zoom_sens)
        jsx = Path(__file__).with_name("ae") / "reFire.jsx"
        print(f"Manifest: {out}")
        print("In After Effects: File > Scripts > Run Script File... ->")
        print(f"  {jsx}")
        print("In the panel: Choose manifest -> Build. Then restyle/animate the "
              "'Caption Style' layer and click Update to propagate.")
        return
    else:
        out = edit(args.video, run_dir=args.run_dir, music=args.music,
                   count=args.count, min_score=args.min_score,
                   order=args.order, encoder=args.encoder, topic=args.topic,
                   model=args.model)
    print(out)


if __name__ == "__main__":
    main()
