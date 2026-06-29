"""CLI: `detect` clip-worthy segments, then `edit` them into a compilation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .ae_export import export_ae
from .assemble import edit
from .pipeline import make, run
from .score import DEFAULT_MODEL
from .select import parse_duration

# Anchor to the repo's assets/, not the CWD: the AE "Make" button launches the CLI
# via a detached `cmd /c start`, whose working dir isn't the repo, so a relative
# "assets" wouldn't resolve and overlays/bgm would come up empty.
_DEFAULT_ASSETS = str(Path(__file__).resolve().parent.parent / "assets")


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from a .env (CWD first, then repo root) into os.environ.

    ponytail: ~10 lines instead of a python-dotenv dependency. Real env wins
    (setdefault), so an exported var still overrides the file. Used so the Claude
    director can read ANTHROPIC_API_KEY from a gitignored .env file.
    """
    repo_root = Path(__file__).resolve().parent.parent
    for env_path in (Path.cwd() / ".env", repo_root / ".env"):
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _progress_writer(path):
    """Return a throttled progress(frac, msg) that writes {pct,msg,done} JSON.

    Only writes when the integer pct changes, so a long transcription emits ~100
    files, not thousands. The GUI polls this file to animate its bar.
    """
    path = Path(path)
    state = {"pct": -1}

    def write(frac, msg, done=False):
        pct = max(0, min(100, int(frac * 100)))
        if pct == state["pct"] and not done:
            return
        state["pct"] = pct
        path.write_text(json.dumps({"pct": pct, "msg": msg, "done": done}),
                        encoding="utf-8")

    return write


def main(argv: list[str] | None = None) -> None:
    _load_dotenv()
    p = argparse.ArgumentParser(prog="refire", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    mk = sub.add_parser("make", help="VOD# + brief + duration -> focused AE manifest (hands-off)")
    mk.add_argument("vod_id", help="Twitch VOD number (auto-downloaded + cached)")
    mk.add_argument("--brief", required=True, help="free text: the subject + vibe you want")
    mk.add_argument("--duration", required=True, help="target runtime: 20m / 20:00 / 1200")
    mk.add_argument("--run-dir", default="run", help="artifact/output directory")
    mk.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model for scoring")
    mk.add_argument("--game", default="", help="game name; LLM-builds a glossary for "
                    "proper-noun mistranscriptions (e.g. \"Genshin Impact\")")
    mk.add_argument("--zoom-sens", type=float, default=1.0,
                    help="zoom amount: >1 more/earlier punches, <1 fewer (default 1.0)")
    mk.add_argument("--words-per-line", type=int, default=3, help="caption grouping")
    mk.add_argument("--tol", type=float, default=0.25, help="duration tolerance band (0-1)")
    mk.add_argument("--cache-dir", default="vods", help="VOD download cache directory")
    mk.add_argument("--assets-dir", default=_DEFAULT_ASSETS,
                    help="folder with bgm/ sfx/ overlays/ for music + emote punch-ins")
    mk.add_argument("--bgm", default=None, help="music-bed track (overrides a random pick from assets/bgm)")
    mk.add_argument("--title", default="", help="stream title; helps the director understand the story")
    mk.add_argument("--claude-model", default="claude-sonnet-4-6",
                    help="Claude model for the narrative director pass")
    mk.add_argument("--flat", action="store_true",
                    help="skip the Claude director; use flat brief-relevance selection")
    mk.add_argument("--local-director", action="store_true",
                    help="run the narrative director on local Ollama (--model), no API spend")
    mk.add_argument("--render", action="store_true",
                    help="also ffmpeg-render a no-AE rough cut to run/rough.mp4 for eyeballing")
    mk.add_argument("--encoder", default="libx264", help="ffmpeg encoder (e.g. h264_nvenc) for --render")
    mk.add_argument("--progress-file", default=None,
                    help="write {pct,msg,done} JSON here for a GUI progress bar")

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

    if args.cmd == "make":
        file_w = _progress_writer(args.progress_file) if args.progress_file else None
        last = {"pct": -1}

        def prog(frac, msg, done=False):
            pct = max(0, min(100, int(frac * 100)))
            if pct != last["pct"] or done:         # throttle to integer-pct changes
                last["pct"] = pct
                print("[%3d%%] %s" % (pct, msg), flush=True)
            if file_w:
                file_w(frac, msg, done)

        try:
            out = make(args.vod_id, args.brief, parse_duration(args.duration),
                       run_dir=args.run_dir, model=args.model, game=args.game,
                       zoom_sens=args.zoom_sens, words_per_line=args.words_per_line,
                       tol=args.tol, cache_dir=args.cache_dir,
                       assets_dir=args.assets_dir, bgm=args.bgm,
                       title=args.title, claude_model=args.claude_model,
                       flat=args.flat, local_director=args.local_director,
                       render=args.render, encoder=args.encoder, progress=prog)
        except BaseException as e:                 # show the failure in the terminal
            prog(1.0, "error: " + str(e), done=True)
            raise
        prog(1.0, "done", done=True)
        jsx = Path(__file__).with_name("ae") / "reFire.jsx"
        print(f"Manifest: {out}")
        print("In After Effects: File > Scripts > Run Script File... ->")
        print(f"  {jsx}")
        print("In the panel: Choose manifest -> Build.")
        return
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
