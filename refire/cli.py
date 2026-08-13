"""CLI: `make` -- VOD number + brief + duration -> a narrative AE build manifest."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .pipeline import make
from .score import DEFAULT_MODEL
from .select import parse_duration
from .transcribe import DEFAULT_BATCH_SIZE, DEFAULT_WHISPER_MODEL

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
    mk.add_argument("--brief", default=None,
                    help="free text: the subject + vibe you want; omit to let the "
                         "director mine the stream's own best story")
    mk.add_argument("--duration", required=True, help="target runtime: 20m / 20:00 / 1200")
    mk.add_argument("--run-dir", default=None,
                    help="artifact/output directory (default: run/<vod_id>/<vod_id>-<phrase>, "
                         "a fresh auto-named folder each run; the shared VOD cache -- audio/"
                         "transcript/embeddings -- still lives in run/<vod_id>)")
    mk.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model for scoring")
    mk.add_argument("--game", default="", help="game name; LLM-builds a glossary for "
                    "proper-noun mistranscriptions (e.g. \"Genshin Impact\")")
    mk.add_argument("--terms", default="", help="extra names to spell correctly: "
                    "\"Kinich, Arlecchino\" or a path to a file with one per line. "
                    "Merged ahead of --game's glossary; use it for anything the local "
                    "model is too old to know")
    mk.add_argument("--transcriber", choices=["local", "deepgram"], default="local",
                    help="speech-to-text backend: local faster-whisper (default, free) or "
                         "deepgram (cloud; needs DEEPGRAM_API_KEY in .env)")
    mk.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL,
                    help=f"faster-whisper model (default {DEFAULT_WHISPER_MODEL}; "
                         "'large-v3' is slower but marginally more accurate)")
    mk.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                    help=f"transcription batch size (default {DEFAULT_BATCH_SIZE}); "
                         "higher = faster if the VRAM is there")
    mk.add_argument("--compute-type", default="float16",
                    help="faster-whisper compute type (float16 default; int8_float16 "
                         "frees VRAM for a bigger --batch-size)")
    mk.add_argument("--zoom-sens", type=float, default=1.0,
                    help="zoom amount: >1 more/earlier punches, <1 fewer (default 1.0)")
    mk.add_argument("--no-motion-zoom", action="store_true",
                    help="skip OpenCV motion scanning and render/static-fit clips without punch zooms")
    mk.add_argument("--words-per-line", type=int, default=3, help="caption grouping")
    mk.add_argument("--tol", type=float, default=0.35,
                    help="duration slack (0-1): --duration is a target, not a hard cap -- "
                         "the cut may land anywhere within target*(1-tol)..target*(1+tol) "
                         "before beats get trimmed/flagged short")
    mk.add_argument("--cache-dir", default="vods", help="VOD download cache directory")
    mk.add_argument("--assets-dir", default=_DEFAULT_ASSETS,
                    help="folder with bgm/ sfx/ overlays/ for music + emote punch-ins")
    mk.add_argument("--bgm", default=None, help="music-bed track (overrides a random pick from assets/bgm)")
    mk.add_argument("--title", default="", help="stream title; helps the director understand the story")
    mk.add_argument("--claude-model", default="claude-sonnet-5",
                    help="Claude model for the narrative director pass "
                         "(claude-opus-4-8 = pricier/higher quality)")
    mk.add_argument("--director-backend", choices=["cli", "api"], default="cli",
                    help="cli = Claude Code headless on your subscription (~$0, default; "
                         "auto-falls back to api); api = ANTHROPIC_API_KEY pay-as-you-go")
    mk.add_argument("--flat", action="store_true",
                    help="skip the Claude director; use flat brief-relevance selection")
    mk.add_argument("--local-director", action="store_true",
                    help="run the narrative director on local Ollama (--model), no API spend")
    mk.add_argument("--scout", choices=["local", "off"], default="local",
                    help="chapterize scout pass before the story pass: local Ollama "
                         "(free, default) or off (single-shot; short VODs only)")
    mk.add_argument("--review-rounds", type=int, default=2,
                    help="max editor-review revision rounds the Claude critic may make "
                         "(0 = single-pass; each round adds one Claude call)")
    mk.add_argument("--render", action="store_true",
                    help="also ffmpeg-render a no-AE rough cut to run/rough.mp4 for eyeballing")
    mk.add_argument("--encoder", default="libx264", help="ffmpeg encoder (e.g. h264_nvenc) for --render")
    mk.add_argument("--no-emotes", action="store_true",
                    help="no emote/gif overlay punch-ins (also drops their SFX)")
    mk.add_argument("--no-sfx", action="store_true",
                    help="keep the emote punch-ins but place them silently (no impact hits)")
    mk.add_argument("--no-cards", action="store_true",
                    help="no section title cards in the AE Master -- the cut runs clip to clip")
    mk.add_argument("--no-proxy", action="store_true",
                    help="point AE at the raw VOD instead of cutting all-intra (DNxHR LB) "
                         "clip proxies -- skips the transcode, but AE previews far slower")
    mk.add_argument("--no-deadspace", action="store_true",
                    help="keep each clip's internal silence instead of jump-cutting it out")
    mk.add_argument("--silence-pad", type=float, default=0.3,
                    help="breath (s) left around each phrase when dead space is cut; "
                         "smaller = tighter (default 0.3)")
    mk.add_argument("--progress-file", default=None,
                    help="write {pct,msg,done} JSON here for a GUI progress bar")

    args = p.parse_args(argv)

    if args.cmd == "make":
        file_w = _progress_writer(args.progress_file) if args.progress_file else None
        last = {"pct": -1, "frac": 0.0}
        t0 = time.monotonic()

        def prog(frac, msg, done=False):
            if not done:
                frac = max(frac, last["frac"])     # monotonic: never show a backwards %
            last["frac"] = frac
            pct = max(0, min(100, int(frac * 100)))
            if pct != last["pct"] or done:         # throttle to integer-pct changes
                last["pct"] = pct
                # Elapsed minutes on every line: a run self-profiles, so we optimize the
                # stage that is actually slow instead of the one we assume is. Trails the
                # message because the AE panel's /^\[ n%\]\s*(.*)$/ shows group 2 as its
                # status text -- leading with the clock would bury the stage name.
                print("[%3d%%] %s  (+%.1fm)" % (pct, msg, (time.monotonic() - t0) / 60),
                      flush=True)
            if file_w:
                file_w(frac, msg, done)

        try:
            out = make(args.vod_id, args.brief, parse_duration(args.duration),
                       run_dir=args.run_dir, model=args.model, game=args.game,
                       terms=args.terms,
                       zoom_sens=args.zoom_sens, words_per_line=args.words_per_line,
                       tol=args.tol, cache_dir=args.cache_dir,
                       assets_dir=args.assets_dir, bgm=args.bgm,
                       title=args.title, claude_model=args.claude_model,
                       director_backend=args.director_backend,
                       flat=args.flat, local_director=args.local_director,
                       review_rounds=args.review_rounds, scout=args.scout,
                       render=args.render, encoder=args.encoder,
                       silence_pad=args.silence_pad,
                       motion_zoom=not args.no_motion_zoom,
                       emotes=not args.no_emotes, sfx=not args.no_sfx,
                       deadspace=not args.no_deadspace, cards=not args.no_cards,
                       proxy=not args.no_proxy,
                       transcriber=args.transcriber,
                       whisper_model=args.whisper_model,
                       batch_size=args.batch_size,
                       compute_type=args.compute_type, progress=prog)
        except BaseException as e:                 # show the failure in the terminal
            prog(1.0, "error: " + str(e), done=True)
            raise
        prog(1.0, "done", done=True)
        # The AE panel greps this exact line out of stdout to auto-load the manifest.
        # Absolute: ExtendScript's File() resolves a relative path against AE's own
        # working directory, not ours, so a relative path here silently finds nothing.
        print(f"Manifest: {Path(out).resolve()}")
        print("In After Effects: Window > Extensions > reFire -> Build.")


if __name__ == "__main__":
    main()
