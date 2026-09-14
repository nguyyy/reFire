"""CLI: `make` -- VOD number + brief + duration -> a narrative AE build manifest."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .pipeline import make
from .score import DEFAULT_MODEL
from . import loud as loud_mod
from .select import SILENCE_PAD, parse_duration
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

    Writes when the integer pct changes OR the message does, so a long transcription
    (one constant message) still emits ~100 files rather than thousands, while a stage
    that moves through many messages inside one pct point -- the scout, 13 chapters
    across 3 pct -- doesn't leave the panel showing a stale line. The GUI polls this
    file to animate its bar.
    """
    path = Path(path)
    state = {"pct": -1, "msg": ""}

    def write(frac, msg, done=False):
        pct = max(0, min(100, int(frac * 100)))
        if pct == state["pct"] and msg == state["msg"] and not done:
            return
        state["pct"], state["msg"] = pct, msg
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
    mk.add_argument("--style", default=None,
                    help="free text: HOW to cut it, as opposed to --brief's what. "
                         "Structure, pacing, what to favor, what never to do -- e.g. "
                         "\"highlight reel, not a story arc; never open a bit "
                         "mid-sequence; favor the loudest reactions\". It overrides "
                         "the default story-arc framing in both the director and the "
                         "critic. May also be a PATH to a style document (.md): its "
                         "frontmatter sets the flags below, its body is the direction")
    # The five knobs a style document can also set. All default to None so `cli` can tell
    # "the user asked for this" from "nobody said" -- an explicit flag beats frontmatter.
    mk.add_argument("--pace", type=float, default=None,
                    help="cut speed: scales every role's clip-length budget at once "
                         "(0.6 = snappier and more beats, 1.5 = room to breathe; "
                         "default 1.0). --style says it in words; this moves the "
                         "numbers the prompt and the pacing audit actually use")
    mk.add_argument("--order", choices=["chrono", "director", "whiplash"], default=None,
                    help="play order: chrono (default, forward in stream time), "
                         "director (the outline's own order), whiplash (maximize the "
                         "tonal mismatch between adjacent beats -- chaos into calm)")
    mk.add_argument("--snap", choices=["sentence", "phrase", "transient"], default=None,
                    help="where cuts land: sentence (default, never mid-sentence), "
                         "phrase (natural speech pauses -- shorter shots, still never "
                         "mid-word), transient (out-point on the audio peak, cut short "
                         "by --truncate so the moment never fully resolves)")
    mk.add_argument("--stack", type=int, default=None,
                    help="montage-stack cold open: N unexplained moments (1-2.5s each, "
                         "escalating, best last) before beat 1. 0 = the default single "
                         "flash-forward teaser")
    mk.add_argument("--truncate", type=float, default=None,
                    help="--snap transient only: seconds to cut BEFORE the peak "
                         "resolves, so the viewer finishes the joke after the cut "
                         "(default 0.3)")
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
    mk.add_argument("--start", default=None,
                    help="only edit from here into the stream: 4h / 4:00:00 / 14400 "
                         "(downloads + transcribes just that window)")
    mk.add_argument("--end", default=None, help="...and stop here: 5h / 5:00:00 / 18000")
    mk.add_argument("--assets-dir", default=_DEFAULT_ASSETS,
                    help="folder with bgm/ sfx/ overlays/ for music + emote punch-ins")
    mk.add_argument("--bgm", default=None, help="music-bed track (overrides a random pick from assets/bgm)")
    mk.add_argument("--title", default="", help="stream title; helps the director understand the story")
    mk.add_argument("--claude-model", default="claude-opus-5",
                    help="Claude model for the narrative director pass "
                         "(claude-sonnet-5 = cheaper/faster, weaker on story shape)")
    mk.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"],
                    default="xhigh",
                    help="reasoning effort for the director and critic calls "
                         "(default xhigh; lower = faster and less subscription quota)")
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
    mk.add_argument("--no-caption-fix", action="store_true",
                    help="skip the Claude pass that repairs mistranscribed proper nouns "
                         "in the captions (character/place names, the streamer's friends)")
    mk.add_argument("--progress-file", default=None,
                    help="write {pct,msg,done} JSON here for a GUI progress bar")

    sr = sub.add_parser("srt", help="manifest.json -> captions.srt (the Premiere path)")
    sr.add_argument("manifest", help="path to a run's ae/manifest.json")
    sr.add_argument("--offset", type=float, default=0.0,
                    help="shift every cue by this many seconds (caption nudge)")

    rc = sub.add_parser("recap",
                        help="re-caption a Premiere timeline you re-cut by hand")
    rc.add_argument("manifest", help="path to a run's ae/manifest.json")
    rc.add_argument("--timeline", default=None,
                    help="timeline.tsv dumped by the panel (default: beside the manifest)")
    rc.add_argument("--offset", type=float, default=0.0,
                    help="shift every cue by this many seconds (caption nudge)")
    rc.add_argument("--words-per-line", type=int, default=3,
                    help="caption line length -- the same knob `make` uses (default 3)")
    rc.add_argument("--fix", action="store_true",
                    help="also run the Claude proper-noun pass over the new lines; "
                         "corrections_<game>.json is applied either way, for free")
    rc.add_argument("--game", default="", help="game name, e.g. 'genshin impact'")
    rc.add_argument("--title", default="", help="stream title (extra context)")
    rc.add_argument("--terms", default="", help="glossary: 'Kinich, Zajef' or a file path")
    rc.add_argument("--chat", default=None,
                    help="vods/<id>.chat.json -- viewers spell names the transcriber can't")
    rc.add_argument("--claude-model", default="claude-opus-5")
    rc.add_argument("--director-backend", choices=["cli", "api"], default="cli")
    rc.add_argument("--corrections-dir", default="run",
                    help="where corrections_<game>.json lives (learned across runs)")

    fc = sub.add_parser("fixcaps",
                        help="re-run the caption proper-noun fix on an existing manifest")
    fc.add_argument("manifest", help="path to a run's ae/manifest.json")
    fc.add_argument("--game", default="", help="game name, e.g. 'genshin impact'")
    fc.add_argument("--title", default="", help="stream title (extra context)")
    fc.add_argument("--terms", default="", help="glossary: 'Kinich, Zajef' or a file path")
    fc.add_argument("--chat", default=None,
                    help="vods/<id>.chat.json -- viewers spell names the transcriber can't")
    fc.add_argument("--claude-model", default="claude-opus-5")
    fc.add_argument("--director-backend", choices=["cli", "api"], default="cli",
                    help="cli = Claude Code headless on your subscription; api = API key")
    fc.add_argument("--corrections-dir", default="run",
                    help="where corrections_<game>.json lives (learned across runs)")

    ld = sub.add_parser("loud",
                        help="several VOD#s -> a loudest-moments Premiere manifest")
    ld.add_argument("vod_ids", nargs="+", help="Twitch VOD numbers to mine")
    ld.add_argument("--duration", default="8m",
                    help="target compilation length (e.g. 8m, 08:00, 480)")
    ld.add_argument("--per-vod", type=int, default=loud_mod.PER_VOD,
                    help=f"candidate windows scanned per VOD (default {loud_mod.PER_VOD})")
    ld.add_argument("--clip", type=float, default=loud_mod.CLIP_S,
                    help=f"shot length in seconds (default {loud_mod.CLIP_S:g})")
    ld.add_argument("--lead", type=float, default=loud_mod.LEAD_S,
                    help="seconds of run-up kept BEFORE the spike, out of --clip "
                         f"(default {loud_mod.LEAD_S:g})")
    ld.add_argument("--pad", type=float, default=loud_mod.PAD_S,
                    help="extra seconds downloaded each side so the cut can be snapped "
                         f"and hand-nudged in Premiere (default {loud_mod.PAD_S:g})")
    ld.add_argument("--baseline", type=float, default=loud_mod.BASELINE_S,
                    help="rolling-median span the spike is measured against; raise it to "
                         "rank sustained loudness higher, lower it for sharper reactions "
                         f"(default {loud_mod.BASELINE_S:g}s)")
    ld.add_argument("--out", default=None, help="output folder (default run/loud-<name>)")
    ld.add_argument("--cache-dir", default="vods", help="where VOD downloads are cached")
    ld.add_argument("--threads", type=int, default=loud_mod.DL_THREADS,
                    help="parallel download threads; the downloader's own default of 4 "
                         f"throttles well under a fast link (default {loud_mod.DL_THREADS}). "
                         "Back it off if Twitch starts rate limiting.")
    ld.add_argument("--game", default="", help="game name, for transcription hotwords")
    ld.add_argument("--terms", default="", help="glossary: 'Kinich, Zajef' or a file path")
    ld.add_argument("--words-per-line", type=int, default=3,
                    help="caption line length (default 3)")
    ld.add_argument("--no-deadspace", action="store_true",
                    help="keep each moment's internal silence instead of jump-cutting it")
    ld.add_argument("--silence-pad", type=float, default=SILENCE_PAD,
                    help="breath left around each phrase when dead air is cut")
    ld.add_argument("--model", default=DEFAULT_MODEL, help="local Ollama model (glossary)")
    ld.add_argument("--transcriber", choices=["local", "deepgram"], default="local")
    ld.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    ld.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ld.add_argument("--compute-type", default="float16")

    args = p.parse_args(argv)

    if args.cmd == "fixcaps":
        from .caption_fix import fix_manifest
        from .srt import write_srt
        fix_manifest(args.manifest, game=args.game, title=args.title, terms=args.terms,
                     chat_json=args.chat, model=args.claude_model,
                     backend=args.director_backend, corrections_dir=args.corrections_dir)
        # captions.srt is derived, so it must be rebuilt or Premiere keeps the old text
        print(f"Captions: {write_srt(args.manifest).resolve()}")
        return

    if args.cmd == "loud":
        loud_mod.loud(args.vod_ids, parse_duration(args.duration), out_dir=args.out,
                      cache_dir=args.cache_dir, per_vod=args.per_vod, clip_s=args.clip,
                      lead_s=args.lead, pad_s=args.pad, baseline_s=args.baseline,
                      game=args.game, terms=args.terms,
                      words_per_line=args.words_per_line,
                      deadspace=not args.no_deadspace, silence_pad=args.silence_pad,
                      model=args.model, transcriber=args.transcriber,
                      whisper_model=args.whisper_model, batch_size=args.batch_size,
                      compute_type=args.compute_type, threads=args.threads)
        return

    if args.cmd == "srt":
        # Pure JSON -> text, so the Premiere panel re-runs this on every caption
        # nudge instead of re-running the whole `make`.
        from .srt import write_srt
        print(f"Captions: {write_srt(args.manifest, args.offset).resolve()}")
        return

    if args.cmd == "recap":
        # Walks the TIMELINE the panel dumped instead of the manifest, so the captions
        # follow footage a human moved, trimmed, split or widened after `make` was done
        # with it. Pure JSON -> text unless --fix is on, so the panel can re-run it freely.
        from .caption_fix import fix_lines, save_corrections
        from .srt import recaption

        def polish(lines):
            fixed, learned = fix_lines(lines, llm=args.fix, game=args.game,
                                       title=args.title, terms=args.terms,
                                       chat_json=args.chat, model=args.claude_model,
                                       backend=args.director_backend,
                                       corrections_dir=args.corrections_dir)
            save_corrections(args.corrections_dir, args.game, learned)
            n = sum(1 for a, b in zip(lines, fixed) if a != b)
            print(f"[captions] {n}/{len(lines)} lines corrected")
            return fixed

        out = recaption(args.manifest, args.timeline, offset=args.offset,
                        words_per_line=args.words_per_line, polish=polish)
        print(f"Captions: {out.resolve()}")
        return

    if args.cmd == "make":
        file_w = _progress_writer(args.progress_file) if args.progress_file else None
        last = {"pct": -1, "frac": 0.0, "msg": ""}
        t0 = time.monotonic()

        def prog(frac, msg, done=False):
            if not done:
                frac = max(frac, last["frac"])     # monotonic: never show a backwards %
            last["frac"] = frac
            pct = max(0, min(100, int(frac * 100)))
            # Throttle on the integer pct, but never swallow a NEW message: a stage
            # that emits many distinct lines inside one pct point had almost all of
            # them dropped -- the scout spans 0.61..0.64, so 13 chapter lines shared
            # 3 pct points and 10 of them never printed. The run looked wedged on
            # "scouting chapters" for 15 minutes while it was working fine.
            if pct != last["pct"] or msg != last["msg"] or done:
                last["pct"], last["msg"] = pct, msg
                # Elapsed minutes on every line: a run self-profiles, so we optimize the
                # stage that is actually slow instead of the one we assume is. Trails the
                # message because the AE panel's /^\[ n%\]\s*(.*)$/ shows group 2 as its
                # status text -- leading with the clock would bury the stage name.
                print("[%3d%%] %s  (+%.1fm)" % (pct, msg, (time.monotonic() - t0) / 60),
                      flush=True)
            if file_w:
                file_w(frac, msg, done)

        # A style may be a document: its body is the direction, its frontmatter supplies
        # defaults for the knobs below. An explicit flag always wins -- `pick` for the
        # value flags (None = nobody said), and a plain AND for the --no-* switches,
        # which can only ever turn something OFF and so can't be ambiguous.
        from .styles import resolve as _resolve_style
        direction, knobs = _resolve_style(args.style)

        def pick(flag, key, default):
            return flag if flag is not None else knobs.get(key, default)

        try:
            out = make(args.vod_id, args.brief, parse_duration(args.duration),
                       style=direction,
                       pace=pick(args.pace, "pace", 1.0),
                       order=pick(args.order, "order", "chrono"),
                       snap=pick(args.snap, "snap", "sentence"),
                       stack=pick(args.stack, "stack", 0),
                       truncate=pick(args.truncate, "truncate", 0.3),
                       loudnorm=knobs.get("loudnorm", False),
                       keep_build=knobs.get("keep_build", True),
                       captions=knobs.get("captions", "all"),
                       run_dir=args.run_dir, model=args.model, game=args.game,
                       terms=args.terms,
                       zoom_sens=args.zoom_sens,
                       words_per_line=knobs.get("words_per_line", args.words_per_line),
                       tol=args.tol, cache_dir=args.cache_dir,
                       start=parse_duration(args.start) if args.start else None,
                       end=parse_duration(args.end) if args.end else None,
                       assets_dir=args.assets_dir, bgm=args.bgm,
                       title=args.title, claude_model=args.claude_model,
                       effort=args.effort,
                       director_backend=args.director_backend,
                       flat=args.flat, local_director=args.local_director,
                       review_rounds=args.review_rounds, scout=args.scout,
                       render=args.render, encoder=args.encoder,
                       silence_pad=args.silence_pad,
                       motion_zoom=not args.no_motion_zoom and knobs.get("motion_zoom", True),
                       emotes=not args.no_emotes, sfx=not args.no_sfx,
                       deadspace=not args.no_deadspace and knobs.get("deadspace", True),
                       cards=not args.no_cards and knobs.get("cards", True),
                       proxy=not args.no_proxy,
                       caption_fix=not args.no_caption_fix,
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
        print("In After Effects or Premiere Pro: Window > Extensions > reFire -> Build.")


if __name__ == "__main__":
    main()
