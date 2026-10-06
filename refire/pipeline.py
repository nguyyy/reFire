"""Chain the pipeline stages for `make`, caching artifacts in a run directory."""
from __future__ import annotations

import json
import random
import uuid
from pathlib import Path

from .audio import extract_audio
from .chat import chat_signal
from .chunk import make_chunks
from .glossary import game_glossary
from .score import DEFAULT_MODEL, score_chunk
from .transcribe import (DEFAULT_BATCH_SIZE, DEFAULT_WHISPER_MODEL, release_whisper,
                         speech_regions, transcribe)


def _noop(*_a, **_k):
    pass


# docker-style adjective-noun tag so repeat runs on the same vod get their own folder
_ADJ = ["blazing", "cozy", "feral", "lucky", "rowdy", "salty", "cosmic", "rusty",
        "velvet", "sneaky", "gilded", "hollow", "jagged", "mellow", "plucky",
        "scrappy", "vivid", "wistful", "zesty", "brisk"]
_NOUN = ["otter", "comet", "raven", "ember", "goblin", "viper", "lynx", "mirage",
         "phantom", "falcon", "kraken", "meadow", "ferret", "tundra", "copper",
         "willow", "badger", "cinder", "sparrow", "yeti"]


def _run_name(vod_id) -> str:
    return f"{vod_id}-{random.choice(_ADJ)}-{random.choice(_NOUN)}"


# past this the story pass gets a chapter digest instead of the raw map (~4 chars/tok)
FULL_MAP_TOK_LIMIT = 150_000


def _resolve_ollama_model(requested: str) -> str:
    """Keep `requested` if it's installed, else auto-pick the largest installed chat model.

    `make` defaults to qwen2.5:14b, which won't fit an 8GB GPU -- rather than 404 deep into
    a run, fall back to the biggest non-embedding model on the box and warn. Returns
    `requested` unchanged if Ollama can't be listed or nothing qualifies (let the real
    error surface downstream).
    """
    import ollama
    try:
        models = ollama.list().models
    except Exception as e:                       # be loud, silence here = a 404 an hour later
        print(f"[model] can't list Ollama models ({e}); using '{requested}' as-is.")
        return requested
    names = [m.model for m in models]
    base = requested.split(":")[0]
    if requested in names:
        return requested
    for n in names:                          # same family, use the installed tag
        if n.split(":")[0] == base:
            return n
    chat = [m for m in models if "embed" not in m.model.lower()]
    if not chat:
        return requested
    chosen = max(chat, key=lambda m: m.size or 0).model
    print(f"[model] '{requested}' not installed; using '{chosen}' "
          f"(pass --model to override).")
    return chosen


def _detect_core(video, chat, run_dir, model=DEFAULT_MODEL, game="", tx_progress=None,
                 transcriber="local", whisper_model=DEFAULT_WHISPER_MODEL,
                 batch_size=DEFAULT_BATCH_SIZE, compute_type="float16", terms="",
                 chat_offset=0.0):
    """Brief-agnostic, cached front half: audio -> transcribe -> chat -> chunks.

    Returns (words, chunks, chat_z); words is [] when the transcript is empty.
    `tx_progress(frac)` is forwarded to transcription. `transcriber` picks the STT backend
    plugin; `whisper_model`/`batch_size`/`compute_type` tune the local one.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    wav = run_dir / "audio.wav"
    if not wav.exists():
        extract_audio(video, wav)
    hotwords = ", ".join(game_glossary(game, model=model, cache_dir=run_dir,
                                       extra=terms))
    words = transcribe(wav, cache_path=run_dir / "transcript.json",
                       hotwords=hotwords, progress=tx_progress, backend=transcriber,
                       model_size=whisper_model, batch_size=batch_size,
                       compute_type=compute_type)
    # nothing after this transcribes, and every later stage loads an ollama model on the same
    # card. whisper left cached there slowed the scout to ~5 tok/s
    release_whisper()
    if not words:
        return [], [], []
    signal = chat_signal(chat, words[-1]["end"], offset=chat_offset)
    return words, make_chunks(words, signal), signal


def _ensure_embeddings(chunks, run_dir):
    """Embed each chunk's text once and cache to run/chunks.json (heavy, per-VOD).

    Call this lazily -- only the flat fallback and narrative's invalid-bounds branch read
    the vectors, so on a normal director run this whole pass is unnecessary.
    """
    from .retrieve import embed
    cache = Path(run_dir) / "chunks.json"
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if len(cached) == len(chunks) and all("embedding" in c for c in cached):
            return cached
    out = [{**c, "embedding": v} for c, v in zip(chunks, embed([c["text"] for c in chunks]))]
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def make(
    vod_id,
    brief: str | None,
    duration_s: float,
    style: str | None = None,   # free text, how to cut it (structure, pacing, priorities)
    pace: float = 1.0,          # cut speed, scales every role's clip budget
    # style machinery, all identity defaults so an unstyled run is unchanged. cli fills these
    # from --style frontmatter (see styles.py)
    order: str = "chrono",      # play order: chrono | director | whiplash
    snap: str = "sentence",     # cut points: sentence | phrase | transient
    stack: int = 0,             # montage stack cold open: N moments (0 = single teaser)
    truncate: float = 0.3,      # transient snap: seconds cut before the peak resolves
    loudnorm: bool = False,     # ffmpeg loudness norm in the rough cut
    keep_build: bool = True,    # enforce the skipped-build / slow-payoff flags
    captions: str = "all",      # all | emph (only lines with an emphasized word)
    run_dir: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    game: str = "",
    terms: str = "",
    zoom_sens: float = 1.0,
    words_per_line: int = 3,
    tol: float = 0.35,   # duration is a suggestion, wide slack both ways
    silence_pad: float = 0.3,
    cache_dir: str | Path = "vods",
    start: float | None = None,   # only edit this window of the stream (s)
    end: float | None = None,
    assets_dir: str | Path = "assets",
    bgm: str | Path | None = None,
    title: str = "",
    claude_model: str = "claude-opus-5",
    effort: str = "xhigh",    # low|medium|high|xhigh|max, matches director.DEFAULT_EFFORT
                              # (literal since director is imported lazily)
    director_backend: str = "cli",   # "cli" = claude -p on the subscription, "api" = sdk key
    flat: bool = False,
    local_director: bool = False,
    review_rounds: int = 2,
    scout: str = "local",   # chapterize backend: "local" (ollama) or "off"
    render: bool = False,
    encoder: str = "libx264",
    motion_zoom: bool = True,
    emotes: bool = True,      # emote/gif punch-ins
    sfx: bool = True,         # impact hits under those punch-ins
    deadspace: bool = True,   # cut the silence out of each clip
    cards: bool = True,       # section title cards in the AE master
    proxy: bool = True,       # all-intra proxies for AE instead of the long-GOP VOD
    caption_fix: bool = True,  # one claude pass over proper nouns in captions

    transcriber: str = "local",
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    compute_type: str = "float16",
    progress=None,
) -> Path:
    """Hands-off: VOD# + brief + duration -> focused, narrative AE manifest.

    Downloads the VOD (cached), runs the cached detect core, then the narrative half:
    a Claude "director" pass reads the whole stream and writes a story outline (central
    idea + ordered beats), and each beat is cast with the best local clip -> titled
    sections -> manifest. If Claude is unavailable (no ANTHROPIC_API_KEY / API error /
    `flat=True`), falls back to the flat path: retrieve -> LLM-score against the brief
    -> sentence-snap -> fill the duration budget -> one untitled section. Prints any
    shortfall warning. `progress(frac, msg)` (0..1) reports stage progress for the GUI.
    """
    from .ae_export import build_manifest
    from .emphasis import annotate_emphasis
    from .ingest import ensure_vod, window_tag
    from .overlay import pick_bgm, pick_overlays
    from .rank import select_by_signal
    from .reframe import ENTER, EXIT
    from .retrieve import retrieve
    from .select import budget_select, snap_to_sentences

    # a bare "12" means 12 seconds which collapses the cut to one beat, it's almost always a
    # missing unit. lower this if you actually want a sub-30s cut
    if duration_s < 30:
        raise SystemExit(f"--duration is {duration_s:g}s -- did you mean "
                         f"'{duration_s:g}m'? A bare number is seconds.")

    report = progress or _noop
    # namespace by vod so two streams don't share a cache (extract/transcribe skip if the file
    # exists). an explicit run_dir still wins (AE make button sets one) and skips auto naming
    if end is not None and start is not None and end <= start:
        raise SystemExit(f"--end ({end:g}s) must be after --start ({start:g}s)")
    auto_named = run_dir is None
    # a window gets its own cache key so it doesn't reuse or poison the full stream's cache
    tag = window_tag(start, end)
    run_dir = Path("run") / f"{vod_id}{tag}" if auto_named else Path(run_dir)
    # this run's output folder, unique per run (vod# + random phrase) so reruns don't overwrite
    # the last outline/manifest/cut. the per-vod cache (audio/transcript/embeddings) is shared
    run_name = _run_name(vod_id) if auto_named else run_dir.name
    out_dir = (run_dir / run_name) if auto_named else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_name}  (cache: {run_dir}, artifacts: {out_dir})")
    model = _resolve_ollama_model(model)       # don't 404 deep into the run
    report(0.0, "downloading VOD")
    if tag:
        # everything downstream is zero-based off this cropped file, add start back for stream time
        print(f"[window] {(start or 0) / 3600:.2f}h .. "
              f"{'end' if end is None else f'{end / 3600:.2f}h'} of the stream only")
    # the downloader's phases fill 0..0.10, otherwise a 40 min download is one frozen line
    video, chat = ensure_vod(
        vod_id, cache_dir, start=start, end=end,
        progress=lambda f, m: report(0.10 * f, f"downloading VOD -- {m}"))
    # transcription is the slow part, maps to 0.10..0.55
    report(0.10, "transcribing")
    words, chunks, _chat_z = _detect_core(   # chat_z rides on chunks for the flat fallback
        video, chat, run_dir, model=model, game=game, terms=terms,
        transcriber=transcriber, chat_offset=start or 0.0,
        whisper_model=whisper_model, batch_size=batch_size, compute_type=compute_type,
        tx_progress=lambda f: report(0.10 + 0.45 * f, "transcribing"))
    if not words:
        raise SystemExit("Empty transcript -- nothing to edit.")
    print(f"[transcribe] {len(words)} words, ~{words[-1]['end'] / 60:.0f} min of speech, "
          f"{len(chunks)} chunks")

    # fuse loudness into the moment map the director reads. signals.json is also there to
    # inspect. chat rate only stays on chunks for the flat/detect path
    report(0.56, "fusing perception signals")
    from . import perception
    audio_z = perception.loudness_signal(run_dir / "audio.wav")
    perception.save_signals(run_dir / "signals.json", audio_z)
    # emphasis (and per-word rms) has to exist before casting, --snap transient cuts on it
    annotate_emphasis(words, run_dir / "audio.wav")
    # speech whisper got no words for, compress_silence must not cut it as dead air. cached per
    # vod, cast and both renderers share it
    voiced = (speech_regions(run_dir / "audio.wav", cache_path=run_dir / "speech.json")
              if deadspace else None)

    # embeddings aren't computed here, only the flat fallback and narrative's invalid-bounds
    # branch use them. both embed on demand (cached in chunks.json) so a normal run skips a
    # full ollama pass over a 6h vod
    def embed_chunks():
        return _ensure_embeddings(chunks, run_dir)

    # narrative path: claude director writes the outline, then we cast the best local clip into
    # each beat -> titled sections (= AE section cards)
    sections = None
    warning = None
    director_error = None
    if not flat:
        try:
            from . import director, narrative
            report(0.60, "writing story outline")
            smap = perception.moment_map(words, audio_z=audio_z)
            (out_dir / "map_v2.txt").write_text(smap, encoding="utf-8")
            map_tok = len(smap) // 4   # ~4 chars/token, rough but fine for sizing
            print(f"[director] stream map ~{map_tok // 1000}k tokens, "
                  f"~{director._n_beats(duration_s)} beats")

            # scout pass (free, local): chapterize so the story pass gets a guided map, or a digest past
            # the context limit
            chapters = []
            if scout != "off":
                report(0.61, "scouting chapters")
                chapters = director.chapterize(
                    smap, model=model, cache_path=out_dir / "chapters.json",
                    progress=lambda f, m: report(0.61 + 0.03 * f, m))
                print(f"[scout] {len(chapters)} chapters")
                # full-res per-chapter map files for the director's drill-down
                map_dir = out_dir / "map"
                map_dir.mkdir(exist_ok=True)
                for i, c in enumerate(chapters, 1):
                    (map_dir / f"chapter_{i:02d}.txt").write_text(
                        perception.excerpt(smap, c.start_s, c.end_s), encoding="utf-8")
            if map_tok <= FULL_MAP_TOK_LIMIT:
                director_input = (perception.chapter_guide(chapters) + "\n\n" + smap
                                  if chapters else smap)
            elif chapters:
                wins = perception.top_windows(audio_z)
                director_input = perception.digest(chapters, smap, windows=wins)
                print(f"[director] map too big for one call; sending digest "
                      f"~{len(director_input) // 4000}k tokens")
            else:
                director_input = smap
                print(f"[director] WARNING: stream map ~{map_tok // 1000}k tokens may not "
                      f"fit context (--scout off); will fall back to flat selection if "
                      f"the call overflows.")

            trace = out_dir / "trace"   # full prompt/response dumps per claude call
            # one cli session per run, director opens it and review rounds resume it so the ~240KB map
            # isn't resent every round
            session_id = str(uuid.uuid4())
            ol = (director.outline_local(director_input, brief, title, duration_s,
                                         model=model, style=style, pace=pace,
                                         stack=stack, keep_build=keep_build)
                  if local_director
                  else director.outline(director_input, brief, title, duration_s,
                                        model=claude_model, trace=trace,
                                        backend=director_backend, run_dir=out_dir,
                                        style=style, pace=pace, stack=stack,
                                        keep_build=keep_build, effort=effort,
                                        session_id=session_id))
            # review loop: cast, let a claude critic read the actual cut and approve or return a revised
            # outline, recast, repeat. this is what turns a reel into a story. local_director stays
            # single pass since the critic costs money
            report(0.66, "casting beats")
            review_log: list[dict] = []
            rounds = 0 if local_director else max(0, review_rounds)
            for rnd in range(rounds + 1):
                sections, outline_log, warning = narrative.cast(
                    ol, embed_chunks, words, duration_s, model=model, tol=tol,
                    progress=lambda f, m: report(0.66 + 0.24 * f, m),
                    # so the durations the critic and trimmer see are what actually renders
                    deadspace=deadspace, silence_pad=silence_pad, voiced=voiced,
                    order=order, snap=snap, stack=stack, truncate=truncate)
                if outline_log.get("dropped"):
                    print("[cast] dropped for budget: "
                          + ", ".join(d["title"] for d in outline_log["dropped"]))
                if rnd == rounds or not sections:
                    break
                # give the critic the shrink this stream actually had instead of the prior. clamped so one
                # weird measurement doesn't blow up the budget
                measured = outline_log.get("shrink")
                shrink = (min(1.0, max(0.4, measured)) if measured
                          else director.CUT_SHRINK)
                try:
                    rv = director.review(director_input, brief, title, outline_log,
                                         duration_s, model=claude_model, trace=trace,
                                         backend=director_backend, run_dir=out_dir,
                                         shrink=shrink, style=style, pace=pace,
                                         stack=stack, keep_build=keep_build,
                                         effort=effort, session_id=session_id)
                except Exception as re:   # a failed review must not throw away a good cast
                    print(f"[review] round {rnd + 1} unavailable ({re}); keeping current cut")
                    break
                note = (rv.notes or "").splitlines()[0][:140] if rv.notes else ""
                review_log.append({"round": rnd + 1, "approved": rv.approved, "notes": rv.notes})
                if rv.approved:
                    print(f"[review] round {rnd + 1}: approved -- {note}")
                    break
                print(f"[review] round {rnd + 1}: revising -- {note}")
                ol = rv.outline
            if sections:
                outline_log["rounds"] = len(review_log)
                outline_log["review"] = review_log
                (out_dir / "outline.json").write_text(
                    json.dumps(outline_log, indent=2), encoding="utf-8")
                # readable companion, if the plan reads boring so will the video
                (out_dir / "cut_plan.md").write_text(
                    narrative.cut_plan_md(outline_log, pace, keep_build, duration_s),
                    encoding="utf-8")
        except Exception as e:
            director_error = e
            sections = None

    # don't fall back to flat if the director fails. flat ignores every style knob
    # (stack/order/snap/truncate/pace) so a --style run would quietly become a 30s chunk reel.
    # caches survive in run_dir so a rerun only costs one director call
    if not flat and not sections:
        raise SystemExit(
            f"director pass failed: {director_error or 'no castable sections'}\n"
            f"Refusing to fall back to flat selection -- it ignores every structural "
            f"style knob (stack/order/snap/truncate/pace) and would ship a chronological "
            f"reel instead of the cut you asked for.\n"
            f"Caches in {run_dir} are warm, so a re-run is cheap. Pass --flat to take the "
            f"flat path deliberately.")

    if not sections:
        # flat fallback: pool of ~3x the budget in ~60s chunks (floored so short briefs have room),
        # one untitled section. no brief -> rank by raw excitement
        report(0.62, "retrieving candidates")
        k = max(40, int(duration_s / 60.0 * 3))
        # only the brief branch needs vectors, select_by_signal uses raw chat/loudness z
        candidates = (retrieve(brief, embed_chunks(), k) if brief
                      else select_by_signal(chunks, k, audio_z=audio_z))
        scored = []
        for i, c in enumerate(candidates):
            res = score_chunk(c, brief=brief or "", model=model)
            scored.append({"start": c["start"], "end": c["end"],
                           "score": res["llm_score"], "reason": res["reason"]})
            report(0.66 + 0.24 * (i + 1) / len(candidates),
                   f"scoring clip {i + 1}/{len(candidates)}")
        snapped = []
        for s in scored:
            a, b = snap_to_sentences(words, s["start"], s["end"])
            snapped.append({**s, "start": a, "end": b})
        picked, warning = budget_select(snapped, duration_s, tol=tol, order="chrono")
        if not picked:
            raise SystemExit("No clips cleared selection for this brief.")
        sections = [{"title": "", "clips": picked}]

    report(0.90, "selecting clips")
    flat_clips = [clip for sec in sections for clip in sec["clips"]]

    report(0.93, "placing overlays + music")
    overlays = (pick_overlays(words, flat_clips, assets_dir, model=model, sfx=sfx)
                if emotes else [[] for _ in flat_clips])
    music = pick_bgm(assets_dir, override=bgm)

    report(0.95, "building manifest" + (" (motion scan)" if motion_zoom else ""))
    z_enter = min(0.95, ENTER / max(zoom_sens, 1e-3))
    z_exit = min(z_enter * 0.9, EXIT / max(zoom_sens, 1e-3))
    mp = build_manifest(video, out_dir, words, sections,
                        words_per_line, z_enter, z_exit,
                        overlays_by_clip=overlays, bgm=music,
                        motion_zoom=motion_zoom, deadspace=deadspace,
                        silence_pad=silence_pad, cards=cards, proxy=proxy,
                        captions=captions, voiced=voiced)
    if caption_fix:
        # the transcriber never knew the game or the streamer's friends. first point captions exist
        # as lines, before premiere opens. timestamps/line count unchanged
        report(0.96, "fixing captions")
        from .caption_fix import fix_manifest
        fix_manifest(mp, game=game, title=title, terms=terms, chat_json=chat,
                     model=claude_model, backend=director_backend,
                     corrections_dir=run_dir.parent, trace=out_dir / "trace")
    if render:
        # rough cut without AE: ffmpeg trim+reframe+subs+concat+music
        report(0.97, "rendering rough cut (ffmpeg)")
        from .assemble import render_clips
        rough = render_clips(video, out_dir, flat_clips, words,
                             music=music, encoder=encoder, silence_pad=silence_pad,
                             motion_zoom=motion_zoom, deadspace=deadspace,
                             loudnorm=loudnorm, voiced=voiced)
        print("Rough cut:", rough)
    report(1.0, "done")
    if warning:
        print("WARNING:", warning)
    return mp
