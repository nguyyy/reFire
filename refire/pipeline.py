"""Chain the pipeline stages for `make`, caching artifacts in a run directory."""
from __future__ import annotations

import json
import random
from pathlib import Path

from .audio import extract_audio
from .chat import chat_signal
from .chunk import make_chunks
from .glossary import game_glossary
from .score import DEFAULT_MODEL, score_chunk
from .transcribe import (DEFAULT_BATCH_SIZE, DEFAULT_WHISPER_MODEL, transcribe)


def _noop(*_a, **_k):
    pass


# docker-style adjective-noun tag so parallel/repeat runs on the same VOD get their own
# output folder instead of clobbering each other's manifest/outline/rough cut.
_ADJ = ["blazing", "cozy", "feral", "lucky", "rowdy", "salty", "cosmic", "rusty",
        "velvet", "sneaky", "gilded", "hollow", "jagged", "mellow", "plucky",
        "scrappy", "vivid", "wistful", "zesty", "brisk"]
_NOUN = ["otter", "comet", "raven", "ember", "goblin", "viper", "lynx", "mirage",
         "phantom", "falcon", "kraken", "meadow", "ferret", "tundra", "copper",
         "willow", "badger", "cinder", "sparrow", "yeti"]


def _run_name(vod_id) -> str:
    return f"{vod_id}-{random.choice(_ADJ)}-{random.choice(_NOUN)}"


# past this the story pass gets a chapter digest instead of the raw map (rough 4 chars/tok)
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
    except Exception as e:                       # loud: silence here means a 404 an hour later
        print(f"[model] can't list Ollama models ({e}); using '{requested}' as-is.")
        return requested
    names = [m.model for m in models]
    base = requested.split(":")[0]
    if requested in names:
        return requested
    for n in names:                          # same family -> use the installed tag
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
                 batch_size=DEFAULT_BATCH_SIZE, compute_type="float16", terms=""):
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
    if not words:
        return [], [], []
    signal = chat_signal(chat, words[-1]["end"])
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
    run_dir: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    game: str = "",
    terms: str = "",
    zoom_sens: float = 1.0,
    words_per_line: int = 3,
    tol: float = 0.35,   # duration is a suggestion -- wide slack both ways over a hard cap
    silence_pad: float = 0.3,
    cache_dir: str | Path = "vods",
    assets_dir: str | Path = "assets",
    bgm: str | Path | None = None,
    title: str = "",
    claude_model: str = "claude-sonnet-5",
    director_backend: str = "cli",   # "cli" = claude -p on the subscription; "api" = SDK key
    flat: bool = False,
    local_director: bool = False,
    review_rounds: int = 2,
    scout: str = "local",   # chapterize backend: "local" (free Ollama) or "off"
    render: bool = False,
    encoder: str = "libx264",
    motion_zoom: bool = True,
    emotes: bool = True,      # emote/gif punch-ins
    sfx: bool = True,         # impact hits under those punch-ins
    deadspace: bool = True,   # cut the silence out of each clip
    cards: bool = True,       # section title cards in the AE Master
    proxy: bool = True,       # all-intra proxies for AE instead of the long-GOP VOD

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
    from .ingest import ensure_vod
    from .overlay import pick_bgm, pick_overlays
    from .rank import select_by_signal
    from .reframe import ENTER, EXIT
    from .retrieve import retrieve
    from .select import budget_select, snap_to_sentences

    # ponytail: a bare "12" parses as 12 SECONDS, which silently collapses the cut to a
    # single beat (_n_beats -> 2, then _trim_to_budget sheds to 1). Almost always a
    # missing unit. Raise the floor if you ever genuinely want a sub-30s cut.
    if duration_s < 30:
        raise SystemExit(f"--duration is {duration_s:g}s -- did you mean "
                         f"'{duration_s:g}m'? A bare number is seconds.")

    report = progress or _noop
    # Namespace artifacts by VOD so two streams don't collide on a shared run/ (the
    # extract/transcribe steps skip-if-exists and would reuse the wrong stream's cache).
    # An explicit run_dir still wins (the AE Make button isolates per output folder) and
    # skips the auto-naming below.
    auto_named = run_dir is None
    run_dir = Path("run") / str(vod_id) if auto_named else Path(run_dir)
    # This run's own output folder: unique per invocation (vod# + a generated phrase) so
    # re-running the same VOD (different brief/duration/etc) doesn't overwrite the last
    # outline/manifest/rough cut. The expensive per-VOD cache (audio/transcript/embeddings,
    # written straight to run_dir below) stays shared and reused across runs.
    run_name = _run_name(vod_id) if auto_named else run_dir.name
    out_dir = (run_dir / run_name) if auto_named else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_name}  (cache: {run_dir}, artifacts: {out_dir})")
    model = _resolve_ollama_model(model)       # don't 404 deep into the run
    report(0.0, "downloading VOD")
    video, chat = ensure_vod(vod_id, cache_dir)
    # transcription is the long pole -> map its segment progress into 0.10..0.55
    report(0.10, "transcribing")
    words, chunks, _chat_z = _detect_core(   # chat_z rides on chunks for the flat fallback
        video, chat, run_dir, model=model, game=game, terms=terms,
        transcriber=transcriber,
        whisper_model=whisper_model, batch_size=batch_size, compute_type=compute_type,
        tx_progress=lambda f: report(0.10 + 0.45 * f, "transcribing"))
    if not words:
        raise SystemExit("Empty transcript -- nothing to edit.")
    print(f"[transcribe] {len(words)} words, ~{words[-1]['end'] / 60:.0f} min of speech, "
          f"{len(chunks)} chunks")

    # Perception: fuse loudness into the enriched moment map the director reads;
    # signals.json doubles as an inspectable artifact. (Chat-rate signal is no longer fed
    # to the director; it stays on chunks only for the legacy flat/detect path.)
    report(0.56, "fusing perception signals")
    from . import perception
    audio_z = perception.loudness_signal(run_dir / "audio.wav")
    perception.save_signals(run_dir / "signals.json", audio_z)

    # ponytail: embeddings are NOT computed here. Nothing reads `chunk["embedding"]`
    # except retrieve.retrieve_with_vec, which only the flat fallback and narrative's
    # invalid-bounds branch reach -- so on a normal director run this was a full Ollama
    # pass over every chunk of a 6h VOD for nothing. Both call sites now embed on demand
    # (still cached to run_dir/chunks.json, so a real fallback pays once per VOD).
    def embed_chunks():
        return _ensure_embeddings(chunks, run_dir)

    # Narrative path: a Claude director pass writes the story outline, then we cast the
    # best local clip into each beat -> titled sections (= AE section cards).
    sections = None
    warning = None
    if not flat:
        try:
            from . import director, narrative
            report(0.60, "writing story outline")
            smap = perception.moment_map(words, audio_z=audio_z)
            (out_dir / "map_v2.txt").write_text(smap, encoding="utf-8")
            map_tok = len(smap) // 4   # ~4 chars/token; rough but enough to size the call
            print(f"[director] stream map ~{map_tok // 1000}k tokens, "
                  f"~{director._n_beats(duration_s)} beats")

            # Scout pass (free, local): chapterize the stream so the story pass reads a
            # guided map -- and, past the context ceiling, a digest instead of the raw map.
            chapters = []
            if scout != "off":
                report(0.61, "scouting chapters")
                chapters = director.chapterize(
                    smap, model=model, cache_path=out_dir / "chapters.json",
                    progress=lambda f, m: report(0.61 + 0.03 * f, m))
                print(f"[scout] {len(chapters)} chapters")
                # full-res per-chapter map files, for the director's Read drill-down
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

            trace = out_dir / "trace"   # full prompt/response dumps per Claude call
            ol = (director.outline_local(director_input, brief, title, duration_s,
                                         model=model)
                  if local_director
                  else director.outline(director_input, brief, title, duration_s,
                                        model=claude_model, trace=trace,
                                        backend=director_backend, run_dir=out_dir))
            # Editor-review loop: cast the outline, let a Claude critic read the REALIZED
            # cut and either approve or return a revised outline, re-cast, repeat. This is
            # what turns a relevant-but-reel cut into a story (see okay-refer-to-memories).
            # local_director stays single-pass (the critic is a Claude call -> would spend).
            report(0.66, "casting beats")
            review_log: list[dict] = []
            rounds = 0 if local_director else max(0, review_rounds)
            for rnd in range(rounds + 1):
                sections, outline_log, warning = narrative.cast(
                    ol, embed_chunks, words, duration_s, model=model, tol=tol,
                    progress=lambda f, m: report(0.66 + 0.24 * f, m))
                if outline_log.get("dropped"):
                    print("[cast] dropped for budget: "
                          + ", ".join(d["title"] for d in outline_log["dropped"]))
                if rnd == rounds or not sections:
                    break
                try:
                    rv = director.review(director_input, brief, title, outline_log,
                                         duration_s, model=claude_model, trace=trace,
                                         backend=director_backend, run_dir=out_dir)
                except Exception as re:   # a review failure must NOT discard a good cast
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
                # editor-readable companion: if the cut plan reads boring, so will the video
                (out_dir / "cut_plan.md").write_text(
                    narrative.cut_plan_md(outline_log), encoding="utf-8")
        except Exception as e:
            # ponytail: any director failure (no key, API error, parse, empty cast) ->
            # fall back to flat selection so the run still ships a cut.
            print("narrative outline unavailable, using flat selection:", e)
            sections = None

    if not sections:
        # Flat fallback: candidate pool ~3x the budget worth of ~60s chunks, floored so
        # short briefs still get headroom for the LLM scorer; one untitled section.
        # No brief -> nothing to embed against, so rank by raw excitement instead.
        report(0.62, "retrieving candidates")
        k = max(40, int(duration_s / 60.0 * 3))
        # only the brief-driven branch needs vectors; select_by_signal ranks on raw
        # chat/loudness z and never touches an embedding.
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
    annotate_emphasis(words, run_dir / "audio.wav")
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
                        silence_pad=silence_pad, cards=cards, proxy=proxy)
    if render:
        # no-AE rough cut for eyeballing: ffmpeg trim+reframe+subs+concat+music bed.
        report(0.97, "rendering rough cut (ffmpeg)")
        from .assemble import render_clips
        rough = render_clips(video, out_dir, flat_clips, words,
                             music=music, encoder=encoder, silence_pad=silence_pad,
                             motion_zoom=motion_zoom, deadspace=deadspace)
        print("Rough cut:", rough)
    report(1.0, "done")
    if warning:
        print("WARNING:", warning)
    return mp
