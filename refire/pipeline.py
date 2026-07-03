"""Chain the segment-detection stages, caching artifacts in a run directory."""
from __future__ import annotations

import json
from pathlib import Path

from .audio import extract_audio
from .chat import chat_signal
from .chunk import make_chunks
from .glossary import game_glossary
from .rank import rank_select
from .score import DEFAULT_MODEL, score_chunk
from .transcribe import transcribe


def _noop(*_a, **_k):
    pass


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
    except Exception:
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
                 transcriber="local"):
    """Brief-agnostic, cached front half: audio -> transcribe -> chat -> chunks.

    Shared by `run` (legacy detect) and `make` (brief pipeline). Returns
    (words, chunks); words is [] when the transcript is empty. `tx_progress(frac)`
    is forwarded to transcription. `transcriber` picks the STT backend plugin.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    wav = run_dir / "audio.wav"
    if not wav.exists():
        extract_audio(video, wav)
    hotwords = ", ".join(game_glossary(game, model=model, cache_dir=run_dir))
    words = transcribe(wav, cache_path=run_dir / "transcript.json",
                       hotwords=hotwords, progress=tx_progress, backend=transcriber)
    if not words:
        return [], []
    signal = chat_signal(chat, words[-1]["end"])
    return words, make_chunks(words, signal)


def run(
    video: str | Path,
    chat: str | Path,
    run_dir: str | Path,
    model: str = DEFAULT_MODEL,
    w_llm: float = 0.6,
    w_chat: float = 0.4,
    top_n: int | None = None,
    threshold: float | None = None,
    game: str = "",
) -> Path:
    """Run detection end-to-end; return path to segments.json."""
    run_dir = Path(run_dir)
    out = run_dir / "segments.json"
    words, chunks = _detect_core(video, chat, run_dir, model=model, game=game)
    if not words:
        run_dir.mkdir(parents=True, exist_ok=True)
        out.write_text("[]", encoding="utf-8")
        return out

    scored = [{**ch, **score_chunk(ch, model=model)} for ch in chunks]
    segments = rank_select(scored, w_llm=w_llm, w_chat=w_chat,
                           top_n=top_n, threshold=threshold)
    out.write_text(json.dumps(segments, indent=2), encoding="utf-8")
    return out


def _ensure_embeddings(chunks, run_dir):
    """Embed each chunk's text once and cache to run/chunks.json (heavy, per-VOD)."""
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
    brief: str,
    duration_s: float,
    run_dir: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    game: str = "",
    zoom_sens: float = 1.0,
    words_per_line: int = 3,
    tol: float = 0.25,
    silence_pad: float = 0.3,
    cache_dir: str | Path = "vods",
    assets_dir: str | Path = "assets",
    bgm: str | Path | None = None,
    title: str = "",
    claude_model: str = "claude-opus-4-8",
    flat: bool = False,
    local_director: bool = False,
    review_rounds: int = 2,
    render: bool = False,
    encoder: str = "libx264",
    motion_zoom: bool = True,
    transcriber: str = "local",
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
    from .reframe import ENTER, EXIT
    from .retrieve import retrieve
    from .select import budget_select, snap_to_sentences

    report = progress or _noop
    # Namespace artifacts by VOD so two streams don't collide on a shared run/ (the
    # extract/transcribe steps skip-if-exists and would reuse the wrong stream's cache).
    # An explicit run_dir still wins (the AE Make button isolates per output folder).
    run_dir = Path("run") / str(vod_id) if run_dir is None else Path(run_dir)
    print(f"[run] artifacts -> {run_dir}")
    model = _resolve_ollama_model(model)       # don't 404 deep into the run
    report(0.0, "downloading VOD")
    video, chat = ensure_vod(vod_id, cache_dir)
    # transcription is the long pole -> map its segment progress into 0.10..0.55
    report(0.10, "transcribing")
    words, chunks = _detect_core(
        video, chat, run_dir, model=model, game=game, transcriber=transcriber,
        tx_progress=lambda f: report(0.10 + 0.45 * f, "transcribing"))
    if not words:
        raise SystemExit("Empty transcript -- nothing to edit.")
    print(f"[transcribe] {len(words)} words, ~{words[-1]['end'] / 60:.0f} min of speech, "
          f"{len(chunks)} chunks")
    report(0.55, "embedding chunks")
    chunks = _ensure_embeddings(chunks, run_dir)

    # Narrative path: a Claude director pass writes the story outline, then we cast the
    # best local clip into each beat -> titled sections (= AE section cards).
    sections = None
    warning = None
    if not flat:
        try:
            from . import director, narrative
            report(0.60, "writing story outline")
            smap = director.stream_map(words)
            map_tok = len(smap) // 4   # ~4 chars/token; rough but enough to size the call
            print(f"[director] stream map ~{map_tok // 1000}k tokens, "
                  f"~{director._n_beats(duration_s)} beats")
            if map_tok > 800_000:
                # ponytail: single-pass director reads the whole stream in one call; past
                # ~800k tokens it risks overflowing Opus's 1M context (run flat-falls-back
                # if it does). Upgrade path for marathon streams = a windowed/two-pass
                # coarse->fine director.
                print(f"[director] WARNING: stream map ~{map_tok // 1000}k tokens may not "
                      f"fit context; will fall back to flat selection if the call overflows.")
            trace = run_dir / "trace"   # full prompt/response dumps per Claude call
            ol = (director.outline_local(smap, brief, title, duration_s, model=model)
                  if local_director
                  else director.outline(smap, brief, title, duration_s,
                                        model=claude_model, trace=trace))
            # Editor-review loop: cast the outline, let a Claude critic read the REALIZED
            # cut and either approve or return a revised outline, re-cast, repeat. This is
            # what turns a relevant-but-reel cut into a story (see okay-refer-to-memories).
            # local_director stays single-pass (the critic is a Claude call -> would spend).
            report(0.66, "casting beats")
            review_log: list[dict] = []
            rounds = 0 if local_director else max(0, review_rounds)
            for rnd in range(rounds + 1):
                sections, outline_log, warning = narrative.cast(
                    ol, chunks, words, duration_s, model=model, tol=tol,
                    progress=lambda f, m: report(0.66 + 0.24 * f, m))
                if outline_log.get("dropped"):
                    print("[cast] dropped for budget: "
                          + ", ".join(d["title"] for d in outline_log["dropped"]))
                if rnd == rounds or not sections:
                    break
                try:
                    rv = director.review(smap, brief, title, outline_log, duration_s,
                                         model=claude_model, trace=trace)
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
                (run_dir / "outline.json").write_text(
                    json.dumps(outline_log, indent=2), encoding="utf-8")
                # editor-readable companion: if the cut plan reads boring, so will the video
                (run_dir / "cut_plan.md").write_text(
                    narrative.cut_plan_md(outline_log), encoding="utf-8")
        except Exception as e:
            # ponytail: any director failure (no key, API error, parse, empty cast) ->
            # fall back to flat selection so the run still ships a cut.
            print("narrative outline unavailable, using flat selection:", e)
            sections = None

    if not sections:
        # Flat fallback: candidate pool ~3x the budget worth of ~60s chunks, floored so
        # short briefs still get headroom for the LLM scorer; one untitled section.
        report(0.62, "retrieving candidates")
        k = max(40, int(duration_s / 60.0 * 3))
        candidates = retrieve(brief, chunks, k)
        scored = []
        for i, c in enumerate(candidates):
            res = score_chunk(c, brief=brief, model=model)
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
    overlays = pick_overlays(words, flat_clips, assets_dir, model=model)
    music = pick_bgm(assets_dir, override=bgm)

    report(0.95, "building manifest" + (" (motion scan)" if motion_zoom else ""))
    z_enter = min(0.95, ENTER / max(zoom_sens, 1e-3))
    z_exit = min(z_enter * 0.9, EXIT / max(zoom_sens, 1e-3))
    mp = build_manifest(video, run_dir, words, sections,
                        words_per_line, z_enter, z_exit,
                        overlays_by_clip=overlays, bgm=music,
                        motion_zoom=motion_zoom)
    if render:
        # no-AE rough cut for eyeballing: ffmpeg trim+reframe+subs+concat+music bed.
        report(0.97, "rendering rough cut (ffmpeg)")
        from .assemble import render_clips
        rough = render_clips(video, run_dir, flat_clips, words,
                             music=music, encoder=encoder, silence_pad=silence_pad,
                             motion_zoom=motion_zoom)
        print("Rough cut:", rough)
    report(1.0, "done")
    if warning:
        print("WARNING:", warning)
    return mp
