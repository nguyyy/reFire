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


def _detect_core(video, chat, run_dir, model=DEFAULT_MODEL, game="", tx_progress=None):
    """Brief-agnostic, cached front half: audio -> transcribe -> chat -> chunks.

    Shared by `run` (legacy detect) and `make` (brief pipeline). Returns
    (words, chunks); words is [] when the transcript is empty. `tx_progress(frac)`
    is forwarded to transcription.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    wav = run_dir / "audio.wav"
    if not wav.exists():
        extract_audio(video, wav)
    hotwords = ", ".join(game_glossary(game, model=model, cache_dir=run_dir))
    words = transcribe(wav, cache_path=run_dir / "transcript.json",
                       hotwords=hotwords, progress=tx_progress)
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
    run_dir: str | Path = "run",
    model: str = DEFAULT_MODEL,
    game: str = "",
    zoom_sens: float = 1.0,
    words_per_line: int = 3,
    tol: float = 0.25,
    cache_dir: str | Path = "vods",
    assets_dir: str | Path = "assets",
    bgm: str | Path | None = None,
    title: str = "",
    claude_model: str = "claude-sonnet-4-6",
    flat: bool = False,
    local_director: bool = False,
    render: bool = False,
    encoder: str = "libx264",
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
    run_dir = Path(run_dir)
    report(0.0, "downloading VOD")
    video, chat = ensure_vod(vod_id, cache_dir)
    # transcription is the long pole -> map its segment progress into 0.10..0.55
    report(0.10, "transcribing")
    words, chunks = _detect_core(
        video, chat, run_dir, model=model, game=game,
        tx_progress=lambda f: report(0.10 + 0.45 * f, "transcribing"))
    if not words:
        raise SystemExit("Empty transcript -- nothing to edit.")
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
            ol = (director.outline_local(smap, brief, title, duration_s, model=model)
                  if local_director
                  else director.outline(smap, brief, title, duration_s, model=claude_model))
            report(0.66, "casting beats")
            sections, outline_log, warning = narrative.cast(
                ol, chunks, words, duration_s, model=model, tol=tol)
            if sections:
                (run_dir / "outline.json").write_text(
                    json.dumps(outline_log, indent=2), encoding="utf-8")
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
            report(0.66 + 0.24 * (i + 1) / len(candidates), "scoring clips")
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

    report(0.95, "building manifest (motion scan)")
    z_enter = min(0.95, ENTER / max(zoom_sens, 1e-3))
    z_exit = min(z_enter * 0.9, EXIT / max(zoom_sens, 1e-3))
    mp = build_manifest(video, run_dir, words, sections,
                        words_per_line, z_enter, z_exit,
                        overlays_by_clip=overlays, bgm=music)
    if render:
        # no-AE rough cut for eyeballing: ffmpeg trim+reframe+subs+concat+music bed.
        report(0.97, "rendering rough cut (ffmpeg)")
        from .assemble import render_clips
        rough = render_clips(video, run_dir, flat_clips, words,
                             music=music, encoder=encoder)
        print("Rough cut:", rough)
    report(1.0, "done")
    if warning:
        print("WARNING:", warning)
    return mp
