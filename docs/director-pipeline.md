# reFire director / clip-selection / story-building pipeline

Audience: a model (or engineer) with no prior context on this repo, who needs to
understand exactly how `python -m refire make <vod_id> --brief "..." --duration 10m`
turns a raw multi-hour Twitch VOD into a titled, ordered set of clips ("sections")
ready to render. This describes the **narrative path** (`pipeline.make`, the actual
product); the legacy `detect`/`edit`/`ae` verbs are a separate, older flat-ranking
path (see the bottom of this doc) that `make` does not use except as a fallback.

Every stage described here is real, current code as of 2026-07-05. File:function
references are given so claims can be checked directly.

## 0. The one-sentence version

A local speech-to-text + chat/audio/vision fusion pass builds a time-stamped "moment
map" of the whole stream. A **director** (one LLM call, ideally Claude via the user's
subscription CLI) reads that map and writes a **story outline**: a central idea plus
an ordered list of **beats**, each a slot in an arc (hook/setup/escalation/reversal/
climax/payoff/button) with exact in/out timestamps and even which exact sub-spans to
keep (jump cuts). **Casting** (`narrative.cast`) turns each beat into a real clip by
snapping those timestamps onto clean sentence boundaries and scoring the result. A
**critic** pass (the same model, a second call) reads the *realized* cut — what the
video actually now says — and can approve it or send back a revised outline; this
repeats up to `--review-rounds` times. The result is titled "sections" (→ AE section
cards / manifest sections), never a flat top-N highlight list.

## 1. Inputs and the shared "detect core"

`pipeline.make()` (`refire/pipeline.py`) first resolves a working Ollama model
(`_resolve_ollama_model`, auto-swaps to an installed one if the requested model 404s),
downloads/caches the VOD (`ingest.ensure_vod`), then runs `_detect_core`:

1. `audio.py::extract_audio` — ffmpeg → mono 16k WAV, cached at `run/<vod>/audio.wav`.
2. `glossary.py::game_glossary` — one local-Ollama call → proper-noun list for the
   named game, used as Whisper `hotwords` (biases transcription, doesn't correct it).
3. `transcribe.py::transcribe` — faster-whisper (local GPU), word-level timestamps,
   cached at `run/<vod>/transcript.json`. Returns `words: [{text,start,end}, ...]`.
4. `chat.py::chat_signal` — Twitch chat message-rate **z-score** over time buckets:
   `chat_z: [(t_center, z), ...]`.
5. `chunk.py::make_chunks` — groups words into ~30-90s windows (`chunks`), each
   carrying its own `chat_z` for the legacy flat path.

Everything downstream is keyed by `run/<vod_id>/` so two VODs never collide.

## 2. Perception fusion — building the "moment map"

The director must judge the stream like a human editor: what was said, where chat
went wild, where the streamer got loud, what was on screen. `perception.py` fuses
these into one text artifact the director reads.

- `perception.loudness_signal(wav)` — buckets the WAV into 5s RMS z-scores, same
  `(t_center, z)` shape as `chat_signal` so the two fuse symmetrically.
- **Vision (optional, default on):** `frames.sample_frames` ffmpeg-scene-detects
  keyframes, thins them (min 20s / max 600s / backfill gaps), extracts one JPEG per
  timestamp, caches to `vods/<vod>.frames/`. `vlm.caption_frames` runs a local VLM
  (`qwen2.5vl:3b` via Ollama) over each frame → a one-line scene caption, cached
  incrementally (crash-resumable) and self-healing against degenerate/garbage output
  (`vlm._degenerate` drops repeated-token poison like `"@@@@@@@@"` on load and refuses
  to store it again).
- `perception.moment_map(words, chat_z, audio_z, captions)` — the core fusion. It
  reuses the same sentence splitter as the legacy `director.stream_map`
  (`perception._sentences`: break the transcript on `.?!`) so both map formats agree
  on line boundaries. For each sentence `[a, b, text]` it emits one line:

  ```
  [546s] (chat!!)(LOUD) NO WAY, did you see that?!
  ```

  `(chat!)`/`(chat!!)` and `(loud)`/`(LOUD)` are z-score threshold marks (1.5 / 3.0)
  for that sentence's time span. Scene captions interleave as their own lines in time
  order: `[550s] [scene: player dies to a boss]`. **Every timestamp is in raw seconds**
  (`[546s]`, not `[09:06]`) — this is deliberate (see §7, a real bug this fixed).
  With no chat/audio/caption signals the map degrades byte-for-byte to the plain
  transcript map (`director.stream_map`).
- `perception.top_windows(chat_z, audio_z, k=8, span_s=120)` — greedily finds the
  `k` highest-combined-signal, non-overlapping 120s windows (chat + loudness, summed,
  clamped ≥0). These are the windows that get **contact-sheeted** (see §4) so the
  director's vision has something to look at, and also the fallback ranking signal
  for the no-brief flat path (`rank.select_by_signal`).
- `run/<vod>/signals.json` and `run/<vod>/map_v2.txt` are written as inspectable
  artifacts — "did the pipeline actually hear that scream?"

## 3. Scout pass — chaptering (free, local, always runs first)

A multi-hour stream's full moment map can be too large for even a 1M-token context
call reliably, and even when it fits, a flat wall of transcript is a worse director
input than a table of contents. `director.chapterize()` (Ollama, local, free) splits
the map into ~20-minute windows (`director._windows`, splits on the `[Ns]` timestamps)
and, per window, asks a small model for:

```json
{"title": "...", "summary": "...", "notable_moments": [{"t_s": ..., "why": "..."}]}
```

parsed into a `Chapter` (with `start_s`/`end_s` taken from the window itself, not the
model — the model can't get those wrong). A failed window degrades to a stub chapter
(never crashes the run). Cached to `run/<vod>/chapters.json` (skip-if-exists).
`perception.chapter_guide(chapters)` renders the coarse table of contents; full-
resolution per-chapter transcript excerpts are also written to `run/<vod>/map/
chapter_NN.txt` so the director can `Read` into one chapter for precision if the
digest is too coarse (see §4).

## 4. Building the director's input

`pipeline.make` decides how much of the map the director actually gets, by size:

- **Fits** (`map_tok <= FULL_MAP_TOK_LIMIT` = 150k tokens, ~4 chars/token estimate):
  chapter guide + the FULL moment map, concatenated. This is the common case for
  most streams.
- **Too big, but chapters exist:** `perception.digest()` — chapter guide first, then
  full-resolution map *excerpts* only around (a) the top-signal windows and (b) each
  chapter's `notable_moments`, ± 60s padding, merged and capped to a char budget. The
  director effectively sees "coarse everywhere, precise where it matters."
- **Too big, no chapters (scout disabled):** sends the raw map anyway with a printed
  warning; may overflow and fall back to flat selection.

**Vision for the outline call:** if vision is on, `frames.contact_sheet` renders a
3×3 JPEG grid (timestamps burned in) for each of the `top_windows`, saved under
`run/<vod>/sheets/`. These are handed to the director as either file paths (CLI
backend — the model `Read`s up to `director.MAX_READS=3` of them) or inlined base64
image blocks (API backend, capped at 8).

## 5. The director call — `director.outline()`

One LLM call: **stream map (+ chapter guide/digest) + brief + title + target duration
→ `Outline`** (`refire/director.py`).

```python
class Outline(_JsonModel):
    central_idea: str
    story_shape: str        # e.g. "confidence collapses into chaos"
    viewer_promise: str      # what the viewer is promised they're watching
    ending_needed: str       # what kind of ending completes the story
    beats: list[Beat]

class Beat(_JsonModel):
    title: str                                 # section-card title
    role: str                                  # hook|setup|escalation|reversal|climax|payoff|button
    intent: str                                # this beat's job in the story
    viewer_question: str                       # what viewer wonders after this beat
    turn: str                                   # what changes in this beat
    transition_in: str                          # why it follows the previous beat
    texture: str                                # funny|tense|awkward|triumphant|chaotic|sincere
    energy: int = 3                             # 1-5 pacing value
    setup_start_s: float | None                 # earlier context, if start_s isn't enough
    payoff_start_s: float | None                # where the joke/reveal/win/fail lands
    reaction_end_s: float | None                # tail after payoff (the laugh), for breathing room
    start_s: float; end_s: float                # the beat's envelope, in RAW SECONDS
    segments: list[Segment]                     # the EDIT: exact kept sub-spans (jump cuts)
    query: str                                  # fallback retrieval text if timestamps are unusable
```

Key prompt instructions (`director._SYSTEM`):
- Decide `story_shape` / `viewer_promise` / `ending_needed` FIRST, for the whole cut.
- Beats are emitted in **story order** (≈ chronological for a linear gaming stream) —
  the director is not picking independent "interesting moments," it's building an arc:
  hook → setup → escalation → climax → button.
- Each beat's `start_s`/`end_s` must be copied straight off the map's `[Ns]` labels —
  the prompt explicitly warns against misreading `[546s]` as `9.06` (a real bug, see
  §7).
- `segments` is the actual edit within a beat: list only the sub-spans worth keeping;
  jump-cut the rambling between them. A beat's kept footage should total 15-60s and
  essentially never exceed ~90s.
- Never cut before `payoff_start_s`; always give the FINAL beat a `reaction_end_s` so
  the ending breathes instead of stopping dead on the punchline (this was a specific,
  user-reported failure mode — see [[refire-director-tuning]] equivalent in
  `docs`/memory).
- If no `--brief` is given, `director.pick_system(None)` swaps in
  `_SYSTEM_STREAM_INTRO`: "find the ONE story this stream tells" instead of following
  an editorial brief — same beat schema, same arc rules, different framing only.
- `_MAP_LEGEND` (appended to every prompt) explains the `(chat!)`/`(loud)`/`[scene:
  ...]` marks so the model treats scene captions as weak visual hints, not fact.

**Backends** (`outline(backend="cli")`, default): tries `director._complete_cli` first
— shells `claude -p` (Claude Code headless), billed against the user's Claude
subscription rather than the API key, with `--tools Read` enabled when contact sheets
exist (model can `Read` up to 3 images/chapter files). On CLI failure it falls
through to `director._complete` (the Anthropic SDK, `ANTHROPIC_API_KEY`, adaptive
thinking, live-streamed to the console). Either path validates the model's raw JSON
text into `Outline` (`_extract_json` + `model_validate`) — Anthropic's strict
structured-output endpoint was abandoned because this schema is too deep for it (a
real 400 error hit in production, see §7). A completely local, free path also exists,
`director.outline_local()` (plain Ollama `format="json"`), used when `--local-director`
is passed; coarser, and skips the critic loop entirely (see below).

## 6. Casting — `narrative.cast()`: outline → real clips

`narrative.cast(outline, chunks, words, target_s, model, tol)` turns each `Beat` into
actual footage, **in beat order**:

**Primary path** (the director gave usable `start_s`/`end_s`, `narrative._valid_bounds`):
1. Take `beat.segments` if present (else treat the whole `start_s..end_s` as one
   segment) — these are the exact kept spans.
2. Widen only the FIRST segment's start (toward `setup_start_s`, if earlier) and the
   LAST segment's end (toward `max(payoff_start_s, reaction_end_s)`, never earlier) —
   clamped to `WIDEN_SLOP=15s` (`WIDEN_SLOP_FINAL=30s` for the very last beat) so a
   stray anchor can't silently re-inflate a deliberately tight cut.
3. Snap every segment onto clean sentence boundaries (`select.snap_to_sentences`);
   the final beat's last segment gets extra slack (`max_pad=8.0`,
   `phrase_fallback=True` — falls back to a natural pause if no terminal punctuation
   is found) so the video's actual ending lands clean, not mid-word.
4. Merge segments that end up closer than `SEG_MERGE_GAP=0.75s` after snapping (avoids
   stutter-cuts).
5. Score the joined kept text ONCE against the beat's own `intent` as a mini-brief
   (`score.score_chunk`) — cheap, and mainly feeds the budget trimmer + `outline.json`,
   not selection (the director already selected; local scoring here is not re-picking
   clips).

**Fallback path** (bounds missing/invalid — `beat.query` only): the old flat-retrieval
recipe — embed the query, `retrieve_with_vec` the `K_RETRIEVE=6` nearest unused
chunks by cosine similarity, LLM-score the top `N_SCORE=4` against `beat.intent`, cast
the highest-scoring one, snap it to sentences. Each beat still yields exactly one
clip (or a jump-cut group of clips if it had segments).

Multiple segments per beat become **multiple clips within one titled section** — the
manifest/renderer already treats a section's clip list as sequential jump cuts, so a
beat with 3 kept segments plays as 3 quick cuts under one section card.

## 7. Budget trim — protecting the arc, not re-scoring clips

The director is asked to land near the target runtime but routinely overshoots. The
naive fix — "drop lowest-scored beats until under budget" — would silently
re-introduce exactly the score-driven selection the director exists to replace,
knifing a low-scoring-but-structurally-essential setup or connective beat for being
"boring." Instead (`narrative._trim_to_budget`):

- Only trims past a **ceiling** `target_s * (1 + tol)` (default `tol=0.25`) — the
  target is a goal, not a hard cap; overshoot within tolerance ships untouched.
- When it must drop, it drops the **most expendable role first**
  (`_DROP_PRIORITY`): `hook`/`climax`/`button` are protected (priority 0, the arc
  spine), `setup` next (1), `reversal`/`payoff` (2), and `escalation` beats (3, the
  most replaceable "more of the same" middle) go first. Within a tier, lowest local
  score loses.
- If the surviving total falls under the **lower** tolerance band
  `target_s * (1 - tol)`, a `warning` is surfaced (printed, never silently padded
  with filler).
- Dropped beats are recorded (`outline_log["dropped"]`) specifically so the critic
  (§8) sees *why* a beat vanished and doesn't keep re-adding it at full length every
  round, fighting the trimmer.

Roleless beats (legacy/flat outlines with no `role`) get priority 2 — i.e. this
degrades to plain lowest-score-first when there's no arc information, which matches
old behavior.

## 8. The critic loop — `director.review()`

This is what separates "relevant clips in order" from "an edited story." After
casting, `pipeline.make` loops up to `review_rounds` (default 2; `0` = single pass,
`--director-backend`/`--review-rounds` are the cost valves):

1. `narrative._realized_script(outline_log)` renders what the **cast** actually
   produced: central idea, story shape, then each beat with its role/intent/
   transition/viewer-question **and the real transcript text of its chosen span** (not
   the director's plan — the critic judges reality). Dropped-for-budget beats are
   listed explicitly with a note not to re-add them at full length.
2. On the **last scheduled round only** (cost control), if vision is on, a fast 480p
   proxy of the current cut is rendered (`frames.render_proxy`) and cut into
   timestamped contact sheets (`frames.cut_sheet`) — so the final review round can
   literally *look at* the assembled video, not just its transcript.
3. `director.review(map, brief, outline_log, ...)` — one more LLM call. The critic
   judges: HOOK (first ~10s), ARC (does it build, or is it a flat highlight reel?),
   SETUP CLARITY, EXPECTATION (does the next beat pay off what the last one raised?),
   CONNECTIVE TISSUE, REDUNDANCY (cut duplicate beats), DEAD WEIGHT (re-cut rambling
   beats via `segments`), PAYOFF COMPLETION (don't cut before the payoff lands),
   ENDING (same scrutiny as the hook — must breathe), PACING (energy variety).
   Returns `Review{approved: bool, notes: str, outline: Outline}` — either an
   approval or a **full re-plan** (reorder/drop/merge/add connective beats/re-tag
   roles/retighten bounds).
4. If approved, or rounds are exhausted, or the review call itself fails (never
   discard a working cast over a review hiccup), the loop stops. Otherwise `cast()`
   re-runs on the revised outline and the loop repeats.
5. The stream map is sent as a Claude **prompt-cache-marked** block *before* the
   realized cut in the user message, and stays byte-identical across rounds — so
   round 2+ reads the (often huge, multi-hour) map from cache instead of re-paying for
   it every round.

`local_director=True` skips this whole loop (`rounds=0` forced) — the critic is
always a Claude call, so the fully-free local path stays single-pass by design.

## 9. Output

`sections = [{"title", "role", "energy", "clips": [{"start","end","role","energy"}, ...]}, ...]`
— chronologically sorted (guarantees forward playback even though beats/casting
worked beat-by-beat) — plus `outline_log` (written to `run/<vod>/outline.json`, the
full inspectable comprehension artifact: every beat's editorial fields, chosen
span(s), score, realized text, dropped beats, and critic notes per round) and
`cut_plan.md` (`narrative.cut_plan_md`, a human-readable editor's-notebook rendering
of the same thing — "if the cut plan reads boring, the video will too").

From here the pipeline is unremarkable: `style.style_for(role, energy)` maps each
beat's role onto presentation knobs (zoom sensitivity, overlay density, caption
words-per-line, whether it gets a title card), `overlay.pick_overlays`/`pick_bgm`
place emote punch-ins/SFX/music, and `ae_export.build_manifest` (After Effects path)
or `assemble.render_clips` (`--render`, plain ffmpeg rough cut) turn `sections` into
the actual video artifact.

## 10. Flat fallback (when the narrative path can't run)

If `--flat` is passed, or the director/cast raises for any reason (missing key, CLI
+ API both unavailable, JSON parse failure, empty cast — `pipeline.make`'s broad
`except Exception`), the pipeline falls back to the pre-narrative design: retrieve
candidates (`retrieve.retrieve` by brief-embedding similarity, or
`rank.select_by_signal` — chat/loudness excitement — if there's no brief to embed
against), LLM-score each against the brief, snap to sentences, then
`select.budget_select` greedily fills the duration target highest-score-first in
chronological order. Output is a **single untitled section** — no arc, no roles, just
relevant clips in time order. This is the same selection logic the legacy `detect`/
`edit` CLI verbs use (minus the director/critic entirely), kept as the safety net so
`make` always ships *something* even when the LLM story pass can't run.

## Key source files (for further reading)

| File | Role |
|---|---|
| `refire/pipeline.py` | Orchestrates `make()`: detect core → perception → scout → director → cast → critic loop → style/manifest/render |
| `refire/perception.py` | Pure fusion of transcript + chat-z + audio-z + VLM captions into the moment map; `top_windows`, `digest`, `chapter_guide` |
| `refire/director.py` | `Outline`/`Beat`/`Segment`/`Chapter`/`Review` schemas, prompts, `chapterize`, `outline`, `review`, CLI/API/local backends |
| `refire/narrative.py` | `cast()` (outline → sections), `_trim_to_budget`, `cut_plan_md` |
| `refire/frames.py` / `refire/vlm.py` | Frame sampling, contact sheets, local VLM scene captions |
| `refire/select.py` | `snap_to_sentences`, `speech_intervals`, `compress_silence`, `budget_select` (flat path) |
| `refire/rank.py` | Legacy score blending (`rank_select`) + `select_by_signal` (no-brief flat candidate ranking) |
| `refire/style.py` | Role → presentation knobs (zoom/overlay density/caption pacing/title card) |
| `refire/ae_export.py`, `refire/assemble.py` | Manifest / ffmpeg-rendered output from `sections` |
