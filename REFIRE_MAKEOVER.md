# reFire Makeover — Claude Code Brief

You are working in the reFire repo (`C:\dev\reFire`). Read this whole brief before touching code. Work through every phase in order without stopping, unless you hit a blocker listed under "When to stop and ask."

---

## 0. Why this exists

reFire finds good moments in Twitch VODs, but the cut it produces still costs me 2–3 hours of manual editing per video (plus ~20 min of thumbnail work). Most of that time is rewatching the video to fix reFire's mistakes. **Target: ~1 hour of human editing per video, spent only on memes/SFX/stylization, with the base cut left alone.**

The four problems to solve:

1. **Bad boundaries.** Clips start mid-reaction or mid-sequence and end mid-payoff. Selection is good; boundaries ignore the wider context of the footage, so cuts are awkward and jumps are abrupt.
2. **Rough cuts.** Filler stays in: boring, unfunny, random words, dead air. I spend a lot of time trimming spans that don't serve the story.
3. **Not final.** The cut should need near-zero structural fixes. **Do not "solve" this by only rewording prompts.** Every change must be measured (see §3).
4. **Clipping cache (most important).** Captions already have a learn-from-mistakes loop (`caption_fixes.json` → `run/corrections_<game>.json`). Build the same thing for clip selection and boundaries: compare what reFire cut against what actually made it into my finished, hand-edited videos, then use those diffs to (a) score reFire and (b) feed lessons back into future runs.

Data locations (**treat both as READ-ONLY; never modify, move, or re-save anything in them**):
- Past reFire runs: `C:\dev\reFire\run` (per-VOD folders `run/<vod_id>/` with manifests, transcripts, rough.mp4, etc.)
- My finished Premiere Pro projects: `C:\Users\f_yer\Desktop\iphone`

---

## 1. Ground rules

- Create branch `makeover` and commit at the end of each phase with a message like `phase N: <summary>`.
- `uv run pytest` must pass at the end of every phase. Add tests for every new module. Keep the existing 294+ tests green.
- **Don't break existing behavior.** Every existing CLI subcommand (`make`, `loud`, `srt`, `recap`, `fixcaps`), flag, style frontmatter key, the AE/Premiere panels, and `remote.py` must keep working. Put new stages behind flags or style-frontmatter keys. Switch a stage on by default only after the eval in §3 shows it helps, and record that decision in the final report.
- Extend the existing architecture rather than replacing it: director + critic loop, perception fusion (chat z-score, loudness, motion), beat schema, pacing audit, the `cli`/`api`/`--local-director` backends. Reuse existing signals instead of recomputing them.
- **Prefer deterministic code over LLM calls** wherever a rule works: silence detection, signal decay, boundary candidates, referent checks. Use the LLM to choose among candidates, not to invent timestamps.
- **Track cost.** Log Claude calls and token counts per stage per run to `run/<vod_id>/cost.json`. A full `make` run should not need more than roughly 2× today's Claude usage. If a design would blow past that, make it cheaper (smaller context windows, batched spans, fewer rounds).
- Keep `CLAUDE.md` updated. If it doesn't exist, create it in Phase A with an architecture map, the conventions, and the rule "a phase is not done until pytest passes and `refire eval` didn't regress."

### When to stop and ask
- You can't figure out how the Premiere projects map back to VOD time after a real attempt (see Phase B).
- An action would delete or overwrite anything in `run/` or `Desktop\iphone`.
- Fewer than 2 usable run↔project pairs exist. You can still build everything, but tell me the eval is too small to trust.

---

## 2. Phases

### Phase A — Recon (no feature code)
1. Read the codebase. Map the pipeline end-to-end: where moments are selected, where boundaries are set (snap/truncate), where the critic loop runs, where the manifest/Premiere sequence/AE project is emitted. Write this map into `CLAUDE.md`.
2. Inventory `C:\dev\reFire\run`. List each VOD folder and which artifacts it has (manifest, transcript with word timings, chat/perception data, rough.mp4, proxies, `timeline.tsv`).
3. Inventory `C:\Users\f_yer\Desktop\iphone`. List every `.prproj` and any exported videos.
4. **Pair runs with projects.** Match by vod_id in file names, media paths referenced inside the projects, or dates. Print a table of run ↔ project ↔ confidence. Note unmatched items.

### Phase B — Premiere project parser + clip diff (the "clipping cache")
`.prproj` files are gzip-compressed XML. Decompress into memory only; never re-save. Premiere times are in ticks, **254016000000 ticks per second**.

Build `refire/learn/prproj.py`:
- Parse sequences, video/audio tracks, track items (sequence in/out, source in/out, source media path, speed, enabled state). Flatten nested sequences.
- Pick the final sequence: the one I exported. Use heuristics (longest, most edited, name, the one nested inside others) and log the choice and the reason.
- **Inspect real files before writing the parser.** Don't assume the XML layout; open a couple of projects and confirm the element structure.

Build `refire/learn/align.py`, which maps every final-timeline item back to **VOD seconds**:
- If the source media is the raw VOD, use the source in/out directly.
- If it's a reFire proxy or cut clip, map it through that run's manifest.
- If it's `rough.mp4`, map rough-timeline time → manifest clip → VOD time.
- For `loud` runs, respect the virtual timeline (VOD *i* at `i*100000s`, via `sources[]`).
- Flag items with speed changes, missing media, or media that isn't from the VOD (memes, SFX, overlays). Keep the overlays in a separate list; they're useful context, not cut decisions.
- Fallback, only if mapping fails for a project: if an exported video exists, align it to the VOD by audio cross-correlation on downsampled audio. Report which method each project used.

Build `refire/learn/diff.py`. Compare reFire's proposed segments (from the run manifest) against the final segments, both in VOD time. Match by overlap (IoU), then classify every difference:

| Label | Meaning |
|---|---|
| `start_late` | I extended the start earlier (reFire started mid-context) |
| `start_early` | I trimmed the start later (reFire included lead-in junk) |
| `end_early` | I extended the end (reFire cut off the payoff/reaction) |
| `end_late` | I trimmed the end (reFire overstayed) |
| `interior_cut` | I removed a span inside a kept clip (filler) |
| `dropped` | reFire proposed it; I removed it entirely (bad pick) |
| `added` | I used VOD footage reFire never proposed (missed moment) |
| `merged` | I joined two reFire clips into one continuous span |
| `reordered` | Same footage, different order |

Each diff record must include: VOD time range, delta seconds, the transcript text of the affected span (with ~10s of context on each side), the perception signals over that span (chat z-score, loudness, motion), and the beat role/energy reFire had assigned, if any. This is what makes the lessons learnable.

Outputs:
- `run/<vod_id>/clip_diff.json`: all diff records for that VOD.
- `run/clip_lessons_<game>.json`: aggregated across VODs. Per-label counts and median deltas (e.g., "median `end_early` = +3.8s"), plus the most representative examples per label.
- CLI: `refire learn <vod_id|--all> [--project path]` builds both. It must be safe to re-run and must never modify inputs.
- A short human-readable `run/clip_lessons_report.md` summarizing where reFire goes wrong most.

### Phase C — Eval harness
Build `refire eval <vod_id|--all>`. It re-runs (or reads) reFire's cut for a VOD and scores it against my final edit using the Phase B aligner and diff.

Primary metric, in seconds of human correction:

$$\text{fix\_cost} = \sum_{i} \left( |\Delta t_{\text{start},i}| + |\Delta t_{\text{end},i}| \right) + t_{\text{interior\_cut}} + t_{\text{dropped}} + t_{\text{added}}$$

Also report: count of edits per label, boundary error per clip, % of final runtime that came straight from reFire unchanged, and estimated review markers needed.

Rules:
- **Leave-one-out.** When scoring VOD X, any lessons/examples injected into prompts must exclude VOD X's own diffs. No leakage.
- Cache transcripts/perception so eval re-runs don't redo Whisper.
- Write `run/eval/<timestamp>.json` plus a comparison table against the previous eval.
- **Record a baseline with current `main` behavior before any Phase D+ change.** Every later phase reports its fix_cost against that baseline.

### Phase D — Review markers (biggest immediate time saver)
Even before the cut is perfect, I should only have to look at flagged spots instead of rewatching everything.
- Give each cut boundary and each trimmed span a confidence score from the signals and the stages below. Examples: reaction energy still elevated at the end, unresolved referent at the start, critic disagreement.
- Emit low-confidence spots as **Premiere sequence markers** with a short reason (e.g., "end may cut off reaction — chat still 2.1σ"). Use the existing Premiere panel/ExtendScript path; markers are supported there. Also emit `markers.csv` as a fallback, and add markers to the AE comp.
- Add a threshold flag so I can tune how many markers I get. The target is ~10 per video.

### Phase E — Boundary stage (problem 1)
Add a dedicated stage between selection and render:
1. **Deterministic candidates.** For each selected moment, compute candidate starts/ends from silence gaps, sentence ends (word timings), low-motion frames, and chat/loudness returning near baseline.
2. **Backward expansion to the setup's origin.** If the first sentence has an unresolved referent ("he," "that," "this," "again," "I told you," "wait"), or references an event earlier in the VOD, extend back to where that context is established. Do this deterministically first; use an LLM to pick among candidates only when it's ambiguous, with about ±60s of transcript.
3. **Forward expansion through the reaction tail.** Don't end until the streamer's reaction speech finishes AND chat rate/loudness decay toward baseline. Use `clip_lessons_<game>.json` medians to calibrate minimum tails.
4. **Merge pass.** Adjacent selected moments that are close in VOD time and on the same thread become one continuous segment, not two cuts with a jump. Tune the gap threshold from `merged` diffs.
5. Integrate with the existing `--snap`/`--truncate` and the pacing audit (`skipped_build`). Don't duplicate them.

### Phase F — Intra-clip trim (problem 2)
1. Split each kept segment into spans (utterances / sentence groups, with silence as separators).
2. Label each span: `setup`, `payoff`, `reaction`, `context`, `filler`, `dead_air`, `chat_reading`, `off_topic`.
3. **Removal test** (core rule): a span is cut if no later kept span depends on it for meaning AND its energy is low. Have the LLM build a dependency check over spans in batches (span IDs in, JSON out), not free-form judgment.
4. Remove dead air deterministically, but **protect pauses immediately before a punchline/payoff** (comedic timing).
5. Calibrate aggressiveness from `interior_cut` diffs: what I actually cut, how long, and what it sounded like.
6. Snap every trim to word boundaries. Never clip a word.

### Phase G — Smoothing (makes cuts feel less "rough")
- Add short audio crossfades (~10–20ms) at every cut in the ffmpeg rough render and the AE path to remove pops.
- Add simple **L-cuts**: let the reaction audio continue briefly past a visual cut where it helps. This adds the `transition_type` concept that's currently missing.
- In the Premiere path, investigate what's feasible via ExtendScript/QE DOM. If transitions or audio offsets aren't doable, skip them there, note it, and rely on markers.

### Phase H — Structured critics (replaces "personality" critics)
Persona critics (Gen Z / boomer / pessimist) would mostly converge into vague notes and can't feel pacing from text. Instead, extend the existing critic loop with **job-specific critics**. Each returns JSON edits (`span_id`, `action`, `reason`, `confidence`), not prose:
- **Continuity critic:** does every segment start with established context? Are there unresolved referents or missing setup?
- **Payoff critic:** does every segment end after the payoff and reaction land?
- **Filler critic:** flags spans that fail the removal test.
- **Drop-off critic** (the "pessimist," kept in this form): marks where a viewer would click off and why. Calibrate it with real examples from `clip_lessons` (`dropped` / `interior_cut`). If YouTube retention data exists anywhere in the repo, use it.

Rules:
- Critics review the **realized cut**: the transcript as a viewer would hear it after cutting, with explicit `[CUT]` markers and gap durations. Not the original VOD transcript.
- An **arbiter** merges edits and resolves conflicts. Unresolved disagreements become review markers (Phase D).
- Cap at 2 rounds. Run critics in parallel where the backend allows. Respect the cost budget in §1.
- Optional, behind a flag: grab the frame just before and after each cut and run one vision check for jarring visual jumps (menus, loading screens, mid-animation).

### Phase I — Close the loop
- Inject the most relevant `clip_lessons` examples into director/boundary/trim/critic prompts as few-shot "reFire proposed X → final was Y" pairs. Retrieve by similarity (same game, similar beat role/energy), with a small fixed count to keep tokens down.
- Going forward, after I finish editing a new video, running `refire learn <vod_id>` should ingest it automatically, the same way caption corrections accumulate today. Document this in `CLAUDE.md` and the README.

---

## 3. Measurement discipline
- Run `refire eval --all` after every phase from D onward. Keep a stage default-on only if it lowers fix_cost without raising `dropped`/`added` errors meaningfully.
- If a stage doesn't help, leave it behind its flag and say so. Don't quietly keep tuning prompts.
- Watch for overfitting: report the leave-one-out score, not the in-sample score.

---

## 4. Final report
When done, write `MAKEOVER_REPORT.md` with:
- the run↔project pairing table and how each was aligned
- the top failure modes from `clip_lessons_report.md`
- the fix_cost table: baseline vs. after each phase, per VOD and overall
- which stages are default-on, which are behind flags, and why
- Claude usage per run before vs. after
- known limitations, especially Premiere-path limits
- exactly how to use it day to day: `refire make …` → edit in Premiere using the markers → `refire learn <vod_id>`
