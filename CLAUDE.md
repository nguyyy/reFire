# reFire -- working notes for agents

reFire turns a multi-hour Twitch VOD into a story-edited cut (AE / Premiere manifest,
optional ffmpeg rough.mp4). AI supplies editorial judgment; deterministic code owns
exact cut points, caches, captions and manifests. Know which half a change belongs in.

**Rule: a phase is not done until `uv run pytest` passes and `refire eval --all` didn't
regress** (fix_cost vs the previous `run/eval/*.json`, leave-one-out).

## Pipeline map (`refire make`, `pipeline.make`)

| Stage | Where | Notes |
|---|---|---|
| download VOD + chat | `ingest.ensure_vod` | `vods/<id>[@a-b].mp4`, `.chat.json`. A `--start/--end` window is its own cache key; all run times are 0-based in the window |
| audio / glossary / transcribe | `pipeline._detect_core` | `run/<vod>/audio.wav`, `transcript.json` (word timings), `speech.json` (Silero) |
| perception | `perception.loudness_signal`, `chat.chat_signal` | `signals.json` = loudness z (5s buckets). Chat z is computed but not fed to the director |
| moment map | `perception.moment_map` | `map_v2.txt`, `[Ns] (loud) text` lines |
| scout | `director.chapterize` (ollama) | `chapters.json`, `map/chapter_NN.txt` |
| **selection** | `director.outline` (Claude) | beats with `start_s/end_s/segments/setup/payoff/reaction` anchors |
| **boundaries** | `narrative.cast` -> `_snap` (`select.snap_to_sentences/phrase/transient`), anchor widening (`WIDEN_SLOP`), `SEG_MERGE_GAP`, `_one_moment` | `--snap`, `--truncate` live here |
| budget | `narrative._trim_to_budget` | drops by role priority |
| pacing audit | `pacing.audit` | `skipped_build`, `severed_exchange`, ... fed to the critic as facts |
| **critic loop** | `pipeline.make` loop -> `director.review` -> `Review.apply` -> re-`cast` | `--review-rounds` (2), CLI session resume |
| dead air | `select.compress_silence` (used by cast, `ae_export`, `assemble`) | per-clip `keep` spans |
| overlays / bgm | `overlay.pick_overlays`, `pick_bgm` | |
| **manifest** | `ae_export.build_manifest` | `run/<vod>/<run>/ae/manifest.json`, DNxHR proxies `p_dnxhr_sq_<a>-<b>.mov` (offset `a` = clip start - 2s) |
| captions | `caption_fix.fix_manifest` | learns into `run/corrections_<game>.json` |
| rough cut | `assemble.render_clips` -> `render.render_clip` | `--render`, `rough.mp4` |
| Premiere | `ppro/reFirePpro.jsx` `build()` lays one V1 trackItem per `keep` span; `timeline()` dumps V1 to `timeline.tsv`; `srt.recaption` re-captions it | `ppro/remote.py` drives it over CDP |
| After Effects | `ae/reFire.jsx` | |
| `loud` | `loud.py` | several VODs on a virtual timeline (VOD i at `i*100000s`, `sources[]`) |

Other verbs: `srt`, `recap`, `fixcaps`, `loud`. Deep dive: `docs/director-pipeline.md`.

## Learning loop (`refire/learn/`)

`refire learn <vod>` reads the finished Premiere project for a run (read-only, gzip XML
in memory), maps every final V-track item back to run-local VOD seconds (`align.py`),
diffs it against what reFire proposed (`diff.py`), and writes `run/<vod>/clip_diff.json`
+ `run/clip_lessons_<game>.json` + `run/clip_lessons_report.md`. `refire eval` scores a
cut against those finals (`fix_cost`, leave-one-out) into `run/eval/<timestamp>.json`.

## Conventions

- **uv only**: `uv run pytest`, `uv run refire ...`. The package dir is `reFire/` but
  imports as `refire` (`pyproject.toml` package-dir).
- **Console is cp1252**: anything printed during a run must be ASCII.
- Tests are pure local (no network/GPU/LLM), whole suite in a few seconds.
- No Claude call inside a per-clip or per-chunk loop. Bulk work -> ollama or
  deterministic code; Claude only for whole-cut reasoning. Log every Claude call to
  `cost.json` (`director._log_cost`).
- New stages go behind a flag / style key and default OFF until `refire eval` shows they
  lower fix_cost without raising `dropped`/`added`.
- `run/` and the user's Premiere project folder are read-only inputs: never modify, move
  or re-save existing files there. New outputs (clip_diff.json, eval/, cost.json) are ok.
- Comments: lowercase, terse, say why. `ponytail:` marks a deliberate shortcut + ceiling.
