# reFire makeover report

Branch `makeover`, 2026-11-05. Brief: `REFIRE_MAKEOVER.md`. Everything below was
measured with `refire eval --all` (leave-one-out) unless it says otherwise.

**Headline:** the measurement loop works. `refire learn` turns a finished Premiere
project into labelled diffs (parser verified exactly against Premiere's own dumps), and
`refire eval` scores any stage, leave-one-out, in seconds of correction. Measured that
way, **none of the new stages beat today's cut on your four finished edits.**
- Boundary, trim and the structured critics all *raised* fix_cost.
- Review markers landed on real fixes no more often than chance.

So everything that changes the cut, or claims to point at problems, ships behind a flag
and off. The only new default is a 15 ms de-pop fade (measured on real audio). The
eval is small (2 full run↔project pairs + 2 rebuilt), so treat each number as a
direction, not a verdict. The model and the stage caps retrain with every
`refire learn`, so these stages can be re-checked as finished edits accumulate.

## 1. Run ↔ project pairing

Paired by the reFire media paths each project references (a byte scan of the gzip'd
XML), then verified by parsing: the parser reproduces the Premiere panel's own
`timeline.tsv` dumps exactly (196/196 and 128/128 V1 items within 20 ms, checked
against the autosave nearest each dump).

| run (`run/<cache>/<run>`) | project | final sequence (why) | aligned by | confidence |
|---|---|---|---|---|
| `2875394138@6840-8760/…-blazing-viper` | `day2/day2sch.prproj` | `reFire cut 2` (top-level, 198 footage items, 1040s; beat nested `fl`, `Nested Sequence 03`) | manifest `sources[]` (198 items) | high: full run artifacts + timeline.tsv |
| `2875784108/…-feral-ferret` | `day2/threeMan.prproj` | `reFire cut 1` (top-level, 128 items, 752s) | manifest `sources[]` (128) | high: full run artifacts + timeline.tsv |
| `2862200457/…-zesty-viper` | `archeonquester/archeon.prproj` | `reFire cut 1` (top-level, 202 items, 1153s) | proxy file names (202) | medium: run folder deleted, proposal rebuilt (see §6) |
| `2872051525@5400-14400/…-wistful-otter` | `7.1livestream/7.1 livestream testing sphere.prproj` | `reFire cut 1` (top-level, 193 items, 2282s) | proxy file names (193) | medium: same |

Unmatched:
- `2874595666@5400-15300/…-scrappy-ferret`: its proxies are in `day2sch.prproj`'s bin, but no sequence uses them (`reFire cut 1` is empty), so there's no final edit. Skipped.
- `2874595666…-jagged-kraken/-jagged-yeti` and the five `2854020361-*` folders: no manifest (empty or aborted runs).
- No exported videos exist (`pinkP/ppan.mp4` and `pproSIMPLE/simple.mp4` are 2–4 MB shorts unrelated to these runs), so the audio cross-correlation fallback was never needed and wasn't built.

Speed changes: none in any project (source span equals sequence span everywhere).
Overlays (memes, SFX, the numeric caption/graphics media) are kept in each
`clip_diff.json` `overlays` list: 1501 in day2sch, 1244 in threeMan.

## 2. Where reFire goes wrong (`run/clip_lessons_report.md`)

All four edits (counts / seconds of correction / median signed delta):

| label | n | s | median |
|---|---|---|---|
| end_early (you extended the end) | 29 | 280.4 | +3.9 |
| restored (dead air reFire cut, you put back) | 68 | 153.1 | +0.9 |
| start_late (you started earlier) | 21 | 148.3 | +4.7 |
| interior_cut (you cut inside a clip) | 78 | 98.3 | −0.6 |
| start_early (you trimmed the lead-in) | 14 | 70.6 | −3.6 |
| dropped (whole clip removed) | 4 | 58.7 | −14.6 |
| end_late (you trimmed the end) | 13 | 49.6 | −3.3 |
| reordered | 9 | — | — |

The two full pairs alone, which is the trustworthy half: interior_cut 59 / 86s,
start_late 17 / 84s, end_early 19 / 77s, start_early 11 / 56s, end_late 12 / 47s,
dropped 2 / 33s, restored 2 / 1s. 98.9% of the final runtime in both came straight
from reFire unchanged.

What the data says, measured:
- **~80% of clip edges are already where you leave them**, and the ones you move go
  both ways (extend vs trim are nearly equal for starts). The fixes also cluster
  (e.g. clips 0–6 of feral-ferret, 21–31 of blazing-viper) instead of following any
  local signal I could find.
- **Speech right before a cut start** predicts that you extend it (16% vs 4%), but it
  anti-predicts trimming. **A loud end** predicts extending (22% vs 10%) and almost
  never an overstay. Referent words ("he", "that", "wait") barely predict anything
  (12% vs 9%).
- **Interior cuts are content, not silence.** In the full pairs 77s of the 98s you cut
  inside clips was speech: whole utterances like "It's not loading.", "Anyways.", or a
  YouTube-poll aside. Across all four edits that's 3.6% of utterances (36/992). No surface feature (length, filler words, loudness, position)
  separates them.
- **You never merged two reFire clips** and almost never restored a dead-air gap
  (2/141 and 0/50 gaps), so the "protect the pause before a punchline" rule has no
  support in this data (0/24 gaps before a loudness jump were restored).
- The 2862200457 restorations (139.5s) are from a cut built on Sep 1, before
  `compress_silence` kept Silero-voiced audio. You put that wordless game audio back,
  which independently supports the 2026-09-13 `voiced` fix.

## 3. fix_cost (seconds of human correction, leave-one-out)

`fix_cost = Σ(|Δstart| + |Δend|) + interior_cut + dropped + added`, where added
includes restored dead air and merge gaps (all are footage you used that reFire didn't
propose).

| VOD | baseline (shipped cut) | D markers | E boundary | F trim | H critics |
|---|---|---|---|---|---|
| 2862200457 (rebuilt) | 243.4 | 243.4 | 248.6 | 243.8 | 259.4 |
| 2872051525@5400-14400 (rebuilt) | 236.5 | 236.5 | 257.9 | 238.5 | 253.3 |
| 2875394138@6840-8760 | 132.2 | 132.2 | 132.2 | 133.7 | 168.3 |
| 2875784108 | 251.1 | 251.1 | 263.3 | 251.1 | 252.7 |
| **overall** | **863.2** | **863.2** | **902.0** | **867.1** | **933.7** |
| full pairs only | 383.3 | 383.3 | 395.5 | 384.8 | 421.0 |

Markers don't change the cut (fix_cost identical by construction). They're measured by
whether they land within ±3s (VOD time) of a real fix. At the default threshold that's
**11.25 markers/video, 0.24 precision, against 0.30 for marking every edge**. Across
thresholds 0.10–0.40 precision moves between 0.0 and 0.42 with no trend, i.e. chance.
I first reported 0.57 at ~10/video. That was an artifact: cold-open teaser edges are
short and mid-sentence (so they score risky) and "hit" edits through the VOD time
they share with their beat, while in Premiere they sit at the start of the video.
Excluding teasers from both scoring and training removed the lift.

Phase G (fades, L-cuts) changes audio only, so it has no fix_cost effect. It was
measured on real audio instead (§4).

Trim and critics were each run twice. Trim gave 866.7 then 867.1, critics 914.1 then
933.7 (the LLM is stochastic). The table shows the second runs, which used the
corrected lesson pool.

The baseline was recorded once before any stage, then re-recorded (same shipped cuts)
after two learn-side fixes: rebuilt proposals use `voiced` only for projects after
2026-09-13, and the two rebuilt runs were wrongly pooled as game "unknown". Every column
above uses the corrected data. `run/eval/*.json` holds every run.

## 4. Defaults: what's on, what's behind a flag, and why

| stage | default | evidence |
|---|---|---|
| review markers (`--markers`, `--marker-threshold 0.26`) | off | no better than chance under leave-one-out (0.24 precision vs 0.30 base at ~11/video). The plumbing is done: Premiere sequence markers, AE comp markers, `markers.csv`, trimmed-span and critic-conflict markers. Re-check with `refire eval --all --markers` as edits accumulate |
| de-pop fades (`--crossfade-ms 15`, rough render + AE) | **on** | on real feral-ferret audio, the waveform step at each jump cut dropped 6–10× (median 0.40→0.06 of the p99.9 sample delta). Dips, not overlaps, so no A/V drift. `--crossfade-ms 0` restores hard cuts. Note: no hard cut exceeded the p99.9 click level even before, so the audible gain is small |
| boundary stage (`--boundary`) | off | fix_cost 863.2 → 902.0. Back-expansion and tail rules land on edges you left alone |
| trim stage (`--trim`, local ollama) | off | fix_cost 863.2 → 867.1 (first run 866.7). In the first run the local judge cut 4 spans (9.7s) across 4 VODs, ~25% overlapping your cuts. Too timid to matter, and wrong when it acted |
| structured critics (`--critics`, `--critic-effort low`) | off | fix_cost 863.2 → 914.1 / 933.7 (two runs). 46 edits applied in the first run, mostly continuity start-extensions and drop-off cuts you never made, zero conflicts |
| L-cuts (`--lcut`, rough render only) | off | not measurable with fix_cost; unproven taste call |
| past-edit few-shots in director/review (`--lessons`) | off | **unmeasured**: it changes selection, and a re-selected cut can't be fairly scored against edits made from the old proposal's proxies (§6). Needs a fresh `make` A/B with and without it, several VODs each |

Style frontmatter keys added: `boundary`, `trim`, `critics`, `lessons`.

## 5. Claude usage per run

Before (from the trace files of the three manifest runs): one director call
(8–45k prompt tokens, ~8–11k out), two **resumed** review rounds (~9k sent each,
~8–12k out; the session context is re-read from cache), and 2–4 caption batches (~2–3k
each). That's **~35–72k prompt tokens sent and ~28–45k output** per run, not counting
the cached session context the CLI re-reads.

After, with defaults: **unchanged**. Markers, fades and lessons-on-disk make no Claude
calls. Every call is now logged with tokens per stage in `run/<vod>/cost.json`.

Opt-in additions, measured:
- `--critics`: 8 calls per run (4 critics × 2 rounds) at the run's model, effort low,
  inputs 1–4k tokens each (continuity and payoff see only clip edges). Measured over 4
  VODs: 32 calls, ~265k processed input tokens (almost all CLI cache writes), 9.8k
  output, $2.56 notional API price, i.e. **~$0.64 / ~66k processed tokens per run**.
  That's well inside the ≤2× budget.
- `--lessons`: ~2 KB (~500 tokens) added to the director and review system prompts,
  fixed per run so it never breaks the session/cache prefix.
- `--trim` and `refire learn`/`eval`: no Claude (trim is local ollama).

## 6. Known limitations

- **The eval is small**: 2 full pairs + 2 rebuilt. The rebuilt pairs' proposals come
  from proxy names plus a fresh transcript, made without glossary hotwords (the
  glossary model wasn't installed), so their `keep` spans approximate what shipped.
  Their restored/interior labels are the noisiest.
- **Replay measures post-selection stages only.** Your final can only contain footage
  from the shipped proposal's proxies. Proxies carry only **2s of head room**
  (`PROXY_PAD`), so you physically couldn't extend a start further than 2s inside a
  proxy, and observed `start_late` is capped by it. Raising `PROXY_PAD` would let the
  data show how much context you really want.
- Cold opens re-use beat footage, and VOD-time coverage is unioned, so dropping a cold
  open while keeping its beat isn't detected.
- Motion signal in diff records is opt-in (`refire learn --motion`, decodes the VOD).
- **Premiere path**: markers use the documented `Sequence.markers.createMarker` API and
  are placed from where each clip actually lands (no drift). Color is set via
  `setColorByIndex` where that exists. Audio transitions and L-cut offsets are **not**
  built there: they need the undocumented QE DOM, which couldn't be verified with no
  Premiere running here. Rely on the markers. Neither JSX change could be run live in
  Premiere or After Effects this session; both pass a syntax check, and the Python
  side is unit-tested.
- AE gets de-pop dips at every time-remap jump, but not L-cuts (one footage layer
  can't carry audio past its own remapped cut).
- The LLM disambiguation step for back-expansion (brief E2) and the optional vision
  jump check (brief H) were not built. The deterministic stages they would refine
  didn't help, so there was nothing to refine.

## 7. Process notes

- One fix to an existing test: `test_stack_survives_the_stream_driven_intro` still
  expected the old prompt wording "short" (the prompt now says "video"). It was failing
  on `master` before any change.
- On this case-insensitive checkout the git index tracks `refire/…`, and the first
  phase commits staged `reFire/…`. That added new files but silently skipped edits to
  existing ones, so commit `31af4a0` carries the phase B–E edits to existing files.
  Phase commits from F on are complete. CLAUDE.md now documents the trap.
- The phase D commit and `d46cd09` say markers default on at 0.57 / 0.60 precision.
  Those numbers were the cold-open artifact described in §3; `b6ddc85` turned markers
  off.
- Written into `run/` (new files only, nothing existing modified): `clip_diff.json` in
  the four cache folders, `clip_lessons_genshin-impact.json`, `clip_lessons_report.md`,
  `eval/`. Also two **new cache folders**, `run/2862200457/` and
  `run/2872051525@5400-14400/` (audio.wav, transcript, speech, signals), rebuilt from
  `vods/` so the deleted runs could be scored. `Desktop\iphone` was only ever read.

## 8. Day to day

```bash
refire make <vod> --brief "..." --duration 15m --game "genshin impact"   # add --markers once eval shows they beat chance
# Premiere: Window > Extensions > reFire > Build, edit, add memes/SFX/stylization.
refire learn <vod>          # after the edit is done: finds the .prproj, diffs, updates lessons
refire eval --all           # see whether reFire is getting closer to your edits
```

`refire learn` finds the project on its own (`REFIRE_PROJECTS`, default
`~/Desktop/iphone`), so every finished edit adds to the lessons the markers and stages train on. To try a
stage on your next real run, add `--boundary`, `--trim`, `--critics` or `--lessons`
(or the style keys), then score the result with
`refire eval <vod> --manifest run/<vod>/<run>/ae/manifest.json` after you've edited and
learned it.
