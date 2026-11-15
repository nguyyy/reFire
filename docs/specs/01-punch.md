# reFire feature specs · 01 · `refire punch`

> Depends on `00-shared.md` (timeline contract, facecam detector, Premiere notes, spikes S1–S2).

## Goal

On genuinely good moments, the edit hard-cuts to a 210% zoom on zy0x's facecam and hard-cuts back once he's done. One Premiere adjustment layer (with a Transform effect) per punch on V2. No keyframes: a punch is a cut, so every value is static for the layer's whole length.

## Rules from Nguy (non-negotiable)

- **Never punch on** uninteresting short phrases ("okay i see" is out; "OH SHIT" and real exclamations are in).
- **Never punch too often or stack punches.** A punch, then 1 s of normal, then another punch is not allowed: either stay zoomed through or drop one.
- **Never punch on boring dialogue:** reading quest dialogue, listing stats, reading in-game text, plain monologue. Exception: a genuinely interesting or dramatic read can qualify.
- **Laughter counts**, but only when sustained (≥ 1.0 s), so it doesn't trip the other rules.
- **No punch when the facecam shows no face** (cam hidden, fullscreen cutscene).
- **Pin, don't center:** zoom about the facecam's nearest corner or edge.

## Inputs

- `timeline.tsv` (current V1, via recaption's dump) + `timeline_map`.
- Run transcript with word timestamps (faster-whisper / Deepgram output already in the run).
- Signals: audio loudness, chat message-rate z-score (both already in the moment map), plus laughter probability (new, below).
- The existing caption excitement detector (CAPS-on-excitement) as a feature.
- `facecam.json` (shared detector).
- Director backend (`cli` / `api` / `--local-director`) for the one batched judge call.

## Pipeline

### 1. Phrases

Split the transcript (mapped to timeline time) into phrases at pauses > 0.30 s or sentence punctuation. Drop phrases that straddle a V1 cut into a different source.

### 2. Features per phrase

| Feature | Definition |
|---|---|
| `z_loud_peak`, `z_loud_mean` | Short-term loudness vs a rolling median baseline (reuse `loud`'s `--baseline` method) |
| `z_chat` | Max chat-rate z-score in the 6 s after phrase start (reaction lag) |
| `laugh` | Laughter probability and sustained duration from YAMNet (or PANNs) on the phrase ± 1 s |
| `lex` | Matches an exclamation lexicon (seed: oh my god, no way, what, bro, let's go, are you kidding, holy, profanity list); editable list in `refire/punch_lexicon.txt` |
| `caps` | The caption excitement detector fired on any word |
| `n_words`, `dur`, `rate` | Size and speaking rate |
| `face` | `face_present()` on the facecam ROI |
| `reading` | See step 4 |

### 3. Cheap prefilter

Keep a phrase only if at least one is true: $z_{loud,peak} \ge 1.5$, `lex`, `caps`, sustained laughter ≥ 1.0 s, $z_{chat} \ge 2.0$. Then drop it if `face` < 0.5.

### 4. Reading detection

- **Transcript inference (always on, free):** the judge (step 5) gets ± 8 s of context and the game glossary, and labels reading-type speech (NPC names, quest-dialogue register, stat lists).
- **OCR (only if cheap):** for each surviving candidate, OCR 2 frames, restricted to the dialogue-box and tooltip regions of the game frame (RapidOCR). Mark `reading` if fuzzy match ratio between phrase and OCR text ≥ 0.6.
  - Budget: total OCR time ≤ 30 s per video. If the budget runs out, the remaining candidates use transcript inference only. Config `punch.ocr: auto | on | off`.
- A reading candidate can still punch if the judge marks it `dramatic_read` and $z_{loud,peak} \ge 2.0$ (Nguy: "maybe cut on very interesting pieces of quest text").

### 5. Judge (one batched LLM call per video)

- Input per candidate: id, phrase text, ± 8 s context, features, `reading` flag.
- Prompt states Nguy's rules above verbatim, with his examples ("okay i see" → no; "OH SHIT" → yes).
- Output JSON per candidate: `category` (`exclamation | reaction | punchline | roast | laughter | rage | dramatic_read | reading | stats | filler | other`), `worthy` 0–1, one-line `reason`.
- Cache by hash of inputs in `run/<vod_id>/punch_judge.json`. Respect `--local-director`.

### 6. Score

$$s = 0.50\,w + 0.20\,\sigma(z_{loud,peak} - 1.5) + 0.15\,\sigma(z_{chat} - 1.5) + 0.15\,\mathbb{1}[\text{lex} \lor \text{caps} \lor \text{laugh}_{\ge 1\text{s}}]$$

where $w$ is the judge's `worthy` and $\sigma$ is the logistic function. Categories `reading`, `stats`, `filler` force $s = 0$. Keep candidates with $s \ge$ `punch.threshold` (0.60). Weights are config so tuning doesn't need code changes.

### 7. Timing

- $t_{in} = t_{\text{first word}} - 0.10$
- $t_{out} = t_{\text{phrase end}} + 0.10$, where phrase end is the last word's end or the end of trailing sustained laughter, whichever is later.
- **Snap to cuts:** if a V1 cut lies in $[t_{in} - 0.40,\ t_{in}]$, set $t_{in}$ to the cut (punch in on the cut). If a cut lies in $[t_{out},\ t_{out} + 0.40]$, extend $t_{out}$ to the cut (no zoom-out a few frames before a cut).
- **Spanning a cut:** allowed only if the facecam box is the same on both sides (IoU ≥ 0.8) and the phrase continues across it. Otherwise end the punch at the cut.

### 8. Spacing and merging

Process in time order. Let the gap be the time from one punch's end to the next's start.

| Gap | Action |
|---|---|
| < `merge_gap` (2.0 s) | Merge into one punch (stay zoomed through) |
| `merge_gap` ≤ gap < `min_gap` (4.0 s) | Keep the higher score, drop the other |
| ≥ `min_gap` | Keep both |

Then:
- Drop punches shorter than `min_len` (0.5 s), or shorter than `laugh_min_len` (1.0 s) when laughter is the only reason.
- **No hard max length.** Soft rule: a merged punch longer than `soft_long` (6.0 s) must contain a qualifying phrase at least every 3 s; otherwise split it at the lowest-energy gap and drop the weaker side.
- **Soft coverage cap:** punched time ≤ 25% of any rolling 60 s window and ≤ 20% of the whole video. Drop lowest scores until both hold. These are first guesses; Nguy tunes by trial and error.

### 9. Pin and scale

Facecam box $(x_0, y_0, x_1, y_1)$ and center $c$, all normalized 0–1.

Pin per axis:

$$p_x = \begin{cases} 0 & c_x < \tfrac13 \\ 1 & c_x > \tfrac23 \\ c_x & \text{otherwise} \end{cases} \qquad p_y \text{ likewise.}$$

So bottom-left → $(0, 1)$, middle-left → $(0, c_y)$.

Transform settings: **Anchor Point = Position = $p$**, Uniform Scale on, Scale $= 100\,s$.

Scaling about a point inside the frame with $s \ge 1$ always covers the whole frame, so this can never produce black bars.

Keep the whole facecam visible after scaling. After scaling about $p$, an edge $e$ lands at $p + s(e - p)$. For each box edge on each axis:

$$s \le \frac{1 - p}{e - p}\ \text{if } e > p, \qquad s \le \frac{p}{p - e}\ \text{if } e < p$$

Final $s = \min(2.10, \text{all bounds})$. If $s < 1.5$, skip the punch and log `facecam_too_large`.

### 10. Sharpening

Add Premiere's **Unsharp Mask** to each punch layer after Transform: Amount 50, Radius 1.0, Threshold 0 (config). It offsets the softness of showing about $\frac{1920}{2.1} \approx 914$ by $\frac{1080}{2.1} \approx 514$ source pixels at full frame.

## Outputs

`run/<vod_id>/punch.json`:

```json
{ "version": 1,
  "params": { "scale": 2.1, "preroll": 0.1, "postroll": 0.1, "merge_gap": 2.0, "min_gap": 4.0 },
  "punches": [
    { "id": "p07", "t_in": 312.48, "t_out": 314.92,
      "pin": [0.0, 1.0], "scale": 2.1,
      "score": 0.83, "category": "exclamation",
      "text": "OH NO NO NO", "reason": "loud panic reaction to boss phase",
      "source": "vod_2291", "merged_from": ["c41", "c42"], "snapped_to_cut": "in" } ] }
```

`run/<vod_id>/punch_review.tsv`: one row per candidate, **kept and dropped**, with features, score and the drop reason (`below_threshold`, `reading`, `no_face`, `spacing_lost_to:p07`, `coverage_cap`, `too_short`...). This is the tuning tool.

`run/<vod_id>/ppro/ops_punch.json`: tick-exact ops (below).

## Applying in Premiere

Ops, in order, applied by the thin .jsx:

1. `clear_track`: remove every V2 track item whose project item is `refire_punch_adj`.
2. For each punch:
   - `insert_adjustment`: overwrite `refire_punch_adj` onto V2 at `t_in`, set the end to `t_out` (spike S1).
   - `add_effect Transform` (QE): Uniform Scale on, Scale Height = Scale Width = $100\,s$, Anchor Point = Position = `pin` (spike S2).
   - `add_effect Unsharp Mask` with the configured values.

Entry points:
- CLI: `refire punch <run_dir> [--timeline timeline.tsv] [--dry-run] [--apply]`. `--dry-run` writes `punch.json` + review only.
- Panel button **Punch-ins**: dump V1 → run analysis → apply ops. Re-running after a re-cut fully replaces V2's reFire items.

## Config (`punch:` frontmatter map)

| Key | Default |
|---|---|
| `scale` | 2.10 |
| `min_scale` | 1.50 |
| `preroll` / `postroll` | 0.10 s |
| `snap_to_cut` | 0.40 s |
| `merge_gap` | 2.0 s |
| `min_gap` | 4.0 s |
| `min_len` | 0.5 s |
| `laugh_min_len` | 1.0 s |
| `soft_long` | 6.0 s |
| `coverage_window` / `coverage_window_max` / `coverage_total_max` | 60 s / 0.25 / 0.20 |
| `threshold` | 0.60 |
| `weights` | `{judge: 0.5, loud: 0.2, chat: 0.15, flags: 0.15}` |
| `ocr` | `auto` (30 s budget) |
| `sharpen` | `{amount: 50, radius: 1.0, threshold: 0}` |
| `track` | 2 |

## Acceptance criteria

- Fixtures from Nguy's examples: a flat "okay i see" → no punch; a loud "OH SHIT" → punch; a flat read of quest dialogue → no punch; 1.4 s of laughter → punch; 0.6 s chuckle → none.
- No two kept punches have a gap < `min_gap`; every gap < `merge_gap` was merged.
- No punch where `face_present` < 0.5.
- Every punch with a cut within 0.40 s before it starts on that cut.
- Pin math: unit tests for bottom-left, middle-left, top-right and center facecams, and a too-large facecam that triggers the skip.
- Visual smoke test: stills from 5 random punches show no black bars and the full facecam visible.
- Runtime ≤ 60 s for a 15-minute video, excluding the judge call and OCR.

## Later (not v1)

- `refire punch --learn`: Nguy marks keep/kill in `punch_review.tsv`; refit `threshold` and `weights` with simple logistic regression.
