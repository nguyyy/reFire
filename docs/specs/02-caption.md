# reFire feature specs · 02 · `refire caption`

> Depends on `00-shared.md` (timeline contract, `.prfpset` parser, Premiere notes, spikes S3–S7).

## Goal

Replace plain SRT captions in the Premiere path with Nguy's own caption look:
- Every caption block is its own text clip with the **captionIn** animation (fade + short slide up, fade out), with motion blur.
- All blocks live in a nested sequence, `refire_captions`, placed on main **V5**.
- The **ccStyle** look (Drop Shadow, Edge Glow, Camera Shake) is applied **once to the nest clip**, not per block. The ccStyle preset was saved from a 284 s clip, which fits it having lived on a nest. One effect instance instead of hundreds is far faster, and the shake runs continuously instead of restarting per caption.

`refire srt` (plain `captions.srt`) stays as it is.

## Decided

- Text: **Coolvetica Regular, size 55, black stroke 3**. Fill white (*assumed*). Existing lowercase-by-default / CAPS-on-excitement rule kept (*assumed*). Proper-noun repair (`caption_fixes.json`, learned corrections) kept.
- **No word-by-word highlight.**
- Captions **don't** move during punch-ins; they're sized not to interfere.
- Intro and outro have **fixed durations anchored to each block's start and end**. Nguy's example: with a 10-frame intro and 10-frame outro, a 50-frame block animates 0–10 and 40–50; a 100-frame block animates 0–10 and 90–100.
- **Always fade.** Overlapping captions are fine, but the **newer caption is always on top**.
- **Motion blur on** the slide.

## What the presets contain (decoded; use as parser test fixtures)

### `captionIn.prfpset` — preset "blurrcaption with better fade"

One **Transform** effect (`AE.ADBE Geometry2`), saved from a block $\frac{114{,}307{,}200{,}000}{254{,}016{,}000{,}000} = 0.45$ s long. Times below are relative to the block start.

| Param | Value |
|---|---|
| Anchor Point | static $(0.5, 0.5)$ |
| Position | key $0.000$ s: $(0.5,\ 0.5324)$ → key $0.247$ s: $(0.5,\ 0.5)$, Bezier. A slide up of $0.0324 \times 1080 \approx 35$ px. |
| Opacity | $0.000$ s: 0 → $0.058$ s: 100 → hold → $0.267$ s: 100 → $0.435$ s: 0, all Bezier |
| Uniform Scale (ParameterID 11) | on |
| Scale Height / Width | 100 |
| Skew, Skew Axis, Rotation | 0 |
| Use Composition's Shutter Angle (ParameterID 9) | **on** — in Premiere this disables motion blur |
| Shutter Angle | 360 |
| Sampling | 0 |
| Preset `Type` | 0 = **Scale** (confirmed from the MOGRT, see below). Irrelevant at runtime because we re-time keys ourselves. |

Note: the preset as saved renders **without** motion blur because "Use Composition's Shutter Angle" is checked. Nguy wants blur, so the pipeline turns that checkbox **off** and keeps Shutter Angle 360.

### `ccStyle.prfpset` — preset "captionStyling"

Three effects, in this order (keep the order). All static.

**1. Camera Shake** (`AE.Impact_Camera_Shake_FX`, Film Impact): Camera Mode 1, Speed 25, Master 100, Variation 20, Stabilize 25, Lean 0°, Auto Scale on, Enable Motion Blur on, Motion Blur 20, Edge Behavior 0, Seed 0, Apply Prescale off (prescale 50). Five Strafe/Stride/Roll triplets: (5.87, 10.34, 3.95), (8.48, 9.45, 0), (71.49, 56.45, 40.84), (48.24, 55.83, 57.46), (64, 76, 22.22). **Apply every named param from the file at runtime**; these values are for the fixture test only.

**2. Drop Shadow** (`AE.ADBE Drop Shadow`): color black (opaque), Opacity 165.75 on a 0–255 scale (= 65%), Direction 135°, Distance 5, Softness 0, Shadow Only off. Spike S5 confirms whether `setValue` expects 165.75 or 65.

**3. Edge Glow** (`AE.Impact_Edge_Glow_FX`, Film Impact): Color #FFDFB3 (warm cream), Intensity 40, Highlights Only 40, Size 15, Width 15, Height 15, Uniform Size on, Angle 0, Colorize 0, Vibrance 10, Desaturate 0, Chromatic Aberration 10, Blend Mode 0, Source Opacity 100.

Neither preset contains font, size, fill or stroke; those live in the caption MOGRT.

## One-time asset: the caption MOGRT

Received from Nguy as `captionMogrt.mogrt` (first export, below) and `captMograph.mogrt` (clean re-export: only `AE.ADBE Text`, no keyframes, same font/size/stroke/`TextLayer` param — **passes lint; use this one**). Decoded contents of the first export:

| Field | Value |
|---|---|
| Author app | Premiere (`authorApp: "ppro"`) → spike S3 must confirm `importMGT` accepts it |
| Frame | 1920×1080, 60 fps ($\frac{254{,}016{,}000{,}000}{4{,}233{,}600{,}000} = 60$), template length 0.733 s |
| Text | `CoolveticaRg-Regular`, size 55, stroke 3 (read from the Source Text blob), box 1440 wide |
| Editable param | one text control, display name **`TextLayer`** (sample value "fearless tier") |
| Effects | **a Transform with captionIn's keyframes baked in**, scaled to the template's 0.733 s; "Use Composition's Shutter Angle" still on |

**Required fix before use:** the baked Transform must go. If it stays, every caption plays its fade-out at about 0.71 s regardless of block length (longer captions vanish early, shorter ones get cut mid-fade), and the per-clip Transform the pipeline adds would animate a second time on top. Nguy re-exports the MOGRT with the Transform deleted from the graphic clip (text style unchanged). Animation comes only from the per-clip Transform, so it always follows the preset file.

**MOGRT lint** (runs before every caption apply; cheap):
1. Unzip the `.mogrt`; read `definition.json` (author app, frame size, fps, text control display name, fonts).
2. Unzip `project.prgraphic`, gunzip the inner `.prproj`, parse the XML.
3. **Fail** if any `IsTimeVarying` is `true` or any `MatchName` other than `AE.ADBE Text` is present, with the message: "Caption MOGRT contains animation/effects — remove the Transform from the graphic and re-export."
4. **Warn** if the MOGRT fps differs from the main sequence fps, or its fonts aren't installed.

Path in config: `caption.mogrt`.

## Pipeline

### 1. Blocks

Reuse recaption's regrouping of the run transcript against `timeline.tsv` (respects `words_per_line` from the style). Each block: `t_in`, `t_out` (timeline seconds), `text` (case rule and corrections applied). Optional `caption.tail` (default 0.0 s) extends each block's end.

### 2. Animation timing (fixed, anchored)

Derive intro and outro from `captionIn.prfpset` at runtime, per parameter:
- Find plateaus (runs of consecutive keys with equal values).
- Keys **before the end of the first plateau** form that param's intro ramp → anchored to the block start, same relative times.
- Keys **from the start of the last plateau** form its outro ramp → anchored to the block end, keeping their offset from the preset's out point.
- A param with only an intro ramp (Position) gets nothing at the end.

From Nguy's preset this gives $t_{intro} = 0.247$ s (fade-in done at $0.058$ s) and $t_{outro} = 0.45 - 0.267 = 0.183$ s. At 60 fps that's about 15 frames in (fade-in ≈ 3.5 frames) and 11 frames out. Keep key times in seconds; Premiere places them on its own frame grid.

For a block of length $L$:
- Intro keys at $0 \ldots t_{intro}$ from the block start.
- Outro keys at $L - t_{outro} \ldots L$ (the preset's fade ends $0.015$ s before the out point; keep that offset).
- **Short blocks:** if $L < t_{intro} + t_{outro}$, scale all intro and outro key times by $k = \dfrac{L}{t_{intro} + t_{outro}}$ and log it.

### 3. Easing

- `caption.easing: approx` (default): keys at the derived times, interpolation 5 (Bezier). Premiere's default Bezier handles; the preset's exact handle shapes can't be set by script.
- `caption.easing: exact`: dense linear keys, one per frame during intro and outro, sampled from a curve fitted to the preset's tangent fields. If those fields can't be decoded reliably, use ease-out cubic for the slide and ease-in-out for fades. Costs about $f \cdot (t_{intro} + t_{outro})$ keys per animated param per block at frame rate $f$; only switch to it if spike S4 shows `approx` looks visibly different.

### 4. Layering

Layers 1–4 (each becomes its own caption track, then its own graphics track above the nest's anchor on V1, decision R6). In time order:
- If block $i$ doesn't overlap block $i-1$: track 1.
- If it overlaps: track of block $i-1$ plus one (the newer caption sits on top).
- If that would exceed 4: trim block $i-1$ to end at block $i$'s start and put block $i$ on block $i-1$'s track.

### 5. Ops

`run/<vod_id>/ppro/ops_caption.json`, applied in chunks of 60 blocks per .jsx call (progress shown in the panel):

1. `ensure_nest`: create `refire_captions` if missing, with the main sequence's settings (clone main and clear it, or create from a preset with no dialog; spike S5). Ensure exactly one V5 item on main referencing it, spanning the whole main sequence; extend its end if the main sequence grew.
2. `style_nest`: only when the nest clip is newly created or `--restyle` is passed. QE-add Camera Shake, Drop Shadow, Edge Glow to the V5 nest clip in that order; set every named param from `ccStyle.prfpset`.
3. `clear_nest`: remove every track item inside `refire_captions` except the anchor on V1, then the empty video tracks.
4. `caption_tracks`: import `ppro/captions/layer_<k>.dfxp` (one TTML per layer, bottom first; font, size, color, outline and centering from the `caption:` keys) into a temp bin, `createCaptionTrack` each, delete the bin, lock the nest's video tracks (spike S8).
5. `menu`: the applier stops; `refire/ppro/menu.py` focuses the nest's timeline, Select All, Graphics and Titles > Upgrade Caption to Graphic (decision R6), then focuses main again. The applier resumes.
6. `after_upgrade`: unlock; the top `layers` video tracks must hold exactly one graphic per block; graphics from caption tracks the nest inherited from main are removed.
7. Per block: `caption_fx` on its graphic (the block's index on its layer's track): text Position = `text_pos`, then QE-add Transform (static params from the preset; Use Composition's Shutter Angle **off**; Shutter Angle 360; Opacity and Position keys from step 2).

### 6. Review file

`run/<vod_id>/caption_review.tsv`: block id, `t_in`, `t_out`, text, track, `anim_scaled` flag, corrections applied.

## Entry points

- CLI: `refire caption <run_dir> [--timeline timeline.tsv] [--dry-run] [--restyle]`.
- Panel: the existing **recaption** button gains the styled path (setting: `caption.mode: styled | srt`, default `styled`). Re-cuts → recaption → nest contents fully rebuilt; the nest clip and its ccStyle effects stay.

## Config (`caption:` frontmatter map)

| Key | Default |
|---|---|
| `mode` | `styled` |
| `mogrt` | `refire/assets/premiere/captMograph.mogrt` (clean re-export; since R6 only the nest's empty anchor clip) |
| `mogrt_text_param` | `TextLayer` |
| `preset_in` | `refire/assets/premiere/captionIn.prfpset` |
| `preset_style` | `refire/assets/premiere/ccStyle.prfpset` |

Paths are repo-relative by default and resolved to absolute paths before they go into the ops file (`importMGT` needs absolute paths). A style document can point them anywhere else. Parser and lint tests read the same files, so updating a preset in the repo updates both the output and the fixtures (re-baseline the expected values when that happens).
| `easing` | `exact` (spike S4: `approx` looked visibly different) |
| `motion_blur` | `true` (forces Use Composition's Shutter Angle off) |
| `shutter_angle` | 360 |
| `tail` | 0.0 s |
| `max_tracks` | 4 |
| `nest_track` | 5 |
| `chunk` | 60 |
| `font`, `font_size`, `color`, `outline` | `Coolvetica`, 55, `#FFFFFF`, `#000000 3px` (the MOGRT's look, as TTML style) |
| `text_pos` | [0.125, 0.9155] (S8: the MOGRT's text box) |

## Acceptance criteria

- Parser fixtures reproduce every value in "What the presets contain".
- MOGRT lint fails on `captionMogrt.mogrt` (baked Transform) and passes on `captMograph.mogrt`.
- Timing unit tests with a synthetic 10-frame intro / 10-frame outro: 50 f → 0–10 and 40–50; 100 f → 0–10 and 90–100; 12 f → scaled with $k = 0.6$.
- With Nguy's real preset: intro keys at $0, 0.058, 0.247$ s from block start; fade-out from $L - 0.183$ to $L - 0.015$ s.
- Layering tests: newer always on a higher track during any overlap; never more than 4 tracks.
- Visual check: a frame mid-slide shows motion blur; a frame after $t_{intro}$ shows text at rest; the nest shows the drop shadow, cream glow and shake.
- Running caption twice produces an identical nest (same item count, times, text).
- 300 blocks apply in ≤ 5 minutes (spike S6 budget).
