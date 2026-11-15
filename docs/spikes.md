# Phase 0 spike results

Spec: `docs/specs/00-shared.md` (Phase 0). Driven by `tests/smoke/spikes.py` +
`tests/smoke/spikes.jsx` through `refire/ppro/remote.py`. Premiere **26.3.0**, scratch project
`scratch.prproj`, sequence `refire_spike` built from one real DNxHR proxy
(`p_dnxhr_sq_10735.300-10780.800.mov`, 1920x1080). Run 2026-11-14.

## Status

| # | Spike | Result | Notes |
|---|---|---|---|
| S1 | Adjustment layer on V2 | **Fallback** (creation); placement pending | No API creates one: neither the DOM `app.project` nor QE `qe.project` has a method (QE has `newTransparentVideo`, `newColorMatte`, `newBlackVideo` only). Fallback per spec: Nguy creates `refire_punch_adj` once. Frame-exact placement + length still to test once it exists. |
| S2 | QE Transform zoom, no black bars | Pending S1 | Transform itself works (see S4/S7). |
| S3 | `importMGT` + set `TextLayer` text | **Fail** | `importMGT` accepts the Premiere-made MOGRT (`isMGT()` true) and the clip stretches freely (0.47 s template -> 2 s). But `getMGTComponent()` is **null**, so there is no `TextLayer` param. The text lives in component `AE.ADBE Text` > `Source Text`, whose `getValue()` returns 1 garbage char; `setValue()` with JSON or a plain string returns true but **blanks the text**. Needs a decision (below). |
| S4 | Transform keys, Bezier, motion blur | **Pass** (approx easing); hand-preset comparison pending | Opacity + Position keys with `setInterpolationTypeAtKey(t, 5)` all return ok; Use Composition's Shutter Angle off + Shutter Angle 360 set. Frames: 0.03 s faded + low, 0.12 s mid-slide visibly motion-blurred (Laplacian var 960 vs 1873 at rest), 0.30 s sharp at rest, L-0.10 s fading out. Key times are `trackItem.inPoint.seconds + offset`. |
| S5 | Caption nest on V5 + ccStyle effects | **Pass** (order check pending) | See S5 notes. |
| S6 | 300 MOGRT captions in chunks of 50 | **Pass**: 114 s total (budget 300 s) | Each block = import + stretch + Transform + 6 Bezier keys + checkbox. Chunk time grows as the nest fills (11 s -> 21 s per 50). `importMGT` fails ~15% of calls in bursts; verify + retry fixes it (7 retries / 300, 0 failures). Text set not included (S3). |
| S7 | Transform checkbox params by index | **Pass** | QE adds the old Transform (see below), whose 12 params match the preset's `Params Index` order exactly **and carry display names**: 0 Anchor Point, 1 Position, 2 Uniform Scale, 3 Scale Height, 4 Scale Width, 5 Skew, 6 Skew Axis, 7 Rotation, 8 Opacity, 9 Use Composition's Shutter Angle, 10 Shutter Angle, 11 Sampling. Film Impact effects have unnamed params -> map by preset index (step 2 parser). |

## Findings the applier must follow

- **Two effects are named "Transform".** `qe.project.getVideoEffectByName("Transform")` adds
  **`AE.ADBE Geometry`** (old), not the preset's `AE.ADBE Geometry2`. By match name
  (`"AE.ADBE Geometry2"`) or any unknown name, `getVideoEffectByName` returns an object and
  `addVideoEffect` silently adds nothing. `getVideoEffectList()` holds name strings only.
  Geometry has the same 12 params as Geometry2, but different defaults (Uniform Scale off,
  Shutter Angle 0), so **set every param explicitly**. Not yet known: whether Geometry renders
  differently or slower (GPU) than Geometry2.
- **Always verify the added effect's `matchName`** after `addVideoEffect`. Names resolved here:
  `Camera Shake` -> `AE.Impact_Camera_Shake_FX`, `Drop Shadow` -> `AE.ADBE Drop Shadow`,
  `Edge Glow` -> `AE.Impact_Edge_Glow_FX` (first list match; `uni.*`, `BCC*`, `S_*` lookalikes
  are installed too).
- **QE works on the active sequence only.** `seq.clone()` makes the clone active;
  `app.project.activeSequence = seq` switches back. Caption ops must activate
  `refire_captions` while adding effects, then restore main.
- `sequence.importMGT(...)` works on a **non-active** sequence.
- **An empty sequence can't be placed.** `overwriteClip`/`insertClip` of an empty nest's project
  item return true and place nothing. With an item at main end - 1 frame inside the nest, the
  nest lands on V5 spanning the whole main sequence. Plan: keep an invisible, tagged anchor
  (QE `newTransparentVideo`) at the nest tail.
- `qe sequence.exportFramePNG(timecode, path)` wants a **native path without extension**
  (it appends `.png`); a forward-slash path throws "Unknown error exception".
- **Track-item naming:** QE track items have `setName`; DOM `trackItem.name =` still to test (S1).
- **Drop Shadow opacity is 0-255:** default reads 127.5; `setValue(165.75)` reads back 165.75.
- **Colors:** `setColorValue(a, r, g, b, true)` with 8-bit channels works; read back with
  `getColorValue()` (`getValue()` returns the 64-bit packed int, which loses precision in a JS double).
- Film Impact params read back fine by index (Camera Shake 47 props, Edge Glow 28, including
  `_ ` internals and unnamed group rows), e.g. Camera Shake Speed = index 28, Edge Glow
  Intensity = 4, Color = 11.
- **Sequence fps follows the media:** the proxy-built sequence is 60 fps (`timebase` 4233600000)
  while that run's manifest says `fps: 30`. Ops must take fps from the live sequence
  (`seq.timebase`), not the manifest.
- `addTracks(n, after, 0)` on the QE sequence adds video tracks without a dialog.

## Open (needs Nguy)

1. Create the adjustment layer once: File > New > Adjustment Layer, rename it
   `refire_punch_adj` in the Project panel. Then S1 placement + S2 black-bar check run.
2. Drag `captionIn` onto the 0.45 s caption clip at 36 s on V3 of `refire_spike` (S4 comparison).
3. Look at the V5 nest clip in Effect Controls: the order must be Camera Shake, Drop Shadow,
   Edge Glow (the DOM component list shows them reversed, but it also lists Opacity before
   Motion, so it is not the UI order). Play it to confirm the shake.
4. S3 decision: how to set caption text (Premiere-made MOGRT text is not scriptable).
