# Phase 0 spike results

Spec: `docs/specs/00-shared.md` (Phase 0). Driven by `tests/smoke/spikes.py` +
`tests/smoke/spikes.jsx` through `refire/ppro/remote.py`. Premiere **26.3.0**, scratch project
`scratch.prproj`, sequence `refire_spike` built from one real DNxHR proxy
(`p_dnxhr_sq_10735.300-10780.800.mov`, 1920x1080). Run 2026-11-14.

## Status

| # | Spike | Result | Notes |
|---|---|---|---|
| S1 | Adjustment layer on V2 | **Fallback** (creation) + **Pass** (placement) | No API creates one: neither the DOM `app.project` nor QE `qe.project` has a method (QE has `newTransparentVideo`, `newColorMatte`, `newBlackVideo` only). Fallback per spec: Nguy creates `refire_punch_adj` once (done). Placement: `overwriteClip` + `trackItem.end = Time(ticks)` lands frame-exact for 0.5 s, 3.333 s and 60 s (longer than the sequence); DOM `trackItem.name = 'refire:punch:...'` is writable and reads back. The renamed item came back as `' refire_punch_adj'` (leading space): **match project items by trimmed name**. |
| S2 | QE Transform zoom, no black bars | **Pass** | Old Transform on the adjustment layer: anchor = position = pin [0, 1], uniform on, scale 210. Zoomed frame shows the facecam corner magnified with all four edge strips well above black (means 83-122); same check on the unzoomed frame 62-90. |
| S3 | `importMGT` + set `TextLayer` text | **Fail** as specced; **baked MOGRT works** (S3b) | `importMGT` accepts the Premiere-made MOGRT (`isMGT()` true) and the clip stretches freely (0.47 s template -> 2 s). But `getMGTComponent()` is **null**, so there is no `TextLayer` param. The text lives in component `AE.ADBE Text` > `Source Text`, whose `getValue()` returns 1 garbage char; `setValue()` with JSON or a plain string returns true but **blanks the text**. Needs a decision (below). |
| S4 | Transform keys, Bezier, motion blur | Keys/blur **pass**; `approx` easing **visibly different** -> `exact` (S4b) | Opacity + Position keys with `setInterpolationTypeAtKey(t, 5)` all return ok; Use Composition's Shutter Angle off + Shutter Angle 360 set. Frames: 0.03 s faded + low, 0.12 s mid-slide visibly motion-blurred (Laplacian var 960 vs 1873 at rest), 0.30 s sharp at rest, L-0.10 s fading out. Key times are `trackItem.inPoint.seconds + offset`. |
| S5 | Caption nest on V5 + ccStyle effects | **Pass** | Nguy confirmed Effect Controls order Camera Shake, Drop Shadow, Edge Glow (DOM component order is not the UI order). The nest clip showed no text because the nest's 0-46 s window held only the anchor; S6's captions sat at 200 s+ and had no text set. |
| S6 | 300 MOGRT captions in chunks of 50 | **Pass**: 114 s total (budget 300 s) | Each block = import + stretch + Transform + 6 Bezier keys + checkbox. Chunk time grows as the nest fills (11 s -> 21 s per 50). `importMGT` fails ~15% of calls in bursts; verify + retry fixes it (7 retries / 300, 0 failures). Text set not included (S3). |
| S7 | Transform checkbox params by index | **Pass** | QE adds the old Transform (see below), whose 12 params match the preset's `Params Index` order exactly **and carry display names**: 0 Anchor Point, 1 Position, 2 Uniform Scale, 3 Scale Height, 4 Scale Width, 5 Skew, 6 Skew Axis, 7 Rotation, 8 Opacity, 9 Use Composition's Shutter Angle, 10 Shutter Angle, 11 Sampling. Film Impact effects have unnamed params -> map by preset index (step 2 parser). |
| S9 | Sauce zoom targeting (04-sauce P0) | **Pass, with a check** (2026-11-27) | OWLv2 base-ensemble (Apache-2.0), the onnx-community ONNX export, CPU only (no torch here): ~11 s per 1080p frame fp32; int8 is 7.5 s and boxes worse. Fine for his ~0.1 zooms/min. Scores do **not** separate right from wrong boxes (a wrong box on UI scored 0.35, a right one on a held figure 0.30-0.35), so each zoom's actual crop gets one batched Claude look before it ships (`place.verify_zooms`); a wrong 143k-damage zoom was caught that way. Zooms are a hard cut to a static window (Nguy: no keyframed ease), so the Premiere keyframe-density spike isn't needed; a target that leaves the window by the last frame skips the zoom. |
| S8 | Caption track + Upgrade Caption to Graphic | **Pass** (2026-11-23) | `Sequence.createCaptionTrack(item, 0, Sequence.CAPTION_FORMAT_SUBTITLE)` makes a caption track from an imported caption file; scripting has no upgrade, but the Windows menu bar is a native HMENU, so WM_COMMAND runs Graphics and Titles > Upgrade Caption to Graphic: ~1180 captions became graphic clips in under 3 s. Details below. |

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

## S3b: setting caption text (2026-11-14, second pass)

The `TextLayer` control exists in `definition.json` (`clientControls[0]`, type 6, value "list, BRO.")
but Premiere-authored MOGRTs expose it nowhere in the DOM. The text lives in the `Source Text`
param (`ArbVideoComponentParam`, control type 9) as a binary blob, base64 in the MOGRT's inner
`.prproj` (`StartKeyframeValue`):

- layout: u64 little-endian payload length (= blob length - 12), magic `44 33 22 11`, then a
  FlatBuffer. Strings are FlatBuffer strings (u32 length + UTF-8 + NUL): `AnimationType`,
  `CoolveticaRg-Regular`, the caption text. Exactly one forward `uoffset` points at the text string.
- `getValue()` returns **1 char** inside ExtendScript itself (U+01D8 = the first two blob bytes; the
  next u16 is 0x0000), so the live blob can't be read; Premiere truncates at the first NUL.
- `setValue()` with the full patched blob (as UTF-16 units, as bytes, or as base64): returns true,
  text **blank** every time. Not usable.
- **Works: bake the text into a per-caption `.mogrt`.** Python appends the new string at the end of
  the FlatBuffer, repoints the one `uoffset`, fixes the length prefix, writes it back into the inner
  `.prproj`, sets `definition.json`'s control value, and zips a new `.mogrt` (~18 KB, 2 ms each).
  `importMGT` of the baked file renders the new text in Coolvetica with the stroke, any length
  (tested 10-23 chars), same or new `capsuleID`. Style runs did not need patching.
- MOGRT clips have **no project item** (`trackItem.projectItem` null; bin unchanged after 450+
  imports): the graphic is embedded, so baked files are temp files and the bin stays clean.
- Cost: 50 distinct baked imports 10.8 s vs 9.0 s for 50 imports of one file (import only, in the
  nest, same retry loop): ~+10 s per 300 captions. S6's 114 s/300 leaves room.
- Spec impact: `02-caption.md` step 4 sets text through `getMGTComponent()`; this replaces it with
  "Python writes `<id>.mogrt` per block, the .jsx imports that file". Needs Nguy's OK.

## S4b: easing (hand preset vs script)

Nguy applied captionIn by hand to a 0.45 s caption. Premiere added `AE.ADBE Geometry2` and
**scaled** the preset's keys by media/preset duration (0.4667/0.45: position 0 -> 0.256 s, opacity
0/0.060/0.277/0.452 s), left Use Composition's Shutter Angle on (no blur), uniform on. That is the
behaviour the spec's anchored timing avoids. Sampled with `getValueAtTime` (fixture
`tests/fixtures/captionIn_curve.json`, 5 ms steps):

| t (s) | 0.02 | 0.04 | 0.08 | 0.12 | 0.16 | 0.20 | 0.24 |
|---|---|---|---|---|---|---|---|
| preset y | 0.5197 | 0.5143 | 0.5079 | 0.5041 | 0.5018 | 0.5006 | 0.5000 |
| script (interp 5) y | 0.5317 | 0.5305 | 0.5271 | 0.5230 | 0.5179 | 0.5117 | 0.5025 |

The preset slide is a hard ease-out (39% of the travel by 0.02 s); default Bezier handles give
nearly the opposite (slow start, lands hard). Per spec: `approx` is visibly different, so the
caption ops use `caption.easing: exact` (dense linear keys from the preset's tangents, checked
against this fixture). Fade-in differs less (opacity 24/84 vs 27/65 at 0.02/0.04 s).

## Open (needs Nguy)

1. ~~Adjustment layer~~, ~~hand-applied captionIn~~, ~~nest effect order~~: done.
2. ~~S3~~: AE-built MOGRT rejected (AE render times); baked per-caption MOGRTs approved
   (decision R5, Nov 21).
3. ~~Caption smoke + real-size timing~~: `tests/smoke/caption.py` 19/19 in Premiere (Nov 23, R6
   flow); 1068 real captions applied in 343 s, almost all of it the ~22 Transform keys per block.

## S8: captions in one go (2026-11-23, Nguy: "select the whole captions track then convert all of them")

Replaces S3b's baked MOGRT per block (one `importMGT` per caption, ~15% retries, the nest
filling one clip at a time). Premiere 26.3.0, `makeovertest.prproj`, throwaway sequence.

- **Upgrade**: Edit > Select All + Graphics and Titles > Upgrade Caption to Graphic, sent as
  WM_COMMAND to the `Premiere Pro` window (ids found by label; they change per session). Each
  non-empty caption track becomes a graphic clip track (`isMGT()` true, component `AE.ADBE Text`)
  on a **new video track on top**, in caption-track order; the caption tracks are left empty.
- **Focus**: menu commands hit the focused panel. Select All went to the Project panel once
  (bin items selected, nothing upgraded); Window > Timelines > <sequence> focuses that timeline
  (and makes it the active sequence). If clips are selected along with captions, the upgrade
  does nothing: lock the video tracks first (QE `setLock`), locked clips aren't selected.
- **Style**: an SRT gets Premiere's default caption look (`<font color>` works, `face`/`size`
  don't). A TTML (`.dfxp`) style honors `fontFamily`, `fontSize`, `color`, `textOutline` and
  `textAlign` **when set on the referenced `<style>`**; regions, `tts:origin`, `displayAlign`
  and region-level `textAlign` are ignored, and every TTML graphic lands at text Position
  (0.05, 0.05) in a 1440 px box. Coolvetica 55 / outline 3 px renders the same glyph box as the
  caption MOGRT (x 822-1096 for "compare me" on both).
- **Position**: the MOGRT's text box is (0.125, 0.65), bottom-aligned; the TTML box is
  top-aligned. Text component Position = (0.125, 0.9155) puts the TTML glyphs on the MOGRT's
  (top edge equal, bottom within 2 px). Text Anchor Point is in frame units (0.5 moved the text
  half a frame), so it can't center by itself.
- **Clean-up**: a caption track keeps its captions after its source item's bin is deleted;
  QE `removeEmptyVideoTracks()` drops empty video tracks only.

