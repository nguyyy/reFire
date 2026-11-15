# reFire feature specs · 00 · Shared foundations

> Read this first. `01-punch.md`, `02-caption.md` and `03-thumb.md` all build on it.
> Written Nov 2026 for the reFire makeover in Claude Code. Repo facts assumed: Python 3.12 + uv, pytest, Premiere CEP panel at `refire/ppro/`, `refire/ppro/remote.py` drives the panel over CDP (port 8089), the panel's **recaption** dumps V1 to `timeline.tsv`.

## Scope

| Feature | Command | Output |
|---|---|---|
| Punch-ins | `refire punch` | Adjustment layers on V2 that hard-cut to a 210% facecam zoom |
| Captions | `refire caption` | Animated MOGRT caption blocks inside a nested captions sequence on V5 |
| Thumbnails | `refire thumb` | 3 PNG candidates + an editable spec per candidate + a critic report |

**Premiere only.** The After Effects panel is deprecated for reFire. Don't extend it or keep feature parity with it.

## Decisions log

Settled with Nguy unless marked *assumed* (Claude's call, veto-able).

| # | Decision |
|---|---|
| P1 | Punch framing pins the zoom to the facecam's nearest corner or edge. Never translate the frame (translation is what creates black bars). |
| P2 | Laughter qualifies for a punch only if it's sustained (≥ 1.0 s). |
| P3 | Punch in 0.10 s before the phrase, out 0.10 s after it fully ends. If a clip cut is within 0.40 s, punch in on the cut. |
| P4 | No hard max punch length, no hard per-minute cap. A soft coverage cap keeps it from being "the streamer all the time". Tune by trial and error. |
| P5 | OCR for "is he reading on-screen text" only if it's cheap; otherwise infer from the transcript. No punch when no face is visible in the facecam. |
| P6 | Track layout: V1 gameplay, V2 punch layers, V3–V4 user memes, V5 caption nest. |
| P7 | Light sharpening on every punch layer. |
| C1 | Caption text: Coolvetica Regular, size 55, black stroke 3. |
| C2 | Caption intro/outro have fixed durations anchored to each block's start and end, whatever the block length. |
| C3 | Captions always fade. Overlaps are fine; the newer caption is always layered on top. |
| C4 | Caption slide uses motion blur. |
| C5 | No word-by-word highlight. Captions don't move out of the way during punch-ins. |
| T1 | Thumbnail font: Conneqt Regular, heavy stroke, metallic gradient, black backing rectangle per line. |
| T2 | External art is fetched by tool use each run. No curated local library (an automatic download cache is fine). |
| T3 | One candidate uses the pixelated-reveal treatment whenever the video has a reveal. |
| T4 | Face restoration (GFPGAN) **on** by default, Nguy's call (Nov 14). Every restored face must pass the SFace identity check or it's reverted. CodeFormer stays excluded. See license table. |
| T5 | *Assumed:* the critic's top pick is uploaded first in Test & Compare, because an inconclusive test keeps the first uploaded thumbnail. |
| A1 | *Assumed:* caption fill is white; the existing lowercase-by-default / CAPS-on-excitement rule stays. |
| A2 | *Assumed:* one title suggestion per thumbnail concept; thumbnail text never swears. |

## Architecture principle: Python decides, .jsx applies

- All analysis, scoring and layout math lives in Python and is unit-tested.
- Python writes an ops file: `run/<vod_id>/ppro/ops_<feature>.json` (a flat list of operations with tick-exact times and parameter values).
- The panel (or `remote.py`) applies the ops file with a thin, dumb .jsx. No decisions in .jsx.
- Every apply is **idempotent**: it first removes the items reFire previously created in the affected tracks, then re-creates them. It never touches items it didn't create.
- Why: Adobe will eventually move Premiere scripting to UXP. A thin "apply this JSON" layer is a small port later.

## Timeline contract

- **The final timeline is the source of truth.** Features run off the current Premiere timeline, not the original `make` output, so hand re-cuts never orphan punches or captions.
- Reuse recaption's dump: V1 → `timeline.tsv` (per clip: timeline in/out, source id, source in/out).
- Create (or extract from recaption) `refire/timeline_map.py`:
  - `timeline_to_source(t) -> (source_id, source_t) | None`
  - `source_to_timeline(source_id, source_t) -> list[float]` (a source moment can appear more than once)
  - `cuts() -> list[float]` (timeline times of V1 edit points)
- All feature outputs are in **timeline seconds**; conversion to ticks happens once, at ops-file writing.

### Track layout (main sequence)

| Track | Content | Owner |
|---|---|---|
| V1 | Gameplay clips | reFire `make` / user re-cuts |
| V2 | Punch adjustment layers | `refire punch` |
| V3–V4 | Memes, SFX visuals | User only. reFire never touches these. |
| V5 | Caption nest (`refire_captions` sequence) | `refire caption` |

Anything on V1 is zoomed by a punch; V3–V5 are not. reFire edits only V2 and V5, and only items it tagged.

### Tagging

- Punch layers: project item named `refire_punch_adj` (one shared adjustment-layer project item, many track items).
- Caption nest: sequence named `refire_captions`; its project item is the only V5 item reFire touches.
- If track-item names are writable in the spike, also name each track item `refire:<feature>:<id>`.

## Shared module: facecam detector (`refire/facecam.py`)

Used by punch (pin + face-present check) and thumb (cutout source). zy0x's facecam is usually in the same place but not always, and `loud` runs stitch several VODs, so detect per source.

**Algorithm**

1. For each source in the run's `sources[]`, sample frames across the time ranges actually used: one every 20 s, min 30, max 120 frames, decoded at 640 px wide.
2. Run YuNet face detection (OpenCV `FaceDetectorYN`, confidence ≥ 0.8).
3. Cluster face centers (DBSCAN, eps 0.05 in normalized units). The facecam cluster is the one with the highest frame coverage *and* low box-size variance (a webcam face stays roughly the same size; in-game faces don't).
4. Estimate the facecam rectangle around that cluster: look for persistent straight edges (Canny + temporal persistence across samples) forming an axis-aligned rectangle that encloses the face cluster. If none is found, fall back to the median face box expanded by $3.0\times$ width and $3.2\times$ height, clipped to the frame (tune on real VODs). Snap any edge within 1% of a frame border to that border.
5. Label the region by which third the box center falls in, e.g. `bottom-left`, `middle-left`.

**API**

```python
detect(source_id) -> Facecam            # cached
face_present(source_id, t0, t1) -> float # fraction of 2–3 frames in the ROI with a face
```

**Output** (`run/<vod_id>/facecam.json`):

```json
{ "<source_id>": {
    "box": [0.0, 0.62, 0.27, 1.0],
    "face_center": [0.12, 0.80],
    "region": "bottom-left",
    "confidence": 0.93,
    "samples": 64 } }
```

- Override: style frontmatter `facecam: {box: [x0, y0, x1, y1]}` (all sources) or `facecam: {<source_id>: {box: [...]}}`.
- Budget: ≤ 10 s per source on CPU. `face_present` must be a few ms per call (ROI only, 2–3 frames).
- If detection confidence < 0.6, punch skips that source and logs it; thumb falls back to whole-frame face search.

## Shared module: `.prfpset` parser (`refire/ppro/prfpset.py`)

Premiere effect presets are XML object graphs. Captions apply Nguy's presets at runtime from the files themselves, so editing a preset in Premiere and re-exporting it changes the output with no code change.

- Resolve `ObjectRef` → `ObjectID` to walk `FilterPresetItem → FilterPreset → VideoFilterComponent → Params`.
- Per `FilterPreset`: `FilterMatchName`, `AnchorInPoint`, `AnchorOutPoint`, `Type`, list of params in `Params Index` order.
  - `Type` 0 = **Scale** (confirmed: Nguy's caption MOGRT carries captionIn's keyframes at exactly the preset's times $\times \frac{0.733}{0.45}$, its own length). Anchor-to-In/Out values are untested.
- Per param: `Name` (may be missing for checkboxes/group headers), `ParameterID`, `ParameterControlType`, `IsTimeVarying`.
  - Static value = `StartKeyframe` field 2 (after the time). `CurrentValue` is unreliable; ignore it.
  - Animated value = `Keyframes`: `;`-separated entries; each entry is `time,value,interp,...tangent fields`. Parse `time` (ticks), `value`, `interp` (5 = Bezier). Keep the remaining tangent fields raw for experiments.
  - Point values are `x:y`, normalized 0–1.
  - Colors are 64-bit packed ARGB with 16-bit channels: $v = A\cdot 2^{48} + R\cdot 2^{32} + G\cdot 2^{16} + B$. The 8-bit channel is the high byte of each 16-bit channel.
  - Booleans are `true` / `false`.
- Skip params whose name starts with `_ ` (plugin internals like `_ Applied Version`) and pure group headers (control types 11, 12, 16).
- Times: $1\ \text{s} = 254{,}016{,}000{,}000$ ticks. Keyframe times are absolute; subtract `AnchorInPoint` to get clip-relative time.

**Fixture tests** use Nguy's two presets (`captionIn.prfpset`, `ccStyle.prfpset`); the expected decoded values are listed in `02-caption.md`.

## Premiere scripting notes

- **Effects can't be added with official ExtendScript.** Use the undocumented QE DOM: `app.enableQE()`, `qe.project.getVideoEffectByName(name)`, `qeTrackItem.addVideoEffect(effect)`.
  - QE track-item lists include empty gaps as items. Match QE items to DOM items **by start ticks**, never by index.
  - Resolve effect display names by enumerating QE's effect list once per session. Third-party names (Film Impact) can vary by version.
- **Parameters:** `trackItem.components[i].properties[j]` with `setValue`, `setTimeVarying(true)`, `addKey`, `setValueAtKey`, `setInterpolationTypeAtKey(time, type)`. Types: 0 Linear, 4 Hold, 5 Bezier, 6 Time. Bezier handle shape (speed/influence) **can't** be set.
- **Point parameters** are normalized `[x, y]` in 0–1.
- **Colors:** `setColorValue(a, r, g, b, 1)` with 8-bit channels.
- **MOGRTs:** `sequence.importMGT(path, ticks, vTrackOffset, aTrackOffset)` returns a TrackItem; text is set through `trackItem.getMGTComponent()`. The docs describe AE-made MOGRTs; Premiere-made ones need the spike.
- **Nesting:** there's no nest method, and `createSubsequence` is not nesting. Build the caption sequence separately and overwrite its project item onto V5.
- **Platform outlook:** Adobe staff (Aug 2026) said ExtendScript keeps working past Sept 2026 with no end date, until UXP reaches parity; UXP can't yet list a MOGRT's parameters. Stay on CEP/ExtendScript, keep the .jsx thin.

## Phase 0 — API spikes (do these first)

Run each through `remote.py` against a scratch project. Record results in `docs/spikes.md`. Each has a fallback so a failure doesn't block the feature.

| # | Spike | Pass condition | Fallback |
|---|---|---|---|
| S1 | Insert an adjustment-layer project item on V2 at an exact time and duration | Track item lands frame-exact, any length | Panel asks Nguy to create one adjustment layer named `refire_punch_adj` once; script reuses it |
| S2 | QE-add **Transform** to that track item; set Uniform Scale, Scale 210, Anchor = Position = a corner pin | Render shows V1 zoomed, no black bars | None needed if S2 passes; if QE fails, stop and reassess |
| S3 | `importMGT` Nguy's caption MOGRT (Premiere-made: `authorApp: "ppro"`); set its text through `getMGTComponent()`, param display name **`TextLayer`** (value is JSON with `textEditValue`) | Text changes, style intact | Nguy builds the caption MOGRT once in After Effects instead |
| S4 | QE-add Transform to the MOGRT clip; keyframe Opacity + Position with Bezier interp; turn off "Use Composition's Shutter Angle", Shutter Angle 360 | Matches a clip with `captionIn.prfpset` applied by hand, including motion blur | `caption.easing: exact` (dense keys, see 02) |
| S5 | Build `refire_captions` (clone main sequence and clear it, or create from a preset with no dialog); overwrite it onto main V5; QE-add Drop Shadow, Impact Edge Glow, Impact Camera Shake; set colors and values | Nest renders with the ccStyle look | Panel shows a one-time reminder to drag `ccStyle` onto the nest clip |
| S6 | Insert 300 MOGRT captions with effects in chunks of 50 | Total ≤ 5 min, panel stays responsive | Larger chunks with a progress bar; last resort: render captions to a ProRes 4444 overlay (not editable) |
| S7 | Map Transform's unnamed checkbox params (`ParameterID` 11 → Uniform Scale, 9 → Use Composition's Shutter Angle) to `properties[j]` indices | Setting them toggles the right checkbox | Match by index order from the preset's `Params Index` |

## Config

- CLI flags > style-document frontmatter > defaults (existing reFire precedence).
- New keys are nested maps so they can't collide with existing flat keys: `punch:`, `caption:`, `thumb:`, `facecam:`. Each feature spec lists its keys and defaults.

## Models and licenses

The channel is heading for monetization, so every model must allow commercial use. Not legal advice; re-check model cards when pinning versions.

| Component | Used for | License | Status |
|---|---|---|---|
| Real-ESRGAN (`x4plus`, `x4plus_anime_6B`) | Upscaling | BSD-3-Clause | Use. Confirm weights follow the repo license. |
| BiRefNet | Cutouts (zy0x, art without alpha) | MIT | Use. |
| OpenCV YuNet | Face detection | MIT (opencv_zoo) | Use. Confirm when pinning. |
| OpenCV SFace | Identity check (only if face restoration is ever enabled) | Apache 2.0 | Use. |
| MediaPipe Face Landmarker | Expression scoring (blendshapes) | Apache 2.0 | Use. Confirm the model card. |
| RapidOCR / PP-OCR models | OCR | Apache 2.0 | Use. Confirm when pinning. |
| YAMNet (or PANNs) | Laughter detection | Apache 2.0 (PANNs: MIT) | Use. Confirm when pinning. |
| Playwright + Chromium | Thumbnail rendering | Apache 2.0 / BSD | Use. |
| GFPGAN | Face restoration | Apache 2.0 code, but bundles NVIDIA StyleGAN2 (non-commercial) and DFDNet (CC BY-NC-SA) parts | **On by default** — Nguy's decision (Nov 14). Guarded by the SFace identity check. |
| CodeFormer | — | S-Lab License 1.0 (non-commercial) | Excluded. |
| BRIA RMBG 1.4 / 2.0 | — | Non-commercial weights | Excluded. |
| Film Impact (Edge Glow, Camera Shake) | Caption look | Paid plugin Nguy owns | Must be installed wherever the project renders. |
| Coolvetica, Conneqt | Fonts | Licensed by Nguy | Confirmed OK for the monetized channel. |

## Testing conventions

- Pure logic (scoring, timing, layering, pin math, preset parsing, layout fitting) → pytest unit tests with fixtures. No Premiere needed.
- Premiere integration → `remote.py` smoke scripts against a scratch project, one per feature, asserting the resulting track items and parameter values.
- Every feature writes a human-readable review file (`punch_review.tsv`, `caption_review.tsv`, `thumb/report.md`) so tuning is trial-and-error friendly.

## Build order

1. Phase 0 spikes (S1–S7).
2. Shared modules: `timeline_map`, `facecam`, `prfpset`.
3. `refire punch`: analysis → review file → ops → apply.
4. `refire caption`.
5. `refire thumb`: one template rendered end to end first, then concepts, critic, external art, feedback loop.
