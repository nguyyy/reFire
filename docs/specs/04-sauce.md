# reFire `sauce` — engagement layer spec

Status: planning draft, Nov 27 2026
Scope: Premiere only (AE stays deprecated). 16:9 long-form only.
Goal: on a finished reFire cut, add gameplay punch-zooms, vine booms and meme emotes in the @fye_er house style, close enough that a viewer can't tell it from a hand edit.
References: See iphone/rendered out/zy0x rot
---

## 1. Definition of done

The bar is **≥95% zero-touch**. Nguy reviews the timeline, and per video:

$$
A = \frac{N_{\text{kept}}}{N_{\text{placed}} + N_{\text{added}}} \ge 0.95
$$

- $N_{\text{kept}}$: placements left untouched (moving one by less than 3 frames still counts as kept)
- $N_{\text{placed}}$: everything `sauce` put on the timeline
- $N_{\text{added}}$: events Nguy had to add by hand (missed moments count against us)

Precision beats recall. **When confidence is low, skip.** A missing boom costs nothing, but a wrong one costs a fix.

---

## 2. Where it runs

```
refire make  →  (optional hand re-cut in Premiere)  →  panel: dump V1 → timeline.tsv
            →  refire sauce <run>  →  sauce.json + sauce_log.md
            →  panel: apply sauce.json  →  Nguy review  →  panel: dump edits → corrections
```

- Input is the **real timeline** (`timeline.tsv`, same mechanism as recaption), so hand re-cuts are respected.
- `sauce.json` is a flat list of events. The panel is a dumb applier, and all judgment lives in Python and Claude.
- `sauce_log.md` holds one line per event: time, type, asset, confidence, and the reason (e.g. `03:41.20 boom+💀 — "I definitely didn't just die to a slime" right after dying to a slime`).

### Track layout (one delete = one fix)

| Track | Contents |
|---|---|
| V1 | reFire cut (untouched) |
| V2 | gameplay punch-zoom adjustment layers |
| V3 | facecam punch adjustment layers (existing `punch`) |
| V4 | split-screen comp (only when V2 and V3 would overlap) |
| V5 | emotes |
| V6 | nested caption sequence (always on top, never scaled) |
| A1 | stream audio |
| A2 | music bed |
| A3 | vine booms |

---

## 3. Learning the style from references (`sauce mine`)

There are 6–8 finished projects. Each was made on top of a reFire cut, so they encode the **polish layer only**, which is exactly what we want.

1. **Parse `.prproj`** (gzipped XML). Pull every clip on every track with sequence in/out, media path, and Motion/Opacity/Transform keyframes.
2. **Classify events** by media path and track: the vine boom file becomes `boom`, emote image files become `emote:<name>`, scale keyframes or adjustment layers over gameplay become `zoom`, and facecam-region scales become `punch`.
3. **Transcribe the rendered export** with faster-whisper (word timestamps). Render time equals sequence time, so no mapping back to the VOD is needed.
4. **Build `style_profile.json`** from the data. Every hard number in §4 is a default to be **overwritten by measured values**:
   - event rate per minute as a function of position in the video, $\hat r_{\text{ref}}(t)$, per event type
   - gap distributions (same-type and any-type)
   - boom offset from the end of the triggering word, $\Delta_{\text{boom}}$ (median and IQR)
   - zoom duration and scale distributions
   - emote frequency table, which **becomes the closed library**, ranked by usage
   - co-occurrence rates (boom+punch, boom+emote, zoom+emote)
5. **Build `exemplars.jsonl`**: for each reference event, the transcript window ($\pm 8$ s), event type, asset, and a short "why it's funny" label (Claude-written once, Nguy-spot-checked). These are the few-shot examples for the director.

Deliverable: a human-readable `style_report.md` so Nguy can sanity-check that the miner saw his style correctly before anything else is built.

Built (Nov 27), `refire sauce mine` → `run/sauce/`:
- Corpus is every long-form upload on @fye_er (Nguy: "use videos from my channel"): the renders in `Desktop/iphone/sf90/rendered out/zy0x rot/videos`, plus uploads that aren't on disk, downloaded to `vods/sauce_refs/`. 12 videos.
- Only 4 projects are on disk (abyss, tierlist ft. zajef, strongbox, snezhnaya); they give exact events. The other 8 get booms (vine-boom cross-correlation: 38/42 found, 5 false on the project videos) and emotes (centered library match: 4/15 found, 0 false; precision first) from `refire/sauce/detect.py`. Punches, zooms and other SFX come from the 4 projects only.
- Transcripts come from each video's own audio (faster-whisper). The exemplar labels are one Claude call.

---

## 4. Rules

### 4.1 Density

Target rate is 1.5× the reference curve:

$$
r_{\text{target}}(t) = 1.5\,\hat r_{\text{ref}}(t)
$$

- The cold open is front-loaded at about 2× body density and tapers back to body density by about 60 s.
- Soft cap of 3 events per 10 s outside the cold open. It may be exceeded when the footage earns it, but it should be rare.
- The budget is a **ceiling**, never a quota. Never place a weak event just to hit density.

### 4.2 Overlap

- **Different categories may stack.** Zoom + emote + boom on the same moment is good.
- **Same category may not overlap.** No zoom on zoom, no two simultaneous booms, no two emotes on screen.
- **Gameplay zoom + facecam punch at the same time** becomes a **split-screen**: mask the facecam out, mask the zoomed gameplay out, and place them side by side on V4 (layout TBD, §8).

### 4.3 Gameplay punch-zoom ("jump cut" = hard cut to a zoomed region, not a time cut)

- Hard cut in, hard cut out, no easing (matches facecam punch).
- Duration $0.75 \le d \le 4$ s by default, refined from the reference distribution.
- Scale capped at 200% (1080p source).
- **Tracking allowed.** The crop follows a moving target (enemy, teammate, character, artifact, on-screen event).
- Only zoom on what the streamer is **looking at, talking about, or reacting to**.
- Repeating a target is fine, including twice within 30 s.
- No sub-second flicker. Keep at least about 0.5 s at 100% between consecutive zooms (tunable).
- Wishes: a tracking zoom is allowed on a 5★ reveal (and the "building system" case, §8).

### 4.4 Vine boom

- Lands in the **pause after the line**, never over the streamer's words. Snap target is word end $+\,\Delta_{\text{boom}}$.
- Usually pairs with a facecam punch on the same frame.
- **No ducking** of stream audio.
- Booms are spammable when timing and situation earn it. There's no fixed per-30 s cap, only the density ceiling and the no-simultaneous rule.
- Triggers: sus or awkward moments, out-of-pocket statements, something embarrassing, fails. A comedic boom on a sad moment is allowed, and encouraged if it lands.
- A boom on a clip cut is fine. Consecutive clips ending on booms is fine but shouldn't become a pattern.

### 4.5 Emotes

- Closed library only (from §3). Consistent and predictable.
- Always **centered**, so they may cover the facecam.
- One on screen at a time. A new emote replaces the current one.
- Hold is 2 s, with opacity fading through the full duration (exact curve and any pop-in scale measured from references).
- Repeats are allowed and spamming is fine.
- Allowed during gameplay zooms if it adds to the moment.
- The emote must fit the mood of the line (💀 vs 😭 vs 🗿 etc.). The mapping is learned from the exemplars.

### 4.6 Captions

- Always on top (V6), never scaled by zooms, punches, or split-screen.

### 4.7 Content

- Everything on stream is fair game, including zy0x joking about personal stuff and teammates or friends as the butt of the joke.
- Filler, read-aloud quest text, stat lists and chat reading get nothing, **unless** the moment itself is funny or out of pocket.
- Donation and sub alerts get nothing **unless** the dono or readout is funny.

---

## 5. Detection pipeline

### 5.1 Candidate moments (where something *might* go)

The weight is on the streamer's dialogue. Chat is a weak signal only, because it's delayed and often irrelevant.

$$
s(t) = w_{\text{dlg}}\,s_{\text{dlg}}(t) + w_{\text{aud}}\,s_{\text{aud}}(t) + w_{\text{vis}}\,s_{\text{vis}}(t) + w_{\text{chat}}\,s_{\text{chat}}(t),
\qquad w_{\text{chat}} \ll w_{\text{dlg}}
$$

- $s_{\text{dlg}}$: transcript features (exclamations, swears-as-punchline, self-own phrasing, sudden topic swerve, the "sus" lexicon), plus prosody
- $s_{\text{aud}}$: loudness spike against a rolling median (reuse the `loud` logic), laughter detection
- $s_{\text{vis}}$: OpenCV motion, plus cheap Genshin detectors (gold wish flash via color histogram, death or party-wipe screen, low-HP vignette)
- $s_{\text{chat}}$: message velocity and emoji-mood commonality, used **only** as a small mood hint (💀 spam leans 💀), never for timing

### 5.2 Sauce director (Claude, same director+critic pattern as `make`)

- **Input per candidate:** transcript window, signals, 3–5 frames, and the nearest exemplars from `exemplars.jsonl` (retrieved by similarity).
- **Output per candidate:** `{type, asset, target_description, confidence, reason}` or `skip`.
- **Critic pass:** checks against §4 and the "would Nguy delete this?" test, and down-weights anything generic.
- Events below a confidence threshold $\tau$ are dropped. $\tau$ is tuned on the eval set (§7) to hit $A \ge 0.95$.

### 5.3 Placement solver (deterministic, no LLM)

- Snaps booms to word end plus $\Delta_{\text{boom}}$.
- Enforces the overlap rules, the density ceiling (keeping highest confidence first), and minimum gaps.
- Converts V2/V3 overlaps into split-screen events.

### 5.4 Zoom targeting

The director names the target in words ("the enemy that just one-shot him", "the 5★ character", "the artifact substat line"). A local pipeline then finds it on screen:

1. Open-vocabulary detector (OWLv2 or Grounding DINO) on the zoom's first frame gives a bounding box.
2. SAM 2 or a CSRT tracker follows it for the hold.
3. Smooth the crop path, clamp it to the frame, and cap scale at 200%.
4. If localization confidence is low, or the target leaves the frame, **skip the zoom**.

Built (Nov 27, Nguy's calls): zooms and punches stay **hard cuts**. Premiere's script API can't set a key's
ease, and matching his preset would mean hand-built dense keyframes, so neither uses the ease. Zooms are held
2-7 s like his, start at the phrase start, stop at the next V1 cut, and are capped by a whole-video budget
(1.5x his rate). Targeting is OWLv2 ONNX on the CPU (spike S9) with a static window clamped inside the
frame, plus one Claude look at every zoomed view, because detector scores don't separate right from wrong.
No tracking yet: a target that leaves the window skips the zoom. A zoom overlapping a facecam punch is
skipped until split-screen (P5).

---

## 6. Premiere apply (panel side)

- **Booms:** insert the vine boom asset on A3 at event time.
- **Emotes:** pre-render each library emote once as a 2 s ProRes 4444 alpha clip with the fade (and any pop-in) **baked in**, then place it on V5. This avoids Premiere's keyframe-easing limits entirely.
- **Gameplay zooms:** adjustment layer on V2 with Transform keyframes for scale and position (tracking needs a key every N frames).
  - ⚠️ **Spike first:** confirm ExtendScript `setTimeVarying` / `addKey` / `setValueAtKey` on the Transform effect is reliable at this key density.
  - **Fallback:** render each tracking zoom as a short pre-cropped clip via ffmpeg and place it on V2. That's less editable, but still one delete to remove.
- **Split-screen:** if Premiere masks prove painful from script, pre-render the comp with ffmpeg and place it on V4.
- **Corrections dump:** after review, the panel dumps V2–V5 and A3 so the diff against `sauce.json` can be computed (§7).

---

## 7. Evaluation and learning loop

Reference projects can't serve as held-out tests (they were cut from a different cut), so evaluation is done on **fresh runs**.

1. Run `sauce` on a new VOD, then Nguy finalizes it as he normally would.
2. Panel dumps his final tracks. Diff them against `sauce.json`, classifying each event as kept, moved, deleted, or added, and compute $A$.
3. **Style distance:** compare the run's measured stats (rates, gaps, $\Delta_{\text{boom}}$, emote mix, zoom durations) against `style_profile.json`. Large drift gets flagged even if $A$ looks fine.
4. **Learning:** deletions and additions are written to `run/sauce_corrections.jsonl` and fed back as exemplars, with negatives ("Nguy deleted this") included. This is the same pattern as `corrections_<game>.json`.
5. **Ship bar:** $A \ge 0.95$ on 3 consecutive fresh videos.

Built (Nov 27), `refire sauce eval <run_dir> [--project x.prproj]`: reads the saved project read-only (the
same data as a panel dump, no panel needed) and matches each placement by its `refire:sauce:<id>` clip name.
Writes `sauce_eval.json`, appends to `run/sauce/eval.jsonl` (ship bar checked there) and to
`run/sauce_corrections.jsonl`; the next `refire sauce` run shows the director his additions as exemplars and
his deletions as "Nguy deleted these".

---

## 8. Open questions

1. **"Building system"** (wish/zoom rule): what does this refer to? The Serenitea Pot, a character build or showcase screen, or something else? I meant building suspense
2. **Split-screen layout:** is it 50/50, gameplay 2/3 with facecam 1/3, and which side does each go on? Should it be hard cut in and out like punches? I think 1/3 2/3, see references i did this one
3. **Vine boom asset:** is it a single file and level, or a few variants to avoid ear fatigue? (Same sound regardless, just different takes or pitch.) single file, should tone down audio a bit and fade it in and out, i usually do -15, -10, -10, -15db.
4. **Emote assets:** do the media paths in the reference `.prproj` files still resolve? If not, where does the library live? look in desktop, editing pack
5. **Cold-open taper:** is 60 s right, or should it come from the references? 60s is good, see references
6. **Naming:** `refire sauce` OK? sure, should be integrated into the pipeline, so sause has ample context.

---

## 9. Build order

| Phase | What | Why first |
|---|---|---|
| P0 | Spikes: Premiere Transform keyframes from script; OWLv2 + SAM 2 speed on GPU | Decides the zoom architecture |
| P1 | `sauce mine` → `style_profile.json`, `exemplars.jsonl`, `style_report.md` | Everything else tunes against it |
| P2 | Booms + emotes end-to-end (director → solver → panel) | Highest value, no vision needed |
| P3 | Eval harness + corrections loop | Lets every later phase be measured |
| P4 | Gameplay zooms with tracking | Hardest part |
| P5 | Split-screen for zoom + punch overlap | Depends on P4 |
| P6 | Tune $\tau$ and weights until $A \ge 0.95$ ×3 | Ship |
