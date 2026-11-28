# reFire feature specs · 03 · `refire thumb`

> Depends on `00-shared.md` (facecam detector, models and licenses, testing conventions).

## Goal

From a reFire run folder alone, produce **three thumbnail candidates** for YouTube Test & Compare, each a different reason to click, in Nguy's house style. Built like a human editor would (real frames, cutouts, layered text and effects), rendered from HTML/CSS. Zero touch; Claude can tweak a candidate afterwards by editing its spec.

## Non-goals

- No diffusion, no generated pixels, no outpainting. Every pixel comes from the video, official game art, Nguy's emotes, or shapes and text drawn in code.
- No photos of real people from the web. Streamer faces only come from the VOD.
- No Figma, Photoshop or Blender.

## Why three *different* bets

YouTube's Test & Compare shows up to three thumbnails evenly and picks a winner by **watch time share**, not CTR. Three variations of one idea return noise; three different promises return a lesson. And a thumbnail that wins clicks but sells a moment the video doesn't deliver loses on watch time, so every concept must be paid off by the cut.

## Inputs

From `run/<vod_id>/`:
- Transcript (word timestamps), `beats.json` (role, payoff, setup, viewer_question, turn, energy, texture), the moment map (chat z-score, loudness, motion), `timeline.tsv`, `punch.json` (pre-scored reaction moments — the best face candidates), stream title, game glossary, `sources[]` (to re-fetch source-quality frames).
- `facecam.json` (shared detector).
- Nguy's emotes folder (config path).
- Reference thumbnails in `refire/thumb/refs/` (Nguy's 8 past thumbnails; the two "MORE TIER LISTS" ones are near-identical, keep one). These ran at roughly 8–11% CTR, so the style is proven.

## Outputs

```
run/<vod_id>/thumb/
  brief.json          # what the video promises, entities, reveal, top moments
  concepts.json       # the three concepts
  A/ B/ C/
    thumb.png         # 1280×720
    spec.json         # every layout value; editing this + re-render = tweak
    index.html        # rendered page (self-contained)
    assets/           # cutouts, crops, art used
  report.md           # ranking, scores, titles, upload order
```

## House style (from the refs + Nguy)

**Text**
- Font **Conneqt Regular**, all caps, usually **two lines**, bottom-left (bottom-right allowed when the face is left).
- Lines sized so both are roughly the same width; text block 52–62% of canvas width, 40 px margins.
- **Fill:** vertical three-stop metallic gradient: color → same color but whiter → color. One **accent phrase** uses the accent color; the rest uses metallic white ($\#D6D6D6 \to \#FFFFFF \to \#D6D6D6$).
- **Stroke:** heavy black, about $0.11\times$ font size, round joins, painted under the fill.
- **Black backing rectangle per line:** fills the gaps between letter strokes so the outline reads as one shape. It's shorter than the letters (default $0.62\times$ cap height, vertically centered), so the tops and bottoms of the letters rise above and below it. Spans the line's full stroked width, square corners.
- **Block shadow:** hard drop shadow (black 55%, offset about $(6, 8)$ px) plus a soft dark glow behind the block.
- Hooks end in **"?"** or **"..."**. Optional small lowercase annotation, e.g. "(birthday)" above line 2.
- Accent palette, sampled from the refs' accent text. Gradient = base → light → base.

  | Name | Base | Light | Seen in |
  |---|---|---|---|
  | blue | $\#2C66F3$ | $\#64A1FF$ | MORE |
  | sky | $\#0884F2$ | $\#58D3F3$ | SNEZHNAYA |
  | green | $\#26CE27$ | $\#52F753$ | EASIEST, RIGHT? |
  | mint | $\#26C075$ | $\#53DA8E$ | TIER LIST? |
  | gold | $\#D2A51B$ | $\#F8CE4F$ | ARTIFACTS |
  | cyan | $\#1FFFF8$ | $\#65FFFB$ | ZAJEF ACCOUNT |

  Pick by theme (gold: artifacts and rewards; sky: Cryo / Snezhnaya; green or mint: "easy", "right", wins; cyan: people and reviews; blue: series and lists). The accent hue must differ from the background's dominant hue by ≥ 60°.

**zy0x**
- Large chest-up cutout on the right third; face height 30–45% of the canvas; may run off the bottom edge.
- Big readable expression: shock, deadpan, skeptical, hype, laughing. Hand gestures are a plus.
- Rim glow: a tight halo plus a wide soft one, white by default (refs also use blue and yellow).

**Subject (left side)**
- The thing the video is about: a game UI element from the stream (tier list, artifact card, character list, Wordle board) or official splash art.
- Slight rotation (−4° to +4°), drop shadow, optional gentle perspective.

**Background:** a blurred, darkened gameplay frame (blur about 12 px, brightness 0.55–0.7, saturation 1.15, scaled 1.08), optional speed lines or light rays from behind zy0x, vignette.

**Signature move — the pixelated reveal:** 6 of 7 unique refs hide something (a tier placement, stat values, a character, a sign) behind a mosaic, often with a pixel-art "???" speech bubble. When the video has a reveal, one candidate always uses this.

**Badge-safe zone:** YouTube's duration badge covers the bottom-right. Nothing essential (hook text, face core) in the bottom-right 16% width × 12% height.

## Layouts

| Layout | Use when | Composition |
|---|---|---|
| `reaction` | The strongest asset is a face moment | Face large right, subject small left, text bottom-left |
| `subject` | The video is about a thing | Subject panel large left/center, face medium right |
| `reveal` | The video has a reveal | Subject with pixelated target + "???" bubble, shocked/skeptical face, "?" hook |
| `duo` | A guest streamer is in the video | Guest cutout left, zy0x right, subject behind |
| `inset` | A two-beat story ("day two...") | Big deadpan zy0x + small shocked inset zy0x |

The three candidates must use three different concept types. `reveal` is mandatory when `brief.reveal` is not null.

## Pipeline

### 1. Brief (Claude, director backend)

Reads beats, transcript summary, the top moments from the moment map and `punch.json`, stream title, glossary. Writes `brief.json`:
- `promise`: what the video delivers, one sentence.
- `entities`: characters, items, bosses, guests (canonical names via glossary).
- `reveal`: `{what, t, beat_id}` or `null`.
- `moments`: top 8 with timeline time, kind (reaction / subject / reveal), why.
- `hook_ideas`: 6–10 candidate hooks.

### 2. Concepts (Claude)

Writes `concepts.json` with exactly three concepts:
- `type`, `layout`
- `hook`: `{lines, accent_phrase, accent_color, annotation}`. 2–4 words ideal, 5 max, all caps, no swearing, ends in "?" or "..." when it's a question or teaser.
- `face`: `{windows: [[t0, t1]], expression}`
- `subject`: `{source: "video" | "art", t | query, crop_hint}`
- `pixelate`: targets to hide
- `extras`: bubble, emote, speed lines, inset
- `title`: one suggested title paired with the hook (≤ 60 characters)
- `payoff_beats`: beat ids that deliver what this thumbnail promises. **A concept without a payoff beat is rejected.**

### 3. Frames

**Face frames.** For each face window ± 1.5 s:
- Re-fetch the segment at source quality (reuse `loud`'s window fetch); crop to the facecam box.
- Score every frame:

$$q = 0.35\,\hat S + 0.35\,\hat E + 0.15\,\hat A + 0.15\,\hat O$$

  - $\hat S$: sharpness, variance of the Laplacian on the face crop, normalized within the window.
  - $\hat E$: expression match from MediaPipe Face Landmarker blendshapes. Shock = jawOpen + eyeWide + browInnerUp; hype = mouthSmile + jawOpen; deadpan = low activation + mouthPress; skeptical = brow asymmetry + mouthPress.
  - $\hat A$: face size. $\hat O$: eyes open, not mid-blink.
  - Drop frames in the top 20% of face-ROI frame difference (motion blur).
- Keep the top 5; the vision model picks one for the concept.

**Subject crops.** Frames at the subject time ± 2 s. Find the UI panel with rectangle contour detection anchored on OCR text near the expected keyword; take the sharpest frame's crop.

### 4. External art (tool use)

For `subject.source = "art"`, inset characters, and splash-art backdrops:
1. Resolve canonical names from `brief.entities` and OCR'd names.
2. Search sources in priority order:
   1. Game-data asset sites/APIs that serve official art with transparency (evaluate ambr.top, enka.network's asset CDN, Honey Hunter World).
   2. The Genshin Impact fandom wiki via the MediaWiki API (wish and card art files; exact file naming to verify).
   3. General image search, accepted **only** if the vision model confirms it's official art of that character, not fan art.
3. Validate: ≥ 1000 px long edge, name match, no watermark.
4. No alpha → BiRefNet cutout.
5. Cache in `~/.cache/refire/art/` with a sidecar JSON (URL, fetched date). This is an automatic cache, safe to wipe, not a curated library.
6. Etiquette: a descriptive User-Agent, ≤ 1 request/s, respect robots.txt and site terms, retry with backoff.
7. The fetcher refuses anything that isn't game art (no photos of real people).

### 5. Cutout and enhance

- **zy0x cutout:** inset the facecam crop by 1–2% (removes border remnants) → BiRefNet alpha → 1 px feather.
- **Upscale** when an element's source pixel height is below its display height: Real-ESRGAN `x4plus` for the face, `x4plus_anime_6B` for game art and UI; then Lanczos down to display size.
- **Face restoration:** GFPGAN **on** by default (`thumb.face_restore: gfpgan`, Nguy's call). Blend the restored face back at weight 0.5–0.7 (not 1.0) to keep skin texture. Revert any face whose SFace cosine similarity to the original is < 0.55, and log it. Order: upscale → restore → identity check.
- **Grade:** contrast +8%, saturation +12%, light unsharp, per layer.
- **Pixelate:** mosaic with block size $\approx$ target width / 10, nearest-neighbor.

### 6. Compose and render

- Templates: `refire/thumb/templates/<layout>.html` + shared `house.css`, driven entirely by `spec.json` (positions, sizes, rotations, colors, text, asset paths).
- **Text** is drawn as SVG `<text>`: `paint-order="stroke"`, `stroke-linejoin="round"`, gradient fill via `<linearGradient>` (vertical, three stops), backing `<rect>` per line sized from measured text bounds and cap height (font metrics read in Python with fontTools and passed in). A pre-pass measures each line and fits font sizes to equal width.
- Fonts loaded with `@font-face` from local files.
- Render with Playwright + headless Chromium at 1280×720 → PNG. `--hires` renders 2560×1440 and downsamples for crisper text.
- If the PNG exceeds YouTube's current custom-thumbnail size limit (check; historically 2 MB), re-encode as JPEG quality 92.
- The same `spec.json` must always produce the same PNG.

### 7. Critic and revisions (max 2 rounds)

**Deterministic checks** (fail = must fix):

| Check | Threshold |
|---|---|
| Hook readable when small | OCR at 246×138 and 168×94: character accuracy ≥ 0.80 |
| Face size | face box height ≥ 22% of canvas |
| Face clear | nothing over the eyes or mouth (landmark hull) |
| Badge zone | no hook text or face core in the bottom-right 16% × 12% |
| Accent vs background | hue distance ≥ 60° |
| Clutter | ≤ 4 primary elements |
| Distinctness | three different concept types; pairwise pHash Hamming distance ≥ 16 |

**Vision critic** (Claude with images; refs attached). Scores 1–5 on:
- Instant read: what is this video about, in one second?
- Emotion strength.
- Curiosity gap.
- Promise kept: must cite the payoff beats from `beats.json`.
- House-style match vs refs (**weighted highest**: the style already performs; don't drift toward generic advice).
- Feed standout: the candidate placed in a mock feed grid beside 8 recent thumbnails from GlorpDaily, Zyoxtopia and Nguy's own channel.

Output: a 0–100 score plus up to 3 concrete edits expressed as `spec.json` changes. Apply, re-render, re-score; stop after 2 rounds or when the score stops improving.

**Mock feed data:** YouTube Data API v3. Resolve handles once with `channels.list?forHandle=...` (1 quota unit) and cache the channel IDs; then uploads playlist → `playlistItems.list` (1 unit each; never `search.list`, which costs 100). Cache thumbnails daily. Channels: **@GlorpDaily**, **@Zyoxtopia** (competitors), **@fye_er** (own).

### 8. Report

`report.md`:
- Ranking with score breakdowns.
- Per candidate: the bet it makes, its title, the beats that pay it off, any warnings (fallback art source, heavily upscaled face).
- **Upload order:** slot 1 = top pick, because an inconclusive test keeps the first uploaded thumbnail.

### 9. Feedback loop

- YouTube's help pages for Test & Compare don't mention an API or export for results.
- `refire thumb sync`: for videos with a candidate set on file and a test older than 14 days, fetch the live thumbnail with `videos.list` (1 unit) and perceptual-hash match it to A/B/C.
  - The live thumbnail alone is ambiguous (an inconclusive test also leaves slot 1 live), so Nguy records the outcome with one command: `refire thumb result <video_id> winner|same|inconclusive`.
- CTR: from the YouTube Analytics API if it exposes impressions CTR for the channel (verify); otherwise import the Studio CSV exports Nguy already pulls.
  - Verified (Nov 27): the Analytics API has `videoThumbnailImpressions` and `videoThumbnailImpressionsClickRate`, so there's no CSV import. It needs the channel owner's OAuth, and the same token covers the Data API: one Google Cloud OAuth client ("Desktop app", YouTube Data API v3 + YouTube Analytics API enabled) saved at `~/.refire/youtube_client.json`; the first sync opens the browser once.
  - Built as: `sync` lists the channel's 50 newest uploads and pHash-matches each live thumbnail against every candidate set under `run/` (distance ≤ 12), so no command is needed to tell reFire which video used which set. `result ... winner` re-reads the live thumbnail to name the winner.
- Store in `~/.refire/thumbs.sqlite`: video, each candidate's features (type, layout, hook, accent, expression, pixelate, critic scores), upload order, result, CTR.
- Use: the last 10 results go into the concept and critic prompts as context. After ≥ 10 decided tests, compute per-feature win rates and bias concept selection (e.g. if `reveal` keeps winning, always include it).

## Commands

| Command | Does |
|---|---|
| `refire thumb <run_dir>` | Full run. Also runs at the end of `refire make` unless `--no-thumb` (detached, log in `thumb/thumb.log`: the panels wait for make to exit before loading the manifest). |
| `refire thumb render <run_dir> <A\|B\|C>` | Re-render one candidate from its `spec.json` |
| `refire thumb revise <run_dir> <A\|B\|C> "<instruction>"` | Claude edits `spec.json`, re-renders, re-runs the deterministic checks |
| `refire thumb sync` | Match live thumbnails to candidates |
| `refire thumb result <video_id> <winner\|same\|inconclusive>` | Record a test outcome |

## Config (`thumb:` frontmatter map)

| Key | Default |
|---|---|
| `fonts.display` | path to Conneqt Regular |
| `accent_palette` | blue / sky / green / mint / gold / cyan (sampled table above) |
| `text.stroke_ratio` | 0.11 |
| `text.rect_ratio` | 0.62 |
| `text.max_words` | 5 |
| `face_height` | [0.30, 0.45] |
| `face_restore` | `gfpgan` (weight 0.6, identity threshold 0.55) |
| `upscale` | `auto` |
| `art_sources` | ordered list (above) |
| `feed_channels` | `{competitors: ["@GlorpDaily", "@Zyoxtopia"], own: "@fye_er"}` |
| `revisions` | 2 |
| `emotes_dir` | Nguy's emotes folder |
| `refs_dir` | `refire/thumb/refs` |
| `hires` | `false` |

## Budget

≤ 5 minutes per run on the GPU, excluding LLM calls; art downloads ≤ 30 s.

## Acceptance criteria

- Text renderer side-by-side with the refs: font, metallic gradient, continuous stroke via backing rectangles, equal-width lines.
- Golden runs on 3 past videos Nguy thumbnailed by hand: all candidates pass the deterministic checks, are three different concept types, and Nguy rates at least one per video "would upload".
- Determinism: same `spec.json` → byte-identical PNG.
- The art fetcher refuses non-game images; no real-person photos are ever downloaded.
- `report.md` always lists upload order and one title per candidate.
- `revise` turns a plain-language tweak into a spec edit and a re-render in under 30 s (excluding the LLM call).

## Open items for Nguy

- ~~Emotes folder path~~: `refire/assets/emotes/`.
