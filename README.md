# refire: automated narrative-driven video editor

refire is an automated video editor that analyzes twitch vods, structures a story outline, selects relevant clips, refines cuts, reviews the draft script, and compiles the final video or after effects project.

---

## key features

### 1. narrative storyboarding (claude director)
- **story shape selection**: an llm reads the stream transcript map to identify a central story shape (e.g., confidence collapses into chaos) and viewer promise.
- **beat planning**: the director structures the video into sequential beats with roles (hook, setup, escalation, reversal, climax, payoff, button).
- **closed-loop critic pass**: an iterative review loop (up to --review-rounds passes, default 2) reads the transcript of the drafted clips. the critic flags issues with context, pacing, redundancy, or endings, and revises the outline.
- **headless cli execution**: the narrative pass can run via the claude cli (`--director-backend cli`) using a claude subscription, the anthropic api (`--director-backend api`), or locally (`--local-director`) on ollama.

### 2. clip boundaries and cuts
- **sentence alignment**: clips are snapped to sentence boundaries to prevent mid-word cuts.
- **silence compression**: internal silences inside clips are removed using `compress_silence` with a configurable padding buffer.
- **payoff preservation**: cuts are anchored to explicit setups, payoffs, and reaction times defined by the storyboard.

### 3. formatting and styling
- **motion-triggered punch zooms**: opencv detects frame-to-frame motion bursts to apply camera reframing (webcam corner anchor, 100% to 200% zoom) that holds through speech and releases on pauses.
- **karaoke subtitles**: builds ass subtitle tracks with word-by-word highlighting. subtitle text defaults to lowercase and switches to uppercase on excitement (detected via rms audio amplitude analysis or keywords like lol, pog, wtf).
- **caption proper-noun repair**: the transcriber never knew what game was on screen or who the streamer's friends are, so it ships `"dude, zhegef"` (the friend Zajef) and `"mawika"` (mavuika). one claude pass re-reads the built captions with the game glossary, the stream title and the vod's own chat — where viewers spell names correctly at the second they're said — and rewrites only the mistakes. line for line: the line count and every timestamp are untouched, and per-word shouting survives, so after effects and the premiere srt both pick it up for free. every accepted fix is logged to `caption_fixes.json` and learned into `run/corrections_<game>.json`, which is applied for free (no llm) on later runs.
- **overlays and sound effects**: maps sound effects and emotes (using assets/ or betterttv search query) to keywords in excitement intervals.
- **music mixing**: loops background music and ducks it during speech.

### 4. compilation rendering
- **ffmpeg local rendering**: concats clips and mixes music into a finished mp4 draft (`run/<vod_id>/rough.mp4`).
- **after effects export**: writes a build manifest (`run/<vod_id>/ae/manifest.json`). the reFire cep panel (`refire/ae/`) imports the manifest, builds the timeline with keyframed zoom steps, sections, caption styles, sfx tracks, and overlays — and can drive the whole pipeline above without leaving after effects.

---

## prerequisites

- **python 3.10+**
- **ffmpeg** on the system path.
- **nvidia gpu and cuda** for local transcription acceleration.
- **ollama** running locally with:
  - embedding model: `ollama pull nomic-embed-text`
  - chat model: e.g., `ollama pull llama3.1:8b` or `qwen2.5:14b` (refire falls back to the largest installed chat model).
- **twitchdownloadercli.exe** in the root directory.
- (optional) **claude cli** (`npm install -g @anthropic-ai/claude-code`) for subscription completions.
- (optional) **.env file** in the root directory:
  ```env
  ANTHROPIC_API_KEY=key_here
  DEEPGRAM_API_KEY=key_here
  ```

---

## installation

install the package in editable mode:
```bash
# default install
pip install -e .

# developer install with dev dependencies
pip install -e ".[dev]"
```

---

## usage

### the main pipeline: `refire make`
downloads the vod, processes chat logs, transcribes, storyboard and casts the beats, then generates the after effects manifest.

```bash
refire make 1762930614 \
  --brief "the funniest hu tao gacha pulls and rage moments" \
  --duration 20m \
  --game "genshin impact" \
  --render
```

#### arguments and options:
- `vod_id`: twitch vod number (downloaded and cached in `vods/`).
- `--brief`: brief defining the vibe and topic.
- `--duration`: target compilation length (e.g., `20m`, `20:00`, or `1200`).
- `--game`: name of the game (used to build a proper-noun spelling dictionary).
- `--render`: immediately renders the video draft locally to `run/<vod_id>/rough.mp4`.
- `--director-backend <cli|api>`: use `cli` for subscription completions via the claude cli, or `api` for pay-as-you-go anthropic key usage.
- `--local-director`: runs the narrative planner on local ollama instead of claude.
- `--review-rounds <n>`: maximum critic iterations (default `2`).
- `--transcriber <local|deepgram>`: choose between local `faster-whisper` and cloud `deepgram`.
- `--whisper-model <name>`: faster-whisper model (default `large-v3-turbo`; `large-v3` is slower for a marginal accuracy gain).
- `--batch-size <n>`: transcription batch size (default `8`); raise it if you have the vram.
- `--compute-type <type>`: `float16` (default) or `int8_float16` to free vram for a bigger batch.
- `--no-motion-zoom`: disables dynamic reframing zooms.
- `--no-emotes`: no emote/gif overlay punch-ins (drops their sfx too).
- `--no-sfx`: emote punch-ins stay, but land silently.
- `--no-cards`: no section title cards in the AE Master — the cut runs clip to clip.
- `--no-deadspace`: keep each clip's internal silence instead of jump-cutting it out (on by default; `--silence-pad` sets the breath left around each phrase).
- `--no-caption-fix`: skip the claude pass that repairs mistranscribed proper nouns in the captions.
- `--progress-file <path>`: logs execution progress to a json file.

#### re-running the caption fix on an existing run: `refire fixcaps`
`make` already does this, so you only need it to re-run against a manifest you built earlier (or after editing `run/corrections_<game>.json` by hand). it rewrites the manifest's caption text and regenerates `captions.srt`.

```bash
refire fixcaps run/<vod_id>/<run>/ae/manifest.json \
  --game "genshin impact" --terms genshin.txt --chat vods/<vod_id>.chat.json
```

every progress line carries elapsed minutes, so a run tells you which stage it spent them in.

---

## project artifacts

output files are saved under `run/<vod_id>/`:
- `clips/` - temporary sub-clips and subtitle files.
- `trace/` - prompt and response text records from claude passes.
- `audio.wav` - extracted audio file.
- `transcript.json` - transcribed word list cache.
- `transcript.glossary.json` / `transcript.source.json` - cache validation files.
- `chunks.json` - video segment transcription chunks and embeddings.
- `outline.json` - story planning outlines and beat metadata.
- `cut_plan.md` - markdown file of storyboard logs and dialogue segments.
- `rough.mp4` - rendered video compilation.
- `ae/manifest.json` - manifest structure read by the after effects script.
- `ae/captions.srt` - the same captions in master-timeline time, written on demand
  by `refire srt` for the premiere panel (and anything else that eats srt).

---

## after effects panel

a cep panel (local html/js in an embedded chromium view, talking to extendscript)
lives in `refire/ae/`. install it once:

```powershell
powershell -ExecutionPolicy Bypass -File refire\ae\install.ps1
```

that junctions the folder into `%APPDATA%\Adobe\CEP\extensions\` and enables
`PlayerDebugMode` so after effects will load the unsigned extension. restart ae,
then **window > extensions > refire**.

1. **01 source / 02 direction** — vod number, an optional brief, a target duration.
   press **make**: the panel spawns `python -m refire make`, streams its progress
   into the panel, and auto-loads the manifest it writes. **03 tuning** exposes the
   rest of the cli (director backend, scout, transcriber, slack, zoom, rough cut).
2. **04 build** — generates the timeline compositions and queues the master.
3. edit the **caption style**, **zoom style** or **overlay style** layers, or the
   **section card** comp, and click **update** to apply across all segments.

already have a manifest? skip make and hit **load** in 04.

panel misbehaving? with it open, browse to `http://localhost:8088` for devtools.

---

## premiere pro panel

a fork of the ae panel lives in `refire/ppro/`, installed the same way (its own
folder, so both can be installed at once):

```powershell
powershell -ExecutionPolicy Bypass -File refire\ppro\install.ps1
```

restart premiere, then **window > extensions > refire**. devtools on
`http://localhost:8089`.

**scope: clip selection + subtitling, nothing else.** premiere's extendscript
cannot create text layers, cannot set keyframe easing, and has no documented
transition api — so the zoom punches, emote overlays, sfx, section cards and
music bed are the after effects half of reFire and are not built here. what you
get is the tedious part: every chosen clip on **v1** in story order, dead air
already cut (one trackitem per kept span — real cuts, no time remap), plus a
**captions.srt** in the `reFire` bin to drag onto the timeline.

**build** regenerates `captions.srt` (`python -m refire srt <manifest>`) at the
current caption offset and then lays out a fresh `reFire cut N` sequence. the
offset nudge is instant because it only rewrites the srt — no `make` re-run.

### driving it from a terminal

that devtools port is a Chrome DevTools Protocol endpoint, so the panel doubles as a
remote for Premiere. with the panel open:

```powershell
python refire\ppro\remote.py 'reFirePpro.probe()'       # extendscript, inside premiere
python refire\ppro\remote.py --panel '$("#build").click()'   # js, inside the panel
python refire\ppro\remote.py --ae 'app.project.file.fsName'  # the ae panel, port 8088
```

each host call re-evals `reFirePpro.jsx` first, so jsx edits land without restarting
premiere. useful for scripted checks and for letting an agent exercise the panel.

the srt is standalone and useful on its own (youtube, a burn-in):

```powershell
python -m refire srt run\<vod>\<run>\ae\manifest.json --offset -0.15
```

---

## testing

run the test suite:
```bash
pytest
```

---

## architectural documentation
detailed notes on the editor's mechanics are located in the [ai editor workflow memory](file:///c:/deving/reFire/docs/ai-editor-workflow-memory.md) file.
