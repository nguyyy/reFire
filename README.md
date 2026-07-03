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
- **overlays and sound effects**: maps sound effects and emotes (using assets/ or betterttv search query) to keywords in excitement intervals.
- **music mixing**: loops background music and ducks it during speech.

### 4. compilation rendering
- **ffmpeg local rendering**: concats clips and mixes music into a finished mp4 draft (`run/<vod_id>/rough.mp4`).
- **after effects export**: writes a build manifest (`run/<vod_id>/ae/manifest.json`). the script `refire/ae/reFire.jsx` imports the manifest, builds the timeline with keyframed zoom steps, sections, caption styles, sfx tracks, and overlays.

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
- `--no-motion-zoom`: disables dynamic reframing zooms.
- `--progress-file <path>`: logs execution progress to a json file.

---

### utility commands
run individual pipeline steps:

#### 1. highlight detection
scans video and chat logs to output `segments.json`.
```bash
refire detect stream.mp4 chat.json --run-dir run --top-n 30
```

#### 2. assemble clips locally
renders selected segment clips into a single video compilation.
```bash
refire edit stream.mp4 --run-dir run --music music.mp3
```

#### 3. after effects manifest export
compiles segment data into the manifest file.
```bash
refire ae stream.mp4 --run-dir run --words-per-line 3
```

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

---

## after effects script execution

1. open after effects.
2. run **file > scripts > run script file...** and pick `refire/ae/reFire.jsx`.
3. select `manifest.json` from the vod output folder.
4. click **build** to generate the timeline compositions.
5. edit styles in the **caption style** comp layer and click **update** in the panel to apply changes across all segments.

---

## testing

run the test suite:
```bash
pytest
```

---

## architectural documentation
detailed notes on the editor's mechanics are located in the [ai editor workflow memory](file:///c:/deving/reFire/docs/ai-editor-workflow-memory.md) file.
