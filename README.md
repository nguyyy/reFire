# refire: automated narrative-driven video editor

refire is an automated video editor that analyzes twitch vods, structures a story outline, selects relevant clips, refines cuts, reviews the draft script, and compiles the final video, premiere pro (.prproj), or after effects (.aep) project.

---

## key features

### 1. narrative storyboarding (claude director)
- **story shape selection**: an llm reads the stream transcript map to identify a central story shape (e.g., confidence collapses into chaos) and viewer promise.
- **beat planning**: the director structures the video into sequential beats with roles (hook, setup, escalation, reversal, climax, payoff, button).
- **closed-loop critic pass**: an iterative review loop (up to --review-rounds passes, default 2) reads the transcript of the drafted clips. the critic flags issues with context, pacing, redundancy, or endings, and revises the outline.
- **headless cli execution**: the narrative pass can run via the claude cli (`--director-backend cli`) using a claude subscription, the anthropic api (`--director-backend api`), or locally (`--local-director`) on ollama.
- **one session per run**: the director opens a named claude session and every review round resumes it, so the ~230kb stream map is sent once instead of once per call. the prompt is also ordered so everything the review rounds re-measure (the span budget, the per-role seconds) sits *after* the map rather than in front of it -- one changed digit there used to move the cache prefix and cost the whole map on every round. measured on real traces: two consecutive review requests shared 1,695 bytes of 250,000; they now share 100%.

### 2. clip boundaries and cuts
- **sentence alignment**: clips are snapped to sentence boundaries to prevent mid-word cuts. that snap is also the real floor on shot length, so `--snap` offers two finer modes for dense styles: `phrase` lands on natural speech pauses (still never mid-word), and `transient` lands on the audio peak and cuts a few hundred ms *before* it resolves. the last cut of the video always keeps its sentence snap.
- **play order**: the cut runs chronologically by default. `--order whiplash` reorders the beats to maximise the tonal jolt between neighbours instead — chaos into calm — for styles whose comedy is the juxtaposition rather than the story.
- **cold open**: one flash-forward teaser by default, or `--stack n` for a montage stack of `n` unexplained 1–2.5s moments ordered so the best lands last. every moment is checked against footage a later beat actually delivers, so the open can't promise something the cut never reaches.
- **silence compression**: internal silences inside clips are removed using `compress_silence` with a configurable padding buffer. "silence" is checked against a cached silero speech pass (`run/<vod>/speech.json`), not just the transcript: whisper returns no words for some game dialogue under the streamer's mic, and cutting those wordless stretches as dead air used to delete one side of a quest exchange. measured over 7 real cuts, 239s of the 1762s removed was speech.
- **payoff preservation**: cuts are anchored to explicit setups, payoffs, and reaction times defined by the storyboard.
- **build-up preservation**: a beat whose comedy is the repetition (a wordle spiral, a run of failed attempts) scores highest at its punchline, so the cheap edit opens on the last guess and reads as a missing scene. the pacing audit measures the largest jump between a beat's kept segments that lands *before* its payoff and flags it (`skipped_build`), so the critic sees the skipped build-up as a measured number instead of being asked to notice it.
- **exchange continuity**: the transcript is one flat stream -- the streamer's mic and the game's own dialogue, unlabelled -- so a beat used to get cut the instant a character stopped talking, leaving the player's reply outside the span and the moment playing as a non-sequitur. three things now close that loop: segments closer than 2.5s merge (a reply-sized hole is never a jump cut), the realized script handed to the critic marks every jump cut with how long it was and *what was said in it* (it previously joined the kept text into one paragraph, so a severed exchange was literally invisible), and the pacing audit flags a 1-20s hole that contained speech as `severed_exchange` -- the small-gap counterpart to `skipped_build`, and the one continuity check a `--style` cannot switch off. the director is told the rule that matters: cut *between* exchanges, never inside one, and decide whether quest dialogue is load-bearing (keep it whole) or slop (drop it whole) -- never half.
- **one beat, one moment**: segments more than 5 minutes apart were the director stapling two unrelated moments into one beat (one `hook` spliced 349s to 2130s). the stray group is now dropped, loudly. the threshold sits above the build-up machinery on purpose, so a wordle spiral sampled across four minutes still survives.

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

- **python 3.10+** (`.python-version` pins 3.12, which is what uv resolves to)
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

on a fresh machine, [uv](https://docs.astral.sh/uv/) is the only thing to install
(`winget install astral-sh.uv`). everything else — the right python, the venv, and
the exact pinned versions from `uv.lock` — comes from one command:

```bash
uv sync --extra dev   # drop --extra dev for a runtime-only install
uv run pytest         # 311 tests, no activation needed
uv run refire make ...
```

`uv.lock` + `.python-version` are committed, so every machine resolves to the same
versions. after changing deps in `pyproject.toml`, re-run `uv sync` and commit the
updated `uv.lock`.

<details><summary>plain pip instead</summary>

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -e ".[dev]"   # unpinned — resolves fresh versions, ignores uv.lock
```
</details>

---

## usage

### the main pipeline: `refire make`
downloads the vod, processes chat logs, transcribes, storyboard and casts the beats, then generates the after effects manifest.

```bash
refire make 1762930614 \
  --brief "the funniest hu tao gacha pulls and rage moments" \
  --style "highlight reel, not a story arc; never open a bit mid-sequence" \
  --duration 20m \
  --pace 0.8 \
  --game "genshin impact" \
  --render

# ...or hand it a style document and let its frontmatter set the machinery
refire make 1762930614 --duration 8m --style ryukStyl.md
```

#### arguments and options:
- `vod_id`: twitch vod number (downloaded and cached in `vods/`).
- `--brief`: brief defining the vibe and topic — **what** the cut is about.
- `--style`: free text defining **how** to cut it: structure, pacing, what to favor, what never to do. the brief picks the footage; the style shapes the edit. by default the director is told to build a story arc and explicitly forbidden the alternative ("that is a highlight reel, not a story") — a style replaces that mandate, in the director *and* in the critic, so a review round cannot quietly restore the arc. omit it and nothing moves. **it can also be a path to a style document** — see [style documents](#style-documents) below.
- `--pace`: cut speed (default `1.0`). scales every role's clip-length budget and the beat count together: `0.6` is snappier and packs in more beats, `1.5` gives each one room to breathe. `--style` says "fast cuts" in words; `--pace` moves the numbers the prompt states literally and the pacing audit enforces — the part an adjective has no leverage over. the beat count is capped at **1.5× the pace-1.0 count** (`director.BEAT_INFLATION_CAP`): the director returns 15–24 beats whatever it is asked for, so a bigger ask buys no extra story turns and only burns thinking budget — `--pace 0.35` once asked for 57 beats on a 16-minute target, spent 57k output tokens deciding, and returned 20. the cap is a ratio, not a fixed number, so genuinely long targets still scale. below the knee where the cap binds (`pace < 1/1.5`) the per-role budgets stop shrinking with it (`pacing.fill_pace`) — a capped beat count against budgets that kept shrinking cannot fill the target at all, which is how a 16-minute ask at `--pace 0.35` shipped as 8:53. past that point a faster pace buys more *cuts per beat*, not less video.
- `--order <chrono|director|whiplash>`: play order (default `chrono`, forward in stream time). `director` keeps the outline's own sequencing; `whiplash` throws the clock away and orders for tonal jolt — chaos into calm, never two similar beats adjacent.
- `--snap <sentence|phrase|transient>`: where cuts land (default `sentence`, never mid-sentence). this is the real floor on shot length: a sentence runs seconds, so no amount of `--pace` produces a 2s shot out of it. `phrase` lands on natural pauses (still never mid-word); `transient` lands on the audio peak and leaves `--truncate` seconds *before* it resolves, so the viewer finishes the joke after the cut has already moved on. the last cut of the video always keeps its sentence snap — truncating the ending is an abrupt stop, not a style.
- `--truncate <s>`: with `--snap transient`, how far before the peak resolves to cut (default `0.3`).
- `--stack <n>`: replaces the single flash-forward teaser with a **montage stack** of `n` unexplained moments (1–2.5s each, escalating, best last) before beat 1. `0` (default) keeps the teaser.
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

#### less-used knobs
the flags above are the ones a cut actually turns on. the rest, grouped by what they touch:

**scope and caching**
- `--start` / `--end`: edit only a window of the stream (`4h` / `4:00:00` / `14400`). only that window is downloaded and transcribed, so a 6h vod you already know the good hour of costs an hour.
- `--run-dir`: artifact directory (default `run/<vod_id>/<vod_id>-<phrase>`, a fresh auto-named folder per run). the shared vod cache — audio, transcript, embeddings — stays in `run/<vod_id>` and is reused across runs.
- `--cache-dir`: where vods download to (default `vods/`).
- `--assets-dir`: folder holding the `sfx/` and `emotes/` used for the music bed and punch-ins.

**director and cost**
- `--claude-model`: model for the narrative pass (default `claude-opus-5`, up from sonnet -- story shape and continuity are the axis opus is better on; `claude-sonnet-5` is cheaper and faster, but weaker at story shape and continuity).
- `--effort <low|medium|high|xhigh|max>`: reasoning effort for the director call (default `xhigh`). Lower is faster and spends less subscription quota.
- `--review-effort <low|medium|high|xhigh|max>`: reasoning effort for the critic rounds (default `high`). The critic returns edits to the director's outline (`keep` / `edit` / new beat) instead of rewriting it, so it needs less thinking. Pass `xhigh` to restore the old behavior.
- `--scout <local|off>`: the chapterize pass that runs before the story pass — free on local ollama by default. `off` is single-shot and only sane on short vods.
- `--flat`: skip the claude director entirely and fall back to flat brief-relevance selection.
- `--title`: the stream title, handed to the director as context (and to the caption fix).
- `--terms`: extra proper nouns — `"Kinich, Arlecchino"` or a path to a file with one per line. merged *ahead* of `--game`'s generated glossary, so it is the escape hatch for anything the local model is too old to know. `genshin.txt` in the repo root is a worked example.
- `--model`: ollama model used for scoring, and for the director under `--local-director`.

**length and captions**
- `--tol`: duration slack, 0–1 (default `0.35`). `--duration` is a target, not a cap: the cut may land anywhere in `target*(1-tol)..target*(1+tol)` before beats get trimmed or flagged short.
- `--words-per-line`: caption grouping (default `3`).
- `--silence-pad`: breath in seconds left around each phrase when dead space is cut (default `0.3`; smaller is tighter).

**render and output**
- `--zoom-sens`: punch-zoom amount (default `1.0`; `>1` is more and earlier).
- `--bgm`: pick the music bed instead of taking a random one from the assets folder.
- `--encoder`: ffmpeg encoder for `--render` (e.g. `h264_nvenc`).
- `--no-proxy`: point ae at the raw vod instead of cutting all-intra (dnxhr lb) clip proxies — skips the transcode, but ae previews far slower.

#### style documents
`--style` takes free text **or a path to a `.md` file**. prose alone can't reach the numbers the cut actually runs on -- the sort order, the snap mode, the beat budgets -- so a style document carries them in a frontmatter block, and everything below the closing fence is the direction the director reads:

```markdown
---
pace: 0.35            # 1.5-4s shots
order: chrono         # play forward in stream time (the cold-open stack is exempt)
snap: transient       # cut on the peak, not on the sentence
truncate: 0.3         # leave 300ms before the punchline finishes
stack: 8              # montage-stack cold open, best moment last
keep_build: false     # "letting a moment breathe so it lands" is the anti-pattern here
cards: false
motion_zoom: false
loudnorm: true
words_per_line: 2
captions: emph        # caption only the lines that are hard to hear
---

# My Style
...everything from here down is fed to the director and the critic verbatim...
```

every key is also a plain cli flag, and **the flag wins** -- `--style ryukStyl.md --pace 1.0` reads the document but overrides its pace. a `.md` with no frontmatter is prose-only, exactly like passing the text inline. unknown keys and bad values warn and are skipped rather than killing a run that already spent director calls.

`ryukStyl.md` in the repo root is a worked example: a reconstruction of a dense, quick-cut clip-channel style, annotated with which section of the spec each knob comes from.

#### re-captioning a hand-recut premiere timeline: `refire recap`
the panel's **recaption** button (see [premiere pro panel](#premiere-pro-panel))
shells this. it needs a `timeline.tsv` — written by `reFirePpro.timeline()`, one
row per v1 trackitem: `media path \t source in \t source out \t timeline in \t
timeline out`. defaults to the one beside the manifest.

```bash
refire recap run/<vod_id>/<run>/ae/manifest.json \
  --words-per-line 3 --game "genshin impact" --fix
```

`--fix` adds the claude proper-noun pass over the regrouped lines; without it the
learned `corrections_<game>.json` is still applied, for free. prints
`Captions: <path>` to the new `captions.recutN.srt`.

#### re-running the caption fix on an existing run: `refire fixcaps`
`make` already does this, so you only need it to re-run against a manifest you built earlier (or after editing `run/corrections_<game>.json` by hand). it rewrites the manifest's caption text and regenerates `captions.srt`.

```bash
refire fixcaps run/<vod_id>/<run>/ae/manifest.json \
  --game "genshin impact" --terms genshin.txt --chat vods/<vod_id>.chat.json
```

every progress line carries elapsed minutes, so a run tells you which stage it spent them in.

### loudest moments across a month of streams: `refire loud`

`make` reads one stream for a *story*. this reads *many* for the moments the room got
loud -- and never downloads a vod whole to do it:

```bash
refire loud 2854020361 2856544608 2857506789 2859479079   --duration 8m --game "genshin impact"
```

twitch serves an audio-only rendition, and the downloader picks the download type off the
output extension, so a 6h stream costs **~440mb instead of ~15gb**. that audio is scored,
and only the windows that survive get their *video* fetched -- the downloader crops
server-side, so a 14s moment costs 14s of transfer. scanning 4.8h of cached audio takes
**under a second**; there is no whisper pass over the stream and no claude call anywhere.

**what counts as loudest** is a spike against a rolling median, not raw level. on raw
level a stretch that is merely loud *throughout* -- a boss fight, a hot-mixed menu, a
music bed -- outranks a real scream, because a window sum wins on duration rather than on
peak. `--baseline` is that median's span: raise it to rank sustained loudness higher,
lower it for sharper reactions.

captions come from transcribing **only the picked windows** (minutes of audio, not hours).
output is a premiere manifest -- `ae/manifest.json` + `ae/captions.srt` -- so the picks
land on v1 via the panel's **build** button and you cut from there. each moment is
downloaded with `--pad` seconds of headroom each side, so a cut can still be nudged
outward by hand once it is on the timeline.

- `vod_ids`: one or more twitch vod numbers (no channel lookup -- paste the ids).
- `--duration`: target length; moments are taken best-first until it is met (`8m` default).
- `--clip` / `--lead`: shot length, and how much of it is run-up *before* the spike
  (default `14` / `4`).
- `--per-vod`: candidate windows scanned out of each stream (default `12`).
- `--pad`: extra seconds downloaded each side for snapping and hand-nudging (default `5`).
- `--baseline`: rolling-median span the spike is measured against (default `60`s).
- `--out`: output folder (default `run/loud-<name>`).
- `--threads`: parallel window downloads (the scan is cheap; the fetch is the wait).
- `--game` / `--terms` / `--words-per-line` / `--whisper-model` etc. behave as in `make`.

the cut spans several vods, so clips ride one virtual timeline (vod *i* at `i * 100000`s)
and each clip points at its own downloaded window through the manifest's `sources[]`. that
is the same shape the ae proxy path already emitted, so the premiere panel, `refire srt`
and **recaption** all work on a `loud` manifest unchanged.

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
- `ae/timeline.tsv`, `ae/captions.recutN.srt` - written by the premiere panel's
  **recaption** button: what is on v1 right now, and the captions rebuilt to match.

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

no premiere open? `uv run python refire/ppro/preview.py` serves the same panel at
`http://127.0.0.1:8090/` with a fake premiere and a replayed `make` (`?speed=120`,
`?run=fail`), and reloads whenever a file in `refire/ppro/` changes. nothing real runs.

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

**recaption** is for after you re-cut that sequence by hand. the manifest
describes the cut reFire built, so the moment you move, trim, split or delete
clips, every cue after the first drifts — and dragging captions back into place
one at a time is worse than the edit was. the button dumps the active sequence's
**v1** to `timeline.tsv` (media path, source in/out, timeline in/out per
trackitem) and runs `python -m refire recap <manifest>`, which walks the
*timeline* instead of the manifest: source in/out plus the manifest's
`sources[].offset` gives the absolute vod range each clip is really showing, and
the run's `transcript.json` is re-grouped against those ranges. moves, trims,
splits, deletes, retimes and handles pulled **wider** than reFire's own cut all
land back under their footage — the widened case is why it reads the transcript
rather than re-timing the manifest's existing cues.

names this stream already taught reFire (`run/corrections_<game>.json`) are
re-applied for free; tick **proper-noun pass** to also run claude over lines it
has never seen. each press writes a fresh `captions.recutN.srt` — premiere's
`importFiles()` skips a path already in the project, so reusing one name would
hand you back the caption track you just replaced. delete the old caption track
and drag the new one on.

v1 only: that is where **build** lays the spine down, so anything stacked above
it reads as b-roll you did not want captioned. media that is not in the
manifest's `sources[]` is skipped, disabled clips are skipped.

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

```bash
uv run pytest        # 311 tests, ~2s, no network and no gpu
```

---

## architectural documentation

- [director pipeline](docs/director-pipeline.md) — how `make` turns a raw multi-hour vod
  into a titled, ordered set of clips: the scout pass, the story pass, casting, and the
  pacing audit. start here.
- [ai editor workflow memory](docs/ai-editor-workflow-memory.md) — the implementation
  brief the narrative editor was built against. load it before changing `director.py`,
  `narrative.py`, `select.py`, or the transcription path.
