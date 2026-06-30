# reFire AI Editor Workflow Memory

Date: 2026-06-30

Purpose: implementation brief for evolving reFire from a clip scorer into a more human-like long-form editor. This file is meant to be loaded by an AI coding agent before changing `director.py`, `narrative.py`, `select.py`, transcription, or the `make` pipeline.

## Product Goal

reFire should not feel like it selected clips from grading scales. It should feel like an editor found a story inside a long stream, shaped viewer expectation, preserved setups and payoffs, and made deliberate cuts.

The core shift:

- Old mental model: find high-scoring clips.
- New mental model: build a story spine, cast clips into story roles, tighten each clip around setup/payoff/reaction, then review the realized rough cut.

AI should provide editorial judgment. Deterministic code should handle exact cuts, cache reuse, transcription plumbing, silence removal, captions, motion zoom, and manifest generation.

## Human Editor Workflow To Automate

A human editor generally does this:

1. Skims the source.
2. Finds the central thread.
3. Marks possible moments.
4. Chooses a story shape.
5. Pulls clips that serve that shape.
6. Tightens each clip around setup, payoff, and reaction.
7. Watches the rough cut.
8. Fixes awkward jumps, repeated beats, weak openings, and weak endings.
9. Adds style: captions, zooms, cards, overlays, music, and SFX.

The automated pipeline should mirror that workflow:

```text
download/transcribe
-> source understanding / stream map
-> story discovery
-> beat planning
-> candidate retrieval
-> beat casting
-> tight cut refinement
-> rough cut assembly
-> critic review over realized transcript
-> revise
-> manifest/render/style pass
```

## AI Roles

Use AI as three separate editorial workers.

### Logger

The logger summarizes the source into searchable editorial memory. It should identify events, notable lines, emotional tone, chat reaction, possible payoffs, and energy.

Example artifact:

```json
{
  "start": 420,
  "end": 475,
  "summary": "Streamer roasts a strange character build",
  "events": ["build reveal", "chat disbelief", "streamer doubles down"],
  "emotional_tone": "mocking, chaotic",
  "notable_lines": ["..."],
  "chat_reaction": "spike/laughing/confused",
  "possible_payoffs": ["mamma mia line", "chat erupts"],
  "energy": 4
}
```

### Director

The director chooses the story this VOD can actually support. It should not merely list interesting moments.

Useful story shapes for gaming/Twitch:

- Confidence collapses into chaos.
- Bad idea slowly becomes genius.
- Chat doubts him, then he proves it.
- One tiny mistake becomes the whole episode.
- The run starts normal, then gets cursed.
- A running joke evolves into the payoff.

Example artifact:

```json
{
  "story_shape": "confidence collapses into chaos",
  "central_idea": "The streamer tries to evaluate builds seriously, but every build gets more cursed.",
  "viewer_promise": "You are watching the descent from normal review to total disbelief.",
  "ending_needed": "A final absurd build or punchline that feels like the peak."
}
```

### Critic

The critic reads the actual rough cut transcript, not the original plan. It checks whether the viewer experience works.

It should ask:

- Does the first 10 seconds hook?
- What does the viewer expect after each beat?
- Does the next beat answer, twist, or escalate that expectation?
- Are two beats doing the same job?
- Does any clip start without enough context?
- Does any clip end before the payoff or reaction?
- Is the ending satisfying?
- Does the energy vary?

## Best Practices For AI Selection

Do not ask an LLM to pick clips directly from the entire VOD in one shot.

Prefer this pattern:

```text
transcribe
-> chunk/map source
-> retrieve candidates
-> AI labels candidate purpose
-> AI builds story beats
-> cast clips into beats
-> deterministic trim
-> critic reviews actual rough cut
-> revise
```

Guidelines:

- Use candidate pools. Embeddings, chat spikes, audio emphasis, and motion can cheaply find maybe 40-100 possible moments. Let AI judge those.
- Avoid raw `1-10` as the main selection signal. Scores compress and produce grading-scale behavior.
- Prefer structured judgments: `role`, `payoff`, `setup`, `viewer_question`, `transition_in`, `energy`, `why_this_clip`.
- Use AI for comparisons: "Which of these candidates better serves the climax?" Pairwise/ranked choice is often more reliable than isolated scoring.
- Force story roles. Every selected clip should be hook, setup, escalation, reversal, climax, payoff, or button.
- Preserve provenance. `outline.json` should explain why each clip exists and what line/action is the payoff.
- Review the realized rough cut, not the imagined plan.
- Cache everything: transcript, embeddings, stream map, chunk summaries, director outline, review results, and final clip decisions.

## Proposed Beat Schema

Current code has `director.Beat` with `title`, `intent`, `start_s`, `end_s`, and `query`.

Next evolution should add editorial fields. Keep backwards compatibility where possible, or add defaults so tests and older outlines continue to parse.

Suggested shape:

```py
class Beat(BaseModel):
    title: str
    role: str              # hook, setup, escalation, reversal, climax, payoff, button
    intent: str            # this beat's job in the story
    viewer_question: str   # what the viewer is wondering after this beat
    turn: str              # what changes in this beat
    transition_in: str     # why this follows the previous beat
    texture: str           # funny, tense, awkward, triumphant, chaotic, sincere
    energy: int            # 1-5 pacing value
    setup_start_s: float | None
    payoff_start_s: float | None
    reaction_end_s: float | None
    start_s: float
    end_s: float
    query: str
```

Interpretation:

- `start_s` / `end_s`: final intended clip bounds.
- `setup_start_s`: where needed context begins, if different from `start_s`.
- `payoff_start_s`: where the joke, reveal, win, fail, or reaction starts landing.
- `reaction_end_s`: optional tail after the payoff; useful so cuts do not feel abrupt.
- `viewer_question`, `turn`, and `transition_in`: critic-facing fields that make the cut read like an editor's notebook.

## Tight Cut Rules

AI can propose clip structure, but deterministic code should enforce clean cuts:

- Start before the setup.
- Never cut before the payoff.
- Keep a short reaction tail.
- Remove dead air inside the clip.
- Snap to sentence or phrase boundaries.
- Prefer phrase/silence boundaries when punctuation is missing.
- For the final beat, be more generous about ending on a clean button.

Example tight-cut artifact:

```json
{
  "start": 1814.2,
  "end": 1872.5,
  "keep_spans": [
    [1814.2, 1828.0],
    [1849.0, 1872.5]
  ],
  "reason": "Preserves setup, cuts dead silence, ends after laugh."
}
```

Existing useful code:

- `select.snap_to_sentences`
- `select.speech_intervals`
- `select.compress_silence`
- `assemble.render_clips`
- `narrative.cast`

## Review Loop

The review pass should operate over the realized script generated from actual chosen spans.

Useful review output:

```json
{
  "approved": false,
  "notes": "Beat 3 repeats Beat 2. Ending is abrupt.",
  "changes": [
    {"action": "drop", "beat": 3},
    {"action": "extend_end", "beat": 5, "new_end": 2412.4},
    {"action": "add", "role": "button", "search_for": "final chat reaction"}
  ]
}
```

The current repo already has `director.review()` and a bounded review loop in `pipeline.make`. The next upgrade is to make the review rubric more viewer-experience-oriented:

- Hook strength.
- Setup clarity.
- Beat-to-beat expectation.
- Escalation.
- Redundancy.
- Payoff completion.
- Energy variation.
- Closing button.

## Editor-Readable Output

The final `outline.json` should be useful for debugging taste. Consider also writing a Markdown companion such as `run/<vod_id>/cut_plan.md`.

Example:

```md
# Cut Plan

Central idea:
The streamer tries to do a normal build review but slowly loses composure.

## Beat 1 - Cold Open

Role: hook
Why it exists: shows the end-state chaos first.
Viewer question: how did we get here?
Clip: 02:14-02:36
Payoff: "What am I looking at?"

## Beat 2 - The Rules

Role: setup
Why it exists: establishes the build review format.
Clip: 04:01-04:42
```

If the cut plan reads boring, the video will probably be boring.

## Style Pass

Story should come before style. After the rough cut is coherent, story metadata can drive presentation:

- Hook: immediate start, faster captions, punch zoom, minimal title card.
- Setup: calmer music, fewer overlays.
- Escalation: stronger pacing, more motion emphasis.
- Climax: stronger zoom/SFX/overlay density.
- Button: preserve reaction tail; avoid over-covering the final laugh.

Existing style modules:

- `subtitles.py`
- `reframe.py`
- `overlay.py`
- `ae_export.py`
- `refire/ae/reFire.jsx`

Motion zoom is optional style, not story logic. Pass `--no-motion-zoom` (or untick "motion zoom" in the AE panel) to skip the slow OpenCV motion scan and the quick-motion punch zooms: AE manifests get empty `zoom_episodes` and `--render`/`edit` cuts static-fit (bottom-left) instead of dynamic reframe. Default stays on.

Important memory: do not reintroduce camera panning. Current accepted zoom design is fixed bottom-left anchor, motion-triggered episodes, no shaky centroid movement.

## Transcription Strategy For Long Audio

Local `faster-whisper` is free/private but slow for 3+ hour streams. Add pluggable transcription backends while normalizing all outputs to the existing word schema:

```json
[
  {"text": "hello", "start": 0.0, "end": 0.4},
  {"text": "world.", "start": 0.5, "end": 0.9}
]
```

Recommended CLI shape:

```text
refire make <vod> --transcriber local
refire make <vod> --transcriber groq
refire make <vod> --transcriber assemblyai
refire make <vod> --transcriber openai
```

Suggested implementation:

- Keep `local` as default for free/private runs.
- Add `transcribe_backend.py` or split modules under `refire/transcribers/`.
- Preserve the existing `transcript.json` cache.
- Add sidecar metadata, e.g. `transcript.source.json`, with backend, model, input hash, glossary/game, and timestamp.
- Chunk long audio for APIs with file-size or duration limits.
- Recombine chunks into one monotonic word list with chunk offsets.
- Keep hotwords/game glossary support where the backend allows it.

Cloud options worth testing for 3+ hour audio:

- Groq Whisper Large v3 Turbo: very cheap and fast for bulk Whisper-style transcription.
- AssemblyAI: useful free tier and long-file workflow; good candidate for long prerecorded audio.
- OpenAI `gpt-4o-mini-transcribe`: likely good when promptable accuracy and game terms matter, but long files require chunking/compression because upload size limits apply.
- Deepgram: strong long-audio speech-to-text option with free credits and feature-rich diarization/keyterm tooling.

Do not hardcode provider assumptions without checking current docs/pricing. Treat backend pricing and limits as unstable.

## Staged Implementation Plan

### Phase 1: Editorial Schema

- Expand `director.Beat` with role, viewer question, turn, transition, energy, texture, and payoff fields.
- Update `director._SYSTEM` to require a specific story shape and viewer promise.
- Update `narrative.cast` to log all editorial fields into `outline.json`.
- Update `_realized_script` so the critic sees editorial intent plus actual transcript.
- Add/adjust tests around parsing, logging, and realized script rendering.

### Phase 2: Critic Upgrade

- Rewrite review rubric around viewer experience.
- Allow the critic to revise role, order, bounds, and transitions.
- Keep the bounded loop: default two review rounds, stop on approval.
- Ensure review failure keeps the current cast rather than falling back unnecessarily.

### Phase 3: Tightening Upgrade

- Use `setup_start_s`, `payoff_start_s`, and `reaction_end_s` to improve cut bounds.
- Improve final-clip ending logic with phrase/silence fallback when punctuation is missing.
- Keep `compress_silence` for rough renders; later port equivalent retiming into AE if desired.

### Phase 4: Editor-Readable Artifacts

- Write `cut_plan.md` beside `outline.json`.
- Include story shape, central idea, viewer promise, beat roles, transitions, payoff lines, timestamps, and critic notes.
- This makes taste debugging easier for both humans and AI coding agents.

### Phase 5: Transcription Backends

- Add `--transcriber`.
- Keep local faster-whisper behavior unchanged.
- Add one cloud backend first, preferably Groq or AssemblyAI for long-file cost/speed.
- Normalize provider responses to existing `transcript.json` word schema.
- Add unit tests for chunk-offset recombination and cache invalidation metadata.

## Repo Touch Points

Likely files:

- `refire/director.py`: schema, prompts, outline/review parsing.
- `refire/narrative.py`: cast logs, editorial fields, payoff-aware bounds.
- `refire/select.py`: boundary and silence logic.
- `refire/pipeline.py`: make workflow, review rounds, transcription backend selection.
- `refire/transcribe.py`: local backend wrapper or adapter.
- `refire/cli.py`: `--transcriber`, provider/model options.
- `refire/ae_export.py`: optional story metadata in manifest.
- `tests/test_narrative.py`: schema/cast/review tests.
- New tests for transcription backend normalization if added.

## North Star

The output should not say only:

> This clip scored 9.0 because it is funny.

It should say:

> This is the reversal beat. The viewer expects a normal build review, but this moment proves the premise has gone off the rails. The payoff starts at 12:42 and the reaction tail ends at 12:51. It belongs after the setup because it changes the rules of the video.

That is the difference between an automated scorer and an automated editor.
