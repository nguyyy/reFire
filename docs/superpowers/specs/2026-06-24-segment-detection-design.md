# Segment Detection Stage — Design

**Date:** 2026-06-24
**Project:** reFire (automated long-form video editor)
**Scope:** Stage 1 of 4 — transcribe + detect clip-worthy segments. Outputs a JSON
segment list consumed by a later, separate video-assembly stage. This spec does
NOT cover cutting, effects, subtitles, or music.

## Goal

Given a long gaming stream (1–8+ hours) plus its chat replay log, produce a ranked
list of clip-worthy segments fully automatically, with no human review checkpoint.

## Context / constraints

- Local GPU available → run everything locally, zero per-run cost.
- Primary content: gaming streams. Chat replay log is available.
- Fully automatic: detection output feeds straight into the (future) cutting stage.

## Pipeline

```
video.mp4 ──ffmpeg──► audio.wav (mono 16kHz)
                          │
                          ▼
                  faster-whisper ──► transcript (word-level timestamps)
                          │
chat.log ──parse──► chat-rate z-score over time
                          │
                          ▼
        chunk transcript into ~30–90s windows, attach chat score
                          │
                          ▼
              local LLM (Ollama) scores each chunk 1–10 + reason
                          │
                          ▼
      merge LLM score + chat z-score ──► ranked segments JSON
                          │
                          ▼
              select top-N / threshold ──► segments.json
```

## Components

Each is an independent unit with a clear input → output contract.

1. **audio_extract** — `video path → wav path`. One `ffmpeg` call: `-ac 1 -ar 16000`.
2. **transcribe** — `wav path → transcript` (list of `{word, start, end}`).
   `faster-whisper`, GPU. Persist to disk so reruns skip this expensive step.
3. **chat_signal** — `chat log → list of {t, zscore}`. Bin messages into fixed
   windows (e.g. 5s), compute message-rate, z-score across the whole stream.
   Emote/keyword density can be added later; rate alone is the v1 signal.
   *(ponytail: rate-only z-score; add emote weighting if rate misses moments.)*
4. **chunk** — `transcript + chat signal → list of chunks`.
   Each chunk: `{start, end, text, chat_z}`. Window ~30–90s on sentence/pause
   boundaries from word timestamps.
5. **score** — `chunk → {llm_score, reason}`. One Ollama call per chunk with a
   gaming-tuned scoring prompt. Returns structured JSON.
6. **rank_select** — `scored chunks → segments.json`. Final score =
   weighted blend of `llm_score` and `chat_z` (weights configurable).
   Sort, threshold/top-N, emit.

## Output contract (the hand-off to Stage 2)

```json
[
  { "start": 1423.5, "end": 1467.0, "score": 8.7, "reason": "clutch 1v3 win, chat exploded" }
]
```
`start`/`end` in seconds. This is the only thing the cutting stage needs — detection
and editing stay fully decoupled.

## Models / stack

- **ffmpeg** — audio extraction.
- **faster-whisper** (local, GPU) — transcription with word timestamps.
- **Ollama** running a local LLM (`qwen2.5:14b` or `llama3.1:8b`) — chunk scoring.
- Glue: Python.

## Error handling

- Missing/corrupt chat log → fall back to transcript-only scoring (chat_z = 0 for all).
- Whisper / ffmpeg failure → fail loud with the underlying error; don't emit partial output.
- Persist intermediate artifacts (audio, transcript) so a failed late stage doesn't
  force re-running expensive early stages.

## Verification

- One self-check: feed a short known fixture (transcript + chat) and assert
  `rank_select` returns segments sorted by score with valid start/end ordering.

## Out of scope (deliberately, for later stages/specs)

- Video cutting, Ken Burns / zoom-pan, subtitles, music — separate specs.
- Multimodal vision-based excitement detection — revisit only if chat + transcript
  proves to miss too many silent-but-interesting moments.
- Podcast/general-content tuning — design generalizes, but tuned for gaming first.
