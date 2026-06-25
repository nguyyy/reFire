# reFire

Automated long-form video editor. **Stage 1 (this repo): segment detection** —
given a long Twitch gaming stream and its chat replay, it outputs a ranked
`segments.json` of clip-worthy moments. Cutting, effects, subtitles, and music
are later stages.

Everything runs locally: ffmpeg + faster-whisper (GPU) + a local Ollama LLM.
Zero per-run cost.

## Prerequisites

- **ffmpeg** on PATH (system install).
- **NVIDIA GPU + CUDA** for faster-whisper.
- **Ollama** running locally with a model pulled, e.g. `ollama pull qwen2.5:14b`.
- Python 3.10+.

## Install

```bash
pip install -e .
```

## Usage

```bash
refire stream.mp4 chat.json --run-dir run --top-n 30
# or:  python -m refire stream.mp4 chat.json
```

`chat.json` is a [TwitchDownloader](https://github.com/lay295/TwitchDownloader)
chat export (the format with a `comments` array, each having
`content_offset_seconds`).

Output: `run/segments.json` —
```json
[
  { "start": 1423.5, "end": 1467.0, "score": 8.7, "reason": "clutch 1v3, chat exploded" }
]
```

### Options

| flag | default | meaning |
|------|---------|---------|
| `--model` | `qwen2.5:14b` | Ollama model for scoring |
| `--top-n` | none | keep only the top N segments |
| `--threshold` | none | drop segments below this final score |
| `--w-llm` / `--w-chat` | 0.6 / 0.4 | blend weights (LLM vs chat spike) |

Audio and transcript are cached in the run dir, so reruns skip the expensive
ffmpeg/Whisper steps.

## How it works

1. ffmpeg → mono 16kHz wav
2. faster-whisper → word-level transcript
3. chat replay → message-rate z-score over time (spikes = excitement)
4. transcript chunked into ~30–90s windows, each tagged with its chat spike
5. local LLM scores each chunk 1–10 for clip-worthiness
6. blend LLM + chat scores, rank, select → `segments.json`

No chat log? Detection falls back to transcript-only scoring automatically.

## Test

```bash
pip install -e ".[dev]"
pytest
```
