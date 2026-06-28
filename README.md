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
- **Ollama** running locally with a chat model (`ollama pull llama3.1:8b`) and,
  for `refire make`, an embedding model (`ollama pull nomic-embed-text`).
- **TwitchDownloaderCLI.exe** in the repo root (for `refire make`'s auto-download).
- Python 3.10+.

## Install

```bash
pip install -e .
```

## Usage

### Hands-off: VOD# + brief + duration → AE manifest

```bash
refire make 1762930614 --brief "the funniest Hu Tao gacha pulls and rage moments" \
  --duration 20m --model llama3.1:8b --game "Genshin Impact"
```

Downloads the VOD (cached in `vods/`), transcribes + embeds it once, then the
brief drives selection: candidates are retrieved by relevance to the brief,
LLM-scored against it, and packed chronologically into a duration budget
(±`--tol`, default 25%; warns instead of padding if short). Output:
`run/ae/manifest.json` → open `refire/ae/reFire.jsx` in After Effects and Build.

### Lower-level: detect → edit/ae from local files

```bash
refire detect stream.mp4 chat.json --run-dir run --top-n 30
# or:  python -m refire detect stream.mp4 chat.json
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
