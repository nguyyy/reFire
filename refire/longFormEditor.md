I want to create an automated long form editor which:

Takes input of long video streams (from 1-8+ hours of footage, ex. gaming stream, podcast, etc),
Chops into interesting segments (using llm to detect what's interesting),
Outputs then edits them into a long form video/clip compilation and adds music, subtitles, and zoom/panning etc (Ken Burns effect, etc.) Similar to the short form opus ai editor - but for long form

What will be the best way to attack this problem using AI?

Some ideas:
convert .mp4/file into pure audio file, transcribe locally/free online service, use llm to mass process info in order to find interesting segments. Take that data, then go back to the video and chop it up into the interesting segments, then edit them together

What are my options for the tech stack?

- FFmpeg for video/audio processing
- Whisper for transcription
- Local LLM? Or API? for processing info to find segments
- What AI models for facial recognition, automatic zoom/panning, subtitle generation, music generation, video editing, etc.
- Prefer free/local models/services. If API is needed, must be cheap.
-maybe an after effects/premiere pro mcp/script stack? unsure, SDKs are very limited in scope.