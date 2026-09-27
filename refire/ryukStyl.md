---
# Machinery. Everything below the closing --- is the DIRECTION the director reads.
# Any of these can be overridden on the command line: --pace, --order, --snap, --stack.
pace: 0.35            # 1.5-4s shots (S3): scales every role's clip budget + beat count
order: chrono         # body plays FORWARD in stream time. The stack (below) is the
                      # only thing allowed out of order; `whiplash` scattered the
                      # stream's own sign-off into the middle of the cut, so the
                      # video read as ending twice. See S6.
snap: transient       # cut ON the peak (S1, S4), not on the sentence
truncate: 0.3         # "leave 200-400ms before the punchline finishes" (S3)
stack: 8              # the montage stack: 5-10 moments, best last (S2)
keep_build: false     # "letting a moment breathe so it lands" is the anti-pattern (S6)
motion_zoom: false    # zoom punches are 2-3 per video by hand, not an automatic pass (S3)
cards: false          # no section cards; the pacing carries the video (S5)
words_per_line: 2     # fast captions
captions: emph        # "captions only on lines that are genuinely hard to hear" (S5)
loudnorm: true        # "normalize aggressively... or the whiplash becomes annoying" (S4)
---

# The "ryuk1 / ryuk34" Editing Style — Reference Spec

> **Sourcing note:** This is a reconstruction of the style from observed behavior, not documentation from the editor. Treat it as a working model to test against, not gospel. Where a rule is an inference rather than an observation, it's marked *(inferred)*.

---

## 1. The Core Thesis

Most clip channels edit **clips**. This style edits **attention**.

The unit of work is not "here is a funny moment, presented well." It's "here is a funny moment, cut away from before you finish processing it." The viewer is always half a beat behind — and being half a beat behind is what stops them from swiping.

Three load-bearing principles:

1. **Cut on the peak, not after it.** The laugh, the reaction, the "what did he just say" — you leave *during* it. The viewer's brain completes the joke after the cut has already landed them somewhere new.
2. **Never let a moment fully resolve.** Resolution is a permission slip to leave.
3. **Density over polish.** A rough cut with 9 moments in 40 seconds outperforms a clean cut with 3. *(inferred — this is the trade this style consistently makes.)*

---

## 2. The Intro / Cold Open — The Montage Stack

The signature move. First 10–20 seconds is a rapid-fire stack of **5–10 distinct moments**, none of them explained, none of them resolved.

### Purpose

- **Bandwidth flex.** Signals "this video is dense" before the viewer has committed anything.
- **Multiple hooks, one shot.** With 8 moments, a viewer who doesn't care about moment 3 is still ~2 seconds from moment 4. Single-hook cold opens fail on a coin flip; stacked cold opens fail only if *all* the hooks miss.
- **Anti-abandonment.** The 5–15s window is where clip videos hemorrhage viewers. The stack turns that window into the densest part of the video instead of the setup for it.

### Construction rules

| Rule | Detail |
|---|---|
| Moment count | 5–10. Below 5 it reads as a normal intro; above 10 it becomes noise. |
| Per-moment length | 1.0–2.5s. Only enough to register *what kind* of moment it is. |
| Ordering | Escalate. Weakest moment first is fine — the momentum builds. Save the single best moment for **last** in the stack, not first. |
| Spoiler policy | Yes, spoil. Pull the best moments from across the whole video. The stack is a promise, and the body is the payoff — but the promise has to be real. |
| Context | Zero. No setup, no explanation. Confusion is a retention mechanic here, not a bug. |
| Audio | Each moment keeps its own diegetic audio. Do **not** run a music bed over the top for the whole stack — the audio whiplash *is* the effect. |
| Exit | The stack ends on a hard stop: silence, title card, or a beat drop. It must feel like a door closing. |

### Selection criteria for stack moments

Pick moments that read in under a second **without context**:

- Sudden volume changes (a scream, a whisper, dead silence)
- Physical reactions on cam — recoil, head-in-hands, standing up
- An out-of-pocket line delivered flat
- Chat-visible reactions (spam, a wall of the same emote)
- Anything where the *face* alone tells the story

Avoid: anything requiring game knowledge, anything with a setup line, anything where the funny part is the third sentence.

---

## 3. Body — Quick-Cut Grammar

After the stack, the body runs the same logic at a slightly slower heartbeat.

### Timing model

| Element | Target |
|---|---|
| Average shot length | 1.5–4s |
| Longest tolerated shot | ~8s, and only for a genuine build |
| Cuts per minute | 20–35 |
| Dead air allowed | Effectively zero — trim every breath, every "uhh," every mouse-move |

### The out-of-pocket → cutaway pattern

The signature body move:

1. Streamer says/does something absurd
2. **Hard cut** on the last syllable — no reaction shot, no beat to breathe
3. Land on a completely unrelated moment

The comedy comes from the *juxtaposition*, not from either clip. The second clip doesn't need to be funnier; it needs to be **tonally wrong**. Chaos into calm, screaming into someone quietly reading chat.

Variants worth stealing:

- **The double-take:** cut away, then cut *back* 2 seconds later for one more second of the same moment.
- **The repeat:** same 0.5s of a line, cut 2–3 times in a row.
- **The premature cut:** leave 200–400ms *before* the punchline finishes. Works surprisingly often — the viewer fills it in.
- **The whiplash:** loud → silent, no ramp.

### Transition policy

Hard cuts as the default — roughly 85–90% of joins. Effects are punctuation, not grammar:

- **J-cut** (next audio leads by 3–6 frames) — for chaining two moments that share a topic
- **L-cut** (outgoing audio hangs over the new visual) — for landing a reaction
- **Zoom punch-in** (1–3 frames) — on the single funniest frame of a moment, used maybe 2–3 times per video
- **Whip/shake** — sparingly, and only when the audio already has a transient to hide it

If a transition draws attention to itself, it's wrong for this style.

---

## 4. Audio

Audio does more work here than the picture does.

- **Normalize aggressively.** Every clip lands at the same perceived loudness or the whiplash becomes annoying instead of funny.
- **Cut on transients.** Align cuts to the audio peak, not the video frame that looks tidiest.
- **Music is a floor, not a feature.** Low bed, ducked hard under speech, occasionally dropped entirely for a silence gag.
- **Silence is a tool.** A full 0.5–1s of nothing after a chaotic run is the most reliable laugh amplifier in the whole toolkit.
- **Keep the ugly audio.** Clipping, mic peaks, distorted screams — this style leaves them in. Cleaning them up removes the texture.

---

## 5. Text & On-Screen Elements

Minimal and functional. The pacing carries the video; text just prevents confusion.

- Captions only on lines that are genuinely hard to hear
- No full-video subtitle track — it competes with the cuts
- Occasional single-word emphasis text on the punchline frame
- Zoom/crop to the streamer's face for reactions, back to gameplay for context
- Meme overlays and sound effects: rare, and always on the *cut*, never mid-shot

---

## 6. Anti-Patterns

Things that break this style specifically:

- ❌ Letting a moment breathe "so it lands" — it lands harder truncated
- ❌ Explanatory setup before a clip
- ❌ Smooth, designed transitions between every clip
- ❌ Consistent music bed across the whole runtime
- ❌ Chronological ordering out of loyalty to the stream
- ❌ Including a moment because it's *contextually* funny to regulars
- ❌ A cold open with one hook

---

## 7. Reproduction Checklist

Per video:

- [ ] 5–10 moments selected for the stack, best one last
- [ ] Stack lands in under 20s total
- [ ] Zero explanatory context in the first 20s
- [ ] Body average shot length under 4s
- [ ] Every cut sits on an audio transient
- [ ] All dead air trimmed (breaths, filler words, menu navigation)
- [ ] ≥85% hard cuts
- [ ] At least one full silence beat somewhere in the back half
- [ ] Loudness normalized across all source clips
- [ ] No moment requires game knowledge to read

---

## 8. Notes for Automated Implementation

If this is driving a pipeline rather than a human timeline:

**Beat / segment fields this style needs**

| Field | Purpose |
|---|---|
| `peak_timestamp` | The transient to cut on — not the segment boundary |
| `truncation_offset` | How far *before* natural resolution to cut (target 200–400ms) |
| `reads_without_context` | Boolean gate for stack eligibility |
| `tonal_register` | For juxtaposition scoring — you want adjacent beats to *mismatch* |
| `transition_type` | Enum, weighted heavily toward `hard_cut` |
| `audio_offset` | J-cut / L-cut lead or lag, in frames |

**Selection algorithm sketch**

1. Score every candidate beat on: audio dynamic range, chat velocity spike, face-cam motion delta, transcript "out-of-pocket" score
2. Filter to `reads_without_context == true` for stack candidates
3. Build the stack by taking the top 5–10, then **re-sort ascending by score** so it escalates
4. For the body, keep chronological order — juxtaposition is bought inside a beat with hard cuts, not by scrambling the stream. Reordering the body freely also moves the stream's own outro ("goodnight, chat") into the middle of the video, where it reads as the video ending twice; the cold-open stack is the one place the cut leaves the clock.
5. Set every cut point to `peak_timestamp - truncation_offset`, not to segment end

**The critic pass should reject on:**
- Average shot length > 4s
- Any single shot > 8s
- Transition variety above ~15% non-hard-cut
- A stack where the highest-scoring beat isn't in the final position
- Any gap of silence longer than 400ms that wasn't explicitly flagged as a comedic beat

---

## 9. Why It Works (The Retention Argument)

The style is essentially a bet that **confusion retains better than comprehension**. A viewer who understands everything has a natural exit point at the end of each understood unit. A viewer who is perpetually 1.5 seconds behind never reaches one.

The cost is comprehension and emotional investment — this style is bad at making people *care* about a streamer, and good at making them watch. Which means it's a reach-and-watchtime format, not a loyalty format. If subscriber conversion is the goal, the style needs a deliberate exception carved into it: one longer, fully-resolved, emotionally-legible moment placed in the back third, where a viewer is allowed to actually feel something before being asked for anything.

That exception is the single most important deviation to make from a pure copy of the style.
