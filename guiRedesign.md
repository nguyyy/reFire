# reFire panel — visual redesign spec

Redesign the panel's appearance and interaction feel. Don't change pipeline behavior, CLI contracts, or what any control does — only how it looks and how it moves between states.

Do this in one pass. Build it, then critique and revise your own work before you hand it back. Don't stop to ask me to approve a plan.

**First, read the panel source and enumerate the real thing:** every parameter control it exposes, every pipeline stage it can report, every state it can be in (idle, validating, running, stage-failed, cancelled, complete). Design against that actual list, not a guess. If something in this spec conflicts with an existing token or class in the codebase, this spec wins — update the token.

---

## Direction

**Constructivism × Swiss, grayscale.** Swiss supplies the structure: one grid, flush-left ragged-right, hierarchy from size/weight/space alone, mathematical spacing. Constructivism supplies the energy: hard weight and scale contrast, solid tonal mass used as structure, one asymmetric gesture that organizes the whole panel.

It's a console someone stares at for hours while a long job runs. Calm and legible beats expressive.

## Tokens — use these exact values

```
--ground    #191C1F   panel background
--surface   #212429   row/control fill, raised blocks
--edge      #2B2F35   rules, inactive fill, disabled edges
--dim       #6E747D   labels, secondary meta, completed stages
--text      #A9AFB8   body, control values
--bright    #E6E8EB   display type, active stage, progress fill
--signal    #CE2B21   live / error / needs-decision only
```

Most of the panel is `--ground`, `--surface`, `--dim`, `--text`. `--bright` is rationed — display numerals, the executing stage, filled progress. `--signal` never appears on idle chrome, static labels, borders, or a resting button. If red is visible when nothing is happening and nothing is wrong, you've used it as decoration; remove it.

**Type:** one neutral grotesque (Inter or Inter Tight, bundled locally, falling back to Helvetica Neue / Arial). Weights 400 / 500 / 700 only. `font-feature-settings: "tnum" 1` globally — timecodes, percentages and counters change in place and must not jitter.

```
11px / 500 / +0.01em   parameter labels, stage names, meta
13px / 400             body, control values, log lines
20px / 500 / -0.01em   executing stage name
44px / 700 / -0.03em   the one display numeral
```

That 11→44 jump is the constructivist contrast. Don't soften it with intermediate sizes; four sizes is the whole scale.

**Spacing:** 4px base unit, everything a multiple. Parameter rows on a 32px rhythm. Panel padding 20px. No value that isn't a multiple of 4.

**Radius:** 0 on structural blocks and the progress bar. 2px on interactive controls only, so touchable things are distinguishable from mass. No third value.

**Mono** (ui-monospace stack) is allowed for paths, IDs and log output. Never for labels to make them look technical.

## Layout

One vertical hairline in `--edge` runs the full panel height at the label/control boundary — this is the Swiss spine and the only rule in the interface. No horizontal dividers between parameter rows; separate them with rhythm instead. Labels sit flush-left of the spine, controls flush-left immediately right of it, so both columns have a hard edge.

The constructivist gesture is a single solid block at the top of the panel holding the run control and the display numeral — full-bleed to the panel edges, `--surface`, no radius, visually heavier than everything under it. That block is the only place boldness is spent. Everything below it stays quiet.

Layout must hold across the panel's full docked width range without reflowing into something unrecognizable.

## Continuity — the part that matters most

The panel is one object changing state, not two screens swapping.

- Parameter rows **persist** through a run. They don't hide, and a progress view doesn't replace them. Running, they de-emphasize to `--dim` and become non-interactive, so the operator can still read what the run was configured with.
- The run control's **own geometry** becomes the progress bar. It grows to full block width in place; it does not disappear and get replaced by a different element.
- The display numeral in the top block shows elapsed time while running, and holds the final duration on completion.

**Progress bar:** determinate and segmented by real pipeline stage, with segment widths proportional to expected duration — a uniform bar lies, because transcription and render dwarf the story pass. Weight the segments from observed run times if the codebase has them; otherwise hardcode a sane weighting and leave a comment naming it as an estimate.

Completed segments fill `--bright`. The executing segment fills as it goes and carries a `--signal` leading edge — that hairline is where red lives. Pending segments are `--edge`.

Inside the executing segment, a 2px indeterminate sub-line moves continuously. This is the anti-hang signal: during a 20-minute transcription the bar barely advances, and without local motion the panel reads as frozen. This sub-line is the one piece of non-user-triggered motion in the interface and it is earned.

**Motion budget:** state transitions 180ms `cubic-bezier(0.2, 0, 0, 1)`. Progress fill 300ms linear. The idle→running transition 320ms, animating position and mass continuously. Nothing else animates — no load-in fades, no hover transitions on every control, no scattered effects. Under `prefers-reduced-motion`, all of it goes instant except the progress fill, and the indeterminate sub-line becomes a static pulse in opacity.

## Do not

- Tracked-out ALL-CAPS eyebrow labels. Caps only inside the top block's display, if at all.
- Identical rounded cards, one radius on everything, `rgba(0,0,0,.1)` shadows.
- Gradients, glass blur, glow, neon, bright acid accents on near-black.
- Middle-dot meta strings, `WORD — fragment` with a spaced em dash, `→` on button text.
- `01 / 02 / 03` markers anywhere except the pipeline stages, which genuinely are a sequence.
- Emoji, decorative icons, dividers that separate nothing.

## Copy

Rewrite the interface text as part of this. Sentence case, plain verbs, no filler. Name things as the operator understands them. An action keeps its name through its whole lifecycle — the button that says "Run" produces a state that says "Running" and a result that says "Run complete." Errors state what broke and the next action, in the interface's voice, without apologizing. Idle state is an invitation to start a run, not a mood.

## Floor

Visible keyboard focus, styled to the palette, on every interactive element. Verify contrast actually passes on the gray-on-gray ramp rather than assuming — `--dim` on `--surface` is the one to check. Watch specificity between structural and component selectors so spacing rules don't cancel. Cancel must be reachable at all times during a run.

## Before you hand it back

Screenshot idle and running if you can render it. Look at both cold and fix what's wrong. Cut one element that isn't carrying weight. Then tell me, in a short list: what you cut, anything in this spec you deviated from and why, and any value you had to invent (segment weights especially) that I should verify against a real run.
