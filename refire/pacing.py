"""Pacing audit: measure the realized cut on the FINISHED video's timeline.

The critic reads `director._realized_script`, which stamps every beat with its SOURCE
timestamp -- where the moment sits in the stream, not where it sits in the cut. So the
review rubric's "does the energy vary?" was asking a model to do arithmetic it had no
inputs for: nothing told it that beats 5-11 are minutes 4-10 of the finished video and
never rise above energy 2. A sagging middle survived two review rounds because the sag
was invisible.

This module supplies those numbers as measured facts. Pure -- no I/O, no model calls --
so the thresholds are unit-testable and cheap to retune when the flags start measuring
the wrong thing (which is the first thing to suspect if a cut still drags with a clean
audit; tune here before touching the prompts).
"""
from __future__ import annotations

# Thresholds. Deliberately blunt: these exist to point the critic at a stretch of the cut,
# not to grade it. Tune here, not in the prompt.
FLAT_ENERGY = 2       # energy at or below this reads as a lull
FLAT_RUN_S = 90.0     # a lull only matters once it runs this long in the finished cut
SLOW_PAYOFF_S = 12.0  # this much setup before the point lands is where viewers leave
DENSITY_FRAC = 0.6    # words/sec under this fraction of the cut's median reads as thin
MIN_DENSITY_S = 20.0  # short beats have noisy words/sec -- don't flag them
FRONT_LOAD_FRAC = 1 / 3.0  # both peaks inside this fraction of the runtime = front-loaded
BUILD_GAP_S = 60.0    # an internal jump this long BEFORE the payoff skipped the build-up.
# Deliberately the same 60 the critic rubric already states in prose ("MISSING BUILD-UP"),
# so the measurement and the instruction it enforces cannot drift apart.
AVG_SHOT_S = 12.0     # mean shot ceiling when build-up is deliberately skipped, at pace
# 1.0. Scales with `pace` like every other budget here, so `--pace 0.35` asks for the
# ~4s average a dense clip reel actually runs at. Only measured when `keep_build=False`:
# at the default the opposite flag (`skipped_build`) is the one that matters.
# NOTE this is a SHOT (one kept segment), not a beat -- see `cut_timeline`'s `shots`.
SHORT_CUT_FRAC = 0.8  # finished runtime under this fraction of target = a short cut.
# Deliberately looser than `narrative` tolerance (0.35) because this flag is advice to the
# critic, not a trim: it should fire when the cut is meaningfully thin, not on every run.

# Kept footage a beat may spend, in FINISHED seconds: role -> (typical_min, ceiling).
# One flat ceiling for every role is what starved build-up: an escalation whose comedy IS
# the repetition (failed guesses, a spiral getting worse) has to show the repetition, and
# at 60-90s of allowance the director can only keep an opener and the payoff and skip the
# minutes between -- a jump to the punchline with no build. A connective setup running
# that long is padding. `director` reads this table into the outline/review prompts, so
# the budget the model is told and the budget the audit enforces cannot drift apart.
ROLE_BUDGET = {
    "hook": (12, 35), "setup": (15, 40), "escalation": (30, 90),
    "reversal": (30, 90), "climax": (45, 130), "payoff": (20, 60),
    "button": (15, 45),
}
DEFAULT_BUDGET = (15, 60)   # unknown or blank role

# How far above the pace-1.0 count `pace` may push the beat ask (`director._n_beats`).
# Lives here, next to the budgets it has to stay reconciled with, because the two numbers
# multiply into the finished runtime and drifting apart is exactly how a 16-minute ask
# shipped as 8:53 -- see `fill_pace`.
BEAT_INFLATION_CAP = 1.5


def fill_pace(pace: float = 1.0) -> float:
    """The scale `ROLE_BUDGET` is actually read at, once the beat COUNT has been capped.

    `pace` is supposed to scale the per-beat budgets and the beat count TOGETHER, so their
    product -- the runtime -- stays put and only the cut speed changes. `BEAT_INFLATION_CAP`
    breaks that in one direction: it clamps the count (the director returns ~20-30 beats
    however many are asked for, so a bigger ask is pure token waste) while nothing clamps
    the budgets. At `pace 0.35` on a 16-minute target that left 30 beats x 48s x 0.35 =
    8:24 of allowance -- the run could not reach the target even with every beat at its
    ceiling, and it shipped 8:53 against 16:00.

    The compensation is a pure function of `pace`, not of the target, because the cap is a
    RATIO: once it binds, `n_beats` is `target / SEC_PER_BEAT * CAP`, so the per-beat
    allowance the target implies is `target / (n_beats * SEC_PER_BEAT)` = `1 / CAP`
    exactly. Below that the budgets stop shrinking and the beats simply hold more shots.

    Identity above the knee (pace >= 1/CAP), so a normal-speed cut is byte-identical.
    """
    return max(pace, 1.0 / BEAT_INFLATION_CAP)


def scaled_budget(pace: float = 1.0) -> dict:
    """`ROLE_BUDGET` with both bounds x `pace` -- the cut-speed knob behind `--pace`.

    <1 = snappier (every role gets less room), >1 = room to breathe. Scaling the table is
    what makes cut speed adjustable at all: these numbers are printed verbatim into the
    director prompt AND enforced as the audit ceiling, so an adjective in the style text
    has no leverage over them. Both readers call this, so the two units stay in lockstep
    the way this module's docstring requires.
    """
    if pace == 1.0:
        return dict(ROLE_BUDGET)
    return {r: (lo * pace, hi * pace) for r, (lo, hi) in ROLE_BUDGET.items()}


def _build_gap(beat: dict) -> float:
    """Largest jump between a beat's kept segments that lands BEFORE its payoff.

    A gap AFTER the payoff is a legitimate tail trim; a gap before it means the build-up
    the moment needs was skipped -- the wordle sequence cut straight to its last guess,
    which scores well and plays as a missing scene. With no `payoff_start_s` anchor every
    gap counts, since nothing marks where the point lands.
    """
    segs = beat.get("segments") or []
    payoff = beat.get("payoff_start_s")
    cut = float(payoff) if payoff is not None else float("inf")
    gaps = [float(nxt[0]) - float(cur[1])
            for cur, nxt in zip(segs, segs[1:]) if float(nxt[0]) <= cut]
    return max(gaps, default=0.0)


def _mmss(s) -> str:
    s = max(0, int(s or 0))
    return f"{s // 60:02d}:{s % 60:02d}"


def _span(a: float, b: float) -> str:
    return f"{_mmss(a)}-{_mmss(b)}"


def _shots(beat: dict, dur: float) -> list[float]:
    """A beat's kept segments as FINISHED seconds -- one entry per shot the viewer sees.

    `segments` are raw stream spans and `dur` is the beat after silence compression, so a
    segment's finished length is its share of the beat, prorated. Exact per-segment
    compression would need the word list, which this module deliberately does not take;
    the ratio is uniform enough that the mean this feeds is right to a fraction of a
    second, and the alternative -- measuring shot length as BEAT length, which is what
    `avg_shot` used to do -- is wrong by the number of segments in a beat.
    """
    segs = beat.get("segments") or []
    raw = [float(b) - float(a) for a, b in segs if float(b) > float(a)]
    total = sum(raw)
    if not raw or total <= 0:
        return [dur] if dur > 0 else []
    return [r * dur / total for r in raw]


def cut_timeline(outline_log: dict) -> list[dict]:
    """Beats re-expressed on the finished cut's clock.

    `at` is the beat's position in the OUTPUT video, accumulated from kept footage (`dur`,
    the sum of a beat's segments) rather than its source envelope -- dead air and
    jump-cut gaps never reach the viewer, so they must not count toward the timeline the
    critic reasons about. Beats are taken in list order, which `narrative.cast` has
    already put in play order.
    """
    out: list[dict] = []
    # whatever plays before beat 1 -- one flash-forward teaser or a whole montage stack --
    # pushes every beat that much later in the finished video than its kept footage alone
    # would put it. `cold_open` is a LIST; a bare dict is an outline from before it was.
    cold = outline_log.get("cold_open") or []
    at = sum(float(c.get("dur") or 0.0)
             for c in ([cold] if isinstance(cold, dict) else cold))
    for i, b in enumerate(outline_log.get("beats") or [], 1):
        start, end = b.get("start"), b.get("end")
        dur = float(b.get("dur") or 0.0)
        if dur <= 0 and start is not None and end is not None:
            dur = max(0.0, float(end) - float(start))
        n_words = len(str(b.get("text") or "").split())
        # how long the viewer waits for the point of the beat to land
        payoff, lead = b.get("payoff_start_s"), None
        if payoff is not None and start is not None and float(payoff) >= float(start):
            lead = float(payoff) - float(start)
        out.append({
            "n": i,
            "title": b.get("title", ""),
            "role": b.get("role", ""),
            "texture": b.get("texture", ""),
            "energy": int(b.get("energy") or 3),
            "at": at,                       # position in the finished cut
            "dur": dur,                     # kept footage
            "end_at": at + dur,
            "src": (start, end),            # where it came from in the stream
            "wps": (n_words / dur) if dur > 0 else 0.0,
            "setup_lead": lead,
            "build_gap": _build_gap(b),     # build-up skipped inside the beat
            "shots": _shots(b, dur),        # per-segment finished lengths (avg_shot)
        })
        at += dur
    return out


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def _flat_runs(timeline: list[dict]) -> list[dict]:
    """Maximal runs of consecutive low-energy beats that occupy real screen time.

    One dull beat is texture; two and a half minutes of them is the sag. Only runs at or
    past `FLAT_RUN_S` of cut time are reported, longest first.
    """
    runs, cur = [], []
    for b in timeline + [None]:                     # sentinel closes a trailing run
        if b is not None and b["energy"] <= FLAT_ENERGY:
            cur.append(b)
            continue
        if cur:
            length = sum(x["dur"] for x in cur)
            if length >= FLAT_RUN_S:
                runs.append({"beats": cur, "at": cur[0]["at"],
                             "end_at": cur[-1]["end_at"], "len": length})
            cur = []
    return sorted(runs, key=lambda r: -r["len"])


def audit(timeline: list[dict], pace: float = 1.0, keep_build: bool = True,
          stack: int = 0, target_s: float = 0.0) -> list[dict]:
    """Measured pacing problems, worst first. Each flag is `{kind, at, text}`.

    `text` is written to be pasted straight into a prompt or a markdown report -- it names
    a position in the finished cut and a number, so the critic can act on a specific
    stretch instead of re-judging the whole thing.

    `keep_build=False` silences the two flags that assume a moment should be allowed to
    land -- `skipped_build` and `slow_payoff`. They are correct for a story and exactly
    backwards for a cut built on skipped setup and premature cuts, where opening
    mid-sequence IS the style. A measurement that fires on every beat of a deliberate
    style is noise, and noise is what teaches the critic to ignore the audit.

    `target_s` (the run's requested runtime) enables the `short_cut` flag. Without it the
    audit could measure everything about a cut EXCEPT that it is half the length it was
    asked for -- which is the one defect the critic will otherwise never act on, because
    the rubric tells it the target is a rough guide.

    Role ceilings are read at `fill_pace(pace)`, not `pace`: the budgets and the beat
    count have to multiply back out to the target (see `fill_pace`), and grading beats
    against a ceiling the runtime cannot afford is what asked for a 16-minute cut and
    flagged it for being over-long at 8:53.
    """
    flags: list[dict] = []
    if not timeline:
        return flags
    total = timeline[-1]["end_at"]
    fill = fill_pace(pace)
    budget = scaled_budget(fill)

    for r in _flat_runs(timeline):
        ns = ", ".join(f"#{b['n']}" for b in r["beats"])
        flags.append({
            "kind": "flat_run", "at": r["at"], "ns": [b["n"] for b in r["beats"]],
            "text": (f"{_span(r['at'], r['end_at'])} ({int(r['len'])}s, "
                     f"{len(r['beats'])} beats: {ns}) never rises above energy "
                     f"{FLAT_ENERGY} -- this is the flattest stretch of the cut."),
        })

    for b in timeline:
        ceiling = budget.get(b["role"], (DEFAULT_BUDGET[0] * fill,
                                         DEFAULT_BUDGET[1] * fill))[1]
        if b["dur"] > ceiling:
            flags.append({
                "kind": "long_beat", "at": b["at"], "ns": [b["n"]],
                "text": (f"beat #{b['n']} at {_mmss(b['at'])} ('{b['title']}') keeps "
                         f"{int(b['dur'])}s against a {int(ceiling)}s ceiling for a "
                         f"{b['role'] or 'plain'} beat -- tighten it or re-role it."),
            })

    if keep_build:
        for b in timeline:
            if b["setup_lead"] is not None and b["setup_lead"] > SLOW_PAYOFF_S:
                flags.append({
                    "kind": "slow_payoff", "at": b["at"], "ns": [b["n"]],
                    "text": (f"beat #{b['n']} at {_mmss(b['at'])} runs "
                             f"{int(b['setup_lead'])}s of setup before its payoff lands."),
                })

        for b in timeline:
            if b["build_gap"] > BUILD_GAP_S:
                flags.append({
                    "kind": "skipped_build", "at": b["at"], "ns": [b["n"]],
                    "text": (f"beat #{b['n']} at {_mmss(b['at'])} ('{b['title']}') jumps "
                             f"{int(b['build_gap'])}s of its own build-up before the "
                             f"payoff -- it opens mid-sequence, with the setup skipped."),
                })
    else:
        # The mirror measurement: with build-up deliberately skipped, the failure mode
        # flips from "the setup was deleted" to "this stopped being dense". Both numbers
        # come straight from the style doc's own reject criteria.
        # A SHOT is one kept segment, not a beat. Measuring beats made this flag read
        # the wrong number twice over: it reported a 30-beat cut's 17.4s mean beat as its
        # "average shot" (the real shots averaged 12.2s), and it could only ever be
        # cleared by shortening BEATS -- i.e. by shipping less video -- when the fix is
        # to split the same footage into more cuts.
        shots = [s for b in timeline for s in (b.get("shots") or []) if s > 0]
        ceiling = AVG_SHOT_S * pace
        mean = sum(shots) / len(shots) if shots else 0.0
        if mean > ceiling:
            want = int(sum(shots) / ceiling)
            flags.append({
                "kind": "avg_shot", "at": 0.0, "ns": [b["n"] for b in timeline],
                "text": (f"average shot runs {mean:.1f}s across {len(shots)} segments "
                         f"against a {ceiling:.1f}s ceiling for this cut speed -- the cut "
                         f"is not dense enough for the direction it was given. This much "
                         f"footage wants about {want} segments, not {len(shots)}: split "
                         f"the long beats into more `segments`, do not shorten them."),
            })

    # Runtime. The critic is told (correctly) that the target is a rough guide, so
    # nothing in the rubric reacts to a cut that came in at 55% of it -- and the shortfall
    # is invisible in the beat list, where every individual beat looks reasonable. Stated
    # as a fact with the fix attached, because "add beats" is the wrong one: an
    # under-filled cut is beats that kept 12s of a 30s allowance.
    if target_s > 0 and total < target_s * SHORT_CUT_FRAC:
        flags.append({
            "kind": "short_cut", "at": 0.0, "ns": [b["n"] for b in timeline],
            "text": (f"finished runtime {_mmss(total)} against a {_mmss(target_s)} target "
                     f"-- {100 * (1 - total / target_s):.0f}% short. The beats are "
                     f"under-filled, not too few: widen the strongest ones toward their "
                     f"role ceilings with MORE `segments` (never longer ones), and add "
                     f"beats only where the footage genuinely supports another one. Do "
                     f"not pad with slow footage."),
        })

    # Thin beats: measured against this cut's own median, so a naturally quiet stream
    # doesn't flag wholesale and a chatty one still surfaces its dead patches.
    dense = [b["wps"] for b in timeline if b["dur"] >= MIN_DENSITY_S and b["wps"] > 0]
    med = _median(dense)
    if med > 0:
        for b in timeline:
            if b["dur"] >= MIN_DENSITY_S and b["wps"] < med * DENSITY_FRAC:
                flags.append({
                    "kind": "low_density", "at": b["at"], "ns": [b["n"]],
                    "text": (f"beat #{b['n']} at {_mmss(b['at'])} says "
                             f"{b['wps']:.1f} words/s against a cut median of "
                             f"{med:.1f} -- thin, mostly dead air or silence."),
                })

    # A cut whose two biggest peaks both land early has nothing left to climb toward;
    # the sag is then structural, not a tightening problem.
    if len(timeline) >= 4 and total > 0:
        peaks = sorted(timeline, key=lambda b: (-b["energy"], b["at"]))[:2]
        if all((p["at"] + p["dur"] / 2) < total * FRONT_LOAD_FRAC for p in peaks):
            flags.append({
                "kind": "peak_position", "at": peaks[0]["at"], "ns": [p["n"] for p in peaks],
                "text": (f"both highest-energy beats (#{peaks[0]['n']}, #{peaks[1]['n']}) "
                         f"land in the first third of a {_mmss(total)} cut -- everything "
                         f"after them is downhill."),
            })

    for a, b in zip(timeline, timeline[1:]):
        if a["role"] and a["role"] == b["role"] and a["texture"] and a["texture"] == b["texture"]:
            flags.append({
                "kind": "sameness", "at": b["at"], "ns": [a["n"], b["n"]],
                "text": (f"beats #{a['n']} and #{b['n']} at {_mmss(a['at'])} are both "
                         f"{a['role']}/{a['texture']} back to back."),
            })

    order = {"short_cut": 0, "flat_run": 1, "peak_position": 2, "stack_order": 3,
             "skipped_build": 4, "avg_shot": 4, "long_beat": 5, "low_density": 6,
             "slow_payoff": 7, "sameness": 8}
    return sorted(flags, key=lambda f: (order.get(f["kind"], 9), f["at"]))


def _stack_flag(cold: list[dict]) -> dict | None:
    """A montage stack that doesn't escalate. The style doc's own reject criterion: a
    stack whose strongest moment isn't last has nowhere to climb and reads as an ordinary
    intro -- the one thing the move exists to avoid."""
    if len(cold) < 2:
        return None
    energies = [float(c.get("energy") or 0) for c in cold]
    if max(energies) <= 0 or energies[-1] >= max(energies):
        return None
    best = energies.index(max(energies)) + 1
    return {
        "kind": "stack_order", "at": 0.0, "ns": [],
        "text": (f"the cold-open stack's strongest moment is #{best} of {len(cold)}, "
                 f"not the last -- it peaks early and then coasts. Reorder the "
                 f"cold_open list so the best moment is last."),
    }


def pacing_note(flags: list[dict], total_s: float = 0.0) -> str:
    """The audit as a prompt/report block. Empty audit still says so -- silence would read
    as "not measured" and invite the critic to invent pacing problems."""
    head = "PACING AUDIT (measured from the realized cut, not opinion)"
    if total_s:
        head += f" -- finished runtime {_mmss(total_s)}"
    if not flags:
        return head + ":\n- no flat runs, over-long beats, or thin patches measured."
    return head + ":\n" + "\n".join(f"- {f['text']}" for f in flags)


def audit_note(outline_log: dict, pace: float = 1.0, keep_build: bool = True,
               target_s: float = 0.0) -> str:
    """`outline_log` -> the audit block, in one call (the form every caller wants).

    `target_s` is the run's requested runtime; 0 (the default) simply omits the
    `short_cut` flag, so every existing caller keeps its old output verbatim.
    """
    tl = cut_timeline(outline_log)
    flags = audit(tl, pace, keep_build, target_s=target_s)
    cold = outline_log.get("cold_open") or []
    stack = _stack_flag([cold] if isinstance(cold, dict) else cold)
    if stack:
        flags.insert(0, stack)
    return pacing_note(flags, tl[-1]["end_at"] if tl else 0.0)
