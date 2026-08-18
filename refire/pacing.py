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
LONG_BEAT_S = 75.0    # kept footage past this wants tightening (prompt guidance says ~90 max)
SLOW_PAYOFF_S = 12.0  # this much setup before the point lands is where viewers leave
DENSITY_FRAC = 0.6    # words/sec under this fraction of the cut's median reads as thin
MIN_DENSITY_S = 20.0  # short beats have noisy words/sec -- don't flag them
FRONT_LOAD_FRAC = 1 / 3.0  # both peaks inside this fraction of the runtime = front-loaded


def _mmss(s) -> str:
    s = max(0, int(s or 0))
    return f"{s // 60:02d}:{s % 60:02d}"


def _span(a: float, b: float) -> str:
    return f"{_mmss(a)}-{_mmss(b)}"


def cut_timeline(outline_log: dict) -> list[dict]:
    """Beats re-expressed on the finished cut's clock.

    `at` is the beat's position in the OUTPUT video, accumulated from kept footage (`dur`,
    the sum of a beat's segments) rather than its source envelope -- dead air and
    jump-cut gaps never reach the viewer, so they must not count toward the timeline the
    critic reasons about. Beats are taken in list order, which `narrative.cast` has
    already put in play order.
    """
    out: list[dict] = []
    # the flash-forward teaser plays before beat 1, so every beat sits that much later in
    # the finished video than its kept footage alone would put it
    at = float((outline_log.get("cold_open") or {}).get("dur") or 0.0)
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


def audit(timeline: list[dict]) -> list[dict]:
    """Measured pacing problems, worst first. Each flag is `{kind, at, text}`.

    `text` is written to be pasted straight into a prompt or a markdown report -- it names
    a position in the finished cut and a number, so the critic can act on a specific
    stretch instead of re-judging the whole thing.
    """
    flags: list[dict] = []
    if not timeline:
        return flags
    total = timeline[-1]["end_at"]

    for r in _flat_runs(timeline):
        ns = ", ".join(f"#{b['n']}" for b in r["beats"])
        flags.append({
            "kind": "flat_run", "at": r["at"], "ns": [b["n"] for b in r["beats"]],
            "text": (f"{_span(r['at'], r['end_at'])} ({int(r['len'])}s, "
                     f"{len(r['beats'])} beats: {ns}) never rises above energy "
                     f"{FLAT_ENERGY} -- this is the flattest stretch of the cut."),
        })

    for b in timeline:
        if b["dur"] > LONG_BEAT_S:
            flags.append({
                "kind": "long_beat", "at": b["at"], "ns": [b["n"]],
                "text": (f"beat #{b['n']} at {_mmss(b['at'])} ('{b['title']}') keeps "
                         f"{int(b['dur'])}s -- the longest single beat should earn it."),
            })

    for b in timeline:
        if b["setup_lead"] is not None and b["setup_lead"] > SLOW_PAYOFF_S:
            flags.append({
                "kind": "slow_payoff", "at": b["at"], "ns": [b["n"]],
                "text": (f"beat #{b['n']} at {_mmss(b['at'])} runs "
                         f"{int(b['setup_lead'])}s of setup before its payoff lands."),
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

    order = {"flat_run": 0, "peak_position": 1, "long_beat": 2,
             "low_density": 3, "slow_payoff": 4, "sameness": 5}
    return sorted(flags, key=lambda f: (order.get(f["kind"], 9), f["at"]))


def pacing_note(flags: list[dict], total_s: float = 0.0) -> str:
    """The audit as a prompt/report block. Empty audit still says so -- silence would read
    as "not measured" and invite the critic to invent pacing problems."""
    head = "PACING AUDIT (measured from the realized cut, not opinion)"
    if total_s:
        head += f" -- finished runtime {_mmss(total_s)}"
    if not flags:
        return head + ":\n- no flat runs, over-long beats, or thin patches measured."
    return head + ":\n" + "\n".join(f"- {f['text']}" for f in flags)


def audit_note(outline_log: dict) -> str:
    """`outline_log` -> the audit block, in one call (the form both callers want)."""
    tl = cut_timeline(outline_log)
    return pacing_note(audit(tl), tl[-1]["end_at"] if tl else 0.0)
