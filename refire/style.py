"""Role -> presentation: let the story drive the style (the doc's "Style Pass").

Story comes before style. Once the director has cast each beat into a story `role`
(hook, setup, escalation, reversal, climax, payoff, button) the cut's presentation
should follow it -- a hook starts hot with fast captions and a punch zoom, a setup
breathes with fewer overlays, a climax stacks zoom + SFX + overlays, a button stays
out of the way of the final laugh. This module is the single mapping; `ae_export`,
`overlay`, and the rough render apply it. A clip with no role (legacy detect path /
flat fallback) gets the NEUTRAL identity, so those paths are unchanged.

Knobs (per clip):
  zoom_sens       multiplier on motion-zoom sensitivity (>1 = more/earlier punches)
  overlay_density multiplier on how many emote/SFX punch-ins this clip earns (0 = none)
  words_per_line  caption pacing (fewer = faster, snappier captions)
  card            show a section title card (hook = minimal, no card)
"""
from __future__ import annotations

NEUTRAL = {"zoom_sens": 1.0, "overlay_density": 1.0, "words_per_line": 3, "card": True}

# baselines from the style pass doc, energy nudges zoom/overlay around these
_ROLE_STYLE = {
    "hook":       {"zoom_sens": 1.3, "overlay_density": 1.0, "words_per_line": 2, "card": False},
    "setup":      {"zoom_sens": 0.8, "overlay_density": 0.4, "words_per_line": 3, "card": True},
    "escalation": {"zoom_sens": 1.1, "overlay_density": 1.0, "words_per_line": 3, "card": True},
    "reversal":   {"zoom_sens": 1.2, "overlay_density": 1.1, "words_per_line": 3, "card": True},
    "climax":     {"zoom_sens": 1.5, "overlay_density": 1.6, "words_per_line": 2, "card": True},
    "payoff":     {"zoom_sens": 1.3, "overlay_density": 1.3, "words_per_line": 2, "card": True},
    "button":     {"zoom_sens": 0.9, "overlay_density": 0.3, "words_per_line": 3, "card": True},
}


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def style_for(role: str, energy: int = 3) -> dict:
    """Presentation knobs for a beat's `role`, tilted by its 1-5 `energy`.

    Unknown/empty role -> NEUTRAL (identity), so non-narrative clips render as before.
    energy 3 is neutral; higher pushes zoom + overlay density up, lower calms them.
    """
    base = _ROLE_STYLE.get((role or "").strip().lower())
    if base is None:
        return dict(NEUTRAL)
    s = dict(base)
    e = _clamp(int(energy or 3), 1, 5)
    s["zoom_sens"] = round(_clamp(s["zoom_sens"] + (e - 3) * 0.06, 0.5, 2.0), 3)
    s["overlay_density"] = round(_clamp(s["overlay_density"] + (e - 3) * 0.12, 0.0, 2.0), 3)
    return s


def _demo() -> None:
    assert style_for("")["zoom_sens"] == 1.0 and style_for("")["card"] is True   # neutral
    assert style_for("hook")["card"] is False                                    # no card
    assert style_for("hook")["words_per_line"] < NEUTRAL["words_per_line"]        # faster caps
    assert style_for("climax")["zoom_sens"] > style_for("setup")["zoom_sens"]     # punchier
    assert style_for("button")["overlay_density"] < style_for("climax")["overlay_density"]
    # energy tilts around the baseline, clamped
    assert style_for("climax", 5)["zoom_sens"] >= style_for("climax", 3)["zoom_sens"]
    assert style_for("setup", 1)["overlay_density"] >= 0.0
    assert style_for("unknown role")["zoom_sens"] == 1.0                          # safe default
    print("style ok")


if __name__ == "__main__":
    _demo()
