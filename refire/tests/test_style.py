"""Style pass: role -> presentation knobs, and how they reach the AE manifest."""
import json

from refire import overlay
from refire.ae_export import build_manifest
from refire.style import NEUTRAL, style_for


def test_style_for_role_and_energy():
    assert style_for("")["zoom_sens"] == 1.0 and style_for("")["card"] is True   # neutral
    assert style_for("hook")["card"] is False                                    # no title card
    assert style_for("hook")["words_per_line"] < NEUTRAL["words_per_line"]        # faster caps
    assert style_for("climax")["zoom_sens"] > style_for("setup")["zoom_sens"]     # punchier
    assert style_for("button")["overlay_density"] < style_for("climax")["overlay_density"]
    assert style_for("climax", 5)["zoom_sens"] >= style_for("climax", 1)["zoom_sens"]  # energy
    assert style_for("???")["zoom_sens"] == 1.0                                   # safe default


def test_overlay_density_by_role():
    assert overlay._overlay_density({}) == 1.0                                    # no role
    assert (overlay._overlay_density({"role": "climax"})
            > overlay._overlay_density({"role": "button"}))


def _words(text):
    # 0.5s apart, 0.4s long -> gaps 0.1s (< PAUSE_GAP) so only words_per_line breaks lines
    return [{"text": t, "start": i * 0.5, "end": i * 0.5 + 0.4, "emph": False}
            for i, t in enumerate(text.split())]


def test_build_manifest_applies_role_style(tmp_path):
    """A hook section: no card, role on the clip, captions paced by the role (wpl=2)."""
    words = _words("a b c d e f")                       # 6 words within [0, 4)
    sections = [{"title": "Cold Open", "role": "hook", "energy": 5,
                 "clips": [{"start": 0.0, "end": 4.0, "role": "hook", "energy": 5}]}]
    mp = build_manifest("x.mp4", tmp_path, words, sections, words_per_line=3,
                        motion_zoom=False)             # motion_zoom off -> no cv2 needed
    m = json.loads(mp.read_text())
    assert m["clips"][0]["role"] == "hook"
    assert m["clips"][0]["zoom_episodes"] == []        # motion scan skipped
    assert m["sections"][0]["card"] is False           # hook -> minimal, no title card
    assert len(m["clips"][0]["captions"]) == 3         # wpl=2 (hook), not the passed 3


def test_build_manifest_no_role_is_unchanged(tmp_path):
    """A roleless section (legacy/flat) keeps the old schema: no role/card keys, passed wpl."""
    words = _words("a b c d e f")
    sections = [{"title": "", "clips": [{"start": 0.0, "end": 4.0}]}]
    mp = build_manifest("x.mp4", tmp_path, words, sections, words_per_line=3,
                        motion_zoom=False)
    m = json.loads(mp.read_text())
    assert "role" not in m["clips"][0]
    assert "card" not in m["sections"][0]
    assert len(m["clips"][0]["captions"]) == 2         # passed wpl=3 -> 2 lines of 3
