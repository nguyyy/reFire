"""Organize selected clips into ordered, titled sections for a coherent reel.

One local-LLM pass groups the chosen highlight segments under the user-provided
topic and orders them into a watchable narrative. The topic is context for
grouping/ordering, NOT a filter -- every chosen clip lands in some section.
"""
from __future__ import annotations

import json
from typing import TypedDict

from .score import DEFAULT_MODEL


class Section(TypedDict):
    title: str
    clips: list


_SYSTEM = (
    "You are editing a Twitch highlights compilation. You are given the stream's "
    "topic and a numbered list of highlight clips (each with a one-line reason and "
    "its time). Group the clips into a few titled sections and order the sections "
    "and the clips within them so the reel plays as one coherent, watchable story. "
    "Use every clip exactly once. Reply ONLY with JSON: "
    '{"sections": [{"title": "<short>", "clips": [<clip numbers>]}]}.'
)


def _fallback(clips) -> list[Section]:
    """One untitled section, chronological -- the reel always builds."""
    return [{"title": "", "clips": sorted(clips, key=lambda c: c["start"])}]


def organize(clips, topic: str, model: str = DEFAULT_MODEL) -> list[Section]:
    """Chosen clips -> ordered titled sections. Falls back to one chrono section."""
    if not clips:
        return []
    if not topic:
        return _fallback(clips)

    listing = "\n".join(
        f"{i}: [{c['start']:.0f}-{c['end']:.0f}s] {c.get('reason', '')}"
        for i, c in enumerate(clips)
    )
    try:
        import ollama  # local import: optional heavy dep
        resp = ollama.chat(
            model=model, format="json",
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": f"Topic: {topic}\n\nClips:\n{listing}"}],
        )
        data = json.loads(resp["message"]["content"])
        sections: list[Section] = []
        used: set[int] = set()
        for sec in data["sections"]:
            idxs = [i for i in sec.get("clips", []) if 0 <= i < len(clips)]
            picked = [clips[i] for i in idxs if i not in used]
            used.update(idxs)
            if picked:
                sections.append({"title": str(sec.get("title", "")), "clips": picked})
        leftover = [c for i, c in enumerate(clips) if i not in used]
        if leftover:   # never drop a clip the model forgot
            sections.append({"title": "", "clips": sorted(leftover, key=lambda c: c["start"])})
        return sections or _fallback(clips)
    except Exception:
        # ponytail: any failure (no ollama, bad JSON, connection) -> still ship a reel
        return _fallback(clips)


def _demo() -> None:
    clips = [{"start": 30.0, "end": 40.0, "reason": "a"},
             {"start": 10.0, "end": 20.0, "reason": "b"}]
    assert organize([], "x") == []
    secs = organize(clips, "")            # no topic -> chrono fallback, no LLM
    assert len(secs) == 1 and secs[0]["clips"][0]["start"] == 10.0, secs
    print("organize fallback ok")


if __name__ == "__main__":
    _demo()
