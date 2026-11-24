"""Premiere's menu bar from outside, for the commands ExtendScript lacks (spike S8).

The scripting DOM can make a caption track (`Sequence.createCaptionTrack`) but has no "Upgrade
Caption to Graphic"; the menu does, and on Windows the menu bar is a native HMENU, so a
WM_COMMAND to the main window runs it like a click. Items are found by label (ids change
between sessions). Menu commands act on the focused panel: Window > Timelines > <seq> focuses
that timeline first, or Select All selects bin items instead.

    python -m refire.ppro.menu upgrade refire_captions
"""
from __future__ import annotations

import sys

WM_COMMAND = 0x0111


def label(s: str) -> str:
    """'Select &All\\tCtrl+A' -> 'Select All' (mnemonic and shortcut dropped)."""
    return s.split("\t")[0].replace("&", "")


def find(tree: list, path: list[str]) -> int:
    """command id of `path` in [(label, id, children)], or KeyError naming the missing step."""
    for i, step in enumerate(path):
        hit = next((n for n in tree if label(n[0]) == step), None)
        if hit is None:
            raise KeyError(" > ".join(path[:i + 1]))
        if i == len(path) - 1:
            return hit[1]
        tree = hit[2]
    raise KeyError("empty path")


def _tree(hmenu, depth=0) -> list:
    import ctypes
    u = ctypes.windll.user32
    out = []
    for i in range(u.GetMenuItemCount(hmenu)):
        buf = ctypes.create_unicode_buffer(256)
        u.GetMenuStringW(hmenu, i, buf, 256, 0x400)                 # MF_BYPOSITION
        sub = u.GetSubMenu(hmenu, i)
        out.append((buf.value, u.GetMenuItemID(hmenu, i), _tree(sub, depth + 1) if sub and depth < 3 else []))
    return out


def run(*paths: list[str], timeout_ms: int = 600_000) -> None:
    """send each menu path to Premiere in order; each waits until Premiere has handled it."""
    import ctypes
    import ctypes.wintypes as W
    u = ctypes.windll.user32
    hwnd = u.FindWindowW("Premiere Pro", None)
    if not hwnd:
        raise RuntimeError("Premiere's main window not found")
    tree = _tree(u.GetMenu(hwnd))
    for path in paths:
        res = W.DWORD()
        if not u.SendMessageTimeoutW(hwnd, WM_COMMAND, find(tree, path), 0, 2, timeout_ms, ctypes.byref(res)):
            raise RuntimeError("Premiere did not answer " + " > ".join(path))


def upgrade_captions(seq: str, back: str = "") -> None:
    """every caption on `seq`'s caption tracks -> graphic clips, one new video track per caption
    track, top-most last (S8: ~1200 captions in under 3 s). Lock video tracks first, or Select
    All takes their clips too and the upgrade does nothing. Focusing a timeline also makes it
    the active sequence, so `back` (main) is focused again after."""
    run(["Window", "Timelines", seq], ["Edit", "Deselect All"], ["Edit", "Select All"],
        ["Graphics and Titles", "Upgrade Caption to Graphic"], ["Edit", "Deselect All"],
        *([["Window", "Timelines", back]] if back else []))


COMMANDS = {"upgrade_captions": upgrade_captions}


if __name__ == "__main__":
    name, *args = sys.argv[1:]
    COMMANDS[{"upgrade": "upgrade_captions"}.get(name, name)](*args)
    print(f"menu: {name} done")
