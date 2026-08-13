"""Pick emoji/gif overlay punch-ins + impact SFX for the funniest moments.

One overlay max per clip, placed on the clip's strongest keyword-emphasized word
(pog/lol/clutch/...), kept only for the top clips (~1 per 30s of footage) so they
land on spikes, not everywhere. The emote is chosen by the LLM to fit the moment's
vibe (clutch -> pog, fail -> sadge), preferring the local pool's actual filenames;
SFX are cycled uniquely. Emote files come from the local pool, else BetterTTV
(gif/png — the AE importer takes both; 7TV's webp/avif would need conversion).

Asset folders are scanned recursively and by name, so the user's layout works as-is:
  assets/emotes/**            -> emote images (png/jpg/gif; avif/webp skipped — AE can't import)
  assets/sfx/** (non-music)   -> impact SFX
  assets/bgm/**, assets/sfx/**background|music|song** -> music beds
"""
from __future__ import annotations

import json
import random
import urllib.parse
import urllib.request
from pathlib import Path

from .emphasis import _CLEAN, _is_keyword  # reuse keyword detection
from .score import DEFAULT_MODEL
from .style import style_for


def _overlay_density(clip) -> float:
    """Role-driven overlay/SFX density for a clip (1.0 when it carries no story role)."""
    role = clip.get("role")
    return style_for(role, clip.get("energy", 3))["overlay_density"] if role else 1.0

OVERLAY_S = 3.0                 # each impact punch-in lasts ~3s
SECONDS_PER_OVERLAY = 30.0      # density cap: ~1 overlay per 30s of footage
_IMG_EXTS = (".png", ".gif", ".jpg", ".jpeg")     # AE-importable stills/anims
_AUDIO_EXTS = (".wav", ".mp3")
_BGM_HINT = ("background", "music", "song", "bgm")  # folder-name hints for the music bed
_BTTV_SEARCH = "https://api.betterttv.net/3/emotes/shared/search?query={q}&offset=0&limit=1"
_BTTV_CDN = "https://cdn.betterttv.net/emote/{id}/3x.{ext}"


def _posix(p) -> str:
    return str(Path(p).resolve()).replace("\\", "/")


def _scan(folder, exts, exclude_hint=(), only_hint=()):
    """Recursively collect files with `exts`, filtered by folder-name hints."""
    folder = Path(folder)
    if not folder.exists():
        return []
    out = []
    for p in folder.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        # match hints on FOLDER names only -- a meme sfx named "clownMusic.mp3" must
        # not count as a music bed just because its filename contains "music".
        parts = " ".join(p.parent.parts).lower()
        if exclude_hint and any(h in parts for h in exclude_hint):
            continue
        if only_hint and not any(h in parts for h in only_hint):
            continue
        out.append(p)
    return out


def _emote_dirs(assets_dir):
    return [Path(assets_dir) / "emotes", Path(assets_dir) / "overlays"]


def _cache_dir(assets_dir):
    return Path(assets_dir) / "emotes" / "_cache"


class _Cycler:
    """Yield items shuffled, with no repeat until the pool is exhausted."""

    def __init__(self, items):
        self._items = list(items)
        self._bag: list = []

    def next(self):
        if not self._items:
            return None
        if not self._bag:
            self._bag = self._items[:]
            random.shuffle(self._bag)
        return self._bag.pop()


def _clip_keyword_moment(words, start, end):
    """Strongest keyword-emphasized word in [start, end) -> (t_rel, word, context) or None."""
    hits = [w for w in words
            if w.get("emph") and start <= w["start"] < end and _is_keyword(w["text"])]
    if not hits:
        return None
    w = hits[0]                 # first strong beat in the clip; they're all hot
    ctx = " ".join(x["text"] for x in words
                   if w["start"] - 6 <= x["start"] <= w["start"] + 6)
    word = w["text"].lower().translate(_CLEAN)
    return round(w["start"] - start, 3), word, ctx


def _llm_emotes(moments, pool_names, model):
    """[{word, context}] -> [emote_name] aligned, preferring `pool_names`.

    Any failure -> the keyword itself (often already an emote name).
    """
    fallback = [m["word"] for m in moments]
    if not moments:
        return []
    try:
        import ollama  # optional heavy dep
        avail = ", ".join(sorted(pool_names)) or "(none)"
        system = (
            "You pick ONE Twitch emote that best fits the vibe of each moment from a "
            "gaming stream (a clutch play -> 'pog', a death/fail -> 'sadge', something "
            "funny -> 'kekw'). PREFER a name from AVAILABLE; only invent another "
            "well-known emote name if nothing available fits. AVAILABLE: " + avail + ". "
            "Reply ONLY with JSON: {\"emotes\": [\"<name>\", ...]} — one lowercase name "
            "per moment, in order."
        )
        user = json.dumps([{"said": m["word"], "context": m["context"]} for m in moments])
        resp = ollama.chat(model=model, format="json", messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        names = json.loads(resp["message"]["content"]).get("emotes", [])
        return [str(names[i]).strip().lower() if i < len(names) and names[i] else fallback[i]
                for i in range(len(moments))]
    except Exception:           # ponytail: LLM down/bad JSON -> keyword as emote name
        return fallback


def _find_local(name, folders):
    for folder in folders:
        folder = Path(folder)
        if not folder.exists():
            continue
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in _IMG_EXTS and p.stem.lower() == name.lower():
                return p
    return None


def _download_bttv(name, cache_dir):
    """Search BetterTTV for `name`, download the top emote (gif/png) -> path or None."""
    try:
        url = _BTTV_SEARCH.format(q=urllib.parse.quote(name))
        req = urllib.request.Request(url, headers={"User-Agent": "reFire"})
        with urllib.request.urlopen(req, timeout=10) as r:
            hits = json.loads(r.read().decode("utf-8"))
        if not hits:
            return None
        eid = hits[0]["id"]
        ext = "gif" if hits[0].get("imageType") == "gif" else "png"
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        dst = cache_dir / f"{name}.{ext}"
        cdn = _BTTV_CDN.format(id=eid, ext=ext)
        req2 = urllib.request.Request(cdn, headers={"User-Agent": "reFire"})
        with urllib.request.urlopen(req2, timeout=20) as r, open(dst, "wb") as f:
            f.write(r.read())
        return dst
    except Exception:           # ponytail: BTTV/offline -> caller falls back to local pool
        return None


def resolve_emote(name, emote_dirs, cache_dir, pool):
    """name -> image path. local pool -> cache -> BTTV -> random local fallback (or None)."""
    p = _find_local(name, emote_dirs) or _find_local(name, [cache_dir])
    if p:
        return p
    p = _download_bttv(name, cache_dir)
    if p:
        return p
    return random.choice(pool) if pool else None


def pick_bgm(assets_dir, override=None):
    """Path to the music bed: explicit override, else a random track (bgm/ or *music* folders)."""
    if override:
        return Path(override)
    files = (_scan(Path(assets_dir) / "bgm", _AUDIO_EXTS)
             + _scan(Path(assets_dir) / "sfx", _AUDIO_EXTS, only_hint=_BGM_HINT))
    return random.choice(files) if files else None


def pick_overlays(words, clips, assets_dir, model=DEFAULT_MODEL, sfx: bool = True):
    """Per-clip overlay lists aligned with `clips` (empty = no overlay for that clip).

    Each overlay is {start (clip-relative s), duration, asset (abs posix), sfx (abs posix
    or None)}. Only the top ~1-per-30s clips that contain a keyword beat get one.
    `sfx=False` places the emotes silently (no impact hits).
    """
    assets_dir = Path(assets_dir)
    emote_dirs = _emote_dirs(assets_dir)
    cache_dir = _cache_dir(assets_dir)
    pool = [p for d in emote_dirs for p in _scan(d, _IMG_EXTS)]
    pool_names = {p.stem for p in pool}
    # impact SFX = everything under assets/sfx that isn't a music/background track
    sfx_files = _scan(assets_dir / "sfx", _AUDIO_EXTS, exclude_hint=_BGM_HINT) if sfx else []

    cand = [_clip_keyword_moment(words, c["start"], c["end"]) for c in clips]
    total_s = sum(c["end"] - c["start"] for c in clips)
    budget = max(1, int(total_s / SECONDS_PER_OVERLAY))
    # Style pass: a clip's story role tilts overlay/SFX density -- a climax stacks them,
    # a button stays out of the final laugh (density 0 opts out). Rank by score * density
    # so the hottest, highest-energy beats win the limited overlay budget. Clips with no
    # role (legacy/flat) get density 1.0 -> ranked by score alone, as before.
    ranked = sorted((i for i, m in enumerate(cand) if m and _overlay_density(clips[i]) > 0),
                    key=lambda i: clips[i].get("score", 0.0) * _overlay_density(clips[i]),
                    reverse=True)
    chosen = sorted(ranked[:budget])
    if not chosen:
        return [[] for _ in clips]

    emotes = _llm_emotes([{"word": cand[i][1], "context": cand[i][2]} for i in chosen],
                         pool_names, model)
    sfx_cycle = _Cycler(sfx_files)
    out = [[] for _ in clips]
    for name, i in zip(emotes, chosen):
        asset = resolve_emote(name, emote_dirs, cache_dir, pool)
        if not asset:
            continue
        sfx = sfx_cycle.next()
        out[i] = [{"start": cand[i][0], "duration": OVERLAY_S,
                   "asset": _posix(asset), "sfx": _posix(sfx) if sfx else None}]
    return out


def _demo() -> None:
    cy = _Cycler(["a", "b", "c"])
    first = sorted(cy.next() for _ in range(3))
    assert first == ["a", "b", "c"], first          # no repeat within a cycle
    assert cy.next() in {"a", "b", "c"}              # next cycle reshuffles the full pool
    assert _Cycler([]).next() is None

    words = [{"text": "POG", "start": 5.0, "end": 5.2, "emph": True},
             {"text": "hello", "start": 6.0, "end": 6.2, "emph": False}]
    m = _clip_keyword_moment(words, 0.0, 10.0)
    assert m and m[0] == 5.0 and m[1] == "pog", m    # clip-relative, cleaned word
    assert _clip_keyword_moment(words, 0.0, 4.0) is None  # outside window

    # music vs impact split by folder-name hint
    assert _scan("nonexistent_dir_xyz", _AUDIO_EXTS) == []
    print("overlay ok")


if __name__ == "__main__":
    _demo()
