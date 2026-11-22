"""Official game art by tool use (docs/specs/03-thumb.md step 4). Genshin for now.

Sources in priority order, all official art with transparency:
  1. gi.yatta.moe (Project Amber; api.ambr.top's successor): character list + wish splash
  2. enka.network's UI asset CDN (same files)
  3. the Genshin fandom wiki via the MediaWiki API ("File:<Name> Wish.png")
General image search is not used (the spec only allows it with a vision check; not built).
The fetcher only talks to an allowlist of game-art hosts, checks robots.txt, waits >= 1 s
between requests to a host, retries with backoff, and refuses anything that isn't an image.
Results cache in ~/.cache/refire/art/ with a sidecar JSON (url, source, fetched date):
an automatic cache, safe to wipe.
"""
from __future__ import annotations

import difflib
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from datetime import date
from pathlib import Path

UA = "refire-thumb/0.1 (zero-touch thumbnail tool; fetches official game art only)"
CACHE = Path.home() / ".cache" / "refire" / "art"
HOSTS = {"gi.yatta.moe", "enka.network", "genshin-impact.fandom.com", "static.wikia.nocookie.net"}
MIN_EDGE = 1000
YATTA = "https://gi.yatta.moe"
FANDOM_API = "https://genshin-impact.fandom.com/api.php"


class ArtError(RuntimeError):
    pass


class Fetcher:
    """polite GETs: allowlisted hosts, robots.txt, >= 1 s per host, retry with backoff."""

    def __init__(self, delay: float = 1.0, tries: int = 3, opener=None):
        self.delay, self.tries = delay, tries
        self._last: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._open = opener or (lambda req, timeout: urllib.request.urlopen(req, timeout=timeout))

    def _allowed(self, url: str) -> bool:
        host = urllib.parse.urlsplit(url).hostname or ""
        if host not in HOSTS:
            return False
        if host not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                with self._open(urllib.request.Request(f"https://{host}/robots.txt", headers={"User-Agent": UA}), 15) as r:
                    rp.parse(r.read().decode("utf-8", "replace").splitlines())
            except (urllib.error.URLError, OSError):
                rp = None                                  # no robots.txt reachable: allowed
            self._robots[host] = rp
        rp = self._robots[host]
        return rp is None or rp.can_fetch(UA, url)

    def get(self, url: str, accept: tuple = ()) -> tuple[bytes, str, str]:
        """-> (body, content type, final url). raises ArtError."""
        if not self._allowed(url):
            raise ArtError(f"refused (not an allowed game-art host or robots.txt): {url}")
        host = urllib.parse.urlsplit(url).hostname
        err = None
        for k in range(self.tries):
            wait = self._last.get(host, 0) + self.delay - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last[host] = time.monotonic()
            try:
                with self._open(urllib.request.Request(url, headers={"User-Agent": UA}), 60) as r:
                    final = r.geturl()
                    ctype = r.headers.get("content-type", "").split(";")[0].strip()
                    if (urllib.parse.urlsplit(final).hostname or "") not in HOSTS:
                        raise ArtError(f"redirected off the allowlist: {final}")
                    if accept and ctype not in accept:
                        raise ArtError(f"{url}: got {ctype}, wanted {accept}")
                    return r.read(), ctype, final
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise ArtError(f"404 {url}") from None
                err = e
            except (urllib.error.URLError, OSError) as e:
                err = e
            time.sleep(2 ** k)                              # 1, 2, 4 s backoff
        raise ArtError(f"{url}: {err}")


IMAGES = ("image/png", "image/webp", "image/jpeg")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def characters(f: Fetcher, cache: Path = CACHE) -> dict[str, str]:
    """canonical name -> icon id (UI_AvatarIcon_X), refreshed daily from yatta."""
    p = cache / "yatta_avatar.json"
    if p.exists() and json.loads(p.read_text(encoding="utf-8")).get("date") == date.today().isoformat():
        return json.loads(p.read_text(encoding="utf-8"))["items"]
    body, _, _ = f.get(f"{YATTA}/api/v2/en/avatar", accept=("application/json",))
    items = json.loads(body)["data"]["items"]
    out = {v["name"]: v["icon"] for v in items.values() if v.get("name") and v.get("icon")}
    cache.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"date": date.today().isoformat(), "items": out}, indent=1), encoding="utf-8")
    return out


def resolve(name: str, roster: dict[str, str]) -> str | None:
    """'ayaka' / 'Kamisato Ayaka' / 'furina' -> the canonical roster name, or None."""
    low = {n.lower(): n for n in roster}
    q = name.strip().lower()
    if q in low:
        return low[q]
    parts = [n for n in roster if q in n.lower().split()]          # a given or family name
    if len(parts) == 1:
        return parts[0]
    hit = difflib.get_close_matches(q, list(low), n=1, cutoff=0.8)
    return low[hit[0]] if hit else None


def _validate(body: bytes) -> tuple[int, int, bool]:
    from PIL import Image
    im = Image.open(io.BytesIO(body))
    im.load()
    w, h = im.size
    if max(w, h) < MIN_EDGE:
        raise ArtError(f"too small ({w}x{h}, need >= {MIN_EDGE} px long edge)")
    alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
    return w, h, alpha


def splash(name: str, f: Fetcher | None = None, cache: Path = CACHE, cutout_fn=None) -> Path:
    """the character's official wish splash with transparency -> cached PNG path."""
    f = f or Fetcher()
    roster = characters(f, cache)
    canon = resolve(name, roster)
    if not canon:
        raise ArtError(f"{name!r} is not a Genshin character on the official roster")
    out = cache / f"{slug(canon)}_wish.png"
    if out.exists():
        return out
    gacha = roster[canon].replace("UI_AvatarIcon_", "UI_Gacha_AvatarImg_")
    tries = [("yatta", f"{YATTA}/assets/UI/{gacha}.png"), ("enka", f"https://enka.network/ui/{gacha}.png")]
    errors = []
    for src, url in tries + [("fandom", None)]:
        try:
            if src == "fandom":
                q = urllib.parse.urlencode({"action": "query", "titles": f"File:{canon} Wish.png", "prop": "imageinfo",
                                            "iiprop": "url|size|mime", "format": "json"})
                meta, _, _ = f.get(f"{FANDOM_API}?{q}", accept=("application/json",))
                pages = json.loads(meta)["query"]["pages"]
                info = next((p["imageinfo"][0] for p in pages.values() if p.get("imageinfo")), None)
                if not info:
                    raise ArtError("no wish file on the wiki")
                url = info["url"]
            body, ctype, final = f.get(url, accept=IMAGES)
            w, h, alpha = _validate(body)
            cache.mkdir(parents=True, exist_ok=True)
            if alpha:
                out.write_bytes(body)
            else:                                           # no transparency: BiRefNet it
                import cv2
                import numpy as np
                img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
                from .cutout import cutout
                cv2.imwrite(str(out), (cutout_fn or cutout)(img, inset_frac=0.0))
            out.with_suffix(".json").write_text(json.dumps(
                {"name": canon, "source": src, "url": final, "fetched": date.today().isoformat(),
                 "size": [w, h], "had_alpha": alpha}, indent=1), encoding="utf-8")
            return out
        except ArtError as e:
            errors.append(f"{src}: {e}")
    raise ArtError(f"no official art for {canon}: " + "; ".join(errors))


ITEM_MIN_EDGE = 256                 # item / weapon / enemy images on the wiki are icon-sized


def item(query: str, f: Fetcher | None = None, cache: Path = CACHE, cutout_fn=None) -> Path:
    """any other game thing by name (weapon, material, enemy, boss, location, ...): the lead
    image of its Genshin wiki page (MediaWiki search -> pageimages original), cut out with
    BiRefNet when it has no transparency -> cached PNG."""
    f = f or Fetcher()
    out = cache / f"{slug(query)}_item.png"
    if out.exists():
        return out
    q = urllib.parse.urlencode({"action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": 1,
                                "gsrnamespace": 0, "prop": "pageimages", "piprop": "original", "format": "json"})
    meta, _, _ = f.get(f"{FANDOM_API}?{q}", accept=("application/json",))
    pages = (json.loads(meta).get("query") or {}).get("pages") or {}
    page = next(iter(pages.values()), None)
    if not page or not page.get("original"):
        raise ArtError(f"no wiki image for {query!r}")
    body, _, final = f.get(page["original"]["source"], accept=IMAGES)
    from PIL import Image
    im = Image.open(io.BytesIO(body))
    im.load()
    if max(im.size) < ITEM_MIN_EDGE:
        raise ArtError(f"wiki image for {query!r} too small ({im.size})")
    alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
    cache.mkdir(parents=True, exist_ok=True)
    if alpha:
        im.convert("RGBA").save(out)
    else:
        import cv2
        import numpy as np
        from .cutout import cutout
        img = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out), (cutout_fn or cutout)(img, inset_frac=0.0))
    out.with_suffix(".json").write_text(json.dumps(
        {"name": page.get("title"), "query": query, "source": "fandom", "url": final,
         "fetched": date.today().isoformat(), "size": list(im.size), "had_alpha": alpha}, indent=1), encoding="utf-8")
    return out


def figure(png: Path, alpha_fn=None) -> Path:
    """just the figure out of an official image: a wish splash carries its effects swirl and
    painted backdrop on the alpha (the boxy crops), so BiRefNet picks the character and that
    mask is multiplied into the image's own alpha. older splashes paint a whole medallion behind
    the character, which the model keeps as one object (wide, not figure-shaped): then it runs
    again on a crop around the figure, where the medallion is background. cached as *_fig.png."""
    out = png.with_name(png.stem + "_fig.png")
    if out.exists():
        return out
    import numpy as np
    from PIL import Image

    from .cutout import alpha as birefnet
    model = alpha_fn or birefnet
    im = Image.open(png).convert("RGBA")
    a0 = np.asarray(im)[..., 3].astype(np.float32) / 255
    rgb = np.asarray(im)[..., :3].astype(np.float32)
    flat = (rgb * a0[..., None] + 128 * (1 - a0[..., None])).astype(np.uint8)       # over mid-gray
    H, W = a0.shape
    a = model(flat[..., ::-1].copy()).astype(np.float32) / 255                       # model takes bgr
    ys, xs = np.nonzero(a > 0.5)
    if len(xs) and (ys.max() - ys.min()) < FIGURE_ASPECT * (xs.max() - xs.min()) and W > H:
        cols = (a > 0.5).sum(0).astype(float)
        mid = cols[W // 3:2 * W // 3]
        cx = W // 3 + int(np.average(np.arange(len(mid)), weights=mid + 1e-6))
        half = int(0.3 * W / 2)
        x0, x1 = max(0, cx - half), min(W, cx + half)
        a = np.zeros_like(a)
        a[:, x0:x1] = model(flat[:, x0:x1, ::-1].copy()).astype(np.float32) / 255
    keep = np.clip(a0 * a, 0, 1)
    arr = np.asarray(im).copy()
    arr[..., 3] = (keep * 255).round().astype(np.uint8)
    res = Image.fromarray(arr)
    res.crop(res.getchannel("A").point(lambda v: 255 if v > 24 else 0).getbbox() or (0, 0, *res.size)).save(out)
    return out


FIGURE_ASPECT = 1.25        # a standing figure's mask is taller than wide by at least this
