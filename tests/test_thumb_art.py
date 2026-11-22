"""art fetcher (refire.thumb.art) against a fake network: allowlist, robots, fallback, cache."""
import io
import json
import urllib.error

import numpy as np
import pytest
from PIL import Image

from refire.thumb import art

ROSTER = {"response": 200, "data": {"items": {
    "1": {"name": "Furina", "icon": "UI_AvatarIcon_Furina"},
    "2": {"name": "Kamisato Ayaka", "icon": "UI_AvatarIcon_Ayaka"},
    "3": {"name": "Neuvillette", "icon": "UI_AvatarIcon_Neuvillette"}}}}


def png(w, h, alpha=True):
    b = io.BytesIO()
    Image.new("RGBA" if alpha else "RGB", (w, h)).save(b, "PNG")
    return b.getvalue()


class Resp(io.BytesIO):
    def __init__(self, body, ctype, url):
        super().__init__(body)
        self.headers, self._url = {"content-type": ctype}, url

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def opener(routes, log):
    def op(req, timeout):
        url = req.full_url
        log.append(url)
        assert req.headers["User-agent"].startswith("refire-thumb")
        r = routes.get(url)
        if r is None and url.endswith("/robots.txt"):
            return Resp(b"User-agent: *\nDisallow:\n", "text/plain", url)
        if r is None or r == 404:
            raise urllib.error.HTTPError(url, 404, "nf", {}, None)
        if isinstance(r, Exception):
            raise r
        return Resp(r[0], r[1], r[2] if len(r) > 2 else url)
    return op


def test_resolve_names():
    roster = {n["name"]: n["icon"] for n in ROSTER["data"]["items"].values()}
    assert art.resolve("furina", roster) == "Furina"
    assert art.resolve("Ayaka", roster) == "Kamisato Ayaka"
    assert art.resolve("neuvilette", roster) == "Neuvillette"      # typo
    assert art.resolve("Elon Musk", roster) is None


def test_refuses_off_allowlist_and_robots_disallow():
    log = []
    f = art.Fetcher(delay=0, opener=opener({"https://enka.network/robots.txt":
                                           (b"User-agent: *\nDisallow: /ui/\n", "text/plain")}, log))
    with pytest.raises(art.ArtError, match="refused"):
        f.get("https://example.com/cat.jpg")
    with pytest.raises(art.ArtError, match="refused"):
        f.get("https://enka.network/ui/x.png")
    assert not any("example.com" in u for u in log)                # never even asked


def test_redirect_off_allowlist_and_non_images_are_rejected():
    log = []
    f = art.Fetcher(delay=0, opener=opener({
        "https://gi.yatta.moe/a.png": (png(10, 10), "image/png", "https://evil.example/a.png"),
        "https://gi.yatta.moe/b.png": (b"<html>", "text/html")}, log))
    with pytest.raises(art.ArtError, match="redirected"):
        f.get("https://gi.yatta.moe/a.png", accept=art.IMAGES)
    with pytest.raises(art.ArtError, match="wanted"):
        f.get("https://gi.yatta.moe/b.png", accept=art.IMAGES)


def test_splash_falls_back_validates_cuts_out_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(art.time, "sleep", lambda s: None)
    big_rgb = png(2048, 1024, alpha=False)
    routes = {
        "https://gi.yatta.moe/api/v2/en/avatar": (json.dumps(ROSTER).encode(), "application/json"),
        "https://gi.yatta.moe/assets/UI/UI_Gacha_AvatarImg_Furina.png": (png(400, 200), "image/png"),  # too small
        "https://enka.network/ui/UI_Gacha_AvatarImg_Furina.png": (big_rgb, "image/png"),               # no alpha
    }
    log, cut = [], []
    f = art.Fetcher(delay=0, opener=opener(routes, log))

    def fake_cut(img, inset_frac=0.0):
        cut.append(img.shape)
        return np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])

    p = art.splash("furina", f, cache=tmp_path, cutout_fn=fake_cut)
    meta = json.loads(p.with_suffix(".json").read_text())
    assert meta["source"] == "enka" and meta["had_alpha"] is False and cut == [(1024, 2048, 3)]
    assert Image.open(p).mode == "RGBA"
    n = len(log)
    assert art.splash("Furina", f, cache=tmp_path) == p and len(log) == n   # cached: no network


def test_unknown_character_and_transient_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(art.time, "sleep", lambda s: None)
    calls = {"n": 0}
    roster_url = "https://gi.yatta.moe/api/v2/en/avatar"

    def flaky(req, timeout):
        if req.full_url.endswith("robots.txt"):
            return Resp(b"", "text/plain", req.full_url)
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("reset")
        return Resp(json.dumps(ROSTER).encode(), "application/json", roster_url)

    f = art.Fetcher(delay=0, opener=flaky)
    with pytest.raises(art.ArtError, match="not a Genshin character"):
        art.splash("Elon Musk", f, cache=tmp_path)
    assert calls["n"] == 3                                          # two failures, then success


def test_item_takes_the_wiki_lead_image(tmp_path):
    api = ("https://genshin-impact.fandom.com/api.php?action=query&generator=search&gsrsearch=Primogem&gsrlimit=3"
           "&gsrnamespace=0&prop=pageimages&piprop=original&format=json")
    img = "https://static.wikia.nocookie.net/gensin-impact/images/primogem.png"
    meta = {"query": {"pages": {"7": {"title": "Primogem/Gallery", "index": 2, "original": {"source": "nope"}},
                                "5": {"title": "Primogem Shop", "index": 1},                      # no image
                                "1": {"title": "Primogem", "index": 3, "original": {"source": img}}}}}
    meta["query"]["pages"]["7"]["index"] = 4                     # best-ranked hit with an image wins
    assert not art._names("Zabai", "Lauma") and art._names("Primogem", "Primogem")
    assert art._names("Deepwood Memories", "Deepwood Memories (Set)")
    log = []
    f = art.Fetcher(delay=0, opener=opener({api: (json.dumps(meta).encode(), "application/json"),
                                           img: (png(300, 300), "image/png")}, log))
    p = art.item("Primogem", f, cache=tmp_path)
    assert p.exists() and json.loads(p.with_suffix(".json").read_text())["name"] == "Primogem"
    tiny = {"query": {"pages": {"1": {"title": "X", "original": {"source": img}}}}}
    f2 = art.Fetcher(delay=0, opener=opener({api.replace("Primogem", "X"): (json.dumps(tiny).encode(), "application/json"),
                                            img: (png(64, 64), "image/png")}, []))
    with pytest.raises(art.ArtError, match="too small"):
        art.item("X", f2, cache=tmp_path)


def test_figure_keeps_only_what_the_model_calls_foreground(tmp_path):
    im = Image.new("RGBA", (400, 300), (200, 50, 50, 255))          # an opaque painted backdrop
    im.save(tmp_path / "s.png")

    def model(bgr):                                               # stand-in: a figure in the middle
        a = np.zeros(bgr.shape[:2], np.uint8)
        a[50:250, 150:250] = 255
        return a
    out = art.figure(tmp_path / "s.png", alpha_fn=model)
    assert Image.open(out).size == (100, 200)                     # cropped to the figure
    assert art.figure(tmp_path / "s.png", alpha_fn=None) == out    # cached: model not run again


def test_figure_recrops_when_the_model_keeps_a_painted_medallion(tmp_path):
    Image.new("RGBA", (1000, 500), (90, 40, 140, 255)).save(tmp_path / "m.png")
    calls = []

    def model(bgr):
        calls.append(bgr.shape)
        a = np.zeros(bgr.shape[:2], np.uint8)
        if bgr.shape[1] == 1000:                 # full splash: one wide round object
            a[50:450, 250:750] = 255
        else:                                    # the crop around the figure: just the figure
            a[60:480, bgr.shape[1] // 2 - 40:bgr.shape[1] // 2 + 40] = 255
        return a
    out = art.figure(tmp_path / "m.png", alpha_fn=model)
    assert len(calls) == 2 and calls[1][1] == 300                  # re-run on a 30%-wide crop
    w, h = Image.open(out).size
    assert h > 2 * w                                                # figure-shaped, medallion gone
