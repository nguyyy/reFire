"""YouTube Data API v3 + YouTube Analytics API for `refire thumb sync` / `result` (03-thumb.md step 9).

Analytics needs the channel owner's OAuth (it has videoThumbnailImpressions + ...ClickRate, so the
spec's Studio-CSV fallback isn't needed), and the same token covers the Data API: one setup. Put a
Google Cloud OAuth client ("Desktop app", YouTube Data API v3 + YouTube Analytics API enabled) at
~/.refire/youtube_client.json; the first call opens the browser once and the refresh token lands
in ~/.refire/youtube_token.json. ponytail: stdlib loopback + PKCE instead of google-auth-oauthlib.
Quota: channels / playlistItems / videos.list are 1 unit each (never search.list, 100).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HOME = Path.home() / ".refire"
CLIENT = HOME / "youtube_client.json"
TOKEN = HOME / "youtube_token.json"
SCOPES = "https://www.googleapis.com/auth/youtube.readonly https://www.googleapis.com/auth/yt-analytics.readonly"
AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
DATA = "https://www.googleapis.com/youtube/v3/"
ANALYTICS = "https://youtubeanalytics.googleapis.com/v2/reports"


def _post(url: str, fields: dict) -> dict:
    req = urllib.request.Request(url, urllib.parse.urlencode(fields).encode(), method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _client() -> dict:
    if not CLIENT.exists():
        raise RuntimeError(f"no YouTube OAuth client at {CLIENT}: create a Desktop-app OAuth client in Google "
                           "Cloud (enable YouTube Data API v3 + YouTube Analytics API) and save its JSON there")
    c = json.loads(CLIENT.read_text(encoding="utf-8"))
    return c.get("installed") or c.get("web") or c


def _consent(c: dict) -> dict:
    """browser sign-in once -> token response (with a refresh token)."""
    import base64
    import hashlib
    import http.server
    import secrets
    import webbrowser
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    state, got = secrets.token_urlsafe(16), {}

    class Back(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            got.update({k: v[0] for k, v in q.items()})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"reFire: signed in to YouTube, you can close this tab.")

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Back)
    redirect = f"http://127.0.0.1:{srv.server_port}"
    url = AUTH + "?" + urllib.parse.urlencode({
        "client_id": c["client_id"], "redirect_uri": redirect, "response_type": "code", "scope": SCOPES,
        "access_type": "offline", "prompt": "consent", "state": state,
        "code_challenge": challenge, "code_challenge_method": "S256"})
    print(f"[youtube] sign in (opening the browser): {url}", flush=True)
    webbrowser.open(url)
    try:
        while "code" not in got and "error" not in got:     # a favicon request carries neither
            srv.handle_request()
    finally:
        srv.server_close()
    if got.get("state") != state or "code" not in got:
        raise RuntimeError(f"youtube sign-in failed: {got.get('error', 'state mismatch')}")
    return _post(TOKEN_URL, {"code": got["code"], "client_id": c["client_id"], "client_secret": c.get("client_secret", ""),
                             "redirect_uri": redirect, "grant_type": "authorization_code", "code_verifier": verifier})


class YouTube:
    def _token(self) -> str:
        t = json.loads(TOKEN.read_text(encoding="utf-8")) if TOKEN.exists() else None
        if t and t.get("expires", 0) > time.time() + 60:
            return t["access_token"]
        c = _client()
        new = None
        if t and t.get("refresh_token"):
            try:
                new = _post(TOKEN_URL, {"refresh_token": t["refresh_token"], "client_id": c["client_id"],
                                        "client_secret": c.get("client_secret", ""), "grant_type": "refresh_token"})
                new["refresh_token"] = t["refresh_token"]
            except urllib.error.HTTPError:                  # revoked / expired: sign in again
                new = None
        new = new or _consent(c)
        new["expires"] = time.time() + float(new.get("expires_in", 3600))
        HOME.mkdir(parents=True, exist_ok=True)
        TOKEN.write_text(json.dumps(new), encoding="utf-8")
        return new["access_token"]

    def get(self, url: str, **params) -> dict:
        req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params),
                                     headers={"Authorization": f"Bearer {self._token()}"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"youtube {url.rstrip('/').rsplit('/', 1)[-1]}: {e.code} "
                               f"{e.read()[:300].decode('utf-8', 'replace')}") from None

    def uploads(self, n: int = 50) -> list[str]:
        """the newest n (<= 50) uploads of the signed-in channel. ponytail: one page, months of uploads."""
        ch = self.get(DATA + "channels", part="contentDetails", mine="true")["items"][0]
        pl = ch["contentDetails"]["relatedPlaylists"]["uploads"]
        items = self.get(DATA + "playlistItems", part="contentDetails", playlistId=pl, maxResults=min(50, n))["items"]
        return [i["contentDetails"]["videoId"] for i in items]

    def videos(self, ids: list[str]) -> list[dict]:
        """-> [{id, title, published, thumb}], thumb = the largest live thumbnail's url."""
        out = []
        for i in range(0, len(ids), 50):
            for v in self.get(DATA + "videos", part="snippet", id=",".join(ids[i:i + 50]))["items"]:
                s, th = v["snippet"], v["snippet"].get("thumbnails", {})
                best = next((th[k]["url"] for k in ("maxres", "standard", "high", "medium", "default") if k in th), None)
                out.append({"id": v["id"], "title": s.get("title", ""), "published": s.get("publishedAt", ""),
                            "thumb": best})
        return out

    def ctr(self, video_id: str, since: str) -> tuple[int, float] | None:
        """(impressions, click rate as the api returns it) since `since` (iso date) -> today."""
        r = self.get(ANALYTICS, ids="channel==MINE", startDate=since[:10], endDate=time.strftime("%Y-%m-%d"),
                     metrics="videoThumbnailImpressions,videoThumbnailImpressionsClickRate",
                     filters=f"video=={video_id}")
        rows = r.get("rows") or []
        return (int(rows[0][0]), float(rows[0][1])) if rows else None

    def fetch(self, url: str, out: Path) -> Path:
        """thumbnails are public on i.ytimg.com: no token."""
        out.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "reFire thumb sync"}),
                                    timeout=30) as r:
            out.write_bytes(r.read())
        return out
