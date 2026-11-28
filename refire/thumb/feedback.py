"""Test & Compare feedback (03-thumb.md step 9): `refire thumb sync` / `refire thumb result`.

sync lists the channel's newest uploads, fetches each live thumbnail and perceptual-hash matches it
against every candidate set on file (run/<vod>/<run>/thumb/v<k>/A|B|C/thumb.png): a match links the
video to its set, with each candidate's features and upload slot, in ~/.refire/thumbs.sqlite. The
live thumbnail alone can't tell a win from an inconclusive test (both can leave slot 1 live), so
Nguy records the outcome with `result`, which re-reads the live thumbnail for a winner.
The last 10 results go into the director's and critic's prompts; after >= 10 decided tests a feature
value that keeps winning must appear in at least one concept (pipeline.validate).
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path

from . import checks

DB = Path.home() / ".refire" / "thumbs.sqlite"
LIVE = Path.home() / ".refire" / "live"          # fetched live thumbnails (cache, safe to wipe)
TEST_DAYS = 14                                   # test & compare runs up to two weeks
MATCH = 12                                       # max phash distance to call a live thumbnail ours (candidates differ by >= 16)
OUTCOMES = ("winner", "same", "inconclusive")
FEATURES = ("type", "expression", "accent", "left", "quote", "pixelate")
MIN_TESTS, MIN_WINS, MIN_RATE = 10, 3, 0.5       # chance is 1/3 per candidate

SCHEMA = """
create table if not exists videos (video_id text primary key, title text, published text, set_dir text,
  live text, synced text, result text, winner text, impressions integer, ctr real);
create table if not exists candidates (video_id text, cand text, slot integer, type text, layout text,
  hook text, accent text, expression text, "left" text, quote integer, pixelate integer, critic real,
  primary key (video_id, cand));
"""


def connect(path=None) -> sqlite3.Connection:
    p = Path(path or DB)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


# --------------------------------------------------------------------------- candidate sets

def sets(runs_dir="run") -> list[Path]:
    """every candidate set on file: a folder holding A/thumb.png (thumb/v<k>, or an older thumb/)."""
    r = Path(runs_dir)
    found = {p.parent.parent for pat in ("*/*/thumb/*/A/thumb.png", "*/*/thumb/A/thumb.png") for p in r.glob(pat)}
    return sorted(found)


def hashes(set_dirs: list[Path]) -> list[tuple[Path, str, int]]:
    return [(s, k, checks.phash(s / k / "thumb.png")) for s in set_dirs for k in "ABC" if (s / k / "thumb.png").exists()]


def match(png: Path, table: list) -> tuple[int, Path, str] | None:
    """-> (distance, set_dir, candidate) of the closest candidate, or None when nothing is close."""
    h = checks.phash(png)
    ranked = sorted(((checks.hamming(h, x), str(s), k) for s, k, x in table), key=lambda r: r[0])
    if not ranked or ranked[0][0] > MATCH:
        return None
    d, s, k = ranked[0]
    return d, Path(s), k


def concept_features(c: dict) -> dict:
    """what a concept bets on, comparable across videos (also what validate() enforces)."""
    sub = c.get("subject") or {}
    made = sub.get("made") or {}
    return {"type": c.get("type"), "expression": (c.get("face") or {}).get("expression"),
            "accent": (c.get("hook") or {}).get("accent_color"),
            "left": made.get("kind", "tierlist") if sub.get("source") == "made" else "art",
            "quote": int(bool(c.get("quote"))),
            "pixelate": int(c.get("type") == "reveal" or bool(made.get("hidden")))}


def features(set_dir: Path, cand: str) -> dict:
    cj = set_dir / "concepts.json"
    concepts = {c.get("id"): c for c in json.loads(cj.read_text(encoding="utf-8"))["concepts"]} if cj.exists() else {}
    spec = json.loads((set_dir / cand / "spec.json").read_text(encoding="utf-8"))
    t = spec.get("text") or {}
    f = concept_features(concepts.get(cand, {}))
    f["accent"] = t.get("accent") or f["accent"]        # what viewers saw (auto-fixed against the background)
    f["type"] = f["type"] or spec.get("layout")
    cr = set_dir / cand / "critic.json"
    return {**f, "layout": spec.get("layout"), "hook": " / ".join(t.get("lines", [])),
            "critic": json.loads(cr.read_text(encoding="utf-8")).get("score") if cr.exists() else None}


def upload_order(set_dir: Path) -> list[str]:
    """report.md's upload order (slot 1 first); A, B, C when no report was written."""
    p = set_dir / "upload_order.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    rep = set_dir / "report.md"
    got = re.findall(r"^\d+\. \*\*([ABC])\*\*", rep.read_text(encoding="utf-8"), re.M) if rep.exists() else []
    return got or ["A", "B", "C"]


# --------------------------------------------------------------------------- store

def link(con, v: dict, set_dir: Path, live: str) -> None:
    """video <-> candidate set, keeping any recorded result."""
    con.execute("""insert into videos (video_id, title, published, set_dir, live, synced) values (?, ?, ?, ?, ?, ?)
                   on conflict(video_id) do update set title=excluded.title, published=excluded.published,
                   set_dir=excluded.set_dir, live=excluded.live, synced=excluded.synced""",
                (v["id"], v.get("title", ""), v.get("published", ""), str(set_dir), live, time.strftime("%Y-%m-%d %H:%M:%S")))
    order = upload_order(set_dir)
    for k in "ABC":
        if not (set_dir / k / "spec.json").exists():
            continue
        f = features(set_dir, k)
        con.execute("""insert or replace into candidates (video_id, cand, slot, type, layout, hook, accent, expression,
                       "left", quote, pixelate, critic) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (v["id"], k, order.index(k) + 1 if k in order else None, f["type"], f["layout"], f["hook"],
                     f["accent"], f["expression"], f["left"], f["quote"], f["pixelate"], f["critic"]))
    con.commit()


def _days(published: str) -> float:
    import calendar
    try:
        t = calendar.timegm(time.strptime(published[:19], "%Y-%m-%dT%H:%M:%S"))     # the api's times are utc
    except ValueError:
        return 0.0
    return (time.time() - t) / 86400


def _live(yt, v: dict) -> Path:
    return yt.fetch(v["thumb"], LIVE / f"{v['id']}.jpg")


def sync(yt, runs_dir="run", con=None, n: int = 50, log=print) -> list[dict]:
    """match the channel's newest uploads to candidate sets on file; refresh CTR; say what to record."""
    con = con or connect()
    table = hashes(sets(runs_dir))
    log(f"[thumb sync] {len(table)} candidates on file")
    out, ctr_err = [], None
    for v in yt.videos(yt.uploads(n)):
        if not v.get("thumb"):
            continue
        m = match(_live(yt, v), table)
        if m is None:
            continue
        dist, set_dir, live = m
        link(con, v, set_dir, live)
        row = con.execute("select * from videos where video_id = ?", (v["id"],)).fetchone()
        if ctr_err is None:
            try:
                got = yt.ctr(v["id"], v["published"])
                if got:
                    con.execute("update videos set impressions = ?, ctr = ? where video_id = ?", (*got, v["id"]))
                    con.commit()
            except (RuntimeError, OSError) as e:          # one message, not one per video
                ctr_err = str(e)
                log(f"[thumb sync] no CTR from the Analytics API: {ctr_err[:200]}")
        days = _days(v["published"])
        if row["result"]:
            state = f"recorded: {row['result']}" + (f" ({row['winner']})" if row["winner"] else "")
        elif days >= TEST_DAYS:
            state = f"test over, {live} is live: refire thumb result {v['id']} winner|same|inconclusive"
        else:
            state = f"test running, day {days:.0f} of {TEST_DAYS}"
        log(f"[thumb sync] {v['id']} = {set_dir} {live} (phash {dist}): {state}")
        out.append({"video_id": v["id"], "set_dir": set_dir, "live": live, "distance": dist, "state": state})
    return out


def result(yt, video_id: str, outcome: str, runs_dir="run", con=None, log=print) -> dict:
    """record a test outcome. winner = the candidate live now (re-read; the stored one if offline)."""
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
    con = con or connect()
    row = con.execute("select * from videos where video_id = ?", (video_id,)).fetchone()
    if outcome == "winner" or row is None:
        try:
            v = yt.videos([video_id])[0]
            table = hashes([Path(row["set_dir"])] if row else sets(runs_dir))
            m = match(_live(yt, v), table)
            if m is None:
                raise ValueError(f"{video_id}: the live thumbnail matches none of the candidates on file")
            link(con, v, m[1], m[2])
        except (RuntimeError, OSError, IndexError) as e:
            if not (row and row["live"]):
                raise
            log(f"[thumb result] couldn't re-read the live thumbnail ({str(e)[:120]}); using the synced one")
        row = con.execute("select * from videos where video_id = ?", (video_id,)).fetchone()
    winner = row["live"] if outcome == "winner" else None
    con.execute("update videos set result = ?, winner = ? where video_id = ?", (outcome, winner, video_id))
    con.commit()
    log(f"[thumb result] {video_id}: {outcome}" + (f", {winner} won" if winner else ""))
    return {"video_id": video_id, "result": outcome, "winner": winner}


# --------------------------------------------------------------------------- use

def history(con, n: int = 10) -> list[str]:
    """the last n recorded results, newest first, one line each (* = the winner)."""
    lines = []
    for v in con.execute("select * from videos where result is not null order by published desc limit ?", (n,)):
        cs = con.execute("select * from candidates where video_id = ? order by slot", (v["video_id"],)).fetchall()
        cands = "; ".join(f"{c['cand']}{'*' if c['cand'] == v['winner'] else ''} slot {c['slot']} {c['type']}, "
                          f"{c['expression']}, {c['accent']}, left {c['left']}{', quote' if c['quote'] else ''}: "
                          f"\"{c['hook']}\"" for c in cs)
        ctr = f", CTR {v['ctr']}" if v["ctr"] is not None else ""
        lines.append(f"{(v['published'] or '')[:10]} {v['result']}{ctr}: {cands}")
    return lines


def win_rates(con) -> tuple[int, dict]:
    """over tests with a winner -> (n, {feature: {value: [wins, seen]}})."""
    tests = con.execute("select video_id, winner from videos where result = 'winner' and winner is not null").fetchall()
    rates: dict = {f: {} for f in FEATURES}
    for t in tests:
        for c in con.execute("select * from candidates where video_id = ?", (t["video_id"],)):
            for f in FEATURES:
                w = rates[f].setdefault(str(c[f]), [0, 0])
                w[0] += c["cand"] == t["winner"]
                w[1] += 1
    return len(tests), rates


def favored(con) -> dict:
    """after MIN_TESTS decided tests: per feature, the value that keeps winning (>= MIN_WINS wins,
    win rate >= MIN_RATE, the best rate) -> {feature: value}."""
    n, rates = win_rates(con)
    if n < MIN_TESTS:
        return {}
    out = {}
    for f, vals in rates.items():
        ok = [(w / s, v) for v, (w, s) in vals.items() if w >= MIN_WINS and w / s >= MIN_RATE]
        if ok:
            out[f] = max(ok)[1]
    return out


def context(path=None) -> tuple[str, dict]:
    """-> (prompt text with the last 10 results, favored features); empty before any result."""
    p = Path(path or DB)
    if not p.exists():
        return "", {}
    con = connect(p)
    try:
        hist = history(con)
        if not hist:
            return "", {}
        fav = favored(con)
        n, rates = win_rates(con)
        text = ("Past Test & Compare results on the channel (newest first, * = the winner; same/inconclusive = no "
                "winner; slot 1 stays live when inconclusive):\n" + "\n".join(f"- {h}" for h in hist))
        if n:
            text += f"\nWin rates over {n} tests with a winner: " + "; ".join(
                f"{f} {v} {w}/{s}" for f in FEATURES for v, (w, s) in sorted(rates[f].items()) if s)
        if fav:
            text += ("\nThese keep winning; at least one concept must use each: "
                     + ", ".join(f"{f} = {v}" for f, v in fav.items()))
        return text, fav
    finally:
        con.close()
