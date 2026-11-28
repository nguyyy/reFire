"""Where sauce might go and what goes there (04-sauce.md 5.1 + 5.2): booms + emotes (P2).

The run's live timeline (timeline.tsv, else the manifest) -> phrases in timeline time with the
dialogue and audio signals punch already computes (loudness z, laughter, the exclamation lexicon,
shouted words); s(t) ranks them. One director call reads the whole cut as numbered phrases, the
strongest marked, with his nearest exemplars (run/sauce/exemplars.jsonl, by word overlap) and the
closed emote library, and proposes {phrase, type, asset, confidence, reason}. One critic call
re-checks every proposal against section 4 and "would Nguy delete this?". Chat stays out: the
spec allows it only as a faint mood hint and the director already reads the line itself.
The director also Reads frame sheets: 3 full frames around each starred phrase's end (face + game).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

W_DLG, W_AUD = 1.0, 0.6           # s(t) weights: dialogue first (spec 5.1); chat 0
MARK = 40                         # phrases marked as strong candidates for the director
EXEMPLARS = 40                    # retrieved exemplars in the prompt
_TOK = re.compile(r"[a-z0-9']+")


# --------------------------------------------------------------------------- candidates

def load(run_dir, timeline=None, log=print) -> dict:
    """the run's timeline phrases with signals -> {phrases, tmap, words, total, context}."""
    from .. import punch
    from ..emphasis import annotate_emphasis
    from ..loud import spike_signal
    from ..srt import _run_dir
    from ..timeline_map import TimelineMap
    run_dir = Path(run_dir)
    mp = run_dir / "ae" / "manifest.json"
    tmap = TimelineMap.load(mp, timeline)
    if not tmap.ours():
        raise ValueError("no reFire footage on V1 -- dump the timeline from the panel first")
    vod = _run_dir(mp)
    words = sorted(json.loads((vod / "transcript.json").read_text(encoding="utf-8")), key=lambda w: w["start"])
    annotate_emphasis(words, vod / "audio.wav")
    loud = punch.Signal(spike_signal(vod / "audio.wav", window=0.5))
    laughs = []
    try:
        from ..laughter import laugh_segments
        laughs = laugh_segments(vod / "audio.wav", [(s.src_in, s.src_out) for s in tmap.ours()])
    except Exception as e:                         # model fetch / odd wav: sauce still runs without it
        log(f"[sauce] laughter off: {e}")
    phr = punch.phrases(punch.timeline_words(words, tmap), tmap)
    lex = punch.load_lexicon()
    for i, p in enumerate(phr):
        p["id"] = i                                # p<int> in the prompt (punch's ids are "c<i>")
        p["f"] = punch.features(p, loud, punch.Signal([]), lex, laughs)
        p["s"] = round(score(p["f"]), 3)
    out = json.loads((run_dir / "outline.json").read_text(encoding="utf-8")) if (run_dir / "outline.json").exists() else {}
    return {"phrases": phr, "tmap": tmap, "total": max(s.tl_out for s in tmap.spans),
            "context": {"central_idea": out.get("central_idea"), "viewer_promise": out.get("viewer_promise")}}


def score(f: dict) -> float:
    """s(t) = w_dlg s_dlg + w_aud s_aud (spec 5.1), chat left out."""
    dlg = 1.0 * f["lex"] + 0.7 * f["caps"]
    aud = max(0.0, f["z_loud_peak"]) / 2 + min(f["laugh_dur"], 3.0) / 1.5
    return W_DLG * dlg + W_AUD * aud


SHEET_TILE = (480, 270)
SHEET_ROWS = 4                    # one starred phrase per row: 1 s before its end, its end, 0.7 s after
OFFSETS = (-1.0, 0.0, 0.7)


def sheets(run_dir, phr: list[dict], ids: set, root: Path) -> list[str]:
    """frame sheets of the starred phrases (spec 5.2: frames per candidate; the face and the game
    both matter for a boom) -> file names in root, labelled p<id> and the offset."""
    import cv2
    import numpy as np

    from ..facecam import Facecams
    cams = Facecams(Path(run_dir) / "ae" / "manifest.json")
    tw, th = SHEET_TILE
    rows = []
    for p in (p for p in phr if p["id"] in ids):
        tiles = []
        for dt in OFFSETS:
            got = cams.frames(p["source"], [max(0.0, p["src_t1"] + dt)], width=0)
            tile = cv2.resize(got[0], (tw, th), interpolation=cv2.INTER_AREA) if got else np.zeros((th, tw, 3), np.uint8)
            for color, thick in (((0, 0, 0), 5), ((255, 255, 255), 2)):
                cv2.putText(tile, f"p{p['id']} {dt:+.1f}s", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thick, cv2.LINE_AA)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    root.mkdir(parents=True, exist_ok=True)
    names = []
    for k in range(0, len(rows), SHEET_ROWS):
        chunk = rows[k:k + SHEET_ROWS]
        chunk += [np.zeros_like(rows[0])] * (SHEET_ROWS - len(chunk))
        names.append(f"frames_{k // SHEET_ROWS + 1}.jpg")
        cv2.imwrite(str(root / names[-1]), np.vstack(chunk), [cv2.IMWRITE_JPEG_QUALITY, 82])
    return names


def tokens(s: str) -> list[str]:
    return _TOK.findall(s.lower())


def retrieve(rows: list[dict], queries: list[str], k: int = EXEMPLARS) -> list[dict]:
    """exemplars sharing the rarest words with the queries (idf-weighted overlap), the best per
    query first, then every library emote's best example, capped at k. ponytail: word overlap,
    not embeddings; upgrade if the picks read off-topic."""
    if not rows:
        return []
    docs = [set(tokens(r["text"])) for r in rows]
    df = Counter(t for d in docs for t in d)
    idf = {t: math.log(len(docs) / n) for t, n in df.items()}
    picked: list[int] = []
    for qy in queries:
        qt = set(tokens(qy))
        best = max(range(len(rows)), key=lambda i: sum(idf.get(t, 0) for t in qt & docs[i]))
        if best not in picked:
            picked.append(best)
    for name in {r["asset"] for r in rows if r["type"] == "emote"}:
        i = next(i for i, r in enumerate(rows) if r["type"] == "emote" and r["asset"] == name)
        if i not in picked:
            picked.append(i)
    return [rows[i] for i in picked[:k]]


def transcript(phr: list[dict], mark: set) -> str:
    """the cut as numbered phrases: p<id> [m:ss.s] (signals) text, strong ones starred."""
    out = []
    for p in phr:
        f = p["f"]
        sig = [f"loud {f['z_loud_peak']:.1f}" if f["z_loud_peak"] >= 1.5 else "",
               f"laugh {f['laugh_dur']:.1f}s" if f["laugh_dur"] >= 0.6 else "",
               "LEX" if f["lex"] else "", "CAPS" if f["caps"] else ""]
        sig = ", ".join(s for s in sig if s)
        out.append(f"{'*' if p['id'] in mark else ' '}p{p['id']} [{int(p['t0'] // 60)}:{p['t0'] % 60:04.1f}]"
                   f"{' (' + sig + ')' if sig else ''} {p['text']}")
    return "\n".join(out)


# --------------------------------------------------------------------------- director + critic

def system(prof: dict, library: list[dict]) -> str:
    e = prof["emote"]
    lib ="\n".join(f"- {r['name']}" + (f": used for {r['moods']}" if r.get("moods") else "") for r in library)
    return f"""You add the polish layer to zy0x's Genshin Impact stream edit (YouTube @fye_er) the way his editor
Nguy does it: vine booms and meme emotes. A viewer must not be able to tell it from Nguy's hand edit.
Precision beats recall: a missing boom costs nothing, a wrong one costs a fix. When unsure, skip.

What Nguy does (measured on his uploads):
- Vine boom: on the end of the punchline word (the hit lands as the line ends, it can clip the word's
  tail). Triggers: sus or awkward lines, out-of-pocket statements, something embarrassing, fails, a
  self-own, a comedic boom on a sad moment. Spammable when the moment earns it. About
  {prof['rates']['boom']['body_per_min']:.1f}/min in the body, more in the first minute.
- Emote: centered, {e['hold'].get('median', 1.4):.1f} s, fading out; one at a time. Must fit the mood of the line.
  Half of them land with a boom on the same frame. About {prof['rates']['emote']['body_per_min']:.2f}/min.
  Closed library (use only these names):
{lib}
- Nothing for filler, read-aloud quest text, stat lists, chat reading, donation/sub alerts, unless the
  moment itself is funny or out of pocket. Everything on stream is fair game, teammates and friends too.

You get the cut as numbered phrases (p<id> [m:ss] text). Starred phrases have strong loudness /
laughter / exclamation signals; anything can qualify, starred or not. You also get examples of what
Nguy did (the event sat at [*] in the transcript). Propose events on the phrase whose END is the hit.
A phrase may get both a boom and an emote. confidence = how sure you are Nguy would keep it (0-1).

Reply with ONLY JSON: {{"events": [{{"phrase": 12, "type": "boom|emote", "asset": "emote name or null",
"confidence": 0.8, "reason": "short: the line + why it's funny"}}]}}"""


CRITIC = """You are Nguy, zy0x's editor, reviewing polish another editor proposed for his stream edit
(vine booms, meme emotes). For each proposal decide: would you delete it? Delete anything generic (a
boom on a line that isn't a punchline, an emote that doesn't fit the line's mood), anything on filler,
quest text, stat lists or chat reading, and anything that only works if you squint. Keep what lands.
Give your own confidence 0-1 that it stays in your final edit.
Reply with ONLY JSON: {"verdicts": [{"i": 0, "keep": true, "confidence": 0.8, "note": "short"}]}"""


def _call(system_prompt: str, user: str, Out, model: str, backend: str, tag: str, ask=None, cwd=None):
    """cwd: a folder of frame sheets the model may Read (cli only; ponytail: the api fallback goes blind)."""
    from .. import director
    if ask:
        return ask(user, Out)
    content = [{"type": "text", "text": user}]
    if backend == "cli":
        try:
            return director._complete_cli(model, system_prompt, content, Out, tag=tag, effort="medium", live=False,
                                          tools="Read" if cwd else "", cwd=cwd)
        except (RuntimeError, OSError) as e:
            director._api_fallback(tag, e)
    import anthropic
    return director._complete(anthropic.Anthropic(), model, system_prompt, content, Out, tag=tag, effort="medium")


def _models():
    from pydantic import BaseModel

    class Ev(BaseModel):
        phrase: int
        type: str
        asset: str | None = None
        confidence: float = 0.0
        reason: str = ""

    class Out(BaseModel):
        events: list[Ev]

    class Verdict(BaseModel):
        i: int
        keep: bool = True
        confidence: float = 0.0
        note: str = ""

    class Critic(BaseModel):
        verdicts: list[Verdict]

    return Out, Critic


def problems(events: list[dict], ids: set, names: set) -> list[str]:
    errs = []
    for e in events:
        if e["type"] not in ("boom", "emote"):
            errs.append(f"type {e['type']!r}: boom or emote only")
        if e["phrase"] not in ids:
            errs.append(f"no phrase p{e['phrase']}")
        if e["type"] == "emote" and e.get("asset") not in names:
            errs.append(f"emote {e.get('asset')!r} is not in the library")
    return errs


def marked(phr: list[dict]) -> set:
    """the strongest-signal phrases: starred in the prompt, framed in the sheets."""
    return {p["id"] for p in sorted(phr, key=lambda p: -p["s"])[:MARK] if p["s"] > 0}


def direct(data: dict, prof: dict, library: list[dict], exemplars: list[dict], model: str = "claude-opus-5",
           backend: str = "cli", ask=None, sheet_names=(), cwd=None, log=print) -> list[dict]:
    """director (+ one corrective retry) then critic -> [{phrase, type, asset, confidence, reason, critic}].
    sheet_names: frame sheets in cwd (sheets()) the director Reads first."""
    Out, Critic = _models()
    phr = data["phrases"]
    mark = marked(phr)
    ex = retrieve(exemplars, [p["text"] for p in phr if p["id"] in mark])
    ex_txt = "\n".join(f"- [{r['type']}{':' + str(r['asset']) if r['type'] == 'emote' else ''}] {r['text']}"
                       + (f"  (why: {r['why']})" if r.get("why") else "") for r in ex if r["type"] in ("boom", "emote"))
    look = (f"\n\nFrame sheets (Read every one first; each row is one starred phrase at 1 s before its end, its end, "
            f"0.7 s after): {', '.join(sheet_names)}") if sheet_names else ""
    user = (f"Video: {json.dumps(data['context'], ensure_ascii=False)}\n\nWhat Nguy did before:\n{ex_txt}\n\n"
            f"The cut:\n{transcript(phr, mark)}{look}")
    sys_p = system(prof, library)
    ids, names = {p["id"] for p in phr}, {r["name"] for r in library}
    see = cwd if sheet_names else None
    out = _call(sys_p, user, Out, model, backend, "sauce director", ask, see)
    evs = [e.model_dump() for e in out.events]
    errs = problems(evs, ids, names)
    if errs:
        out = _call(sys_p, user + "\n\nYour last answer broke these rules, fix them: " + "; ".join(errs[:20]),
                    Out, model, backend, "sauce director", ask, see)
        evs = [e for e in (x.model_dump() for x in out.events) if not problems([e], ids, names)]
    log(f"[sauce] director: {len(evs)} proposals")
    if not evs:
        return []
    by = {p["id"]: p for p in phr}
    words = [w for p in phr for w in p["words"]]

    def around(t: float) -> str:                   # phrase words carry timeline times in t0/t1
        return (" ".join(w["text"] for w in words if t - 8 <= w["t0"] < t) + " [*] "
                + " ".join(w["text"] for w in words if t <= w["t0"] < t + 8))

    rev = "\n".join(f"{i}. [{e['type']}{':' + str(e['asset']) if e['type'] == 'emote' else ''}] "
                    f"{around(by[e['phrase']]['t1'])}  (proposed because: {e['reason']})" for i, e in enumerate(evs))
    verdicts = {v.i: v for v in _call(CRITIC, rev, Critic, model, backend, "sauce critic", ask).verdicts}
    for i, e in enumerate(evs):
        v = verdicts.get(i)
        e["critic"] = v.note if v else ""
        # both have to believe it: the critic can only lower the director's confidence or drop it
        e["confidence"] = 0.0 if v and not v.keep else round(min(e["confidence"], v.confidence if v else e["confidence"]), 3)
    log(f"[sauce] critic kept {sum(e['confidence'] > 0 for e in evs)} of {len(evs)}")
    return evs
