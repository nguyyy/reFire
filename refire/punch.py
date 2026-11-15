"""`refire punch`: hard-cut 210% zooms on the facecam at genuinely good moments.

docs/specs/01-punch.md. Runs off the live timeline (timeline_map), so a re-cut re-punches.
Pipeline: transcript -> timeline phrases -> features -> cheap prefilter (+ face check) ->
one batched judge call -> score -> timing (pre/post roll, cut snap) -> spacing/merge ->
min length / soft long split / coverage cap -> pin + scale per facecam. Everything here is
deterministic; the judge is injected (`judge(cands) -> {id: {category, worthy, reason}}`).
All times are timeline seconds unless named src_*.
"""
from __future__ import annotations

import math
import re
from bisect import bisect_left
from pathlib import Path

import numpy as np

from .subtitles import MAX_WORD_S

DEFAULTS = {
    "scale": 2.10, "min_scale": 1.50, "preroll": 0.10, "postroll": 0.10, "snap_to_cut": 0.40,
    "merge_gap": 2.0, "min_gap": 4.0, "min_len": 0.5, "laugh_min_len": 1.0, "soft_long": 6.0,
    "coverage_window": 60.0, "coverage_window_max": 0.25, "coverage_total_max": 0.20,
    "threshold": 0.60, "weights": {"judge": 0.5, "loud": 0.2, "chat": 0.15, "flags": 0.15},
    "ocr": "auto", "sharpen": {"amount": 50, "radius": 1.0, "threshold": 0}, "track": 2,
}
PAUSE = 0.30                 # phrase split
LAUGH_MIN = 1.0              # sustained laughter (P2)
SOFT_EVERY = 3.0             # a long merged punch needs a qualifying phrase this often
ZERO_CATS = {"reading", "stats", "filler"}
CATEGORIES = ("exclamation", "reaction", "punchline", "roast", "laughter", "rage",
              "dramatic_read", "reading", "stats", "filler", "other")
LEXICON = Path(__file__).with_name("punch_lexicon.txt")
_SENT_END = re.compile(r"[.!?]['\")\]]*$")
_NORM = re.compile(r"[^a-z0-9' ]+")


def config(overrides: dict | None = None) -> dict:
    cfg = {**DEFAULTS, **(overrides or {})}
    for k in ("weights", "sharpen"):          # partial maps merge into the defaults
        cfg[k] = {**DEFAULTS[k], **((overrides or {}).get(k) or {})}
    return cfg


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# --------------------------------------------------------------------------- phrases

def timeline_words(words: list[dict], tmap) -> list[dict]:
    """transcript words -> copies placed on the timeline (t0/t1), one per appearance."""
    starts = [w["start"] for w in words]
    out = []
    for si, s in enumerate(tmap.spans):
        if s.src_in is None:
            continue
        for w in words[bisect_left(starts, s.src_in):bisect_left(starts, s.src_out)]:
            t0 = s.tl_in + (w["start"] - s.src_in) / s.rate
            out.append({**w, "t0": t0, "t1": min(s.tl_out, s.tl_in + (w["end"] - s.src_in) / s.rate),
                        "span": si, "src": s.source_id})
    return sorted(out, key=lambda w: w["t0"])


def _continuous(a, b) -> bool:
    """two V1 clips butt together with no jump in source (a razor split, not an edit)."""
    return (a.source_id == b.source_id and abs(a.tl_out - b.tl_in) < 0.05
            and abs(a.src_out - b.src_in) < 0.1)


def phrases(tw: list[dict], tmap) -> list[dict]:
    """split at pauses > PAUSE (word length clamped, whisper pads words) or sentence ends.
    a phrase that runs across a cut into other footage is marked `straddle` (dropped)."""
    out, cur = [], []
    for w in tw:
        if cur:
            p = cur[-1]
            gap = w["t0"] - min(p["t1"], p["t0"] + MAX_WORD_S)
            if gap > PAUSE or _SENT_END.search(p["text"].strip()):
                out.append(cur)
                cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    res = []
    for i, ws in enumerate(out):
        straddle = any(a["span"] != b["span"] and not _continuous(tmap.spans[a["span"]], tmap.spans[b["span"]])
                       for a, b in zip(ws, ws[1:]))
        res.append({"id": f"c{i}", "t0": ws[0]["t0"], "t1": ws[-1]["t1"], "words": ws,
                    "text": " ".join(w["text"].strip() for w in ws), "source": ws[0]["src"],
                    "src_t0": ws[0]["start"], "src_t1": ws[-1]["end"], "straddle": straddle})
    return res


def laugh_candidates(laughs, tmap, phr: list[dict], start_id: int) -> list[dict]:
    """sustained laughs with no phrase on them become their own candidates ("[laughter]")."""
    out = []
    for a, b, prob in laughs or []:
        if b - a < LAUGH_MIN:
            continue
        for s in tmap.ours():
            if s.src_in <= a < s.src_out:
                t0 = s.tl_in + (a - s.src_in) / s.rate
                t1 = min(s.tl_out, s.tl_in + (b - s.src_in) / s.rate)
                if any(p["t0"] - 0.3 < t1 and t0 < p["t1"] + 0.3 for p in phr):
                    continue
                out.append({"id": f"c{start_id + len(out)}", "t0": t0, "t1": t1, "words": [],
                            "text": "[laughter]", "source": s.source_id, "src_t0": a, "src_t1": b,
                            "straddle": False})
    return out


# --------------------------------------------------------------------------- features

def load_lexicon(path: Path = LEXICON) -> list[str]:
    if not path.exists():
        return []
    return [_NORM.sub(" ", ln.split("#")[0].lower()).strip()
            for ln in path.read_text(encoding="utf-8").splitlines() if ln.split("#")[0].strip()]


def _lex_hit(text: str, lexicon: list[str]) -> bool:
    t = " " + " ".join(_NORM.sub(" ", text.lower()).split()) + " "
    return any(f" {e} " in t for e in lexicon if e)


class Signal:
    """(t_center, z) series on the transcript clock -> peak/mean over a span."""

    def __init__(self, sig):
        self.t = np.array([t for t, _ in sig or []], float)
        self.z = np.array([z for _, z in sig or []], float)
        self.half = (self.t[1] - self.t[0]) / 2 if len(self.t) > 1 else 0.5

    def _sel(self, a, b):
        return self.z[(self.t >= a - self.half) & (self.t <= b + self.half)]

    def peak(self, a, b) -> float:
        v = self._sel(a, b)
        return float(v.max()) if v.size else 0.0

    def mean(self, a, b) -> float:
        v = self._sel(a, b)
        return float(v.mean()) if v.size else 0.0


def features(p: dict, loud: Signal, chat: Signal, lexicon, laughs) -> dict:
    a, b = p["src_t0"], p["src_t1"]
    near = [(la, lb, pr) for la, lb, pr in laughs or [] if la < b + 1.0 and lb > a - 1.0]
    dur = max(p["t1"] - p["t0"], 1e-3)
    return {
        "z_loud_peak": round(loud.peak(a, b), 3), "z_loud_mean": round(loud.mean(a, b), 3),
        "z_chat": round(chat.peak(a, a + 6.0), 3),
        "laugh": round(max((pr for *_, pr in near), default=0.0), 3),
        "laugh_dur": round(max((lb - la for la, lb, _ in near), default=0.0), 3),
        # trailing laughter: starts inside the phrase or within 1 s of its end
        "laugh_end": max((lb for la, lb, _ in near if a <= la <= b + 1.0 and lb - la >= LAUGH_MIN),
                         default=None),
        "lex": _lex_hit(p["text"], lexicon),
        "caps": any(w.get("emph") for w in p["words"]),
        "n_words": len(p["words"]), "dur": round(dur, 3), "rate": round(len(p["words"]) / dur, 2),
    }


def prefilter_reasons(f: dict) -> list[str]:
    return [k for k, ok in (("loud", f["z_loud_peak"] >= 1.5), ("lex", f["lex"]), ("caps", f["caps"]),
                            ("laugh", f["laugh_dur"] >= LAUGH_MIN), ("chat", f["z_chat"] >= 2.0)) if ok]


# --------------------------------------------------------------------------- score

def score(f: dict, judged: dict, cfg: dict) -> float:
    cat, w = judged.get("category", "other"), float(judged.get("worthy", 0.0))
    reading = f.get("reading") or cat in ("reading", "dramatic_read")
    if cat in ZERO_CATS or (reading and not (cat == "dramatic_read" and f["z_loud_peak"] >= 2.0)):
        return 0.0
    k = cfg["weights"]
    flag = f["lex"] or f["caps"] or f["laugh_dur"] >= LAUGH_MIN
    return round(k["judge"] * w + k["loud"] * sigmoid(f["z_loud_peak"] - 1.5)
                 + k["chat"] * sigmoid(f["z_chat"] - 1.5) + k["flags"] * flag, 4)


# --------------------------------------------------------------------------- timing

def iou(a, b) -> float:
    if not a or not b:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / u if u > 0 else 0.0


def timing(p: dict, f: dict, cuts: list[float], tmap, boxes: dict, cfg: dict) -> tuple[float, float, str | None]:
    """-> (t_in, t_out, snapped) per spec step 7."""
    end = p["t1"]
    if f.get("laugh_end") is not None and p["words"]:        # trailing laughter extends the end
        last = p["words"][-1]
        s = tmap.spans[last["span"]]
        end = max(end, min(s.tl_out, s.tl_in + (f["laugh_end"] - s.src_in) / s.rate))
    t_in, t_out, snapped = p["t0"] - cfg["preroll"], end + cfg["postroll"], []
    w = cfg["snap_to_cut"]
    before = [c for c in cuts if t_in - w <= c <= t_in]
    if before:
        t_in = max(before)
        snapped.append("in")
    after = [c for c in cuts if t_out <= c <= t_out + w]
    if after:
        t_out = min(after)
        snapped.append("out")
    # a cut inside the punch: keep spanning only into the same facecam with the phrase going on
    for c in sorted(x for x in cuts if t_in < x < t_out):
        left, right = tmap.timeline_to_source(c - 1e-3), tmap.timeline_to_source(c)
        same = (left and right and iou(boxes.get(left[0]), boxes.get(right[0])) >= 0.8
                and p["t0"] < c < p["t1"])
        if not same:
            if c <= p["t0"]:
                t_in = c
            else:
                t_out = c
                break
    return round(t_in, 4), round(t_out, 4), "+".join(snapped) or None


# --------------------------------------------------------------------------- spacing

def _merge(a: dict, b: dict) -> dict:
    hi = a if a["score"] >= b["score"] else b
    return {**hi, "t_in": min(a["t_in"], b["t_in"]), "t_out": max(a["t_out"], b["t_out"]),
            "score": max(a["score"], b["score"]), "text": f"{a['text']} / {b['text']}",
            "merged_from": a["merged_from"] + b["merged_from"],
            "parts": a["parts"] + b["parts"], "laugh_only": a["laugh_only"] and b["laugh_only"]}


def space(cands: list[dict], cfg: dict, dropped: dict) -> list[dict]:
    """step 8: merge gaps < merge_gap, keep the better of gaps < min_gap."""
    kept: list[dict] = []
    for c in sorted(cands, key=lambda c: c["t_in"]):
        while True:
            if not kept:
                kept.append(c)
                break
            prev = kept[-1]
            gap = c["t_in"] - prev["t_out"]
            if gap < cfg["merge_gap"]:
                c = _merge(kept.pop(), c)          # may now sit too close to the one before
                continue
            if gap < cfg["min_gap"]:
                win, lose = (prev, c) if prev["score"] >= c["score"] else (c, prev)
                for i in lose["merged_from"]:
                    dropped[i] = f"spacing_lost_to:{win['merged_from'][0]}"
                if win is c:
                    kept.pop()
                    continue                       # re-check c against the new neighbour
                break
            kept.append(c)
            break
    return kept


def _energy(loud: Signal, tmap, a: float, b: float) -> float:
    src = [tmap.timeline_to_source(t) for t in np.linspace(a, b, 5)]
    v = [loud.peak(s[1], s[1]) for s in src if s]
    return float(np.mean(v)) if v else 0.0


def soft_long(k: dict, cfg: dict, loud: Signal, tmap, dropped: dict) -> list[dict]:
    """a long merged punch with a > SOFT_EVERY stretch of no qualifying phrase is split at
    its quietest gap; the weaker side goes. ponytail: inert at defaults (merge_gap 2 < 3)."""
    if k["t_out"] - k["t_in"] <= cfg["soft_long"] or len(k["parts"]) < 2:
        return [k]
    parts = sorted(k["parts"], key=lambda p: p["t0"])
    gaps = [(parts[i]["t1"], parts[i + 1]["t0"], i) for i in range(len(parts) - 1)]
    if max(b - a for a, b, _ in gaps) <= SOFT_EVERY:
        return [k]
    a, b, i = min(gaps, key=lambda g: _energy(loud, tmap, g[0], g[1]))
    sides = [parts[:i + 1], parts[i + 1:]]
    keep, lose = sorted(sides, key=lambda s: max(p["score"] for p in s), reverse=True)
    for p in lose:
        dropped[p["id"]] = "soft_long_split"
    t_in = keep[0]["t0"] - cfg["preroll"] if keep[0] is not parts[0] else k["t_in"]
    t_out = keep[-1]["t1"] + cfg["postroll"] if keep[-1] is not parts[-1] else k["t_out"]
    nk = {**k, "t_in": t_in, "t_out": t_out, "parts": keep, "merged_from": [p["id"] for p in keep],
          "score": max(p["score"] for p in keep)}
    return soft_long(nk, cfg, loud, tmap, dropped)


def coverage(kept: list[dict], total: float, cfg: dict, dropped: dict) -> list[dict]:
    """drop lowest scores until punched time <= window_max of any rolling window and
    <= total_max of the video."""
    kept = list(kept)
    W = cfg["coverage_window"]

    def overlap(a, b):
        return sum(max(0.0, min(b, k["t_out"]) - max(a, k["t_in"])) for k in kept)

    while kept:
        worst = None
        if sum(k["t_out"] - k["t_in"] for k in kept) > cfg["coverage_total_max"] * total:
            worst = kept
        else:
            for a in sorted({k["t_in"] for k in kept} | {k["t_out"] - W for k in kept}):
                a = min(max(0.0, a), max(0.0, total - W))
                if overlap(a, a + W) > cfg["coverage_window_max"] * min(W, total) + 1e-9:
                    worst = [k for k in kept if k["t_out"] > a and k["t_in"] < a + W]
                    break
        if worst is None:
            return kept
        lo = min(worst, key=lambda k: k["score"])
        for i in lo["merged_from"]:
            dropped[i] = "coverage_cap"
        kept.remove(lo)
    return kept


# --------------------------------------------------------------------------- pin + scale

def pin(box: list[float]) -> list[float]:
    def axis(c):
        return 0.0 if c < 1 / 3 else 1.0 if c > 2 / 3 else round(c, 4)
    return [axis((box[0] + box[2]) / 2), axis((box[1] + box[3]) / 2)]


def max_scale(box: list[float], p: list[float], cap: float) -> float:
    """largest s <= cap that keeps the whole box on screen when scaling about p."""
    s = cap
    for e, q in ((box[0], p[0]), (box[2], p[0]), (box[1], p[1]), (box[3], p[1])):
        if e > q + 1e-9:
            s = min(s, (1 - q) / (e - q))
        elif e < q - 1e-9:
            s = min(s, q / (q - e))
    return round(s, 4)


# --------------------------------------------------------------------------- whole pass

def analyze(phr: list[dict], feats: dict, judge, face, cams: dict, cuts: list[float], tmap,
            total: float, cfg: dict, loud: Signal):
    """phrases + features -> (punches, review rows). `face(p) -> fraction`, `cams` = source_id
    -> Facecam|None, `judge(cands) -> {id: verdict}` (called once, prefilter survivors only)."""
    dropped: dict[str, str] = {}
    boxes, framing = {}, {}
    for sid, cam in cams.items():
        if cam is None or not cam.usable:
            framing[sid] = "no_facecam"
            continue
        boxes[sid] = cam.box
        pt = pin(cam.box)
        s = max_scale(cam.box, pt, cfg["scale"])
        framing[sid] = (pt, s) if s >= cfg["min_scale"] else "facecam_too_large"

    survivors = []
    for p in phr:
        f = feats[p["id"]]
        f["prefilter"] = prefilter_reasons(f)
        if p["straddle"]:
            dropped[p["id"]] = "straddles_cut"
        elif not f["prefilter"]:
            dropped[p["id"]] = "no_signal"
        elif not isinstance(framing.get(p["source"]), tuple):
            dropped[p["id"]] = framing.get(p["source"], "no_facecam")
        else:
            f["face"] = round(face(p), 3)
            if f["face"] < 0.5:
                dropped[p["id"]] = "no_face"
            else:
                survivors.append(p)

    verdicts = judge([{"id": p["id"], "text": p["text"], "t0": p["t0"], "t1": p["t1"],
                       "src_t0": p["src_t0"], "src_t1": p["src_t1"], "features": feats[p["id"]]}
                      for p in survivors]) if survivors else {}
    cands = []
    for p in survivors:
        f, v = feats[p["id"]], verdicts.get(p["id"], {})
        f.update(category=v.get("category", "other"), worthy=v.get("worthy", 0.0), reason=v.get("reason", ""))
        f["score"] = score(f, v, cfg)
        if f["score"] < cfg["threshold"]:
            cat = f["category"]
            dropped[p["id"]] = (cat if cat in ZERO_CATS else
                                "reading" if cat == "dramatic_read" and not f["score"] else "below_threshold")
            continue
        t_in, t_out, snapped = timing(p, f, cuts, tmap, boxes, cfg)
        part = {"id": p["id"], "t0": p["t0"], "t1": p["t1"], "score": f["score"]}
        cands.append({"t_in": t_in, "t_out": t_out, "score": f["score"], "category": f["category"],
                      "text": p["text"], "reason": f["reason"], "source": p["source"],
                      "merged_from": [p["id"]], "parts": [part], "snapped_to_cut": snapped,
                      "laugh_only": f["prefilter"] == ["laugh"]})

    kept = space(cands, cfg, dropped)
    out = []
    for k in kept:
        n = k["t_out"] - k["t_in"]
        if n < cfg["min_len"] or (k["laugh_only"] and n < cfg["laugh_min_len"]):
            for i in k["merged_from"]:
                dropped[i] = "too_short"
            continue
        out.extend(soft_long(k, cfg, loud, tmap, dropped))
    out = coverage(out, total, cfg, dropped)

    punches = []
    for n, k in enumerate(sorted(out, key=lambda k: k["t_in"]), 1):
        pt, s = framing[k["source"]]
        punches.append({"id": f"p{n:02d}", "t_in": k["t_in"], "t_out": k["t_out"], "pin": pt, "scale": s,
                        "score": k["score"], "category": k["category"], "text": k["text"],
                        "reason": k["reason"], "source": k["source"], "merged_from": k["merged_from"],
                        "snapped_to_cut": k["snapped_to_cut"]})
    owner = {i: p["id"] for p in punches for i in p["merged_from"]}
    for i, r in dropped.items():                 # name the winner by its punch id when it survived
        if r.startswith("spacing_lost_to:") and r.split(":")[1] in owner:
            dropped[i] = "spacing_lost_to:" + owner[r.split(":")[1]]
    review = []
    for p in phr:
        f = feats[p["id"]]
        review.append({"id": p["id"], "t0": round(p["t0"], 3), "t1": round(p["t1"], 3), "text": p["text"],
                       **{k: f.get(k, "") for k in ("z_loud_peak", "z_loud_mean", "z_chat", "laugh", "laugh_dur",
                                                    "lex", "caps", "n_words", "dur", "rate", "face",
                                                    "category", "worthy", "score", "reason")},
                       "prefilter": "+".join(f.get("prefilter", [])),
                       "status": f"kept:{owner[p['id']]}" if p["id"] in owner else "dropped",
                       "drop_reason": "" if p["id"] in owner else dropped.get(p["id"], "?")})
    return punches, review


# --------------------------------------------------------------------------- judge

JUDGE_SYSTEM = """You judge candidate punch-in moments for a gaming stream edit (streamer: zy0x).
A punch-in is a hard cut to a 210% zoom on the streamer's facecam while he reacts, then a hard
cut back. The editor's rules, verbatim:

- Never punch on uninteresting short phrases ("okay i see" is out; "OH SHIT" and real
  exclamations are in).
- Never punch too often or stack punches.
- Never punch on boring dialogue: reading quest dialogue, listing stats, reading in-game text,
  plain monologue. Exception: a genuinely interesting or dramatic read can qualify.
- Laughter counts, but only when sustained.

Examples: "okay i see" -> filler, worthy ~0. "OH SHIT" (loud) -> exclamation, worthy ~0.9.
A flat read of quest dialogue -> reading. Rattling off artifact substats -> stats.

Each candidate has its text, about 8 s of surrounding stream transcript, and measured
features (z_loud_peak = loudness spike vs the local baseline, z_chat = chat rate spike,
laugh_dur = sustained laughter seconds, lex/caps = exclamation word / shouted word).
Use the context and the game glossary to spot reading: NPC names, quest-dialogue register,
lists of stats or item text. A reading that is genuinely dramatic is `dramatic_read`.

Categories: exclamation, reaction, punchline, roast, laughter, rage, dramatic_read, reading,
stats, filler, other. `worthy` is 0-1: how much the editor would want this punched.
Reply with ONLY a JSON object: {"verdicts": [{"id": "...", "category": "...",
"worthy": 0.0, "reason": "one short line"}, ...]} with one entry per candidate, same ids."""


def _verdict_models():
    from pydantic import BaseModel, field_validator

    class Verdict(BaseModel):
        id: str
        category: str
        worthy: float
        reason: str = ""

        @field_validator("category")
        @classmethod
        def known(cls, v):
            return v if v in CATEGORIES else "other"

        @field_validator("worthy")
        @classmethod
        def clamp(cls, v):
            return min(1.0, max(0.0, v))

    class Verdicts(BaseModel):
        verdicts: list[Verdict]

    return Verdicts


def context_text(words: list[dict], a: float, b: float, pad: float = 8.0) -> str:
    """what was said in the stream around [a, b] (transcript clock)."""
    starts = [w["start"] for w in words]
    return " ".join(w["text"].strip() for w in words[bisect_left(starts, a - pad):bisect_left(starts, b + pad)])


def judge_payload(cands: list[dict], words: list[dict], glossary=()) -> str:
    keys = ("z_loud_peak", "z_chat", "laugh_dur", "lex", "caps", "dur", "rate")
    rows = [{"id": c["id"], "text": c["text"], "context": context_text(words, c["src_t0"], c["src_t1"]),
             "features": {k: c["features"].get(k) for k in keys}} for c in cands]
    import json
    return json.dumps({"glossary": list(glossary), "candidates": rows}, ensure_ascii=False)


def make_judge(run_dir, words: list[dict], glossary=(), backend: str = "cli",
               model: str = "claude-opus-5", local_model: str | None = None, effort: str = "medium"):
    """-> judge(cands) for analyze(): ONE call per video, cached by input hash in
    <run_dir>/punch_judge.json. backend cli (subscription) -> api fallback; local_model =
    --local-director (ollama via local_llm)."""
    import hashlib
    import json

    cache_path = Path(run_dir) / "punch_judge.json"

    def judge(cands):
        if not cands:
            return {}
        user = judge_payload(cands, words, glossary)
        who = f"local:{local_model}" if local_model else f"{backend}:{model}"
        key = hashlib.sha256((who + JUDGE_SYSTEM + user).encode("utf-8")).hexdigest()[:16]
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        if key not in cache:
            Verdicts = _verdict_models()
            if local_model:
                from . import local_llm
                got = Verdicts.model_validate(json.loads(local_llm.chat(
                    local_model, [{"role": "system", "content": JUDGE_SYSTEM},
                                  {"role": "user", "content": user}],
                    options={"num_ctx": 16384}, num_predict=4096, stall_s=300, total_s=1200,
                    tag="punch judge")))
            else:
                from . import director
                content = [{"type": "text", "text": user}]
                got = None
                if backend == "cli":
                    try:
                        got = director._complete_cli(model, JUDGE_SYSTEM, content, Verdicts,
                                                     tag="punch judge", effort=effort, live=False)
                    except (RuntimeError, OSError) as e:
                        director._api_fallback("punch judge", e)
                if got is None:
                    import anthropic
                    got = director._complete(anthropic.Anthropic(), model, JUDGE_SYSTEM, content,
                                             Verdicts, tag="punch judge", effort=effort)
            cache[key] = {v.id: v.model_dump(exclude={"id"}) for v in got.verdicts}
            cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        return cache[key]

    return judge
