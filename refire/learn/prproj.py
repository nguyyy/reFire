"""Read a Premiere Pro project (.prproj) in memory. Never writes, never re-saves.

A .prproj is gzip'd XML: a flat list of objects linked by ObjectID/ObjectRef and
ObjectUID/ObjectURef. Checked against the user's real projects (2026-11), one clip on a
timeline is this chain:

    Sequence/TrackGroups/TrackGroup/Second -> VideoTrackGroup|AudioTrackGroup
      /TrackGroup/Tracks/Track -> VideoClipTrack|AudioClipTrack
      /ClipTrack/ClipItems/TrackItems/TrackItem -> VideoClipTrackItem|AudioClipTrackItem
      /ClipTrackItem/TrackItem/{Start,End}     sequence ticks, Start omitted when 0
      /ClipTrackItem/SubClip -> SubClip/Clip -> VideoClip|AudioClip
      /Clip/{InPoint,OutPoint}                 source ticks
      /Clip/Source -> VideoMediaSource/MediaSource/Media -> Media/FilePath
                   or VideoSequenceSource/SequenceSource/Sequence -> a nested Sequence

No speed or disabled element appears in any of those projects, so speed is taken as the
ratio of source span to sequence span (true whatever Premiere calls it), and an item
only counts as disabled if its ClipTrackItem carries an explicit Disabled/IsDisabled
true or Enabled false.
"""
from __future__ import annotations

import base64
import gzip
import re
import struct
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

TICKS = 254016000000          # premiere ticks per second


def _sec(el, tag: str) -> float:
    t = el.findtext(tag)
    return int(t) / TICKS if t and t.strip().lstrip("-").isdigit() else 0.0


class Project:
    """The parsed object graph. `seqs` maps sequence uid -> {"name", "items"}."""

    def __init__(self, path):
        self.path = Path(path)
        with gzip.open(self.path, "rb") as f:          # read-only, in memory
            self.root = ET.fromstring(f.read())
        self.by_id, self.by_uid = {}, {}
        for el in self.root:
            if "ObjectID" in el.attrib:
                self.by_id[el.attrib["ObjectID"]] = el
            if "ObjectUID" in el.attrib:
                self.by_uid[el.attrib["ObjectUID"]] = el
        self.seqs = {uid: self._sequence(el) for uid, el in self.by_uid.items()
                     if el.tag == "Sequence"}

    def ref(self, el):
        if el is None:
            return None
        a = el.attrib
        if "ObjectRef" in a:
            return self.by_id.get(a["ObjectRef"])
        if "ObjectURef" in a:
            return self.by_uid.get(a["ObjectURef"])
        return None

    def _sequence(self, seq) -> dict:
        items = []
        for g in seq.findall("TrackGroups/TrackGroup"):
            grp = self.ref(g.find("Second"))
            if grp is None or grp.tag not in ("VideoTrackGroup", "AudioTrackGroup"):
                continue
            kind = "video" if grp.tag == "VideoTrackGroup" else "audio"
            for t in grp.findall("TrackGroup/Tracks/Track"):
                track = self.ref(t)
                if track is None:
                    continue
                idx = int(t.attrib.get("Index", 0))
                for ti in track.findall("ClipTrack/ClipItems/TrackItems/TrackItem"):
                    it = self._item(self.ref(ti), kind, idx)
                    if it:
                        items.append(it)
        return {"name": seq.findtext("Name") or "", "items": items}

    def _item(self, el, kind: str, track: int) -> dict | None:
        if el is None:
            return None
        cti = el.find("ClipTrackItem")
        if cti is None:
            return None
        ti = cti.find("TrackItem")
        sub = self.ref(cti.find("SubClip"))
        clip = self.ref(sub.find("Clip")) if sub is not None else None
        if ti is None or clip is None:
            return None
        c = clip.find("Clip")
        src = self.ref(c.find("Source")) if c is not None else None
        media = nested = None
        if src is not None:
            m = self.ref(src.find("MediaSource/Media"))
            if m is not None:
                media = m.findtext("FilePath") or m.findtext("ActualMediaFilePath")
            s = src.find("SequenceSource/Sequence")
            if s is not None:
                nested = s.attrib.get("ObjectURef")
        flags = {d.tag: (d.text or "").strip().lower() for d in cti}
        enabled = not (flags.get("Disabled") == "true" or flags.get("IsDisabled") == "true"
                       or flags.get("Enabled") == "false")
        return {"kind": kind, "track": track,
                "seq_in": _sec(ti, "Start"), "seq_out": _sec(ti, "End"),
                "src_in": _sec(c, "InPoint"), "src_out": _sec(c, "OutPoint"),
                "media": media, "nested": nested, "enabled": enabled}


def speed(it: dict) -> float:
    span = it["seq_out"] - it["seq_in"]
    return (it["src_out"] - it["src_in"]) / span if span > 1e-9 else 1.0


def flatten(proj: Project, uid: str, kinds=("video",), _depth: int = 0) -> list[dict]:
    """Every enabled media item that plays on sequence `uid`, nested sequences resolved.

    A nested item shows its child sequence's [src_in, src_out] window, so each child
    item is clipped to that window and shifted (and speed-scaled) onto the parent's
    clock. `nest` records the path of sequence names it came through.
    """
    out = []
    seq = proj.seqs.get(uid)
    if seq is None or _depth > 8:                     # cycle / runaway guard
        return out
    for it in seq["items"]:
        if it["kind"] not in kinds or not it["enabled"]:
            continue
        if it["media"]:
            out.append({**it, "nest": [seq["name"]]})
            continue
        if not it["nested"]:
            continue
        rate = speed(it)
        for ch in flatten(proj, it["nested"], kinds, _depth + 1):
            a, b = max(ch["seq_in"], it["src_in"]), min(ch["seq_out"], it["src_out"])
            if b - a <= 1e-6:
                continue
            cr = speed(ch)
            out.append({**ch,
                        "seq_in": it["seq_in"] + (a - it["src_in"]) / rate,
                        "seq_out": it["seq_in"] + (b - it["src_in"]) / rate,
                        "src_in": ch["src_in"] + (a - ch["seq_in"]) * cr,
                        "src_out": ch["src_in"] + (b - ch["seq_in"]) * cr,
                        "track": it["track"],
                        "nest": [seq["name"]] + ch["nest"]})
    return sorted(out, key=lambda x: (x["seq_in"], x["track"]))


def nested_uids(proj: Project) -> set[str]:
    """Sequences used as a clip inside another sequence."""
    return {it["nested"] for s in proj.seqs.values() for it in s["items"] if it["nested"]}


def pick_final(proj: Project, keep=None) -> tuple[str | None, str]:
    """(uid, reason) of the sequence the user exported.

    Heuristic, logged: prefer top-level sequences (not nested inside another), then the
    one with the most footage items that `keep(media_path)` accepts (most edited), then
    the longest. A sequence nesting others counts its nested footage. `keep` lets the
    caller restrict to one run's media when a project holds several cuts.
    """
    keep = keep or (lambda _m: True)
    inner = nested_uids(proj)
    scored = []
    for uid, s in proj.seqs.items():
        items = [i for i in flatten(proj, uid) if keep(i["media"])]
        if not items:
            continue
        dur = max(i["seq_out"] for i in items)
        scored.append((uid not in inner, len(items), dur, uid, s["name"]))
    if not scored:
        return None, "no sequence holds matching footage"
    scored.sort(reverse=True)
    top, n, dur, uid, name = scored[0]
    rest = "; ".join(f"'{x[4]}' {x[1]} items" for x in scored[1:4])
    return uid, (f"'{name}': {'top-level' if top else 'nested'}, {n} footage items, "
                 f"{dur:.0f}s" + (f" (beat {rest})" if rest else ""))


_RUN_RE = re.compile(rb"\\run\\([^\\<]+)\\([^\\<]+)\\")


def referenced_runs(path) -> dict[str, int]:
    """'<cache dir>/<run name>' -> media-path count, by a byte scan (no XML parse).

    Cheap enough to run over a whole folder of projects to find which one a run's
    proxies ended up in."""
    with gzip.open(path, "rb") as f:
        data = f.read()
    out: dict[str, int] = {}
    for m in re.finditer(rb"<FilePath>([^<]*)</FilePath>", data):
        r = _RUN_RE.search(m.group(1))
        if r:
            key = f"{r.group(1).decode('utf-8', 'replace')}/{r.group(2).decode('utf-8', 'replace')}"
            out[key] = out.get(key, 0) + 1
    return out


# --------------------------------------------------------------------------- captions

def _reach(proj: Project, el, pred, depth: int = 5):
    """First object reachable from `el` through Object(U)Refs that satisfies pred (BFS)."""
    seen, layer = {id(el)}, [el]
    for _ in range(depth):
        nxt = []
        for e in layer:
            for c in e.iter():
                r = proj.ref(c)
                if r is not None and id(r) not in seen:
                    if pred(r):
                        return r
                    seen.add(id(r))
                    nxt.append(r)
        layer = nxt
    return None


def _strings(b: bytes) -> list[str]:
    """Length-prefixed, NUL-terminated utf-8 strings in a FlatBuffer blob."""
    out = []
    for i in range(len(b) - 4):
        n = struct.unpack_from("<I", b, i)[0]
        if 0 < n < 400 and i + 4 + n < len(b) and b[i + 4 + n] == 0:
            try:
                s = b[i + 4:i + 4 + n].decode("utf-8")
            except UnicodeDecodeError:
                continue
            if all(ch.isprintable() or ch == "\n" for ch in s):
                out.append(s)
    return out


def captions(proj: Project) -> dict[str, list[tuple[float, float, str]]]:
    """Caption files imported into the project -> {file path: [(t0, t1, text)]}.

    Premiere copies an imported .srt's cues into the project (MasterClip -> TranscriptClip
    -> CaptionCollection -> Caption{TimeStart,TimeEnd} -> Block), so reFire's
    ae/captions.srt -- the speech its cut carried, in cut time -- survives the run folder.
    Cue text sits in a base64 FlatBuffer next to style keys and the font name; those
    repeat in nearly every cue, the text doesn't. Sequence caption tracks (owned by the
    Project, not a MasterClip) are the editor's own captions and are skipped.
    """
    # a repeated blob is stored once; later blocks carry only its BinaryHash
    blobs = {el.get("BinaryHash"): el.text.strip() for el in proj.root.iter("FormattedTextData")
             if el.get("BinaryHash") and (el.text or "").strip()}
    raw: dict[str, list] = {}
    for mc in proj.root:
        if mc.tag != "MasterClip":
            continue
        coll = _reach(proj, mc, lambda r: r.tag == "CaptionCollection")
        if coll is None:
            continue
        media = _reach(proj, mc, lambda r: r.tag == "Media" and bool(r.findtext("FilePath")))
        key = media.findtext("FilePath") if media is not None else (mc.findtext("Name") or "")
        cues = []
        for it in coll.iter("CaptionMapItem"):
            cap = proj.ref(it.find("Second"))
            if cap is None:
                continue
            ss = []
            for bi in cap.iter("BlockVectorItem"):
                ftd = proj.ref(bi).find("FormattedTextData") if proj.ref(bi) is not None else None
                data = ftd is not None and ((ftd.text or "").strip() or blobs.get(ftd.get("BinaryHash")))
                if data:
                    ss += _strings(base64.b64decode(data))
            cues.append((_sec(cap, "TimeStart"), _sec(cap, "TimeEnd"), ss))
        raw[key] = cues
    n = sum(1 for c in raw.values() for _, _, ss in c if ss)   # recut cues carry no strings
    freq = Counter(s for cues in raw.values() for _, _, ss in cues for s in set(ss))
    style = {s for s, k in freq.items() if k > max(3, 0.3 * n)}
    return {key: sorted((round(a, 3), round(b, 3),
                         " ".join(s for s in ss if s not in style).strip())
                        for a, b, ss in cues)
            for key, cues in raw.items()}
