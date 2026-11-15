"""Map the live Premiere timeline (V1 dump, `timeline.tsv`) to source time and back.

Features run off the final timeline, not the `make` output, so hand re-cuts never orphan a
punch or caption. Source time is the transcript's clock (run-local VOD seconds; `loud` runs
put VOD i at i * VOD_BASE), so `source_id` is just which VOD that is.

timeline.tsv (reFirePpro.jsx doTimeline): optional `#timebase<TAB>ticks-per-frame` header,
then one row per enabled V1 clip: media path, src in, src out, timeline in, timeline out
(seconds; src = the media file's own clock). Media that isn't one of the manifest's
sources (b-roll the user dropped in) still counts as a cut but maps to no source.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

TICKS = 254_016_000_000     # premiere ticks per second
VOD_BASE = 100_000.0        # = loud.VOD_BASE (not imported: loud pulls in numpy/ingest)


def _norm(p) -> str:
    return str(p).replace("\\", "/").lower()


@dataclass(frozen=True)
class Span:
    tl_in: float
    tl_out: float
    media: str
    src_in: float | None = None     # transcript clock; None = foreign media
    src_out: float | None = None
    offset: float = 0.0             # transcript time of the media's frame 0 (media t = src t - offset)

    @property
    def rate(self) -> float:
        """source seconds per timeline second (>1 where the clip was sped up)."""
        return (self.src_out - self.src_in) / (self.tl_out - self.tl_in)

    @property
    def source_id(self) -> int | None:
        return None if self.src_in is None else int(self.src_in // VOD_BASE)


def read_tsv(path) -> tuple[list[tuple[str, float, float, float, float]], int | None]:
    """-> (rows, ticks per frame or None for dumps older than the header)."""
    rows, tpf = [], None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        cols = line.split("\t")
        if cols[0] == "#timebase" and len(cols) > 1:
            tpf = int(cols[1])
            continue
        if len(cols) < 5:
            continue
        try:
            rows.append((cols[0], *(float(c) for c in cols[1:5])))
        except ValueError:          # a malformed row is skipped, not fatal
            continue
    return rows, tpf


class TimelineMap:
    def __init__(self, spans: list[Span], ticks_per_frame: int | None = None):
        self.spans = sorted(spans, key=lambda s: s.tl_in)
        self.ticks_per_frame = ticks_per_frame

    @classmethod
    def load(cls, manifest_path, timeline_path=None) -> TimelineMap:
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows, tpf = read_tsv(timeline_path or manifest_path.parent / "timeline.tsv")
        # media path -> transcript time of its frame 0
        offsets = {_norm(s["path"]): s.get("offset", 0.0) for s in manifest.get("sources") or []}
        if not offsets and manifest.get("source"):
            offsets[_norm(manifest["source"])] = 0.0
        if tpf is None and manifest.get("fps"):
            # ponytail: pre-header dumps trust the manifest fps, which can be wrong (60 fps
            # sequence vs fps 30, spike notes); ticks still land on the real grid when it divides
            tpf = round(TICKS / manifest["fps"])
        spans = []
        for path, src_in, src_out, at, end in rows:
            if end - at <= 1e-3:
                continue
            off = offsets.get(_norm(path))
            if off is None or src_out - src_in <= 1e-3:
                spans.append(Span(at, end, path))
            else:
                spans.append(Span(at, end, path, src_in + off, src_out + off, off))
        return cls(spans, tpf)

    def ours(self) -> list[Span]:
        return [s for s in self.spans if s.src_in is not None]

    def timeline_to_source(self, t: float) -> tuple[int, float] | None:
        """timeline seconds -> (source_id, source_t); None in a gap or over foreign media.
        a cut point belongs to the clip that starts there."""
        for s in reversed(self.spans):
            if s.tl_in <= t < s.tl_out:
                if s.src_in is None:
                    return None
                return s.source_id, s.src_in + (t - s.tl_in) * s.rate
        return None

    def source_to_timeline(self, source_id: int, source_t: float) -> list[float]:
        """every timeline time showing that source moment (a re-cut can repeat footage)."""
        return sorted(s.tl_in + (source_t - s.src_in) / s.rate for s in self.ours()
                      if s.source_id == source_id and s.src_in <= source_t < s.src_out)

    def cuts(self) -> list[float]:
        """timeline times of V1 edit points: every clip edge except the sequence's ends."""
        if not self.spans:
            return []
        edges = {round(e, 6) for s in self.spans for e in (s.tl_in, s.tl_out)}
        return sorted(edges - {round(self.spans[0].tl_in, 6), round(max(s.tl_out for s in self.spans), 6)})

    def gaps(self) -> list[tuple[float, float]]:
        """empty stretches between V1 clips."""
        out, end = [], None
        for s in self.spans:
            if end is not None and s.tl_in > end + 1e-6:
                out.append((end, s.tl_in))
            end = s.tl_out if end is None else max(end, s.tl_out)
        return out

    def ticks(self, t: float) -> int:
        """timeline seconds -> frame-snapped ticks (the only place seconds become ticks)."""
        if not self.ticks_per_frame:
            raise ValueError("no timebase: re-dump the timeline from the panel")
        return round(t * TICKS / self.ticks_per_frame) * self.ticks_per_frame
