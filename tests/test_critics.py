"""Structured critics: realized-cut rendering, arbiter conflicts, edits, rounds."""
from refire import critics
from refire.learn.diff import Context
from refire.trim import utterances


def W(spec):
    return [{"text": t, "start": a, "end": b} for t, a, b in spec]


WORDS = W([("Earlier.", 0.0, 0.5), ("he", 5.0, 5.3), ("did", 5.4, 5.7), ("it.", 5.8, 6.0),
           ("um", 8.0, 8.3), ("ok", 9.0, 9.3), ("NO", 12.0, 12.4), ("WAY.", 12.5, 13.0),
           ("Next.", 40.0, 40.5), ("thing.", 41.0, 41.5)])


def clips():
    return [{"i": 0, "start": 5.0, "end": 12.4, "keep": [[5.0, 6.3], [7.7, 12.4]]},
            {"i": 1, "start": 40.0, "end": 41.5, "keep": [[40.0, 41.5]]}]


def test_realized_cut_marks_clips_and_cuts_and_hides_cut_text():
    cl = clips()
    txt = critics.realized(cl, utterances(cl, WORDS))
    assert txt.splitlines()[0] == "== CLIP 1 ==" and "[CUT 1.4s]" in txt
    assert "Earlier" not in txt and "== CLIP 2 ==" in txt
    edge = critics.realized(cl, utterances(cl, WORDS), "edges_end")
    assert "S1 he did it." not in edge          # only each clip's last spans


def test_arbiter_conflict_and_low_confidence():
    sp = utterances(clips(), WORDS)
    edits = {"filler": [critics.Edit(span_id="S2", action="cut", confidence=0.9)],
             "continuity": [critics.Edit(span_id="S2", action="keep", confidence=0.8)],
             "dropoff": [critics.Edit(span_id="S3", action="cut", confidence=0.3)]}
    applied, conflicts = critics.arbitrate(edits, sp)
    assert applied == [] and conflicts[0]["span_id"] == "S2"
    out, _ = critics.apply_edits(clips(), sp, applied, conflicts, Context(WORDS))
    assert any(m["kind"] == "conflict" for m in out[0]["markers"])


def test_review_applies_cut_and_extensions_then_stops():
    table = {"continuity": [{"span_id": "S1", "action": "extend_start", "confidence": 0.9,
                             "reason": "opens on a pronoun"}],
             "filler": [{"span_id": "S3", "action": "cut", "confidence": 0.95}]}
    calls = []

    def ask(job, system, user):
        calls.append(job)
        assert "Reply ONLY with JSON" in system and "== CLIP 1 ==" in user
        rows = table.get(job, []) if len(calls) <= 4 else []
        return critics.Edits(edits=[critics.Edit(**e) for e in rows])

    many = clips() + [{"i": k, "start": 100.0 + k, "end": 100.5 + k,
                       "keep": [[100.0 + k, 100.5 + k]]} for k in range(2, 40)]
    words = WORDS + W([(f"w{k}.", 100.0 + k, 100.4 + k) for k in range(2, 40)])
    out, log = critics.review_clips(many, Context(words), {}, ask)
    c0 = out[0]
    assert c0["start"] == 0.0                     # back to the thread head 'Earlier.'
    assert all(not (a <= 9.0 < b) for a, b in c0["keep"])   # 'ok' cut, from the silence
    assert len(log["rounds"]) == 2 and len(calls) == 8
    names = [m["name"] for c in out for m in c.get("markers", [])]
    assert any("continuity" in n for n in names) and any("filler" in n for n in names)


def test_failed_critic_yields_no_edits():
    def boom(job, system, user):
        raise RuntimeError("cli down")
    out, log = critics.review_clips(clips(), Context(WORDS), {}, boom)
    assert out[0]["start"] == 5.0 and log["rounds"][0]["applied"] == 0
