"""refire.learn: prproj parsing, VOD-time alignment, diff labels, lessons. Synthetic data."""
import gzip
import json

import pytest

from refire.learn import align, diff, lessons, prproj

T = prproj.TICKS
RUN = r"C:\dev\reFire\run\100@0-500\100-a-b\ae\proxies"


def _proxy(a, b):
    return rf"{RUN}\p_dnxhr_sq_{a:.3f}-{b:.3f}.mov"


def make_project(path, seqs):
    """seqs: {uid: (name, [(track, seq_in, seq_out, src_in, src_out, media|('seq', uid))])}"""
    n = [100]

    def oid():
        n[0] += 1
        return str(n[0])

    objs, top = [], []
    for uid, (name, items) in seqs.items():
        grp, trk = oid(), f"T-{uid}"
        tracks = {}
        for it in items:
            tracks.setdefault(it[0], []).append(it)
        track_xml = []
        for k, (tr, its) in enumerate(sorted(tracks.items())):
            tuid = f"{trk}-{tr}"
            refs = []
            for track, si, so, ci, co, media in its:
                ti, sc, cl, src = oid(), oid(), oid(), oid()
                refs.append(f'<TrackItem Index="{len(refs)}" ObjectRef="{ti}"/>')
                start = f"<Start>{int(si * T)}</Start>" if si else ""
                objs.append(f'<VideoClipTrackItem ObjectID="{ti}"><ClipTrackItem><TrackItem>'
                            f'{start}<End>{int(so * T)}</End></TrackItem>'
                            f'<SubClip ObjectRef="{sc}"/></ClipTrackItem></VideoClipTrackItem>')
                objs.append(f'<SubClip ObjectID="{sc}"><Clip ObjectRef="{cl}"/></SubClip>')
                objs.append(f'<VideoClip ObjectID="{cl}"><Clip><Source ObjectRef="{src}"/>'
                            f'<InPoint>{int(ci * T)}</InPoint><OutPoint>{int(co * T)}'
                            f'</OutPoint></Clip></VideoClip>')
                if isinstance(media, tuple):
                    objs.append(f'<VideoSequenceSource ObjectID="{src}"><SequenceSource>'
                                f'<Sequence ObjectURef="{media[1]}"/></SequenceSource>'
                                f'</VideoSequenceSource>')
                else:
                    mu = f"M{src}"
                    objs.append(f'<VideoMediaSource ObjectID="{src}"><MediaSource>'
                                f'<Media ObjectURef="{mu}"/></MediaSource></VideoMediaSource>')
                    objs.append(f'<Media ObjectUID="{mu}"><FilePath>{media}</FilePath></Media>')
            objs.append(f'<VideoClipTrack ObjectUID="{tuid}"><ClipTrack><ClipItems><TrackItems>'
                        + "".join(refs) + "</TrackItems></ClipItems></ClipTrack></VideoClipTrack>")
            track_xml.append(f'<Track Index="{tr}" ObjectURef="{tuid}"/>')
        objs.append(f'<VideoTrackGroup ObjectID="{grp}"><TrackGroup><Tracks>'
                    + "".join(track_xml) + "</Tracks></TrackGroup></VideoTrackGroup>")
        top.append(f'<Sequence ObjectUID="{uid}"><Name>{name}</Name><TrackGroups>'
                   f'<TrackGroup Index="0"><Second ObjectRef="{grp}"/></TrackGroup>'
                   f'</TrackGroups></Sequence>')
    xml = "<PremiereData>" + "".join(top + objs) + "</PremiereData>"
    with gzip.open(path, "wb") as f:
        f.write(xml.encode("utf-8"))
    return path


@pytest.fixture
def proj(tmp_path):
    p = make_project(tmp_path / "x.prproj", {
        "S1": ("main", [(0, 0, 5, 3, 8, _proxy(98, 140)),          # vod 101..106
                        (0, 5, 7, 0, 2, ("seq", "S2")),             # nested window
                        (1, 1, 2, 0, 1, r"C:\memes\boom.png")]),
        "S2": ("inner", [(0, 0, 10, 2, 12, _proxy(198, 230))]),     # vod 200..210
    })
    return prproj.Project(p)


def test_parse_items_ticks_and_default_start(proj):
    main = proj.seqs["S1"]["items"]
    first = next(i for i in main if i["media"] and "98.000" in i["media"])
    assert (first["seq_in"], first["seq_out"], first["src_in"], first["src_out"]) == (0, 5, 3, 8)
    assert first["kind"] == "video" and first["enabled"]


def test_flatten_resolves_nested_and_pick_final(proj):
    proj.seqs["S1"]["items"][1].update(src_in=4.0, src_out=6.0)   # parent shows inner 4..6
    flat = prproj.flatten(proj, "S1")
    inner = next(i for i in flat if "198.000" in i["media"])
    assert inner["seq_in"] == pytest.approx(5.0) and inner["seq_out"] == pytest.approx(7.0)
    assert inner["src_in"] == pytest.approx(6.0) and inner["nest"] == ["main", "inner"]
    uid, why = prproj.pick_final(proj, keep=lambda m: "proxies" in (m or ""))
    assert uid == "S1" and "top-level" in why


def test_referenced_runs_byte_scan(tmp_path):
    p = make_project(tmp_path / "y.prproj", {"S": ("s", [(0, 0, 1, 0, 1, _proxy(0, 20))])})
    assert prproj.referenced_runs(p) == {"100@0-500/100-a-b": 1}


def test_map_items_methods_and_overlays():
    items = [
        {"kind": "video", "track": 0, "seq_in": 0, "seq_out": 2, "src_in": 1, "src_out": 3,
         "media": _proxy(50, 80)},                                  # proxy name
        {"kind": "video", "track": 0, "seq_in": 2, "seq_out": 3, "src_in": 0, "src_out": 2,
         "media": r"C:\v\src.mp4"},                                 # sources[] + speed 2x
        {"kind": "video", "track": 1, "seq_in": 0, "seq_out": 1, "src_in": 0, "src_out": 1,
         "media": r"C:\memes\boom.png"},
        {"kind": "video", "track": 0, "seq_in": 3, "seq_out": 4, "src_in": 0, "src_out": 1,
         "media": r"C:\dev\reFire\run\999\9-x\ae\proxies\p_dnxhr_sq_5.000-9.000.mov"},
    ]
    man = {"sources": [{"path": "C:/v/src.mp4", "offset": 1000.0}], "clips": []}
    out = align.map_items(items, man, run_key="100@0-500/100-a-b")
    vods = [(s["vod_start"], s["vod_end"], s["method"]) for s in out["segments"]]
    assert vods == [(51.0, 53.0, "proxy_name"), (1000.0, 1002.0, "sources")]
    assert len(out["overlays"]) == 2                  # meme + another run's proxy
    assert any(f["flag"] == "speed" and f["speed"] == 2.0 for f in out["flags"])


def test_rough_mp4_maps_through_keep_spans():
    man = {"clips": [{"start": 100.0, "end": 120.0, "keep": [[0, 4], [10, 14]]},
                     {"start": 300.0, "end": 305.0}],
           "sections": [{"clip_indices": [0, 1]}]}
    it = {"kind": "video", "track": 0, "seq_in": 0, "seq_out": 9, "src_in": 3, "src_out": 9,
          "media": r"C:\out\rough.mp4"}
    segs = align.map_items([it], man)["segments"]
    assert [(s["vod_start"], s["vod_end"]) for s in segs] == [(103.0, 104.0), (110.0, 114.0),
                                                            (300.0, 301.0)]


def test_proposal_from_proxy_names_undoes_pad_and_tail():
    prop = align.proposal_from_proxies([_proxy(0, 15), _proxy(98, 140)])
    assert [(p["start"], p["end"]) for p in prop] == [(0.0, 5.0), (100.0, 130.0)]


def test_proposal_from_proxies_keeps_bin_order_and_marks_teasers():
    # bin order is play order (story order, not chronological); a clip inside a
    # later-playing clip is the cold-open teaser cut from it
    prop = align.proposal_from_proxies([_proxy(108, 125), _proxy(98, 140), _proxy(0, 15),
                                        _proxy(98, 140)])
    assert [(p["start"], p["end"]) for p in prop] == [(110.0, 115.0), (100.0, 130.0), (0.0, 5.0)]
    assert [p["section"] for p in prop] == ["Cold Open", "", ""]
    assert [p["i"] for p in prop] == [0, 1, 2]


def test_projects_skips_autosaves_and_reads_several_folders(tmp_path):
    import os
    a, b = tmp_path / "a", tmp_path / "b"
    for p in (a / "x.prproj", a / "Adobe Premiere Pro Auto-Save" / "x--1.prproj", b / "y.prproj"):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    got = lessons.projects(f"{a}{os.pathsep}{b}")
    assert [p.name for p in got] == ["x.prproj", "y.prproj"]


def _blob(*strings):
    """A FlatBuffer-ish caption block: length-prefixed, NUL-terminated utf-8 strings."""
    import base64
    import struct
    b = b"\x00" * 8
    for s in strings:
        e = s.encode("utf-8")
        b += struct.pack("<I", len(e)) + e + b"\x00" * 4
    return base64.b64encode(b).decode()


def test_captions_read_from_project_with_shared_blobs(tmp_path):
    srt = r"C:\dev\reFire\run\100@0-500\100-a-b\ae\captions.srt"
    cues = [(0.0, 0.5, "hello there"), (0.6, 1.0, "general"), (1.2, 1.5, "kenobi"),
            (2.0, 2.4, "hello there"), (3.0, 3.3, "bold one")]
    objs = [f'<MasterClip ObjectUID="MC"><Name>captions.srt</Name><Clips><Clip ObjectRef="900"/>'
            f'</Clips><Transcript ObjectRef="901"/></MasterClip>',
            '<DataClip ObjectID="900"><Source ObjectRef="902"/></DataClip>',
            '<DataMediaSource ObjectID="902"><MediaSource><Media ObjectURef="M"/></MediaSource>'
            '</DataMediaSource>',
            f'<Media ObjectUID="M"><FilePath>{srt}</FilePath></Media>',
            '<TranscriptClip ObjectID="901"><Captions ObjectRef="903"/></TranscriptClip>']
    items, seen = [], set()
    for n, (a, b, text) in enumerate(cues):
        items.append(f'<CaptionMapItem Index="{n}"><Second ObjectRef="{910 + n}"/></CaptionMapItem>')
        objs.append(f'<Caption ObjectID="{910 + n}"><BlockVector><BlockVectorItem Index="0" '
                    f'ObjectRef="{920 + n}"/></BlockVector><TimeStart>{int(a * T)}</TimeStart>'
                    f'<TimeEnd>{int(b * T)}</TimeEnd></Caption>')
        data = "" if text in seen else _blob("AnimationType", "LucidaConsole", text)
        seen.add(text)                                  # a repeat carries only its hash
        objs.append(f'<Block ObjectID="{920 + n}"><FormattedTextData Encoding="base64" '
                    f'BinaryHash="h-{text}">{data}</FormattedTextData></Block>')
    objs.append('<CaptionCollection ObjectID="903"><CaptionMap>' + "".join(items)
                + "</CaptionMap></CaptionCollection>")
    p = tmp_path / "c.prproj"
    with gzip.open(p, "wb") as f:
        f.write(("<PremiereData>" + "".join(objs) + "</PremiereData>").encode("utf-8"))
    got = prproj.captions(prproj.Project(p))
    assert list(got) == [srt]
    assert got[srt] == [(a, b, t) for a, b, t in cues]       # style strings dropped


def _w(text, a):
    return {"text": text, "start": a, "end": a + 0.5}


# today's transcript: A = 100-130 (two phrases), B = 200-220, C = 300-302 (one stray word
# the build's transcript never heard), D = 400-410. reFire played B, A, C, D.
WORDS = ([_w(t, a) for t, a in [("alpha", 101.0), ("beta", 101.6), ("gamma", 102.2),
                                 ("delta", 120.0), ("epsilon", 120.6), ("zeta", 121.2)]]
         + [_w(t, a) for t, a in [("eta", 201.0), ("theta", 201.6), ("iota", 202.2)]]
         + [_w("noise", 301.0)]
         + [_w(t, a) for t, a in [("kappa", 401.0), ("lambda", 401.6), ("mu", 402.2)]])
# the build's captions.srt in cut time: B 0-2.3, A 2.3-6.9, C kept whole 6.9-8.9, D 8.9-11.2
CUES = [(0.3, 1.4, "eta theta"), (1.5, 2.0, "iota"),
        (2.6, 3.7, "alpha beta"), (3.8, 4.3, "gamma"), (4.9, 5.4, "delta"),
        (5.5, 6.0, "epsilon"), (6.1, 6.6, "zeta"),
        (9.2, 10.3, "kappa lambda"), (10.4, 10.9, "mu")]


def _rebuilt(order):
    spans = {"A": (100.0, 130.0), "B": (200.0, 220.0), "C": (300.0, 302.0), "D": (400.0, 410.0)}
    return align.proposal_from_proxies([_proxy(spans[k][0] - 2, spans[k][1] + 10)
                                        for k in order], WORDS)


def test_locate_cues_continues_runs_and_jumps_on_unique_hits():
    spans = [(100.0, 130.0), (200.0, 220.0), (300.0, 302.0), (400.0, 410.0)]
    assert align.locate_cues(CUES, WORDS, spans) == [201.0, 202.2, 101.0, 102.2, 120.0,
                                                     120.6, 121.2, 401.0, 402.2]


def test_use_captions_checks_order_and_takes_the_keep_the_captions_support():
    prop, chk = align.use_captions(_rebuilt("BACD"), CUES, WORDS)
    assert chk["order_agree_pct"] == 100.0 and chk["clips_reached"] == 3
    assert chk["located_pct"] == 100.0
    # the build heard nothing in C and kept it whole; today's words would cut it to 1.1s
    assert chk["keep_from"] == "transcript + captions"
    assert prop[2]["keep"] == [[300.0, 302.0]]
    assert chk["rebuilt_s"] == pytest.approx(10.3) and chk["with_cues_s"] == pytest.approx(11.2)
    _p, wrong = align.use_captions(_rebuilt("ABCD"), CUES, WORDS)
    assert wrong["order_agree_pct"] < 100.0


def test_use_captions_leaves_sparse_captions_as_a_check_only():
    sparse = [CUES[0], CUES[-2]]                  # emphasis-only captions, 2 cues / 10s
    prop, chk = align.use_captions(_rebuilt("BACD"), sparse, WORDS)
    assert chk["keep_from"] == "transcript" and "with_cues_s" not in chk
    assert prop[2]["keep"] == [[300.7, 301.8]]


def _clip(i, a, b, keep=None):
    return {"i": i, "start": a, "end": b, "keep": keep or [[a, b]], "role": ""}


def _seg(a, b, at):
    return {"vod_start": a, "vod_end": b, "seq_in": at, "seq_out": at + b - a}


def test_diff_labels_and_fix_cost():
    prop = [_clip(0, 100, 120, [[100, 108], [110, 120]]),   # dead air 108-110
            _clip(1, 200, 220),
            _clip(2, 300, 310),
            _clip(3, 321, 330),                              # 11s after clip 2
            _clip(4, 400, 410)]
    final = [_seg(97, 105, 0), _seg(107, 115, 8),           # start_late 3, restored 108-110
             _seg(117, 118, 16),                            # interior cut 115-117, end_late 2
             _seg(300, 330, 30),                            # clips 2+3 merged (gap 11s)
             _seg(205, 224, 70),                            # clip 1 after 2/3: reordered
             _seg(600, 604, 90)]                            # added; clip 4 dropped
    res = diff.diff(prop, final)
    got = {(r["label"], r["delta_s"]) for r in res["records"]}
    assert ("start_late", 3.0) in got
    assert ("restored", 2.0) in got
    assert ("interior_cut", -2.0) in got
    assert ("end_late", -2.0) in got
    assert ("start_early", -5.0) in got and ("end_early", 4.0) in got
    assert ("merged", 11.0) in got
    assert ("dropped", -10.0) in got and ("added", 4.0) in got
    assert [r["clips"] for r in res["records"] if r["label"] == "reordered"] == [[1]]
    fc = res["summary"]["fix_cost"]
    assert fc["boundary"] == 3 + 2 + 5 + 4
    assert fc["interior_cut"] == 4 and fc["dropped"] == 10   # holes 105-107 + 115-117 and fc["added"] == 2 + 11 + 4
    assert fc["total"] == 14 + 4 + 10 + 17


def test_diff_ignores_frame_jitter_and_reports_unchanged():
    prop = [_clip(0, 10, 20)]
    res = diff.diff(prop, [_seg(10.05, 19.95, 0)])
    assert res["records"] == [] and res["summary"]["fix_cost"]["total"] == 0
    assert res["summary"]["unchanged_pct"] == 100.0


def test_context_text_signals_and_beat():
    words = [{"text": t, "start": float(i), "end": i + 0.5} for i, t in enumerate("a b c d e".split())]
    ctx = diff.Context(words, loud=[(2.5, 1.0), (7.5, 3.0)],
                       beats=[{"title": "x", "role": "payoff", "energy": 4,
                               "segments": [[0, 5]]}])
    d = ctx.describe(1, 3)
    assert d["text"] == "b c" and d["beat"]["role"] == "payoff"
    assert d["signals"]["loud_z"]["max"] == 1.0   # 7.5s bucket is out of reach


def test_lessons_aggregate_leave_one_out(tmp_path):
    for vod, delta in (("A", 2.0), ("B", 6.0)):
        rec = {"label": "end_early", "delta_s": delta, "vod_start": 1, "vod_end": 3,
               "text": "hey", "before": "", "after": "", "signals": {}, "beat": None}
        (tmp_path / vod).mkdir()
        (tmp_path / vod / "clip_diff.json").write_text(json.dumps({
            "vod": vod, "run": "r", "game": "g", "project": "p.prproj", "sequence": "s",
            "records": [rec], "summary": {"fix_cost": {"total": delta}}}))
    both = lessons.aggregate(lessons.load_diffs(tmp_path))
    assert both["labels"]["end_early"]["median_delta"] == 4.0
    loo = lessons.load_lessons(tmp_path, "g", exclude=("A",))
    assert loo["labels"]["end_early"]["median_delta"] == 6.0
    assert {e["vod"] for e in loo["labels"]["end_early"]["examples"]} == {"B"}


def _diff_file(root, vod, prop, final, game="g"):
    from refire.learn import diff as D
    (root / vod).mkdir(parents=True, exist_ok=True)
    res = D.diff(prop, final)
    (root / vod / "clip_diff.json").write_text(json.dumps({
        "vod": vod, "run": "r", "game": game, "project": "p.prproj", "sequence": "s",
        "proposal": prop, "final": final, **res}))


def test_eval_replays_scores_and_compares(tmp_path, capsys):
    from refire.learn import evaluate
    _diff_file(tmp_path, "A", [_clip(0, 10, 20)], [_seg(8, 20, 0)])          # start_late 2
    _diff_file(tmp_path, "B", [_clip(0, 10, 20)], [_seg(10, 20, 0)])         # perfect
    base = evaluate.evaluate(run_root=tmp_path, tag="baseline")
    assert base["overall"]["total"] == 2.0
    assert [p["fix_cost"]["total"] for p in base["per_vod"]] == [2.0, 0.0]
    again = evaluate.evaluate(vod="A", run_root=tmp_path)
    out = capsys.readouterr().out
    assert again["per_vod"][0]["vod"] == "A" and "baseline" in out
    assert len(list((tmp_path / "eval").glob("*.json"))) >= 1
