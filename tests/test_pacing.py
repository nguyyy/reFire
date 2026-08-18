"""The pacing audit is the critic's only source of truth about the FINISHED cut's
timeline, so its arithmetic gets tested: if `at` drifts, every flag points the critic at
the wrong stretch of video and the review round is wasted."""
from refire.pacing import audit, cut_timeline, pacing_note


def beat(n, energy=3, dur=30.0, role="escalation", texture="funny",
         words=60, start=100.0, payoff=None):
    """One outline_log beat. `text` carries the word count the density check reads."""
    return {"title": f"beat {n}", "role": role, "texture": texture, "energy": energy,
            "start": start, "end": start + dur, "dur": dur,
            "payoff_start_s": payoff, "text": " ".join(["word"] * words)}


def log(beats):
    return {"central_idea": "x", "beats": beats}


# --- timeline -----------------------------------------------------------

def test_positions_accumulate_over_kept_footage_not_source_envelopes():
    # source spans are far apart and out of scale with the cut; only `dur` may drive `at`
    tl = cut_timeline(log([beat(1, dur=20.0, start=1000.0),
                           beat(2, dur=40.0, start=5000.0),
                           beat(3, dur=15.0, start=9000.0)]))
    assert [b["at"] for b in tl] == [0.0, 20.0, 60.0]
    assert tl[-1]["end_at"] == 75.0
    assert tl[1]["src"] == (5000.0, 5040.0)


def test_dur_falls_back_to_the_envelope_when_segments_never_ran():
    b = beat(1, dur=30.0, start=10.0)
    b["dur"] = 0
    assert cut_timeline(log([b]))[0]["dur"] == 30.0


def test_density_and_setup_lead_are_derived():
    tl = cut_timeline(log([beat(1, dur=20.0, words=50, start=100.0, payoff=118.0)]))
    assert tl[0]["wps"] == 2.5
    assert tl[0]["setup_lead"] == 18.0


def test_payoff_before_the_realized_start_is_not_a_negative_lead():
    # casting can snap a beat's start PAST the director's payoff anchor
    tl = cut_timeline(log([beat(1, start=100.0, payoff=90.0)]))
    assert tl[0]["setup_lead"] is None


def test_empty_log_is_not_a_crash():
    assert cut_timeline({}) == []
    assert audit([]) == []


# --- the sag ------------------------------------------------------------

def kinds(beats):
    return [f["kind"] for f in audit(cut_timeline(log(beats)))]


def test_a_flat_middle_is_found_with_its_position_in_the_cut():
    beats = ([beat(1, energy=5), beat(2, energy=4)]
             + [beat(i, energy=2) for i in range(3, 7)]      # 4 x 30s = 120s lull
             + [beat(7, energy=5), beat(8, energy=4)])
    flags = audit(cut_timeline(log(beats)))
    flat = [f for f in flags if f["kind"] == "flat_run"]
    assert len(flat) == 1
    # the lull starts after two 30s beats and runs 120s -> 01:00-03:00 of the finished cut
    assert flat[0]["at"] == 60.0
    assert "01:00-03:00" in flat[0]["text"]
    assert "120s" in flat[0]["text"]


def test_a_varied_cut_has_no_flat_run():
    assert "flat_run" not in kinds([beat(1, energy=4), beat(2, energy=2),
                                    beat(3, energy=5), beat(4, energy=2),
                                    beat(5, energy=4)])


def test_a_short_lull_is_texture_not_a_sag():
    # two low beats, but only 60s of cut time -- under FLAT_RUN_S
    assert "flat_run" not in kinds([beat(1, energy=5), beat(2, energy=2, dur=30.0),
                                    beat(3, energy=2, dur=30.0), beat(4, energy=5)])


def test_a_trailing_lull_still_closes():
    beats = [beat(1, energy=5)] + [beat(i, energy=1) for i in range(2, 6)]
    assert "flat_run" in kinds(beats)


def test_long_and_thin_beats_are_flagged_with_their_cut_position():
    beats = [beat(1), beat(2, dur=100.0, words=200), beat(3)]
    flags = {f["kind"]: f for f in audit(cut_timeline(log(beats)))}
    assert "long_beat" in flags
    assert flags["long_beat"]["at"] == 30.0        # after beat 1's 30s

    thin = [beat(1, words=90), beat(2, words=90), beat(3, words=10), beat(4, words=90)]
    assert "low_density" in kinds(thin)


def test_front_loaded_peaks_are_structural_not_a_tightening_problem():
    peaks_early = [beat(1, energy=5), beat(2, energy=5)] + [beat(i, energy=3)
                                                            for i in range(3, 9)]
    assert "peak_position" in kinds(peaks_early)
    peaks_spread = [beat(1, energy=5), beat(2, energy=3), beat(3, energy=3),
                    beat(4, energy=3), beat(5, energy=3), beat(6, energy=5)]
    assert "peak_position" not in kinds(peaks_spread)


def test_back_to_back_same_role_and_texture():
    assert "sameness" in kinds([beat(1, role="escalation", texture="funny"),
                                beat(2, role="escalation", texture="funny")])
    assert "sameness" not in kinds([beat(1, role="escalation", texture="funny"),
                                    beat(2, role="escalation", texture="tense")])


# --- the prompt block ---------------------------------------------------

def test_a_clean_audit_says_so_rather_than_going_silent():
    # silence would read to the critic as "not measured" and invite invented problems
    note = pacing_note([], total_s=600.0)
    assert "no flat runs" in note
    assert "10:00" in note


def test_the_note_lists_flags_worst_first():
    beats = ([beat(1, energy=5)] + [beat(i, energy=1) for i in range(2, 6)]
             + [beat(6, dur=100.0, words=200)])
    note = pacing_note(audit(cut_timeline(log(beats))))
    assert note.index("flattest stretch") < note.index("longest single beat")
