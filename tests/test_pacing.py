"""The pacing audit is the critic's only source of truth about the FINISHED cut's
timeline, so its arithmetic gets tested: if `at` drifts, every flag points the critic at
the wrong stretch of video and the review round is wasted."""
from refire.pacing import (ROLE_BUDGET, audit, cut_timeline, pacing_note,
                          scaled_budget)


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
    assert note.index("flattest stretch") < note.index("ceiling for a")


def test_the_long_beat_ceiling_is_per_role_not_flat():
    # 70s: comfortably inside an escalation's 90s budget, way past a setup's 40s.
    # The old flat 75s ceiling flagged neither, which is how beats that were never
    # long got "tightened" while a sprawling setup went unmentioned.
    assert "long_beat" not in kinds([beat(1, dur=70.0, role="escalation")])
    assert "long_beat" in kinds([beat(1, dur=70.0, role="setup")])
    # and the flag says which ceiling was broken, so the critic can re-role instead
    flags = audit(cut_timeline(log([beat(1, dur=70.0, role="setup")])))
    assert "40s ceiling for a setup beat" in flags[0]["text"]


def test_a_climax_may_run_longer_than_any_other_role():
    assert "long_beat" not in kinds([beat(1, dur=120.0, role="climax")])
    assert "long_beat" in kinds([beat(1, dur=120.0, role="payoff")])


# --- cut speed (--pace) -------------------------------------------------

def test_scaled_budget_is_identity_at_pace_one():
    """pace 1.0 must not perturb the table -- every default run depends on it."""
    assert scaled_budget() == dict(ROLE_BUDGET)
    assert scaled_budget(1.0) == dict(ROLE_BUDGET)


def test_scaled_budget_scales_both_bounds():
    assert scaled_budget(0.5)["climax"] == (22.5, 65.0)
    assert scaled_budget(2.0)["hook"] == (24.0, 70.0)


def test_pace_moves_the_long_beat_ceiling():
    """The ceiling the audit enforces has to travel with the budget the director was
    told, or a deliberately slow cut gets flagged for obeying its own direction."""
    tl = cut_timeline(log([beat(1, dur=100.0, role="escalation")]))   # 90s ceiling
    assert any(f["kind"] == "long_beat" for f in audit(tl))
    assert not any(f["kind"] == "long_beat" for f in audit(tl, pace=1.5))
    # and the other way: a snappy cut flags a beat that was fine at 1.0
    tl2 = cut_timeline(log([beat(2, dur=80.0, role="escalation")]))
    assert not any(f["kind"] == "long_beat" for f in audit(tl2))
    assert any(f["kind"] == "long_beat" for f in audit(tl2, pace=0.6))


# --- skipped build-up ---------------------------------------------------

def test_gap_before_the_payoff_is_a_skipped_build():
    """Cutting from a bit's opener straight to its punchline reads as a missing scene --
    the wordle sequence jumped to its last guess because that scores highest."""
    b = beat(1, dur=50.0, start=0.0, payoff=200.0)
    b["segments"] = [[0.0, 10.0], [190.0, 240.0]]     # 180s of build-up skipped
    flags = audit(cut_timeline(log([b])))
    hit = [f for f in flags if f["kind"] == "skipped_build"]
    assert len(hit) == 1
    assert "180s of its own build-up" in hit[0]["text"]


def test_gap_after_the_payoff_is_not_flagged():
    """A jump past the payoff is a tail trim, not a skipped build."""
    b = beat(1, dur=50.0, start=0.0, payoff=5.0)
    b["segments"] = [[0.0, 10.0], [190.0, 240.0]]
    assert not any(f["kind"] == "skipped_build"
                   for f in audit(cut_timeline(log([b]))))


def test_small_internal_jumps_are_not_flagged():
    """Jump-cutting the rambling out of a moment is the normal edit, not a defect."""
    b = beat(1, dur=50.0, start=0.0, payoff=200.0)
    b["segments"] = [[0.0, 30.0], [55.0, 90.0], [120.0, 160.0]]   # 25s, 30s gaps
    assert not any(f["kind"] == "skipped_build"
                   for f in audit(cut_timeline(log([b]))))


def test_beat_without_segments_has_no_build_gap():
    """The fallback cast path emits no segments; it must not flag."""
    tl = cut_timeline(log([beat(1)]))
    assert tl[0]["build_gap"] == 0.0
    assert not any(f["kind"] == "skipped_build" for f in audit(tl))


def test_skipped_build_outranks_tightening_flags():
    """A structural break is worth the critic's attention before a long/thin beat is."""
    b = beat(1, dur=200.0, start=0.0, payoff=300.0, words=10)
    b["segments"] = [[0.0, 10.0], [290.0, 480.0]]
    kinds = [f["kind"] for f in audit(cut_timeline(log([b])))]
    assert "skipped_build" in kinds and "long_beat" in kinds
    assert kinds.index("skipped_build") < kinds.index("long_beat")


# --- the audit must not fight a deliberate style -------------------------

def _build_gap_timeline():
    """One beat whose segments skip 100s of its own build-up before the payoff, plus a
    slow lead-in -- the two flags a story wants and a dense clip reel does not."""
    log = {"cold_open": [], "beats": [{
        "title": "wordle", "role": "escalation", "energy": 4, "texture": "funny",
        "start": 0.0, "end": 200.0, "dur": 80.0, "text": "w " * 200,
        "payoff_start_s": 190.0, "setup_start_s": None,
        "segments": [[0.0, 10.0], [110.0, 190.0]]}]}
    return log


def test_keep_build_off_silences_the_flags_a_style_contradicts():
    log = _build_gap_timeline()
    tl = cut_timeline(log)
    kinds_on = {f["kind"] for f in audit(tl)}
    kinds_off = {f["kind"] for f in audit(tl, keep_build=False)}
    assert "skipped_build" in kinds_on
    assert "skipped_build" not in kinds_off and "slow_payoff" not in kinds_off


def test_average_shot_is_measured_only_when_build_up_is_skipped():
    """The mirror measurement: with setup deliberately gone, the failure flips from
    'the build was deleted' to 'this stopped being dense'."""
    log = _build_gap_timeline()
    tl = cut_timeline(log)
    assert not [f for f in audit(tl) if f["kind"] == "avg_shot"]
    flags = [f for f in audit(tl, keep_build=False) if f["kind"] == "avg_shot"]
    # a SHOT is one kept segment, not the beat: 80s over two segments is a 40s mean.
    # Measured as beats this read 80.0s, and the only way to clear it was to ship less.
    assert flags and "average shot runs 40.0s across 2 segments" in flags[0]["text"]


def test_average_shot_counts_segments_not_beats():
    """The flag has to be clearable by CUTTING MORE, not by shipping less footage: one
    60s beat split into ten 6s segments is a dense cut, and grading its beat length said
    the opposite."""
    from refire.pacing import audit, cut_timeline

    segs = [[float(i * 6), float(i * 6 + 6)] for i in range(10)]
    log = {"cold_open": [], "beats": [
        {"title": "a", "role": "escalation", "energy": 3, "dur": 60.0,
         "start": 0.0, "end": 60.0, "text": "x " * 120, "segments": segs}]}
    tl = cut_timeline(log)
    assert tl[0]["shots"] == [6.0] * 10
    assert not [f for f in audit(tl, keep_build=False) if f["kind"] == "avg_shot"]


def test_a_short_cut_is_flagged_against_its_target():
    """The one defect invisible beat-by-beat: every beat looks fine and the video is half
    the length it was asked for. Silent, the critic has no reason to act on it."""
    from refire.pacing import audit, cut_timeline

    log = {"cold_open": [], "beats": [
        {"title": "a", "role": "escalation", "energy": 3, "dur": 60.0,
         "start": 0.0, "end": 60.0, "text": "x y z", "segments": [[0.0, 60.0]]}]}
    tl = cut_timeline(log)
    assert not [f for f in audit(tl) if f["kind"] == "short_cut"]        # no target given
    assert not [f for f in audit(tl, target_s=60.0) if f["kind"] == "short_cut"]
    short = [f for f in audit(tl, target_s=960.0) if f["kind"] == "short_cut"]
    assert short and "94% short" in short[0]["text"]
    assert short[0]["kind"] == audit(tl, target_s=960.0)[0]["kind"]      # ranked first


def test_average_shot_ceiling_scales_with_pace():
    from refire.pacing import AVG_SHOT_S

    log = {"cold_open": [], "beats": [
        {"title": "a", "role": "", "energy": 3, "dur": 6.0, "start": 0.0, "end": 6.0,
         "text": "x y z", "segments": [[0.0, 6.0]]}]}
    tl = cut_timeline(log)
    assert 6.0 < AVG_SHOT_S                                   # under the pace-1.0 ceiling
    assert not [f for f in audit(tl, keep_build=False) if f["kind"] == "avg_shot"]
    # at pace 0.35 the ceiling is 4.2s, so a 6s average is now too slow
    assert [f for f in audit(tl, 0.35, keep_build=False) if f["kind"] == "avg_shot"]


def test_a_stack_that_peaks_early_is_flagged():
    from refire.pacing import audit_note

    beats = [{"title": "a", "role": "", "energy": 3, "dur": 10.0, "start": 0.0,
              "end": 10.0, "text": "x", "segments": [[0.0, 10.0]]}]
    good = {"cold_open": [{"start": 1.0, "end": 3.0, "dur": 2.0, "energy": 2},
                          {"start": 5.0, "end": 7.0, "dur": 2.0, "energy": 5}],
            "beats": beats}
    bad = {"cold_open": [{"start": 1.0, "end": 3.0, "dur": 2.0, "energy": 5},
                         {"start": 5.0, "end": 7.0, "dur": 2.0, "energy": 2}],
           "beats": beats}
    assert "strongest moment" not in audit_note(good)
    assert "strongest moment is #1 of 2" in audit_note(bad)


def test_a_single_teaser_is_never_a_stack_order_problem():
    from refire.pacing import audit_note

    log = {"cold_open": [{"start": 1.0, "end": 3.0, "dur": 2.0, "energy": 5}],
           "beats": [{"title": "a", "role": "", "energy": 3, "dur": 10.0, "start": 0.0,
                      "end": 10.0, "text": "x", "segments": [[0.0, 10.0]]}]}
    assert "strongest moment" not in audit_note(log)


def test_audit_defaults_are_unchanged():
    """keep_build defaults to True everywhere, so an unstyled run measures exactly what
    it measured before."""
    log = _build_gap_timeline()
    tl = cut_timeline(log)
    assert audit(tl) == audit(tl, 1.0, True)


def _beat(n_title, segs, seams, role="setup", dur=30.0):
    return {"title": n_title, "role": role, "energy": 3, "dur": dur,
            "start": segs[0][0], "end": segs[-1][1], "segments": segs, "seams": seams}


def test_severed_exchange_fires_on_a_short_hole_that_had_speech_in_it():
    """The Archon-Quest bug as a number. A 3.5s hole with talking in it is a line removed
    from inside a moment -- nothing measured this before, because `skipped_build` does not
    fire below 60s and only looks before the payoff."""
    from refire.pacing import audit, cut_timeline

    log = {"beats": [_beat("The Archon's Warning", [[100.0, 120.0], [123.5, 140.0]],
                           [{"at": 120.0, "dur": 3.5,
                             "text": "and what do you intend to do about it"}])]}
    kinds = [f["kind"] for f in audit(cut_timeline(log))]
    assert "severed_exchange" in kinds
    said = next(f for f in audit(cut_timeline(log)) if f["kind"] == "severed_exchange")["text"]
    assert "3.5s of SPEECH" in said and "what do you intend" in said


def test_severed_exchange_ignores_dead_air_and_deliberate_skips():
    """Two things it must NOT fire on, or the critic drowns in noise: a silent hole (the
    edit cutting dead air, which is the feature) and a long one (a skip BETWEEN moments,
    which is `skipped_build`'s job)."""
    from refire.pacing import audit, cut_timeline

    silent = {"beats": [_beat("A", [[100.0, 120.0], [124.0, 140.0]],
                              [{"at": 120.0, "dur": 4.0, "text": "   "}])]}
    assert not [f for f in audit(cut_timeline(silent)) if f["kind"] == "severed_exchange"]

    wide = {"beats": [_beat("B", [[100.0, 120.0], [400.0, 430.0]],
                            [{"at": 120.0, "dur": 280.0, "text": "lots of talking"}])]}
    assert not [f for f in audit(cut_timeline(wide)) if f["kind"] == "severed_exchange"]


def test_severed_exchange_survives_a_dense_style():
    """keep_build=False silences skipped_build -- a dense cut skips moments on purpose.
    It must NOT silence this one: severing an exchange is wrong under every style."""
    from refire.pacing import audit, cut_timeline

    log = {"beats": [_beat("A", [[100.0, 120.0], [123.0, 140.0]],
                           [{"at": 120.0, "dur": 3.0, "text": "the reply that got deleted"}])]}
    kinds = [f["kind"] for f in audit(cut_timeline(log), keep_build=False)]
    assert "severed_exchange" in kinds
    assert "skipped_build" not in kinds
