"""thumb pipeline pieces that need no model/network: validation, accent fix, layout, made graphics, snap."""
import numpy as np

from refire.thumb import pipeline


def concepts(types=("reaction", "subject", "reveal"), lines=("BIG", "HOOK?"), payoff=(1,)):
    Out = pipeline._models()
    return Out.model_validate({"brief": {"promise": "x"}, "concepts": [
        {"type": t, "hook": {"lines": list(lines)}, "face": {"t": 10}, "subject": {"source": "none"},
         "title": "t", "payoff_beats": list(payoff)} for t in types]})


def test_validate_house_rules():
    assert pipeline.validate(concepts(), 5) == []
    assert any("one of each type" in e for e in pipeline.validate(concepts(types=("reaction",) * 3), 5))
    assert any("too long" in e for e in pipeline.validate(concepts(lines=("ONE TWO THREE", "FOUR FIVE SIX")), 5))
    assert any("out of range" in e for e in pipeline.validate(concepts(payoff=(9,)), 5))


def test_concept_without_payoff_is_rejected():
    import pydantic
    try:
        concepts(payoff=())
    except pydantic.ValidationError as e:
        assert "payoff" in str(e)
    else:
        raise AssertionError("accepted a concept with no payoff beats")


def test_accent_keeps_or_swaps_by_hue():
    blue = np.zeros((20, 20, 3), np.uint8); blue[..., 2] = 220
    assert pipeline._accent("gold", blue) == "gold"                 # gold on blue is fine
    gold = np.zeros((20, 20, 3), np.uint8); gold[..., 0] = 210; gold[..., 1] = 165; gold[..., 2] = 27
    assert pipeline._accent("gold", gold) != "gold"                  # gold on gold gets swapped


def test_hook_lines_stay_short():
    assert pipeline.validate(concepts(lines=("STRONGBOX...", "GONE RIGHT?")), 5) == []
    errs = pipeline.validate(concepts(lines=("GENSHIN OWED ME", "THIS")), 5)
    assert any("characters" in e for e in errs)


def test_left_element_is_half_the_left_side_and_centered():
    for wh in ((1000, 640), (492, 690), (300, 900)):
        sw, sh = pipeline.screen_size(wh)
        assert sw <= pipeline.LEFT_MAX[0] + 1e-6 and sh <= pipeline.LEFT_MAX[1] + 1e-6
        assert sw * sh >= 0.4 * 640 * 720 or sh >= pipeline.LEFT_MAX[1] - 1e-6       # big unless height-capped


def test_snap_moves_times_into_the_cut():
    class Cams:
        _ranges = [(0, "a.mov", 100.0, 110.0, 0.0), (0, "a.mov", 200.0, 205.0, 0.0)]
    assert pipeline.snap(Cams, 0, 105.0) == 105.0
    assert pipeline.snap(Cams, 0, 150.0) == 110.0 - 0.05
    assert pipeline.snap(Cams, 0, 190.0) == 200.05


def test_gameplay_crops_avoid_the_stream_facecam():
    left_cam = (0.0, 0.6, 0.25, 1.0)
    assert pipeline.gameplay((0, 0, 1, 1), left_cam) == (0.26, 0, 1, 1)
    assert pipeline.gameplay((0.64, 0.06, 0.99, 0.94), left_cam) == (0.64, 0.06, 0.99, 0.94)   # already clear
    assert pipeline.gameplay((0, 0, 0.5, 1), left_cam)[0] == 0.26
    right_cam = (0.75, 0.6, 1.0, 1.0)
    assert pipeline.gameplay((0, 0, 1, 1), right_cam) == (0, 0, 0.74, 1)


def test_made_graphics_html():
    from refire.thumb import made
    from pathlib import Path
    fake = Path(__file__).parent / "golden" / "thumb" / "assets" / "face.png"
    h = made.tierlist_html([{"label": "GOD TIER", "chars": ["A", "B"]}, {"label": "RUN AWAY", "color": "#7FFFFF",
                            "chars": ["C"]}], icon_fn=lambda n: fake, hidden=[[0, 1]])
    assert h.count("<img") == 3 and h.count('class="px"') == 1 and "#FF7F7F" in h and "RUN AWAY" in h


def test_artifact_card_is_the_in_game_layout():
    import pytest
    from refire.thumb import made, text
    if not text._installed(made.GAME_FONT).exists():
        pytest.skip("game font not installed")
    piece = {"set": "Night of the Sky's Unveiling", "bonus": ["Increases Elemental Mastery by 80.", "Long."],
             "icon": made.ASSETS / "artifact_buttons.png"}
    c = made.stat_card_svg({"title": "Revelation's Toll", "subtitle": "Sands of Eon", "main": ["DEF", "8.7%"],
                            "level": "+0", "rows": [["Elemental Mastery", "19"], ["CRIT DMG", "+5.4%"],
                                                    ["ATK", "19", "unactivated"]], "hidden": [1]}, piece)
    assert "Elemental Mastery+19" in c and "5.4%" not in c and "???" in c       # hidden value -> mosaic
    assert "(unactivated)" in c and made.CARD["unact"] in c
    assert "Night of the Sky&#x27;s Unveiling:" in c and "2-Piece Set: Increases" in c
    assert 'font-family="SDKJP"' in c and c.count("<polygon") == 5               # five stars by default
    assert c.count('y="382.0"') >= 2                                             # first substat on its measured baseline


def test_artifact_lookup_by_set_then_piece(tmp_path):
    from refire.thumb import made
    import json
    sets = {"data": {"items": {"15041": {"name": "Night of the Sky's Unveiling"}, "1": {"name": "Gladiator's Finale"}}}}
    detail = {"data": {"name": "Night of the Sky's Unveiling", "affixList": {"b": "4pc <color=#FFF>text</color>", "a": "2pc"},
                       "suit": {"EQUIP_SHOES": {"name": "Revelation's Toll", "icon": "UI_RelicIcon_15041_5"},
                                "EQUIP_BRACER": {"name": "Bloom of the Mind's Desire", "icon": "UI_RelicIcon_15041_4"}}}}
    calls = []

    class F:
        def get(self, url, accept=()):
            calls.append(url)
            if url.endswith("/reliquary"):
                return json.dumps(sets).encode(), "application/json", url
            if url.endswith("/reliquary/15041"):
                return json.dumps(detail).encode(), "application/json", url
            if url.endswith("/reliquary/1"):
                return json.dumps({"data": {"suit": {}}}).encode(), "application/json", url
            return b"png", "image/png", url
    got = made.artifact({"title": "Bloom of the Mind's Desire", "set": "night of the skys unveiling",
                         "subtitle": "Flower of Life"}, F(), tmp_path)
    assert got["icon"].name == "UI_RelicIcon_15041_4.png" and got["bonus"] == ["2pc", "4pc text"]
    # no set given: found by its piece name across the sets
    got = made.artifact({"title": "Revelation's Toll"}, F(), tmp_path)
    assert got["icon"].name == "UI_RelicIcon_15041_5.png"
    assert made.artifact({"title": "Nope", "set": "Gladiator's Finale"}, F(), tmp_path) is None


def test_fit_card_lands_its_projected_corners_on_the_box():
    box = pipeline.left_box(*pipeline.screen_size((492, 690)))
    assert box[0] >= 40 and box[1] >= 30 and box[2] <= pipeline.LEFT_RIGHT
    card = pipeline.fit_card("a.png", 492 / 690, box, -70, [3, 16, -2])
    got = pipeline.project(card, 492 / 690)
    assert abs((got[0] + got[2]) - (box[0] + box[2])) < 2 and abs((got[1] + got[3]) - (box[1] + box[3])) < 2
    assert abs((got[2] - got[0]) - (box[2] - box[0])) < 2 or abs((got[3] - got[1]) - (box[3] - box[1])) < 2
    assert got[0] >= box[0] - 1 and got[1] >= box[1] - 1 and got[2] <= box[2] + 1 and got[3] <= box[3] + 1


def test_left_element_lookups_stay_on_plan():
    c = concepts().concepts[0]
    c.subject.query, c.subject.alt_query = "Skirk", "Skirk"
    assert pipeline.left_queries(c, {"about_art": "Escoffier", "entities": ["Qiqi"]}) == ["Skirk", "Escoffier"]
    c.subject.query = c.subject.alt_query = None
    assert pipeline.left_queries(c, {"entities": ["Qiqi"]}) == []        # never a passing mention


def test_frame_sheets_label_every_beat_frame(tmp_path):
    import json

    import cv2
    (tmp_path / "outline.json").write_text(json.dumps({"beats": [
        {"segments": [[0, 10], [20, 30]]}, {"start": 40, "end": 50}]}), encoding="utf-8")
    times = pipeline.beat_times(tmp_path)
    assert times == [(0, 5.0), (0, 25.0), (1, 45.0)]

    class Cams:
        def detect(self, sid):
            return None

        def frames(self, sid, ts, width=0):
            return [] if ts[0] == 25.0 else [np.full((720, 1280, 3), 90, np.uint8)]   # 25 s: outside the cut
    names = pipeline.frame_sheets(Cams(), times * 5, tmp_path)                    # 10 frames found -> 1 sheet
    assert names == ["frames_1.jpg"]
    sheet = cv2.imread(str(tmp_path / names[0]))
    assert sheet.shape == (3 * 270, 4 * 480, 3)
    assert sheet[200, 240].mean() > 60 and sheet[800, 1800].mean() < 10        # tiles filled, the rest black


def test_annotation_may_not_repeat_the_hook():
    out = concepts(lines=("BIRTHDAY", "BEST PIECE?"))
    out.concepts[0].hook.annotation = "(birthday)"
    assert any("repeats the hook" in e for e in pipeline.validate(out, 5))
    out.concepts[0].hook.annotation = "(day 3)"
    assert pipeline.validate(out, 5) == []


def test_seen_screen_only_for_a_frame_the_director_saw(tmp_path):
    import json

    class Cams:
        def frames(self, sid, ts, width=0):
            return [np.full((720, 1280, 3), 77, np.uint8)]
    (tmp_path / "frames.json").write_text(json.dumps([100.0, 250.0]), encoding="utf-8")
    cam = (0.0, 0.0, 0.2, 0.4)                                  # stream cam on the left
    got = pipeline.seen_screen(Cams(), 0, 101.5, cam, "full", tmp_path)
    assert got is not None and got.shape[1] < 1280 * 0.8 + 2      # cam column cropped off
    assert pipeline.seen_screen(Cams(), 0, 180.0, cam, "full", tmp_path) is None    # never blind


def test_team_puts_the_middle_figure_in_front_and_larger():
    from PIL import Image
    figs = [Image.new("RGBA", (300, 400), c) for c in ((255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255))]
    g = pipeline.team(figs)
    px = np.asarray(g)
    assert g.height == round(900 * 1.12)                      # the middle one sets the height
    cx = g.width // 2
    assert px[g.height - 5, cx, 1] == 255                     # green (middle) drawn over its neighbours
    assert px[g.height - 5, 5, 0] == 255 and px[g.height - 5, g.width - 5, 2] == 255
    assert px[2, 5, 3] == 0                                   # the smaller side figures stand lower


def test_text_boxes_find_a_reveals_giveaways():
    img = np.zeros((100, 200, 3), np.uint8)

    def ocr(im):                                              # rapidocr's shape: [[pts, text, score]], elapse
        return [[[[20, 10], [120, 10], [120, 30], [20, 30]], "Challenge Completed", 0.99],
                [[[20, 60], [80, 60], [80, 70], [20, 70]], "Leave for Now", 0.98]], 0.1
    got = pipeline.text_boxes(img, ["challenge completed"], ocr)
    assert len(got) == 1 and abs(got[0][0] - (0.1 - 0.012)) < 1e-9 and got[0][3] < 0.35
    assert pipeline.text_boxes(img, ["143,229"], ocr) == [] and pipeline.text_boxes(img, None, ocr) == []


def test_head_to_thigh_skips_thin_effects_above_the_head():
    from PIL import Image
    a = np.zeros((1000, 400), np.uint8)
    a[0:200, 190:210] = 255                                   # a thin ribbon above the head
    a[200:1000, 100:300] = 255                                # the body
    im = Image.fromarray(np.dstack([a, a, a, a]), "RGBA")
    out = np.asarray(pipeline.head_to_thigh(im).getchannel("A"))
    assert (out[0] > 0).sum() > 150                           # starts at the body, not the ribbon
    assert out.shape[0] <= round(800 * 0.6) + 1
