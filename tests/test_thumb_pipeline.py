"""thumb pipeline pieces that need no model/network: validation, accent fix, layout, made graphics, snap."""
import numpy as np

from refire.thumb import pipeline


def concepts(types=("reaction", "subject", "reveal"), lines=("BIG", "HOOK?"), payoff=(1,)):
    Out = pipeline._models()
    return Out.model_validate({"brief": {"promise": "x"}, "concepts": [
        {"type": t, "hook": {"lines": list(lines)}, "face": {"t": 10}, "subject": {"source": "art"},
         "title": "t", "payoff_beats": list(payoff)} for t in types]})


def test_validate_house_rules():
    assert pipeline.validate(concepts(), 5) == []
    assert any("one of each type" in e for e in pipeline.validate(concepts(types=("reaction",) * 3), 5))
    assert any("too long" in e for e in pipeline.validate(concepts(lines=("ONE TWO THREE", "FOUR FIVE SIX")), 5))
    assert any("out of range" in e for e in pipeline.validate(concepts(payoff=(9,)), 5))
    out = concepts()
    out.concepts[2].subject.source = "panel"                  # nguy: never a stream screen
    assert any("never a stream screen" in e for e in pipeline.validate(out, 5))


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


def test_left_element_fills_the_left_and_middle():
    for wh in ((1000, 640), (492, 690), (300, 900)):
        sw, sh = pipeline.screen_size(wh)
        assert sw <= pipeline.LEFT_MAX[0] + 1e-6 and sh <= pipeline.LEFT_MAX[1] + 1e-6
        assert abs(sw - pipeline.LEFT_MAX[0]) < 1e-6 or abs(sh - pipeline.LEFT_MAX[1]) < 1e-6   # as big as fits
    x0, y0, x1, y1 = pipeline.left_box(*pipeline.screen_size((1000, 640)), pipeline.LEFT_CENTER, margin=(12, 10))
    assert x0 <= 20 and x1 >= 800                                           # left edge into the middle


def test_a_characters_face_lands_in_the_middle_of_the_top_left_quarter(tmp_path, monkeypatch):
    from PIL import Image
    Image.new("RGBA", (400, 800), (200, 100, 50, 255)).save(tmp_path / "art0.png")
    monkeypatch.setattr(pipeline, "art_face", lambda im: (0.5, 0.2))
    c = concepts().concepts[0]
    cards, box = pipeline.cards_for(c, tmp_path / "art0.png", "art")
    assert cards[0]["face"] == list(pipeline.ART_FACE)


def test_quotes_stay_short_and_cards_differ():
    out = concepts()
    out.concepts[0].quote = "THIS IS WAY TOO LONG FOR A BUBBLE"
    for k in (1, 2):
        out.concepts[k].subject.source, out.concepts[k].subject.made = "made", {"kind": "result"}
    errs = pipeline.validate(out, 5)
    assert any("quote over" in e for e in errs) and any("same kind of card" in e for e in errs)


def test_bottom_shade_sits_under_the_hook_only():
    from refire.thumb import render
    html = render._layers([{"kind": "face", "src": "f.png"}, {"kind": "hook", "src": "h.png"}],
                          {"height": 0.2, "alpha": 0.3}, 720)
    assert html.index('class="shade"') > html.index("f.png") and html.index('class="shade"') < html.index("h.png")
    assert "rgba(0,0,0,0.3000) 100%" in html and "height: 144.0px" in html
    assert 'class="shade"' not in render._layers([{"kind": "hook", "src": "h.png"}], None, 720)


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


def test_team_puts_the_first_named_in_the_middle_in_front_and_larger():
    from PIL import Image
    figs = [Image.new("RGBA", (300, 400), c) for c in ((255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255))]
    g = pipeline.team(figs)                                   # red = the one the hook is about
    px = np.asarray(g)
    assert g.height == round(900 * 1.12)                      # the middle one sets the height
    assert px[g.height - 5, g.width // 2, 0] == 255           # red stands in the middle, drawn over the others
    assert px[g.height - 5, 5, 1] == 255 and px[g.height - 5, g.width - 5, 2] == 255
    assert px[2, 5, 3] == 0                                   # the smaller side figures stand lower


def test_head_to_thigh_skips_thin_effects_above_the_head():
    from PIL import Image
    a = np.zeros((1000, 400), np.uint8)
    a[0:200, 190:210] = 255                                   # a thin ribbon above the head
    a[200:1000, 100:300] = 255                                # the body
    im = Image.fromarray(np.dstack([a, a, a, a]), "RGBA")
    out = np.asarray(pipeline.head_to_thigh(im).getchannel("A"))
    assert (out[0] > 0).sum() > 150                           # starts at the body, not the ribbon
    assert out.shape[0] <= round(800 * 0.6) + 1


def test_art_grows_from_the_top_left_until_it_covers_its_share():
    def cover(m, sw, sh):
        vis = min(sh, 720 - pipeline.ART_AT[1])
        return m[:max(1, int(m.shape[0] * vis / sh))].mean() * sw * vis / (1280 * 720)
    wide = np.zeros((500, 900), bool)
    wide[:, 100:800] = True                                   # wide and dense: reaches its share
    sw, sh = pipeline.art_size(wide, 1100)
    assert pipeline.ART_COVER <= cover(wide, sw, sh) < pipeline.ART_COVER + 0.02
    tall = np.zeros((900, 450), bool)
    tall[:, 100:350] = True                                   # tall and thin: stops at the height cap,
    sw, sh = pipeline.art_size(tall, 1100)                    # its face stays above the hook
    assert sh <= pipeline.ART_MAX_H and cover(tall, sw, sh) < pipeline.ART_COVER
    assert pipeline.art_size(tall, 300)[0] <= 300             # never past the width cap


def test_stream_frame_fallback_is_cropped_to_the_canvas_aspect():
    assert pipeline.cover(np.zeros((720, 950, 3), np.uint8)).shape[:2] == (534, 950)    # cam column cut off
    assert pipeline.cover(np.zeros((1000, 3000, 3), np.uint8)).shape[:2] == (1000, 1778)
    assert pipeline.backdrop(None) is None                                             # no place named


def test_backdrop_falls_back_to_the_next_place(monkeypatch, tmp_path):
    import cv2
    from refire.thumb import art
    cv2.imwrite(str(tmp_path / "m.jpg"), np.full((90, 160, 3), 200, np.uint8))

    def fake(name):
        if name == "Stygian Onslaught":
            raise art.ArtError("banner")
        return tmp_path / "m.jpg"
    monkeypatch.setattr(art, "backdrop", fake)
    seen = []
    assert pipeline.backdrop("Stygian Onslaught, Mondstadt", seen.append).shape == (90, 160, 3)
    assert len(seen) == 1 and "Stygian" in seen[0]


def test_result_card_hides_its_headline_and_drops_the_artifact_bits():
    import pytest
    from refire.thumb import made, text
    if not text._installed(made.GAME_FONT).exists():
        pytest.skip("game font not installed")
    c = made.stat_card_svg({"kind": "result", "title": "Wavebinder: Stormfang", "subtitle": "Stygian Onslaught",
                            "main": ["Time Elapsed", "117s"], "rows": [["Team", "Sucrose"], ["Swirls", "1"]],
                            "hidden": ["main", 1]}, None)
    assert "117s" not in c and c.count("???") == 2 and "Team: Sucrose" in c and "Swirls: " in c
    assert 'width="116" height="58"' not in c and "<polygon" not in c and "#3A434E" not in c   # no lock, stars, level


def test_body_drops_a_thin_ribbon_at_the_side():
    from PIL import Image
    a = np.zeros((400, 600), np.uint8)
    a[100:110, 0:300] = 255                                   # a thin ribbon trailing left
    a[50:400, 300:500] = 255                                  # the body
    out = pipeline.body(Image.fromarray(np.dstack([a, a, a, a]), "RGBA"))
    assert out.width == 200                                   # just the body's columns
