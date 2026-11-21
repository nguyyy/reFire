"""thumb pipeline pieces that need no model/network: validation, accent fix, art crop, snap."""
import numpy as np
from PIL import Image

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


def test_art_crop_keeps_the_middle_of_the_figure(tmp_path):
    im = Image.new("RGBA", (1000, 500), (0, 0, 0, 0))
    im.paste((255, 255, 255, 255), (100, 50, 900, 450))
    pipeline.art_crop_out = None
    out = pipeline.art_crop(_save(im, tmp_path / "s.png"), tmp_path / "c.png")
    assert Image.open(out).size == (480, 400)                         # 60% of the 800 px wide figure


def _save(im, p):
    im.save(p)
    return p


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
