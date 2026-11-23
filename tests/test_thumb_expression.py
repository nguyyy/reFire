"""expression recipes (no model): blendshape dicts -> scores."""
from refire.thumb import expression as X


def bs(**kw):
    return kw


def test_recipes_pick_the_matching_face():
    shock = bs(jawOpen=0.8, eyeWideLeft=0.7, eyeWideRight=0.7, browInnerUp=0.6)
    grin = bs(mouthSmileLeft=0.9, mouthSmileRight=0.9, jawOpen=0.5, eyeSquintLeft=0.6, eyeSquintRight=0.6)
    flat = bs(mouthPressLeft=0.6, mouthPressRight=0.6)
    assert X.score("shock", shock) > X.score("shock", grin) > X.score("shock", flat)
    assert X.score("laughing", grin) > X.score("laughing", shock)
    assert X.score("hype", grin) > X.score("hype", flat)
    assert X.score("deadpan", flat) > X.score("deadpan", shock)
    sk = bs(browOuterUpLeft=0.8, browOuterUpRight=0.1, mouthPressLeft=0.5, mouthPressRight=0.5)
    assert X.score("skeptical", sk) > X.score("skeptical", shock)


def test_facing_drops_for_closed_eyes_looking_down_and_tilt():
    good = {"bs": {}, "pitch": 3.0, "yaw": -4.0}
    assert X.facing(good) == 1.0
    assert X.facing({"bs": {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.9}, "pitch": 0, "yaw": 0}) < 0.2
    assert X.facing({"bs": {"eyeLookDownLeft": 0.9, "eyeLookDownRight": 0.9}, "pitch": 0, "yaw": 0}) < 0.3
    assert X.facing({"bs": {}, "pitch": 40.0, "yaw": 0}) == 0.0


def test_closed_eyes_never_win_even_when_laughing():
    laugh = dict(mouthSmileLeft=0.9, mouthSmileRight=0.9, jawOpen=0.6, eyeSquintLeft=0.7, eyeSquintRight=0.7)
    shut = {**laugh, "eyeBlinkLeft": 0.95, "eyeBlinkRight": 0.95}
    assert X.score("laughing", shut) < 0.1 < X.score("laughing", laugh)
