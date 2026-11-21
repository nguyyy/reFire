"""cutout plumbing with a stand-in alpha model (the real BiRefNet run is in the smoke/golden path)."""
import numpy as np

from refire.thumb import cutout


def test_inset_trims_every_side():
    img = np.zeros((200, 400, 3), np.uint8)
    assert cutout.inset(img, 0.015).shape == (194, 388, 3)
    assert cutout.inset(img, 0).shape == (200, 400, 3)


def test_cutout_is_bgra_with_feathered_alpha():
    img = np.full((100, 100, 3), 128, np.uint8)
    hard = lambda im: np.where(np.arange(im.shape[1])[None, :] < im.shape[1] // 2, 255, 0).astype(np.uint8).repeat(im.shape[0], 0)  # noqa: E731
    out = cutout.cutout(img, inset_frac=0.0, feather=1.0, alpha_fn=hard)
    assert out.shape == (100, 100, 4) and out.dtype == np.uint8
    edge = out[50, 47:53, 3]
    assert edge[0] == 255 and edge[-1] == 0 and 0 < edge[2] < 255 or 0 < edge[3] < 255   # soft, not a step
    assert (out[:, :, :3] == 128).all()


def test_models_are_pinned():
    name, sha = cutout.MODELS["lite"]
    assert name.endswith(".onnx") and len(sha) == 64
