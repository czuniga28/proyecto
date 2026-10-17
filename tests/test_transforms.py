import numpy as np
import pytest

from src.data.transforms import normalize_volume, pad_volume


def test_normalize_gives_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    vol = rng.integers(0, 4096, size=(5, 32, 32), dtype=np.uint16)

    out = normalize_volume(vol, 0.5, 99.5)

    assert out.dtype == np.float32
    assert out.shape == vol.shape
    assert abs(out.mean()) < 1e-4
    assert abs(out.std() - 1) < 1e-4


def test_normalize_does_not_wrap_uint16():
    vol = np.full((2, 4, 4), 3000, dtype=np.uint16)
    vol[0] = 10  # half the voxels well below the mean

    out = normalize_volume(vol, 0, 100)

    assert out.min() < 0  # uint16 arithmetic would have wrapped these to huge positives
    assert out.max() < 2


def test_normalize_clips_outliers():
    vol = np.ones((1, 100, 100), dtype=np.uint16) * 100
    vol[0, 0, 0] = 4095  # one saturated voxel

    out = normalize_volume(vol, 0.5, 99.5)

    assert np.isfinite(out).all()
    assert out.max() == pytest.approx(0.0)  # outlier clipped to the bulk value


def test_constant_volume_becomes_zeros():
    out = normalize_volume(np.full((2, 3, 3), 7, dtype=np.uint16), 0.5, 99.5)

    assert not out.any()


def test_pad_keeps_original_pixels_at_origin():
    vol = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    out = pad_volume(vol, 5, 6)

    assert out.shape == (2, 5, 6)
    np.testing.assert_array_equal(out[:, :3, :4], vol)  # ROI coordinates stay valid
    assert (out[:, 3:, :] == vol.min()).all()
    assert (out[:, :, 4:] == vol.min()).all()


def test_pad_rejects_larger_input():
    with pytest.raises(ValueError):
        pad_volume(np.zeros((1, 330, 320), np.float32), 320, 320)
