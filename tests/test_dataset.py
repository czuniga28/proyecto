from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.slices import build_slice_index
from src.train.dataset import (
    SliceDataset,
    denormalize_boxes,
    hflip,
    shift_image,
    stack_25d,
)

BACKGROUND = -1.0
BRIGHT = 5.0
ROI = (10, 20, 30, 50)  # Not symmetric, so a missed flip or a swapped axis shows up
POSITIVE = 2
NEGATIVE = 0


def make_dataset_files(root: Path, height: int = 64, width: int = 64) -> pd.DataFrame:
    """Save a volume with a bright rectangle exactly inside the ROI of slices 1-3."""
    x0, y0, x1, y1 = ROI
    vol = np.full((5, height, width), BACKGROUND, dtype=np.float16)
    vol[1:4, y0:y1, x0:x1] = BRIGHT

    (root / 'volumes').mkdir()
    np.save(root / 'volumes' / '1-5.npy', vol)

    return build_slice_index(pd.DataFrame([{
        'examId': 1, 'seriesNo': 5, 'depth': 5, 'roiX': x0, 'roiY': y0, 'roiZ': 1,
        'roiWidth': x1 - x0, 'roiHeight': y1 - y0, 'roiDepth': 3,
        'aclDiagnosis': 1, 'split': 'fold_0',
    }]))


def bright_bbox(image: torch.Tensor) -> tuple[int, int, int, int]:
    """End-exclusive bounding box of the bright pixels of the central channel."""
    rows, cols = np.nonzero(image[1].numpy() > 0)
    return cols.min(), rows.min(), cols.max() + 1, rows.max() + 1


@pytest.fixture
def slices(tmp_path) -> pd.DataFrame:
    return make_dataset_files(tmp_path)


def test_stack_25d_repeats_extreme_slice_at_the_edges():
    vol = np.arange(5, dtype=np.float16)[:, None, None] * np.ones((5, 4, 6), dtype=np.float16)

    assert stack_25d(vol, 2)[:, 0, 0].tolist() == [1, 2, 3]
    assert stack_25d(vol, 0)[:, 0, 0].tolist() == [0, 0, 1]
    assert stack_25d(vol, 4)[:, 0, 0].tolist() == [3, 4, 4]
    assert stack_25d(vol, 2).shape == (3, 4, 6)
    assert stack_25d(vol, 2).dtype == np.float32


@pytest.mark.parametrize('s', [-1, 5])
def test_stack_25d_rejects_slice_out_of_range(s):
    with pytest.raises(ValueError, match='out of range'):
        stack_25d(np.zeros((5, 4, 6)), s)


def test_hflip_swaps_box_edges_and_returns_positive_strides():
    img = np.arange(12, dtype=np.float32).reshape(3, 4)

    flipped, box = hflip(img, (1, 0, 3, 2))

    assert box == (1, 0, 3, 2)
    assert flipped[0].tolist() == [3, 2, 1, 0]
    assert hflip(img, (0, 1, 1, 3))[1] == (3, 1, 4, 3)
    torch.from_numpy(flipped)


def test_shift_image_fills_uncovered_pixels_without_wrapping():
    img = np.arange(1, 13, dtype=np.float32).reshape(3, 4)

    shifted, _ = shift_image(img, (0, 0, 1, 1), dx=1, dy=1, fill=-9.0)

    assert shifted.tolist() == [[-9, -9, -9, -9], [-9, 1, 2, 3], [-9, 5, 6, 7]]


def test_shift_image_clips_box_to_the_image():
    img = np.zeros((10, 10), dtype=np.float32)

    assert shift_image(img, (7, 0, 10, 3), dx=2, dy=0, fill=0.0)[1] == (9, 0, 10, 3)
    assert shift_image(img, (0, 1, 3, 4), dx=-1, dy=-2, fill=0.0)[1] == (0, 0, 2, 2)


def test_box_stays_on_bright_region_without_augmentation(tmp_path, slices):
    item = SliceDataset(slices, tmp_path, size=64)[POSITIVE]

    assert bright_bbox(item['image']) == ROI
    assert (item['box'] * 64).tolist() == list(ROI)


def test_box_stays_on_bright_region_after_flip(tmp_path, slices):
    item = SliceDataset(slices, tmp_path, size=64, augment=True, hflip_p=1.0, max_shift=0)[POSITIVE]

    assert bright_bbox(item['image']) == (34, 20, 54, 50)
    assert (item['box'] * 64).tolist() == [34, 20, 54, 50]


@pytest.mark.parametrize('hflip_p', [0.0, 1.0])
def test_box_stays_on_bright_region_after_shift(tmp_path, slices, hflip_p):
    dataset = SliceDataset(slices, tmp_path, size=64, augment=True, hflip_p=hflip_p, max_shift=8)
    torch.manual_seed(0)

    boxes = set()
    for _ in range(50):
        item = dataset[POSITIVE]
        box = tuple(round(c) for c in (item['box'] * 64).tolist())
        assert bright_bbox(item['image']) == box
        boxes.add(box)

    assert len(boxes) > 1, 'the shift never moved the box'


def test_box_stays_on_bright_region_after_resizing_a_non_square_image(tmp_path):
    slices = make_dataset_files(tmp_path, height=64, width=128)
    dataset = SliceDataset(slices, tmp_path, size=128, augment=True, hflip_p=1.0, max_shift=8)
    torch.manual_seed(0)

    for _ in range(20):
        item = dataset[POSITIVE]
        box = np.array((item['box'] * 128).tolist())
        assert np.abs(np.array(bright_bbox(item['image'])) - box).max() <= 2


# Between 1x and 2x the width, unclamped slice stops turn negative and count from the end
@pytest.mark.parametrize('dx', [100, 5, -5])
def test_shift_image_past_the_axis_leaves_only_fill_and_an_empty_box(dx):
    img = np.arange(1, 13, dtype=np.float32).reshape(3, 4)

    shifted, (x0, y0, x1, y1) = shift_image(img, (1, 0, 3, 2), dx=dx, dy=0, fill=-9.0)

    assert (shifted == -9.0).all()
    assert x0 == x1
    assert (y0, y1) == (0, 2)


def test_uncovered_pixels_use_the_image_minimum(tmp_path, slices):
    dataset = SliceDataset(slices, tmp_path, size=64, augment=True, max_shift=8)
    torch.manual_seed(0)

    shifted = 0
    for _ in range(20):
        item = dataset[POSITIVE]
        image = item['image']
        is_known = torch.isclose(image, torch.tensor(BACKGROUND)) | torch.isclose(image, torch.tensor(BRIGHT))
        assert is_known.all(), 'some pixel is neither background nor ROI'
        shifted += (item['box'] * 64).tolist() != list(ROI)

    assert shifted > 0, 'the shift never uncovered any pixel'


def test_returned_fields_of_a_positive_slice(tmp_path, slices):
    item = SliceDataset(slices, tmp_path, size=32)[POSITIVE]

    assert item['image'].shape == (3, 32, 32)
    assert item['image'].dtype == torch.float32
    assert item['label'] == 1.0
    assert item['has_box']
    assert item['index'] == POSITIVE
    assert item['box'].tolist() == pytest.approx([10 / 64, 20 / 64, 30 / 64, 50 / 64])


@pytest.mark.parametrize('augment', [False, True])
def test_negative_slice_has_zero_box_and_no_mask(tmp_path, slices, augment):
    dataset = SliceDataset(slices, tmp_path, size=32, augment=augment, hflip_p=1.0)

    item = dataset[NEGATIVE]

    assert item['label'] == 0.0
    assert not item['has_box']
    assert item['box'].tolist() == [0, 0, 0, 0]


def test_volumes_are_memory_mapped(tmp_path, slices):
    dataset = SliceDataset(slices, tmp_path, size=32)

    assert isinstance(dataset._load_volume('volumes/1-5.npy'), np.memmap)


def test_torch_seed_makes_augmentation_reproducible(tmp_path, slices):
    dataset = SliceDataset(slices, tmp_path, size=32, augment=True, hflip_p=0.5, max_shift=8)

    torch.manual_seed(3)
    first = dataset[POSITIVE]
    torch.manual_seed(3)
    second = dataset[POSITIVE]

    assert torch.equal(first['image'], second['image'])
    assert torch.equal(first['box'], second['box'])


def test_denormalize_boxes_inverts_the_normalization(tmp_path, slices):
    item = SliceDataset(slices, tmp_path, size=32)[POSITIVE]

    pixels = denormalize_boxes(item['box'][None], 64, 64)

    assert pixels.tolist() == [list(ROI)]
    assert denormalize_boxes(np.array([0.5, 0.25, 1.0, 0.75]), 320, 160).tolist() == [160, 40, 320, 120]


def test_stack_25d_context_0_is_2d_and_context_2_skips_a_slice():
    vol = np.arange(5, dtype=np.float16)[:, None, None] * np.ones((5, 4, 6), dtype=np.float16)

    assert stack_25d(vol, 2, context=0)[:, 0, 0].tolist() == [2, 2, 2]
    assert stack_25d(vol, 2, context=2)[:, 0, 0].tolist() == [0, 2, 4]
    assert stack_25d(vol, 1, context=2)[:, 0, 0].tolist() == [0, 1, 3]
    with pytest.raises(ValueError, match='context'):
        stack_25d(vol, 2, context=-1)


def test_dataset_passes_context_to_the_stack(tmp_path, slices):
    item = SliceDataset(slices, tmp_path, size=64, context=0)[1]

    # Slice 1 is bright and slice 0 is not: with context 0 every channel is slice 1
    assert torch.equal(item['image'][0], item['image'][1])
    assert torch.equal(item['image'][2], item['image'][1])
