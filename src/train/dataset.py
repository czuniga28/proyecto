from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

BOX_COLUMNS = ['x0', 'y0', 'x1', 'y1']


def stack_25d(vol: np.ndarray, s: int) -> np.ndarray:
    """
    Stack slice s and its two neighbours as the channels of a 2.5D image.

    Args:
        vol: Volume of shape (D, H, W).
        s: Index of the central slice.

    Returns:
        A float32 array of shape (3, H, W) with slices s-1, s and s+1. At the
        edges of the volume the extreme slice is repeated.

    Raises:
        ValueError: If s is not a slice of vol.
    """
    # np.clip would turn an invalid s into a valid-looking stack of the extreme slice
    if not 0 <= s < len(vol):
        raise ValueError(f"Slice {s} out of range for a volume of depth {len(vol)}")

    neighbours = np.clip([s - 1, s, s + 1], 0, len(vol) - 1)
    return vol[neighbours].astype(np.float32)


def hflip(img: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Flip an image and its box horizontally.

    Args:
        img: Image of shape (..., H, W).
        box: End-exclusive box (x0, y0, x1, y1).

    Returns:
        The flipped image and the box (W - x1, y0, W - x0, y1).
    """
    x0, y0, x1, y1 = box
    width = img.shape[-1]

    # copy: torch.from_numpy rejects the negative strides of the flipped view
    return img[..., ::-1].copy(), (width - x1, y0, width - x0, y1)


def _shift_slices(shift: int, size: int) -> tuple[slice, slice]:
    """Return the source and destination slices that move an axis of length size by shift."""
    # Past the axis length nothing survives; without this the stops would turn negative
    shift = max(-size, min(shift, size))
    source = slice(max(-shift, 0), size - max(shift, 0))
    destination = slice(max(shift, 0), size + min(shift, 0))
    return source, destination


def shift_image(
    img: np.ndarray,
    box: tuple[int, int, int, int],
    dx: int,
    dy: int,
    fill: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Translate an image and its box without wrapping around the borders.

    Args:
        img: Image of shape (..., H, W).
        box: End-exclusive box (x0, y0, x1, y1).
        dx: Columns to move; positive moves right.
        dy: Rows to move; positive moves down.
        fill: Value of the uncovered pixels, normally the image minimum.

    Returns:
        The shifted image and the moved box clipped to the image. A box that
        leaves the image entirely ends up empty, with x0 == x1 or y0 == y1.
    """
    x0, y0, x1, y1 = box
    height, width = img.shape[-2:]

    source_y, destination_y = _shift_slices(dy, height)
    source_x, destination_x = _shift_slices(dx, width)

    shifted = np.full_like(img, fill)
    shifted[..., destination_y, destination_x] = img[..., source_y, source_x]

    x0, x1 = np.clip([x0 + dx, x1 + dx], 0, width).tolist()
    y0, y1 = np.clip([y0 + dy, y1 + dy], 0, height).tolist()

    return shifted, (x0, y0, x1, y1)


def denormalize_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Convert boxes normalized to [0, 1] back to pixel coordinates.

    Args:
        boxes: Array or CPU tensor of shape (..., 4) with boxes (x0, y0, x1, y1).
        width: Width of the original image, in pixels.
        height: Height of the original image, in pixels.

    Returns:
        A float array of the same shape, in the coordinates aggregate.py expects.
    """
    return np.asarray(boxes, dtype=float) * np.array([width, height, width, height], dtype=float)


class SliceDataset(Dataset):
    """
    Serve one 2.5D image per slice, with its label and its box.

    Args:
        slices: Slice index with a RangeIndex, as returned by train_slices.
        processed_dir: Directory that the 'volume_file' column is relative to.
        size: Side of the square image returned.
        augment: Whether to apply the augmentations below.
        hflip_p: Probability of a horizontal flip. Left at 0: orientation is
            consistent (anterior on the left), checked in the kneeLR figure.
        max_shift: Largest translation per axis, in pixels of the original image.
    """

    def __init__(
        self,
        slices: pd.DataFrame,
        processed_dir: str | Path,
        size: int = 224,
        augment: bool = False,
        hflip_p: float = 0.0,
        max_shift: int = 20,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.size = size
        self.augment = augment
        self.hflip_p = hflip_p
        self.max_shift = max_shift

        self.volume_files = slices['volume_file'].to_numpy()
        self.slice_idx = slices['slice_idx'].to_numpy(dtype=int)
        self.contains_acl = slices['contains_acl'].to_numpy(dtype=bool)
        self.boxes = slices[BOX_COLUMNS].to_numpy(dtype=float, na_value=np.nan)

    def __len__(self) -> int:
        return len(self.volume_files)

    def _load_volume(self, volume_file: str) -> np.ndarray:
        # Not cached: every memmap holds a file descriptor, and keeping one per
        # volume can exceed the open-file limit (256 by default on macOS)
        return np.load(self.processed_dir / volume_file, mmap_mode='r')

    def _augment(self, img: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if torch.rand(1).item() < self.hflip_p:
            img, box = hflip(img, box)
        if self.max_shift > 0:
            dx, dy = torch.randint(-self.max_shift, self.max_shift + 1, (2,)).tolist()
            img, box = shift_image(img, box, dx, dy, fill=img.min())
        return img, box

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | int]:
        has_box = bool(self.contains_acl[i])
        img = stack_25d(self._load_volume(self.volume_files[i]), self.slice_idx[i])

        # Negatives carry a placeholder through the geometry: their box is reset below
        box = self.boxes[i] if has_box else np.zeros(4)
        if self.augment:
            img, box = self._augment(img, box)

        height, width = img.shape[-2:]
        box = np.asarray(box, dtype=float) / np.array([width, height, width, height], dtype=float)
        if not has_box:
            box = np.zeros(4)

        image = F.interpolate(
            torch.from_numpy(np.asarray(img))[None],
            size=(self.size, self.size),
            mode='bilinear',
            align_corners=False,
            antialias=True,
        )[0]

        return {
            'image': image,
            'label': torch.tensor(has_box, dtype=torch.float32),
            'box': torch.tensor(box, dtype=torch.float32),
            'has_box': torch.tensor(has_box),
            'index': i,
        }
