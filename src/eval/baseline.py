import numpy as np
import pandas as pd

from src.eval.metrics import iou_3d, roi_to_box3d


def mean_box(boxes: np.ndarray) -> np.ndarray:
    """
    Fixed box at the average position and size (Phase 1 baseline).

    Averaging each coordinate (x0, y0, z0, x1, y1, z1) is the same as averaging
    the start and the size separately, so this matches the Phase 1 definition.

    Args:
        boxes: Array of shape (n, 6) in BOX3D_COLUMNS order.

    Returns:
        Array of shape (6,).
    """
    boxes = np.asarray(boxes, dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 6 or len(boxes) == 0:
        raise ValueError(f"expected a non-empty (n, 6) array, got {boxes.shape}")
    return boxes.mean(axis=0)


def cv_fixed_box(volumes: pd.DataFrame, n_cv_folds: int) -> pd.DataFrame:
    """
    Cross-validate the fixed-box baseline on the frozen folds of splits.csv.

    For fold k the box is the mean of the other folds' boxes; the test split is
    dropped before anything is computed, so it never contributes to a box.
    Phase 1 computed the baseline on all data at once; refitting per fold makes
    it comparable with the models of this phase, which only see training folds.

    Args:
        volumes: One row per volume with examId, seriesNo, aclDiagnosis, split
            and the roi* columns.
        n_cv_folds: Number of CV folds (fold_0 .. fold_{n_cv_folds-1}).

    Returns:
        One row per (volume, fold): examId, seriesNo, aclDiagnosis, fold,
        part ('train' or 'val') and iou. Every development volume appears once
        as 'val' and n_cv_folds - 1 times as 'train'.
    """
    dev = volumes[volumes['split'] != 'test'].reset_index(drop=True)
    folds = [f'fold_{k}' for k in range(n_cv_folds)]
    unknown = set(dev['split']) - set(folds)
    if unknown:
        raise ValueError(f"unexpected split labels: {sorted(unknown)}")

    boxes = roi_to_box3d(dev)
    parts = []
    for fold in folds:
        is_val = (dev['split'] == fold).to_numpy()
        box = mean_box(boxes[~is_val])
        ious = iou_3d(boxes, box)
        parts.append(dev[['examId', 'seriesNo', 'aclDiagnosis']].assign(
            fold=fold, part=np.where(is_val, 'val', 'train'), iou=ious,
        ))
    return pd.concat(parts, ignore_index=True)
