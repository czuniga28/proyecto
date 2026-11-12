import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

# Boxes are end-exclusive, the same convention as slices.py and the QA bounds check:
#   2D: (x0, y0, x1, y1)            x = column (roiX), y = row (roiY)
#   3D: (x0, y0, z0, x1, y1, z1)    z = slice index (roiZ)
BOX3D_COLUMNS = ['x0', 'y0', 'z0', 'x1', 'y1', 'z1']


def iou_1d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    IoU between intervals (start, end), vectorized over leading dimensions.

    Used for the z extent of 3D boxes, to tell slice-range errors from in-plane ones.

    Args:
        a, b: Arrays of shape (..., 2), broadcastable against each other.

    Returns:
        Array of shape (...,) with values in [0, 1]. Two empty intervals give 0.
    """
    return _iou(np.asarray(a, dtype=float), np.asarray(b, dtype=float), ndim=1)


def iou_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    IoU between 2D boxes (x0, y0, x1, y1), vectorized over leading dimensions.

    Args:
        a, b: Arrays of shape (..., 4), broadcastable against each other.

    Returns:
        Array of shape (...,) with values in [0, 1]. Two empty boxes give 0.
    """
    return _iou(np.asarray(a, dtype=float), np.asarray(b, dtype=float), ndim=2)


def iou_3d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    IoU between 3D boxes (x0, y0, z0, x1, y1, z1), vectorized over leading dimensions.

    Args:
        a, b: Arrays of shape (..., 6), broadcastable against each other.

    Returns:
        Array of shape (...,) with values in [0, 1]. Two empty boxes give 0.
    """
    return _iou(np.asarray(a, dtype=float), np.asarray(b, dtype=float), ndim=3)


def _iou(a: np.ndarray, b: np.ndarray, ndim: int) -> np.ndarray:
    lo = np.maximum(a[..., :ndim], b[..., :ndim])
    hi = np.minimum(a[..., ndim:], b[..., ndim:])
    inter = np.clip(hi - lo, 0, None).prod(axis=-1)

    vol_a = np.clip(a[..., ndim:] - a[..., :ndim], 0, None).prod(axis=-1)
    vol_b = np.clip(b[..., ndim:] - b[..., :ndim], 0, None).prod(axis=-1)
    union = vol_a + vol_b - inter

    # Avoid 0/0 when both boxes are empty: define IoU as 0 there
    return np.divide(inter, union, out=np.zeros_like(union), where=union > 0)


def roi_to_box3d(df: pd.DataFrame) -> np.ndarray:
    """
    Convert KneeMRI ROI columns into end-exclusive 3D boxes.

    Args:
        df: DataFrame with roiX, roiY, roiZ, roiWidth, roiHeight, roiDepth.

    Returns:
        Array of shape (len(df), 6) in BOX3D_COLUMNS order.
    """
    x0, y0, z0 = df['roiX'], df['roiY'], df['roiZ']
    return np.column_stack([
        x0, y0, z0,
        x0 + df['roiWidth'], y0 + df['roiHeight'], z0 + df['roiDepth'],
    ]).astype(float)


def summarize_iou(ious: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """
    Per-class and global summary of volume-level IoU.

    Reported per class because with ~6 % ruptures the global mean can hide a
    complete failure on that class (Phase 2, imbalance strategy).

    Args:
        ious: Array of shape (n,) with one IoU per volume.
        labels: Array of shape (n,) with aclDiagnosis per volume.

    Returns:
        DataFrame indexed by 'all', 0, 1, 2 (classes present) with columns
        n, mean, std, median, frac_ge_05, frac_ge_03.
    """
    ious = np.asarray(ious, dtype=float)
    labels = np.asarray(labels)
    groups = {'all': np.ones(len(ious), dtype=bool)}
    groups.update({int(c): labels == c for c in np.unique(labels)})

    rows = {}
    for name, mask in groups.items():
        x = ious[mask]
        rows[name] = {
            'n': len(x),
            'mean': x.mean(),
            'std': x.std(ddof=1) if len(x) > 1 else np.nan,
            'median': np.median(x),
            'frac_ge_05': (x >= 0.5).mean(),
            'frac_ge_03': (x >= 0.3).mean(),
        }
    return pd.DataFrame.from_dict(rows, orient='index')


def slice_classification_metrics(
    y_true: np.ndarray, p: np.ndarray, threshold: float = 0.5,
) -> dict[str, float]:
    """
    Metrics for the "does this slice contain the ACL?" head.

    Accuracy is deliberately left out: with ~8:1 negatives it rewards predicting
    "no ACL" everywhere.

    Args:
        y_true: Array of shape (n,) with contains_acl (bool or 0/1).
        p: Array of shape (n,) with predicted probabilities.
        threshold: Probability above which a slice counts as positive for F1.

    Returns:
        Dict with 'auprc' (average precision) and 'f1'.
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    return {
        'auprc': float(average_precision_score(y_true, p)),
        'f1': float(f1_score(y_true, (p >= threshold).astype(int), zero_division=0)),
    }
