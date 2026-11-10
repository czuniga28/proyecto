import numpy as np
import pandas as pd

from src.eval.metrics import BOX3D_COLUMNS, iou_3d


def best_run(p: np.ndarray, tau: float) -> tuple[int, int]:
    """
    Contiguous slice range that maximizes sum(p - tau) (maximum subarray, Kadane).

    Slices with p > tau add to the run and slices below it subtract, so a single
    weak slice between two strong ones is kept while a long weak tail is cut.
    The run is never empty: if every p < tau it is the single most likely slice,
    because the task always expects one box per volume.

    Args:
        p: Array of shape (depth,) with per-slice probabilities, in slice order.
        tau: Threshold, tuned on validation (see select_tau).

    Returns:
        (z0, z1), end-exclusive. Ties keep the earliest, shortest run.
    """
    gain = np.asarray(p, dtype=float) - tau
    if gain.size == 0:
        raise ValueError("empty probability vector")

    best_sum, best = gain[0], (0, 1)
    cur_sum, cur_start = gain[0], 0
    for i in range(1, len(gain)):
        # Restart the run when carrying the previous sum would only hurt
        if cur_sum < 0:
            cur_sum, cur_start = gain[i], i
        else:
            cur_sum += gain[i]
        if cur_sum > best_sum:
            best_sum, best = cur_sum, (cur_start, i + 1)
    return best


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Lower weighted median; falls back to equal weights if all weights are 0.

    A median rather than a mean so that one bad per-slice box does not drag the
    3D box.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        weights = np.ones_like(values)

    order = np.argsort(values, kind='stable')
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, cum[-1] / 2)])


def aggregate_volume(p: np.ndarray, boxes: np.ndarray, tau: float) -> np.ndarray:
    """
    Combine per-slice predictions of one volume into a single 3D box.

    The z extent comes from the classification head (best_run); x and y from the
    p-weighted median of the box head's outputs on the slices of that run.

    Args:
        p: Array of shape (depth,) with per-slice probabilities, in slice order.
        boxes: Array of shape (depth, 4) with predicted 2D boxes (x0, y0, x1, y1)
            in original 320x320 pixel coordinates.
        tau: Threshold for best_run.

    Returns:
        Array of shape (6,) in BOX3D_COLUMNS order.
    """
    boxes = np.asarray(boxes, dtype=float)
    if boxes.shape != (len(p), 4):
        raise ValueError(f"boxes shape {boxes.shape} != ({len(p)}, 4)")

    z0, z1 = best_run(p, tau)
    w = np.asarray(p, dtype=float)[z0:z1]
    x0, y0, x1, y1 = (weighted_median(boxes[z0:z1, j], w) for j in range(4))
    return np.array([x0, y0, z0, x1, y1, z1], dtype=float)


def aggregate_predictions(preds: pd.DataFrame, tau: float) -> pd.DataFrame:
    """
    Apply aggregate_volume to every volume in a table of per-slice predictions.

    Args:
        preds: One row per slice with examId, seriesNo, slice_idx, p and the
            predicted 2D box px0, py0, px1, py1. Every volume must have all its
            slices 0..depth-1, otherwise z indices would be shifted.
        tau: Threshold for best_run.

    Returns:
        One row per volume: examId, seriesNo and BOX3D_COLUMNS.
    """
    rows = []
    for (exam, series), g in preds.groupby(['examId', 'seriesNo'], sort=True):
        g = g.sort_values('slice_idx')
        if not np.array_equal(g['slice_idx'].to_numpy(), np.arange(len(g))):
            raise ValueError(f"volume {exam}-{series} is missing slices")
        box = aggregate_volume(
            g['p'].to_numpy(), g[['px0', 'py0', 'px1', 'py1']].to_numpy(), tau,
        )
        rows.append([exam, series, *box])
    return pd.DataFrame(rows, columns=['examId', 'seriesNo', *BOX3D_COLUMNS])


def select_tau(
    preds: pd.DataFrame, gt: pd.DataFrame, taus: np.ndarray,
) -> tuple[float, pd.Series]:
    """
    Pick the tau that maximizes mean 3D IoU. Call it on validation folds only:
    tuning on the test set would leak.

    Args:
        preds: Per-slice predictions, as in aggregate_predictions.
        gt: One row per volume with examId, seriesNo and BOX3D_COLUMNS.
        taus: Candidate thresholds.

    Returns:
        (best tau, Series of mean IoU indexed by tau). Ties keep the first tau in taus.
    """
    key = ['examId', 'seriesNo']
    scores = {}
    for tau in taus:
        pred_boxes = aggregate_predictions(preds, tau)
        m = pred_boxes.merge(gt[key + BOX3D_COLUMNS], on=key, suffixes=('', '_gt'),
                             validate='one_to_one')
        if len(m) != len(pred_boxes):
            raise ValueError("some predicted volumes have no ground truth")
        ious = iou_3d(m[BOX3D_COLUMNS].to_numpy(),
                      m[[f'{c}_gt' for c in BOX3D_COLUMNS]].to_numpy())
        scores[float(tau)] = ious.mean()

    curve = pd.Series(scores, name='mean_iou').rename_axis('tau')
    return float(curve.idxmax()), curve
