import numpy as np
import pandas as pd

from src.eval.metrics import BOX3D_COLUMNS, iou_1d, iou_2d, iou_3d

KEY = ['examId', 'seriesNo']


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


def _single_most_likely(p: np.ndarray) -> tuple[int, int]:
    i = int(np.argmax(p))
    return i, i + 1


def threshold_span(p: np.ndarray, tau: float) -> tuple[int, int]:
    """
    From the first to the last slice with p > tau (ablation alternative to best_run).

    A single false positive far from the ACL stretches the span to reach it.
    Falls back to the most likely slice when no p exceeds tau.

    Returns:
        (z0, z1), end-exclusive.
    """
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        raise ValueError("empty probability vector")
    above = np.flatnonzero(p > tau)
    if above.size == 0:
        return _single_most_likely(p)
    return int(above[0]), int(above[-1]) + 1


def longest_run(p: np.ndarray, tau: float) -> tuple[int, int]:
    """
    Longest contiguous run of slices with p > tau (ablation alternative to best_run).

    Unlike best_run it never bridges a slice below tau. Ties go to the run with
    the larger sum of p, then to the earliest. Falls back to the most likely
    slice when no p exceeds tau.

    Returns:
        (z0, z1), end-exclusive.
    """
    p = np.asarray(p, dtype=float)
    if p.size == 0:
        raise ValueError("empty probability vector")
    above = p > tau
    if not above.any():
        return _single_most_likely(p)

    best, best_key = (0, 1), None
    i = 0
    while i < len(p):
        if not above[i]:
            i += 1
            continue
        j = i
        while j < len(p) and above[j]:
            j += 1
        key = (j - i, p[i:j].sum())
        if best_key is None or key > best_key:
            best, best_key = (i, j), key
        i = j
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


def _xy_wmedian(boxes: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.array([weighted_median(boxes[:, j], w) for j in range(4)])


def _xy_median(boxes: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.median(boxes, axis=0)


def _xy_wmean(boxes: np.ndarray, w: np.ndarray) -> np.ndarray:
    if w.sum() <= 0:
        w = np.ones_like(w)
    return np.average(boxes, axis=0, weights=w)


def _xy_argmax(boxes: np.ndarray, w: np.ndarray) -> np.ndarray:
    return boxes[int(np.argmax(w))]


# Interchangeable rules for the ablation (E1, E2). The defaults are the ones in
# aggregate_volume; the others exist to show whether that choice is justified.
Z_METHODS = {'kadane': best_run, 'threshold': threshold_span, 'longest': longest_run}
XY_METHODS = {'wmedian': _xy_wmedian, 'median': _xy_median, 'wmean': _xy_wmean, 'argmax': _xy_argmax}


def _method(registry: dict, name: str, kind: str):
    if name not in registry:
        raise ValueError(f"Unknown {kind} {name!r}. Valid: {sorted(registry)}")
    return registry[name]


def aggregate_volume(
    p: np.ndarray,
    boxes: np.ndarray,
    tau: float,
    z_method: str = 'kadane',
    xy_method: str = 'wmedian',
) -> np.ndarray:
    """
    Combine per-slice predictions of one volume into a single 3D box.

    The z extent comes from the classification head (best_run); x and y from the
    p-weighted median of the box head's outputs on the slices of that run.

    Args:
        p: Array of shape (depth,) with per-slice probabilities, in slice order.
        boxes: Array of shape (depth, 4) with predicted 2D boxes (x0, y0, x1, y1)
            in original 320x320 pixel coordinates.
        tau: Threshold for best_run.
        z_method: Name in Z_METHODS: how the slice range is chosen.
        xy_method: Name in XY_METHODS: how the boxes of that range are combined.

    Returns:
        Array of shape (6,) in BOX3D_COLUMNS order.
    """
    choose_z = _method(Z_METHODS, z_method, 'z_method')
    combine_xy = _method(XY_METHODS, xy_method, 'xy_method')
    boxes = np.asarray(boxes, dtype=float)
    if boxes.shape != (len(p), 4):
        raise ValueError(f"boxes shape {boxes.shape} != ({len(p)}, 4)")

    z0, z1 = choose_z(p, tau)
    w = np.asarray(p, dtype=float)[z0:z1]
    x0, y0, x1, y1 = combine_xy(boxes[z0:z1], w)
    return np.array([x0, y0, z0, x1, y1, z1], dtype=float)


def aggregate_predictions(
    preds: pd.DataFrame,
    tau: float,
    z_method: str = 'kadane',
    xy_method: str = 'wmedian',
) -> pd.DataFrame:
    """
    Apply aggregate_volume to every volume in a table of per-slice predictions.

    Args:
        preds: One row per slice with examId, seriesNo, slice_idx, p and the
            predicted 2D box px0, py0, px1, py1. Every volume must have all its
            slices 0..depth-1, otherwise z indices would be shifted.
        tau: Threshold for best_run.
        z_method, xy_method: Passed to aggregate_volume.

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
            z_method, xy_method,
        )
        rows.append([exam, series, *box])
    return pd.DataFrame(rows, columns=['examId', 'seriesNo', *BOX3D_COLUMNS])


def select_tau(
    preds: pd.DataFrame,
    gt: pd.DataFrame,
    taus: np.ndarray,
    z_method: str = 'kadane',
    xy_method: str = 'wmedian',
) -> tuple[float, pd.Series]:
    """
    Pick the tau that maximizes mean 3D IoU. Call it on validation folds only:
    tuning on the test set would leak.

    Args:
        preds: Per-slice predictions, as in aggregate_predictions.
        gt: One row per volume with examId, seriesNo and BOX3D_COLUMNS.
        taus: Candidate thresholds.
        z_method, xy_method: Passed to aggregate_predictions, so each
            aggregation rule gets its own best tau.

    Returns:
        (best tau, Series of mean IoU indexed by tau). Ties keep the first tau in taus.
    """
    key = ['examId', 'seriesNo']
    scores = {}
    for tau in taus:
        pred_boxes = aggregate_predictions(preds, tau, z_method, xy_method)
        m = pred_boxes.merge(gt[key + BOX3D_COLUMNS], on=key, suffixes=('', '_gt'),
                             validate='one_to_one')
        if len(m) != len(pred_boxes):
            raise ValueError("some predicted volumes have no ground truth")
        ious = iou_3d(m[BOX3D_COLUMNS].to_numpy(),
                      m[[f'{c}_gt' for c in BOX3D_COLUMNS]].to_numpy())
        scores[float(tau)] = ious.mean()

    curve = pd.Series(scores, name='mean_iou').rename_axis('tau')
    return float(curve.idxmax()), curve


def score_volumes(pred_boxes: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """
    Per-volume IoU of predicted 3D boxes, overall and by component.

    iou_z (the slice range) and iou_xy (the box in the sagittal plane) show
    whether an error comes from the classification head or from the box head.

    Args:
        pred_boxes: Output of aggregate_predictions.
        gt: One row per volume with examId, seriesNo and BOX3D_COLUMNS; other
            columns, such as aclDiagnosis, are kept.

    Returns:
        gt without BOX3D_COLUMNS, in gt's order, plus iou, iou_z and iou_xy.

    Raises:
        ValueError: If a ground-truth volume has no prediction.
    """
    m = gt.reset_index(drop=True).merge(
        pred_boxes[KEY + BOX3D_COLUMNS], on=KEY, how='left',
        suffixes=('_gt', ''), validate='one_to_one',
    )
    if m[BOX3D_COLUMNS].isna().any().any():
        raise ValueError("some ground-truth volumes have no predictions")

    pred = m[BOX3D_COLUMNS].to_numpy()
    true = m[[f'{c}_gt' for c in BOX3D_COLUMNS]].to_numpy()
    z, xy = [2, 5], [0, 1, 3, 4]
    return gt.drop(columns=BOX3D_COLUMNS).reset_index(drop=True).assign(
        iou=iou_3d(pred, true),
        iou_z=iou_1d(pred[:, z], true[:, z]),
        iou_xy=iou_2d(pred[:, xy], true[:, xy]),
    )
