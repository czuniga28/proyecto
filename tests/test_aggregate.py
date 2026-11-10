import numpy as np
import pandas as pd
import pytest

from src.eval.aggregate import (
    aggregate_predictions, aggregate_volume, best_run, select_tau, weighted_median,
)
from src.eval.metrics import BOX3D_COLUMNS


def test_best_run_finds_the_high_probability_block():
    p = np.array([0.0, 0.1, 0.9, 0.8, 0.95, 0.1, 0.0])
    assert best_run(p, tau=0.5) == (2, 5)


def test_best_run_bridges_one_weak_slice_between_strong_ones():
    # Gains: +0.45, -0.1, +0.45 -> keeping the gap (0.8) beats either side (0.45)
    p = np.array([0.0, 0.95, 0.4, 0.95, 0.0])
    assert best_run(p, tau=0.5) == (1, 4)


def test_best_run_never_empty_when_all_below_tau():
    p = np.array([0.1, 0.3, 0.2])
    assert best_run(p, tau=0.5) == (1, 2)


def test_best_run_at_volume_edges():
    assert best_run(np.array([0.9, 0.9, 0.0, 0.0]), tau=0.5) == (0, 2)
    assert best_run(np.array([0.0, 0.0, 0.9, 0.9]), tau=0.5) == (2, 4)


def test_best_run_empty_input_raises():
    with pytest.raises(ValueError):
        best_run(np.array([]), tau=0.5)


def test_weighted_median_ignores_outlier_and_falls_back_to_equal_weights():
    assert weighted_median([10, 11, 200], [1, 1, 1]) == 11
    # Heavy weight pulls the median to its value
    assert weighted_median([10, 11, 200], [0.1, 0.1, 5]) == 200
    assert weighted_median([10, 11, 12], [0, 0, 0]) == 11


def test_aggregate_volume_recovers_known_box():
    depth = 8
    p = np.zeros(depth)
    p[3:6] = 0.9
    boxes = np.tile([100.0, 110.0, 190.0, 200.0], (depth, 1))
    boxes[4] = [101, 111, 191, 201]      # small jitter on the median slice
    boxes[0] = [0, 0, 10, 10]            # garbage outside the run must not matter

    box = aggregate_volume(p, boxes, tau=0.5)

    np.testing.assert_array_equal(box, [100, 110, 3, 190, 200, 6])


def test_aggregate_volume_checks_shapes():
    with pytest.raises(ValueError):
        aggregate_volume(np.zeros(4), np.zeros((3, 4)), tau=0.5)


def synthetic_preds() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two volumes whose predictions are perfect at tau=0.5 and too wide at tau=0.1."""
    rows, gt = [], []
    for exam, (z0, z1) in [(1, (2, 5)), (2, (4, 6))]:
        for s in range(10):
            inside = z0 <= s < z1
            near = z0 - 1 <= s < z1 + 1
            p = 0.9 if inside else (0.3 if near else 0.0)
            rows.append({'examId': exam, 'seriesNo': 5, 'slice_idx': s, 'p': p,
                         'px0': 100, 'py0': 100, 'px1': 180, 'py1': 180})
        gt.append({'examId': exam, 'seriesNo': 5,
                   'x0': 100, 'y0': 100, 'z0': z0, 'x1': 180, 'y1': 180, 'z1': z1})
    # Shuffle rows: aggregation must not depend on input order
    preds = pd.DataFrame(rows).sample(frac=1, random_state=0)
    return preds, pd.DataFrame(gt)


def test_aggregate_predictions_one_row_per_volume():
    preds, gt = synthetic_preds()

    out = aggregate_predictions(preds, tau=0.5)

    assert list(out.columns) == ['examId', 'seriesNo', *BOX3D_COLUMNS]
    np.testing.assert_array_equal(out[BOX3D_COLUMNS].to_numpy(), gt[BOX3D_COLUMNS].to_numpy())


def test_aggregate_predictions_rejects_missing_slices():
    preds, _ = synthetic_preds()
    preds = preds[~((preds.examId == 1) & (preds.slice_idx == 0))]

    with pytest.raises(ValueError, match="missing slices"):
        aggregate_predictions(preds, tau=0.5)


def test_select_tau_prefers_the_threshold_that_matches_gt():
    preds, gt = synthetic_preds()

    tau, curve = select_tau(preds, gt, taus=np.array([0.1, 0.5]))

    assert tau == 0.5
    assert curve[0.5] == pytest.approx(1.0)
    assert curve[0.1] < 1.0
