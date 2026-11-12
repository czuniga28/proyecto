import numpy as np
import pandas as pd
import pytest

from src.eval.aggregate import (
    XY_METHODS,
    Z_METHODS,
    aggregate_predictions,
    aggregate_volume,
    best_run,
    longest_run,
    score_volumes,
    select_tau,
    threshold_span,
    weighted_median,
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


# --- Ablation alternatives (E1: z, E2: xy) ---

# ACL at 2-4 and an isolated false positive at 8
P_FALSE_POSITIVE = np.array([0.0, 0.1, 0.9, 0.9, 0.9, 0.1, 0.0, 0.0, 0.8, 0.0])
# Two strong blocks split by one weak slice; the right one is longer
P_GAP = np.array([0.0, 0.95, 0.4, 0.9, 0.9, 0.0])


def test_isolated_false_positive_stretches_only_threshold_span():
    assert threshold_span(P_FALSE_POSITIVE, tau=0.5) == (2, 9)
    assert longest_run(P_FALSE_POSITIVE, tau=0.5) == (2, 5)
    assert best_run(P_FALSE_POSITIVE, tau=0.5) == (2, 5)


def test_only_kadane_bridges_a_weak_slice():
    assert best_run(P_GAP, tau=0.5) == (1, 5)
    assert longest_run(P_GAP, tau=0.5) == (3, 5)


def test_longest_run_breaks_length_ties_by_probability():
    p = np.array([0.6, 0.6, 0.0, 0.9, 0.9])
    assert longest_run(p, tau=0.5) == (3, 5)


@pytest.mark.parametrize('z_method', sorted(Z_METHODS))
def test_every_z_method_falls_back_to_the_most_likely_slice(z_method):
    assert Z_METHODS[z_method](np.array([0.1, 0.3, 0.2]), 0.5) == (1, 2)


@pytest.mark.parametrize('z_method', sorted(Z_METHODS))
def test_every_z_method_rejects_empty_input(z_method):
    with pytest.raises(ValueError):
        Z_METHODS[z_method](np.array([]), 0.5)


def test_xy_methods_differ_on_an_outlier():
    p = np.array([0.9, 0.9, 0.6])
    boxes = np.array([[100, 100, 180, 180], [102, 102, 182, 182], [0, 0, 20, 20]], dtype=float)

    def xy(name):
        return aggregate_volume(p, boxes, tau=0.5, z_method='threshold', xy_method=name)[[0, 1, 3, 4]]

    np.testing.assert_array_equal(xy('wmedian'), [100, 100, 180, 180])
    np.testing.assert_array_equal(xy('median'), [100, 100, 180, 180])
    np.testing.assert_array_equal(xy('argmax'), [100, 100, 180, 180])
    # The mean is the only one the outlier drags
    assert xy('wmean')[0] < 90


def test_median_ignores_the_weights_and_wmedian_does_not():
    # Two weak slices on the left, two confident ones on the right
    p = np.array([0.55, 0.55, 0.99, 0.99])
    boxes = np.array([[10, 10, 20, 20], [11, 11, 21, 21], [50, 50, 60, 60], [52, 52, 62, 62]], dtype=float)

    median = aggregate_volume(p, boxes, 0.5, 'threshold', 'median')
    wmedian = aggregate_volume(p, boxes, 0.5, 'threshold', 'wmedian')

    assert median[0] == 30.5
    assert wmedian[0] == 50


def test_default_methods_are_kadane_and_wmedian():
    p = P_FALSE_POSITIVE
    boxes = np.tile([100.0, 110.0, 190.0, 200.0], (len(p), 1))
    np.testing.assert_array_equal(
        aggregate_volume(p, boxes, 0.5), aggregate_volume(p, boxes, 0.5, 'kadane', 'wmedian'),
    )
    assert set(Z_METHODS) == {'kadane', 'threshold', 'longest'}
    assert set(XY_METHODS) == {'wmedian', 'median', 'wmean', 'argmax'}


@pytest.mark.parametrize('kwargs', [{'z_method': 'kadan'}, {'xy_method': 'mean'}])
def test_unknown_method_names_the_valid_ones(kwargs):
    with pytest.raises(ValueError, match='Valid'):
        aggregate_volume(np.ones(3), np.zeros((3, 4)), 0.5, **kwargs)


def test_select_tau_picks_a_threshold_per_method():
    # An isolated slice at p=0.6: threshold_span must raise tau above it, Kadane need not
    p = np.array([0.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0])
    preds = pd.DataFrame({'examId': 1, 'seriesNo': 5, 'slice_idx': range(10), 'p': p,
                          'px0': 100, 'py0': 100, 'px1': 180, 'py1': 180})
    gt = pd.DataFrame([{'examId': 1, 'seriesNo': 5,
                        'x0': 100, 'y0': 100, 'z0': 1, 'x1': 180, 'y1': 180, 'z1': 4}])
    taus = np.array([0.5, 0.7])

    assert select_tau(preds, gt, taus, z_method='kadane')[0] == 0.5
    assert select_tau(preds, gt, taus, z_method='threshold')[0] == 0.7


def test_score_volumes_splits_iou_into_z_and_xy():
    gt = pd.DataFrame([
        {'examId': 1, 'seriesNo': 5, 'aclDiagnosis': 2,
         'x0': 0, 'y0': 0, 'z0': 0, 'x1': 10, 'y1': 10, 'z1': 4},
        {'examId': 2, 'seriesNo': 5, 'aclDiagnosis': 0,
         'x0': 0, 'y0': 0, 'z0': 0, 'x1': 10, 'y1': 10, 'z1': 4},
    ])
    # Volume 1: right plane, z half; volume 2: perfect. Given in the other order
    pred = pd.DataFrame([
        {'examId': 2, 'seriesNo': 5, 'x0': 0, 'y0': 0, 'z0': 0, 'x1': 10, 'y1': 10, 'z1': 4},
        {'examId': 1, 'seriesNo': 5, 'x0': 0, 'y0': 0, 'z0': 0, 'x1': 10, 'y1': 10, 'z1': 2},
    ])

    scored = score_volumes(pred, gt)

    assert scored['examId'].tolist() == [1, 2]
    assert scored['aclDiagnosis'].tolist() == [2, 0]
    assert scored['iou'].tolist() == pytest.approx([0.5, 1.0])
    assert scored['iou_z'].tolist() == pytest.approx([0.5, 1.0])
    assert scored['iou_xy'].tolist() == pytest.approx([1.0, 1.0])


def test_score_volumes_rejects_missing_predictions():
    _, gt = synthetic_preds()
    pred = gt.iloc[:1]
    with pytest.raises(ValueError, match='no predictions'):
        score_volumes(pred, gt)
