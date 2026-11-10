import numpy as np
import pandas as pd
import pytest

from src.eval.metrics import (
    iou_2d, iou_3d, roi_to_box3d, slice_classification_metrics, summarize_iou,
)


def test_iou_2d_identical_disjoint_and_partial():
    a = np.array([0, 0, 10, 10])

    assert iou_2d(a, a) == 1.0
    assert iou_2d(a, [20, 20, 30, 30]) == 0.0
    # Overlap 5x10 = 50, union 100 + 100 - 50 = 150
    assert iou_2d(a, [5, 0, 15, 10]) == pytest.approx(50 / 150)


def test_iou_touching_edges_is_zero_because_end_exclusive():
    assert iou_2d([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0
    assert iou_3d([0, 0, 0, 10, 10, 3], [0, 0, 3, 10, 10, 6]) == 0.0


def test_iou_3d_partial_by_hand():
    a = np.array([0, 0, 0, 10, 10, 4])   # volume 400
    b = np.array([5, 0, 2, 15, 10, 6])   # volume 400, overlap 5*10*2 = 100
    assert iou_3d(a, b) == pytest.approx(100 / 700)


def test_iou_3d_box_inside_another():
    outer = np.array([0, 0, 0, 10, 10, 10])
    inner = np.array([2, 2, 2, 7, 7, 7])
    assert iou_3d(outer, inner) == pytest.approx(125 / 1000)


def test_iou_empty_boxes_is_zero_not_nan():
    empty = np.array([5, 5, 5, 5])
    assert iou_2d(empty, empty) == 0.0


def test_iou_is_vectorized_and_symmetric():
    a = np.array([[0, 0, 0, 10, 10, 4], [0, 0, 0, 2, 2, 2]])
    b = np.array([[5, 0, 2, 15, 10, 6], [0, 0, 0, 2, 2, 2]])

    out = iou_3d(a, b)
    assert out.shape == (2,)
    np.testing.assert_allclose(out, iou_3d(b, a))
    assert out[1] == 1.0


def test_roi_to_box3d_matches_slice_index_convention():
    df = pd.DataFrame([{'roiX': 10, 'roiY': 20, 'roiZ': 2,
                        'roiWidth': 30, 'roiHeight': 40, 'roiDepth': 2}])
    # Same numbers as test_slices: x1 = 40, y1 = 60, positives on slices 2-3
    np.testing.assert_array_equal(roi_to_box3d(df), [[10, 20, 2, 40, 60, 4]])


def test_summarize_iou_per_class():
    ious = np.array([0.6, 0.4, 0.2, 0.8])
    labels = np.array([0, 0, 1, 2])

    s = summarize_iou(ious, labels)

    assert list(s.index) == ['all', 0, 1, 2]
    assert s.loc['all', 'n'] == 4
    assert s.loc['all', 'mean'] == pytest.approx(0.5)
    assert s.loc[0, 'mean'] == pytest.approx(0.5)
    assert s.loc[0, 'frac_ge_05'] == pytest.approx(0.5)
    assert s.loc[1, 'frac_ge_03'] == 0.0
    assert np.isnan(s.loc[2, 'std'])


def test_slice_classification_metrics():
    y = np.array([0, 0, 1, 1])
    perfect = slice_classification_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert perfect == {'auprc': 1.0, 'f1': 1.0}

    # Predicting "no ACL" everywhere: F1 is 0, which accuracy (50 %) would hide
    none = slice_classification_metrics(y, np.zeros(4))
    assert none['f1'] == 0.0
