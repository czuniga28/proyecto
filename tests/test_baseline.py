import numpy as np
import pandas as pd
import pytest

from src.eval.baseline import cv_fixed_box, mean_box


def volumes(n_per_fold: int = 4) -> pd.DataFrame:
    """Three CV folds of identical boxes plus one test volume with an absurd box."""
    rows = []
    for k in range(3):
        for i in range(n_per_fold):
            rows.append({'examId': 100 * k + i, 'seriesNo': 5, 'aclDiagnosis': i % 3,
                         'split': f'fold_{k}', 'roiX': 100, 'roiY': 100, 'roiZ': 10,
                         'roiWidth': 80, 'roiHeight': 80, 'roiDepth': 4})
    rows.append({'examId': 999, 'seriesNo': 5, 'aclDiagnosis': 0, 'split': 'test',
                 'roiX': 0, 'roiY': 0, 'roiZ': 0,
                 'roiWidth': 5, 'roiHeight': 5, 'roiDepth': 1})
    return pd.DataFrame(rows)


def test_mean_box_averages_coordinates():
    boxes = np.array([[0, 0, 0, 10, 10, 2], [10, 10, 2, 20, 20, 6]])
    np.testing.assert_array_equal(mean_box(boxes), [5, 5, 1, 15, 15, 4])


def test_mean_box_rejects_empty_or_wrong_shape():
    with pytest.raises(ValueError):
        mean_box(np.empty((0, 6)))
    with pytest.raises(ValueError):
        mean_box(np.zeros((3, 4)))


def test_test_split_never_used():
    out = cv_fixed_box(volumes(), n_cv_folds=3)

    assert 999 not in set(out['examId'])
    # If the test box leaked into the mean, identical dev boxes would not get IoU 1
    np.testing.assert_allclose(out['iou'], 1.0)


def test_each_dev_volume_validated_once_and_trained_on_the_rest():
    out = cv_fixed_box(volumes(), n_cv_folds=3)

    counts = out.groupby(['examId', 'part']).size().unstack()
    assert (counts['val'] == 1).all()
    assert (counts['train'] == 2).all()
    # The val fold is the volume's own split
    val = out[out.part == 'val']
    assert (val['fold'] == 'fold_' + (val['examId'] // 100).astype(str)).all()


def test_validation_fold_excluded_from_its_own_box():
    v = volumes()
    # Move fold_0 far away: its own box (mean of fold_1, fold_2) no longer covers it,
    # but the other folds still get a partial overlap because fold_0 is in their mean
    v.loc[v.split == 'fold_0', 'roiX'] = 220
    out = cv_fixed_box(v, n_cv_folds=3)

    val = out[out.part == 'val']
    assert (val.loc[val.fold == 'fold_0', 'iou'] == 0).all()
    assert (val.loc[val.fold != 'fold_0', 'iou'] > 0).all()
    assert (val.loc[val.fold != 'fold_0', 'iou'] < 1).all()


def test_unknown_split_label_raises():
    v = volumes()
    v.loc[0, 'split'] = 'fold_7'
    with pytest.raises(ValueError, match="unexpected split"):
        cv_fixed_box(v, n_cv_folds=3)
