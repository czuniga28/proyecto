import pandas as pd

from src.data.slices import build_slice_index


def tiny_volumes() -> pd.DataFrame:
    return pd.DataFrame([
        # 6 slices, ROI on slices 2-3
        {'examId': 1, 'seriesNo': 5, 'depth': 6, 'roiX': 10, 'roiY': 20, 'roiZ': 2,
         'roiWidth': 30, 'roiHeight': 40, 'roiDepth': 2, 'aclDiagnosis': 0, 'split': 'test'},
        # 4 slices, ROI on slices 0-2
        {'examId': 2, 'seriesNo': 7, 'depth': 4, 'roiX': 1, 'roiY': 2, 'roiZ': 0,
         'roiWidth': 3, 'roiHeight': 4, 'roiDepth': 3, 'aclDiagnosis': 2, 'split': 'fold_0'},
    ])


def test_one_row_per_slice_and_right_positives():
    s = build_slice_index(tiny_volumes())

    assert len(s) == 10
    assert s[s.examId == 1]['contains_acl'].tolist() == [False, False, True, True, False, False]
    assert s[s.examId == 2]['contains_acl'].tolist() == [True, True, True, False]


def test_dist_to_roi():
    s = build_slice_index(tiny_volumes())

    assert s[s.examId == 1]['dist_to_roi'].tolist() == [2, 1, 0, 0, 1, 2]
    assert s[s.examId == 2]['dist_to_roi'].tolist() == [0, 0, 0, 1]


def test_box_is_end_exclusive_and_empty_on_negatives():
    s = build_slice_index(tiny_volumes())
    pos = s[(s.examId == 1) & s.contains_acl].iloc[0]
    neg = s[(s.examId == 1) & ~s.contains_acl].iloc[0]

    assert (pos.x0, pos.y0, pos.x1, pos.y1) == (10, 20, 40, 60)
    assert neg[['x0', 'y0', 'x1', 'y1']].isna().all()


def test_carries_split_label_and_volume_file():
    s = build_slice_index(tiny_volumes())

    assert set(s[s.examId == 2]['split']) == {'fold_0'}
    assert set(s[s.examId == 2]['aclDiagnosis']) == {2}
    assert set(s[s.examId == 1]['volume_file']) == {'volumes/1-5.npy'}
