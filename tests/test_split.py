import pytest

from src.data.split import assign_splits

# Synthetic data is small, so proportions are checked loosely here; the strict
# 0.02 tolerance is exercised on the real data by the pipeline itself.
LOOSE = 0.15


def test_no_exam_in_more_than_one_split(manifest):
    df = assign_splits(manifest, seed=42, max_prop_diff=LOOSE)

    assert (df.groupby('examId')['split'].nunique() == 1).all()


def test_every_volume_assigned_once_with_expected_names(manifest):
    df = assign_splits(manifest, seed=42, n_test_splits=6, n_cv_folds=5, max_prop_diff=LOOSE)

    assert len(df) == len(manifest)
    assert set(df['split']) == {'test', 'fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4'}


def test_same_seed_same_split(manifest):
    a = assign_splits(manifest, seed=42, max_prop_diff=LOOSE)
    b = assign_splits(manifest, seed=42, max_prop_diff=LOOSE)

    assert a['split'].equals(b['split'])


def test_input_is_not_modified(manifest):
    before = manifest.copy()

    assign_splits(manifest, seed=42, max_prop_diff=LOOSE)

    assert manifest.equals(before)


def test_missing_volume_stops_the_split(manifest):
    bad = manifest.copy()
    bad.loc[3, 'exists'] = False

    with pytest.raises(ValueError, match="missing"):
        assign_splits(bad, seed=42, max_prop_diff=LOOSE)


def test_proportion_check_can_fail(manifest):
    with pytest.raises(ValueError, match="Class proportions"):
        assign_splits(manifest, seed=42, max_prop_diff=0.0001)
