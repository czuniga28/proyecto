import numpy as np
import pandas as pd
import pytest
from conftest import make_manifest

from src.data.slices import build_slice_index
from src.train.sampler import (
    EpochSampler,
    diagnosis_weights,
    epoch_sample,
    pos_weight,
    train_slices,
)

HARD_DIST = 2


def make_slices() -> pd.DataFrame:
    """Slices of a synthetic manifest split by exam into 'test' and 'fold_0'..'fold_4'."""
    manifest = make_manifest(n_exams=300)
    fold = manifest['examId'] % 6
    manifest['split'] = ('fold_' + (fold - 1).astype(str)).where(fold > 0, 'test')
    return build_slice_index(manifest)


@pytest.fixture
def slices() -> pd.DataFrame:
    return make_slices()


@pytest.fixture
def train(slices: pd.DataFrame) -> pd.DataFrame:
    return train_slices(slices, 'fold_0')


def test_train_slices_excludes_validation_and_test(slices, train):
    assert set(train['split']) == {'fold_1', 'fold_2', 'fold_3', 'fold_4'}
    assert train.index.tolist() == list(range(len(train)))
    assert len(train) == len(slices[~slices['split'].isin(['fold_0', 'test'])])


def test_train_slices_without_val_fold_excludes_only_test(slices):
    dev = train_slices(slices, None)
    assert set(dev['split']) == {f'fold_{k}' for k in range(5)}
    assert dev.index.tolist() == list(range(len(dev)))
    assert len(dev) == (slices['split'] != 'test').sum()


def test_train_slices_rejects_exam_on_both_sides(slices):
    shared_exam = slices.loc[slices['split'] == 'fold_0', 'examId'].iloc[0]
    slices.loc[slices['split'] == 'fold_1', 'examId'] = shared_exam

    with pytest.raises(ValueError, match='Leakage'):
        train_slices(slices, 'fold_0')


def test_train_slices_rejects_unknown_fold(slices):
    with pytest.raises(ValueError, match='Unknown val_fold'):
        train_slices(slices, 'fold0')


def test_diagnosis_weights_match_phase_2_figures():
    labels = [0] * 575 + [1] * 143 + [2] * 46
    volumes = pd.DataFrame({
        'examId': range(len(labels)), 'seriesNo': 5, 'aclDiagnosis': labels,
    })
    # Class 2 gets far more slices, which must not change a per-volume frequency
    n_slices = volumes['aclDiagnosis'].map({0: 2, 1: 3, 2: 50})
    slices = volumes.loc[volumes.index.repeat(n_slices)]

    weights = diagnosis_weights(slices)

    assert weights == pytest.approx({0: 1.0, 1: 2.0, 2: 3.5}, abs=0.04)


def test_sample_size_and_hard_negative_count(train):
    idx = epoch_sample(train, np.random.default_rng(0), diagnosis_weights(train), hard_dist=HARD_DIST)
    sampled = train.iloc[idx]
    negatives = sampled[~sampled['contains_acl']]
    n_pos = int(train['contains_acl'].sum())

    assert len(idx) == 2 * n_pos
    assert sampled['contains_acl'].sum() == n_pos
    assert (negatives['dist_to_roi'] <= HARD_DIST).sum() == round(n_pos * 0.5)
    assert not negatives.index.has_duplicates


def test_missing_hard_negatives_are_replaced_by_easy_ones(train):
    # Each volume has 3 positives and only 4 hard negatives, but 6 negatives are requested
    idx = epoch_sample(
        train, np.random.default_rng(0), diagnosis_weights(train),
        neg_per_pos=2.0, hard_frac=1.0, hard_dist=HARD_DIST,
    )
    sampled = train.iloc[idx]
    negatives = sampled[~sampled['contains_acl']]
    is_hard = (train['dist_to_roi'] <= HARD_DIST) & ~train['contains_acl']

    assert (negatives['dist_to_roi'] <= HARD_DIST).sum() == is_hard.sum()
    assert len(negatives) == 2 * train['contains_acl'].sum()


def test_class_2_appears_above_its_natural_frequency(train):
    idx = epoch_sample(train, np.random.default_rng(0), diagnosis_weights(train))

    natural = (train['aclDiagnosis'] == 2).mean()
    sampled = (train.iloc[idx]['aclDiagnosis'] == 2).mean()

    assert sampled > 1.5 * natural


def test_sample_is_shuffled(train):
    idx = epoch_sample(train, np.random.default_rng(0), diagnosis_weights(train))
    n_pos = int(train['contains_acl'].sum())

    assert not train['contains_acl'].to_numpy()[idx[:n_pos]].all()


def test_pos_weight_is_one_on_a_balanced_sample(train):
    idx = epoch_sample(train, np.random.default_rng(0), diagnosis_weights(train))

    assert pos_weight(train, idx) == 1.0


def test_pos_weight_uses_the_sample_not_all_the_slices(train):
    positives = np.flatnonzero(train['contains_acl'])[:2]
    negatives = np.flatnonzero(~train['contains_acl'])[:6]

    assert pos_weight(train, np.concatenate([positives, negatives])) == 3.0


def test_same_seed_and_epoch_give_the_same_sample(train):
    first, second = EpochSampler(train, seed=7), EpochSampler(train, seed=7)
    first.set_epoch(3)
    second.set_epoch(3)

    assert list(first) == list(second)


def test_different_epoch_or_seed_gives_another_sample(train):
    sampler = EpochSampler(train, seed=7)
    sampler.set_epoch(0)
    epoch_0 = list(sampler)
    sampler.set_epoch(1)

    assert list(sampler) != epoch_0
    assert list(EpochSampler(train, seed=8)) != epoch_0


def test_epoch_does_not_depend_on_previous_epochs(train):
    walked = EpochSampler(train, seed=7)
    for epoch in range(3):
        walked.set_epoch(epoch)
        list(walked)
    walked.set_epoch(5)

    direct = EpochSampler(train, seed=7)
    direct.set_epoch(5)

    assert list(walked) == list(direct)


def test_sampler_forwards_kwargs_and_reports_its_length(train):
    sampler = EpochSampler(train, seed=7, neg_per_pos=2.0)
    n_pos = int(train['contains_acl'].sum())

    assert len(sampler) == len(list(sampler)) == 3 * n_pos
    assert all(type(position) is int for position in sampler)
