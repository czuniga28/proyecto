from collections.abc import Iterator

import numpy as np
import pandas as pd
from torch.utils.data import Sampler


def train_slices(slices: pd.DataFrame, val_fold: str) -> pd.DataFrame:
    """
    Return the slices used to train when val_fold is held out for validation.

    Args:
        slices: Slice index with at least the columns 'examId' and 'split'.
        val_fold: Name of the validation fold, e.g. 'fold_0'.

    Returns:
        The slices whose split is neither val_fold nor 'test', with a fresh
        RangeIndex so that positions and index labels match.

    Raises:
        ValueError: If val_fold is not a split in slices, or if an examId appears
            both in training and in the excluded slices.
    """

    if val_fold not in set(slices['split']):
        raise ValueError(f"Unknown val_fold {val_fold!r}. Available splits: {sorted(slices['split'].unique())}")

    excluded = slices['split'].isin([val_fold, 'test'])

    leaked = set(slices.loc[~excluded, 'examId']) & set(slices.loc[excluded, 'examId'])
    if leaked:
        raise ValueError(f"Leakage detected! These examIds appear in training and in excluded splits: {sorted(leaked)}")

    return slices[~excluded].reset_index(drop=True)


def diagnosis_weights(slices: pd.DataFrame) -> dict[int, float]:
    """
    Compute a weight per aclDiagnosis from its frequency in volumes, not slices.

    Args:
        slices: Slice index with at least the columns 'examId', 'seriesNo' and
            'aclDiagnosis'.

    Returns:
        A dict mapping each diagnosis to 1 / sqrt(freq), scaled so that the most
        frequent diagnosis has weight 1.
    """
    volumes = slices.drop_duplicates(['examId', 'seriesNo'])
    freq = volumes['aclDiagnosis'].value_counts(normalize=True)

    weights = 1 / np.sqrt(freq)
    weights = weights / weights.min()

    return {int(diagnosis): float(weight) for diagnosis, weight in weights.items()}


def _weighted_draw(
    pool: np.ndarray,
    slice_weights: np.ndarray,
    size: int,
    replace: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw size positions from pool with probability proportional to their slice weight."""
    # rng.choice rejects an empty p, even when size is 0
    if size == 0:
        return pool[:0]

    p = slice_weights[pool]
    return rng.choice(pool, size=size, replace=replace, p=p / p.sum())


def epoch_sample(
    slices: pd.DataFrame,
    rng: np.random.Generator,
    weights: dict[int, float],
    neg_per_pos: float = 1.0,
    hard_frac: float = 0.5,
    hard_dist: int = 2,
) -> np.ndarray:
    """
    Choose the slice positions to train on during one epoch.

    Positives are drawn with replacement, one draw per positive slice. Negatives
    are drawn without replacement: hard ones lie within hard_dist slices of the
    ROI, easy ones farther away. All draws are weighted by the slice's diagnosis.

    Args:
        slices: Slice index with a RangeIndex and the columns 'contains_acl',
            'dist_to_roi' and 'aclDiagnosis'.
        rng: Random generator, used for every draw and for the final shuffle.
        weights: Weight per diagnosis, as returned by diagnosis_weights.
        neg_per_pos: Number of negatives to draw per positive slice.
        hard_frac: Fraction of the negatives that should be hard.
        hard_dist: Largest dist_to_roi at which a negative counts as hard.

    Returns:
        Shuffled positions into slices. Positives may repeat. Missing hard
        negatives are replaced by easy ones, so fewer negatives than requested
        are returned only when both groups run out.
    """
    contains_acl = slices['contains_acl'].to_numpy()
    is_near = slices['dist_to_roi'].to_numpy() <= hard_dist
    slice_weights = slices['aclDiagnosis'].map(weights).to_numpy(dtype=float)

    positives = np.flatnonzero(contains_acl)
    hard_negatives = np.flatnonzero(~contains_acl & is_near)
    easy_negatives = np.flatnonzero(~contains_acl & ~is_near)

    n_pos = len(positives)
    n_neg = round(neg_per_pos * n_pos)
    n_hard = min(round(n_neg * hard_frac), len(hard_negatives))
    n_easy = min(n_neg - n_hard, len(easy_negatives))

    sample = np.concatenate([
        _weighted_draw(positives, slice_weights, n_pos, True, rng),
        _weighted_draw(hard_negatives, slice_weights, n_hard, False, rng),
        _weighted_draw(easy_negatives, slice_weights, n_easy, False, rng),
    ])
    rng.shuffle(sample)

    return sample


def pos_weight(slices: pd.DataFrame, idx: np.ndarray) -> float:
    """
    Return N_neg / N_pos of the sampled positions idx, not of all the slices.

    Args:
        slices: Slice index with a RangeIndex and the column 'contains_acl'.
        idx: Sampled positions into slices. Repeated positions count every time.

    Returns:
        The ratio of negatives to positives in the sample.
    """
    contains_acl = slices['contains_acl'].to_numpy()[idx]
    n_pos = int(contains_acl.sum())

    return (len(contains_acl) - n_pos) / n_pos


class EpochSampler(Sampler[int]):
    """
    Draw a new epoch_sample every epoch, reproducible from (seed, epoch) alone.

    Call set_epoch before iterating; otherwise every epoch repeats the same sample.

    Args:
        slices: Training slices, as returned by train_slices.
        seed: Base seed of the sampler.
        **kwargs: neg_per_pos, hard_frac and hard_dist, passed to epoch_sample.
    """

    def __init__(self, slices: pd.DataFrame, seed: int, **kwargs) -> None:
        super().__init__()
        self.slices = slices
        self.seed = seed
        self.kwargs = kwargs
        self.weights = diagnosis_weights(slices)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _sample(self) -> np.ndarray:
        rng = np.random.default_rng([self.seed, self.epoch])
        return epoch_sample(self.slices, rng, self.weights, **self.kwargs)

    def __iter__(self) -> Iterator[int]:
        return iter(self._sample().tolist())

    def __len__(self) -> int:
        return len(self._sample())
