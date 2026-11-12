import numpy as np


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    n: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict[str, float]:
    """
    Confidence interval for mean(a - b) when a and b score the same volumes.

    Resamples volumes with replacement and applies the same indices to a and b,
    so the interval reflects only which volumes happened to be in the sample.
    It does not include training noise (seeds); compare against the seed spread
    for that.

    Args:
        a, b: Arrays of shape (n_volumes,), one score per volume, in the same order.
        n: Number of bootstrap resamples.
        seed: Seed of the resampling.
        ci: Coverage of the interval.

    Returns:
        {'delta': mean(a - b), 'lo': lower bound, 'hi': upper bound}. An interval
        that contains 0 means no evidence of a difference.

    Raises:
        ValueError: If a and b differ in shape or are empty.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError(f"a and b must be non-empty 1-D arrays of the same shape, got {a.shape} and {b.shape}")

    ci_ = bootstrap_mean(a - b, n=n, seed=seed, ci=ci)
    return {'delta': ci_['mean'], 'lo': ci_['lo'], 'hi': ci_['hi']}


def bootstrap_mean(
    x: np.ndarray,
    n: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict[str, float]:
    """
    Percentile bootstrap interval for the mean of x.

    Each element of x must be one independent unit (one volume, not one slice),
    otherwise the interval comes out too narrow.

    Args:
        x: Array of shape (n_units,).
        n: Number of bootstrap resamples.
        seed: Seed of the resampling.
        ci: Coverage of the interval.

    Returns:
        {'mean': mean(x), 'lo': lower bound, 'hi': upper bound}.

    Raises:
        ValueError: If x is empty or not 1-D.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"x must be a non-empty 1-D array, got shape {x.shape}")

    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    tail = (1 - ci) / 2 * 100
    lo, hi = np.percentile(means, [tail, 100 - tail])
    return {'mean': float(x.mean()), 'lo': float(lo), 'hi': float(hi)}
