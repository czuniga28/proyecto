import numpy as np
import pytest

from src.eval.bootstrap import bootstrap_mean, paired_bootstrap


def test_identical_scores_give_a_zero_interval():
    a = np.linspace(0, 1, 50)
    assert paired_bootstrap(a, a) == {'delta': 0.0, 'lo': 0.0, 'hi': 0.0}


def test_a_constant_shift_is_detected():
    rng = np.random.default_rng(0)
    b = rng.uniform(0.3, 0.8, size=150)
    a = b + 0.1 + rng.normal(0, 0.02, size=150)

    ci = paired_bootstrap(a, b)

    assert ci['delta'] == pytest.approx(0.1, abs=0.01)
    assert ci['lo'] > 0


def test_pure_noise_interval_contains_zero():
    rng = np.random.default_rng(1)
    b = rng.uniform(0.3, 0.8, size=150)
    a = b + rng.normal(0, 0.1, size=150)

    ci = paired_bootstrap(a, b)

    assert ci['lo'] < 0 < ci['hi']


def test_same_seed_same_interval():
    rng = np.random.default_rng(2)
    a, b = rng.uniform(size=40), rng.uniform(size=40)
    assert paired_bootstrap(a, b, seed=7) == paired_bootstrap(a, b, seed=7)


@pytest.mark.parametrize('a, b', [([1.0, 2.0], [1.0]), ([], [])])
def test_rejects_mismatched_or_empty_inputs(a, b):
    with pytest.raises(ValueError):
        paired_bootstrap(np.array(a), np.array(b))


def test_bootstrap_mean_constant_gives_a_zero_width_interval():
    assert bootstrap_mean(np.full(30, 0.4)) == pytest.approx({'mean': 0.4, 'lo': 0.4, 'hi': 0.4})


def test_bootstrap_mean_interval_covers_the_mean_and_shrinks_with_n():
    rng = np.random.default_rng(3)
    small, large = rng.uniform(size=40), rng.uniform(size=4000)

    ci_small, ci_large = bootstrap_mean(small), bootstrap_mean(large)

    assert ci_small['lo'] < small.mean() < ci_small['hi']
    assert ci_large['hi'] - ci_large['lo'] < ci_small['hi'] - ci_small['lo']


def test_paired_bootstrap_is_bootstrap_mean_of_the_difference():
    rng = np.random.default_rng(4)
    a, b = rng.uniform(size=60), rng.uniform(size=60)
    paired, single = paired_bootstrap(a, b, seed=5), bootstrap_mean(a - b, seed=5)
    assert (paired['delta'], paired['lo'], paired['hi']) == (single['mean'], single['lo'], single['hi'])


@pytest.mark.parametrize('x', [[], [[1.0, 2.0]]])
def test_bootstrap_mean_rejects_empty_or_2d(x):
    with pytest.raises(ValueError):
        bootstrap_mean(np.array(x))
