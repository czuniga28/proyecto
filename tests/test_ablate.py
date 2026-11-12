import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import ablate

ABLATIONS = {'base': {}, 'seed_1': {'seed': 1}, 'ctx_2d': {'context': 0}}


def test_apply_overrides_sets_nested_keys_without_touching_the_base():
    base = {'seed': 42, 'model': {'pretrained': True, 'dropout': 0.2}}

    cfg = ablate.apply_overrides(base, {'model.pretrained': False, 'seed': 0})

    assert cfg == {'seed': 0, 'model': {'pretrained': False, 'dropout': 0.2}}
    assert base['model']['pretrained'] is True


@pytest.mark.parametrize('key', ['model.pretrainde', 'modle.pretrained', 'seed.value'])
def test_apply_overrides_rejects_keys_missing_from_the_base(key):
    with pytest.raises(KeyError):
        ablate.apply_overrides({'seed': 42, 'model': {'pretrained': True}}, {key: False})


@pytest.fixture
def setup(run_cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(ablate, 'OUT_DIR', tmp_path / 'ablations')
    monkeypatch.setattr(ablate, 'SUMMARY_CSV', tmp_path / 'reports' / 'ablations.csv')
    run_cfg['context'] = 1
    config, ablations = tmp_path / 'cfg.yaml', tmp_path / 'ablations.yaml'
    config.write_text(yaml.safe_dump(run_cfg))
    ablations.write_text(yaml.safe_dump(ABLATIONS, sort_keys=False))
    argv = ['ablate', '--config', str(config), '--ablations', str(ablations), '--device', 'cpu']
    monkeypatch.setattr(sys, 'argv', argv)
    return tmp_path


def test_main_runs_every_variant_and_compares_with_base(setup):
    ablate.main()

    table = pd.read_csv(setup / 'reports' / 'ablations.csv')
    assert table['variant'].tolist() == list(ABLATIONS)
    base = table.set_index('variant')
    assert base.loc['base', 'delta'] == 0
    assert np.isfinite(table['seed_sigma']).all()
    # Seed rows measure the noise; only real variants get a verdict
    assert base[['beyond_2sigma']].loc[['base', 'seed_1']].isna().all().all()
    assert base.loc['ctx_2d', ['boot_lo', 'boot_hi']].notna().all()
    assert (setup / 'ablations' / 'ctx_2d' / 'env.txt').exists()
    assert yaml.safe_load((setup / 'ablations' / 'ctx_2d' / 'config.yaml').read_text())['context'] == 0


def test_main_skips_finished_variants(setup, monkeypatch):
    ablate.main()
    first = pd.read_csv(setup / 'reports' / 'ablations.csv')

    def fail(*args, **kwargs):
        raise AssertionError("run called for a finished variant")

    monkeypatch.setattr(ablate, 'run', fail)
    ablate.main()

    pd.testing.assert_frame_equal(first, pd.read_csv(setup / 'reports' / 'ablations.csv'))


def test_main_checks_every_override_before_training(setup, monkeypatch):
    bad = {**ABLATIONS, 'typo': {'model.pretrainde': False}}
    (setup / 'ablations.yaml').write_text(yaml.safe_dump(bad))
    monkeypatch.setattr(ablate, 'run', lambda *a, **k: pytest.fail("trained before validating"))

    with pytest.raises(KeyError, match='pretrainde'):
        ablate.main()


def test_only_rejects_unknown_variants(setup, monkeypatch):
    monkeypatch.setattr(sys, 'argv', sys.argv + ['--only', 'ctx_3d'])
    with pytest.raises(ValueError, match='ctx_3d'):
        ablate.main()
