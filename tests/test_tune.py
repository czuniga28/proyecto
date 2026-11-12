import sys

import optuna
import pandas as pd
import pytest
import yaml

from scripts import tune

PARAMS = {'lr': 3e-4, 'weight_decay': 1e-4, 'dropout': 0.1, 'box_weight': 2.0,
          'max_shift': 10, 'hard_frac': 0.6, 'hard_dist': 3}


def test_suggest_applies_params_without_touching_base(run_cfg):
    base = yaml.safe_load(yaml.safe_dump(run_cfg))
    cfg = tune.suggest(optuna.trial.FixedTrial(PARAMS), run_cfg)

    assert run_cfg == base
    assert cfg['lr'] == 3e-4 and cfg['weight_decay'] == 1e-4 and cfg['box_weight'] == 2.0
    assert cfg['model']['dropout'] == 0.1
    assert cfg['max_shift'] == 10
    assert cfg['sampler']['hard_frac'] == 0.6
    assert cfg['sampler']['hard_dist'] == 3
    assert (cfg['epochs'], cfg['patience']) == (15, 3)


def test_main_writes_reports(run_cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(tune, 'OPTUNA_DIR', tmp_path / 'optuna')
    monkeypatch.setattr(tune, 'TRIALS_CSV', tmp_path / 'reports' / 'optuna_trials.csv')
    monkeypatch.setattr(tune, 'FIGS_DIR', tmp_path / 'figs')
    monkeypatch.setattr(tune, 'BEST_CFG', tmp_path / 'm1_best.yaml')
    # 2 epochs per trial instead of 15 keeps the test fast
    real_suggest = tune.suggest
    monkeypatch.setattr(tune, 'suggest', lambda t, c: {**real_suggest(t, c), 'epochs': 2})

    config = tmp_path / 'cfg.yaml'
    config.write_text(yaml.safe_dump(run_cfg))
    monkeypatch.setattr(sys, 'argv', ['tune', '--config', str(config), '--n-trials', '2', '--device', 'cpu'])
    tune.main()

    trials = pd.read_csv(tmp_path / 'reports' / 'optuna_trials.csv')
    assert len(trials) == 2
    assert {'params_lr', 'params_max_shift', 'user_attrs_auprc'} <= set(trials.columns)
    assert (tmp_path / 'figs' / 'optuna_history.png').exists()
    assert (tmp_path / 'figs' / 'optuna_importance.png').exists()
    assert (tmp_path / 'optuna' / 'trial_0' / 'metrics.json').exists()
    assert (tmp_path / 'optuna' / 'trial_0' / 'env.txt').exists()

    best = yaml.safe_load((tmp_path / 'm1_best.yaml').read_text())
    study = optuna.load_study(study_name='m1', storage=f"sqlite:///{tmp_path / 'optuna' / 'm1.db'}")
    assert best['lr'] == pytest.approx(study.best_params['lr'])
