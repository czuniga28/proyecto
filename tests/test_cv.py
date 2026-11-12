import json
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import ablate, cv
from src.train.loop import load_data

KEY = ['examId', 'seriesNo']


def test_cv_configs_one_copy_per_fold_and_seed():
    cfg = {'seed': 42, 'val_fold': 'fold_0', 'model': {'dropout': 0.1}}

    configs = cv.cv_configs(cfg, ['fold_0', 'fold_3'], [42, 0])

    assert list(configs) == ['fold_0_s42', 'fold_0_s0', 'fold_3_s42', 'fold_3_s0']
    assert configs['fold_3_s0']['val_fold'] == 'fold_3'
    assert configs['fold_3_s0']['seed'] == 0
    configs['fold_3_s0']['model']['dropout'] = 0.9
    assert cfg == {'seed': 42, 'val_fold': 'fold_0', 'model': {'dropout': 0.1}}


def test_cv_configs_rejects_test():
    with pytest.raises(ValueError, match='test'):
        cv.cv_configs({'seed': 42, 'val_fold': 'fold_0'}, ['fold_0', 'test'], [42])


def write_run(runs_dir, name, metrics, volumes):
    d = runs_dir / name
    d.mkdir(parents=True)
    (d / 'metrics.json').write_text(json.dumps(metrics))
    volumes.to_csv(d / 'val_volumes.csv', index=False)


def volumes(exams, labels, ious):
    return pd.DataFrame({'examId': exams, 'seriesNo': 5, 'aclDiagnosis': labels, 'iou': ious})


def test_collect_reads_metrics_and_volumes(tmp_path):
    write_run(tmp_path, 'fold_0_s42', {'iou': 0.5, 'best_epoch': 3, 'tau': 0.4},
              volumes([1, 2], [0, 2], [0.4, 0.6]))
    write_run(tmp_path, 'fold_1_s0', {'iou': 0.7, 'best_epoch': 5, 'tau': 0.5},
              volumes([3], [1], [0.7]))

    folds_df, volumes_df = cv.collect(['fold_0_s42', 'fold_1_s0'], tmp_path)

    assert folds_df[['run', 'fold', 'seed', 'iou']].values.tolist() == [
        ['fold_0_s42', 'fold_0', 42, 0.5], ['fold_1_s0', 'fold_1', 0, 0.7],
    ]
    assert volumes_df.columns.tolist() == KEY + ['aclDiagnosis', 'fold', 'seed', 'iou']
    assert volumes_df[['examId', 'fold', 'seed']].values.tolist() == [
        [1, 'fold_0', 42], [2, 'fold_0', 42], [3, 'fold_1', 0],
    ]


def test_collect_names_the_missing_run(tmp_path):
    write_run(tmp_path, 'fold_0_s42', {'iou': 0.5}, volumes([1], [0], [0.5]))
    with pytest.raises(FileNotFoundError, match='fold_0_s0'):
        cv.collect(['fold_0_s42', 'fold_0_s0'], tmp_path)


def make_cv_tables():
    """
    2 folds x 2 seeds over 4 volumes. Seed-averaged IoU per volume:
    exam 1 (fold_0, cls 0) 0.5, exam 2 (fold_0, cls 2) 0.7,
    exam 3 (fold_1, cls 0) 0.3, exam 4 (fold_1, cls 1) 0.9.
    """
    rows = []
    for exam, fold, label, by_seed in [
        (1, 'fold_0', 0, (0.4, 0.6)), (2, 'fold_0', 2, (0.6, 0.8)),
        (3, 'fold_1', 0, (0.2, 0.4)), (4, 'fold_1', 1, (0.8, 1.0)),
    ]:
        for seed, iou in zip([42, 0], by_seed):
            rows.append([exam, 5, label, fold, seed, iou])
    volumes_df = pd.DataFrame(rows, columns=KEY + ['aclDiagnosis', 'fold', 'seed', 'iou'])
    folds_df = pd.DataFrame({'fold': ['fold_0', 'fold_0', 'fold_1', 'fold_1'], 'seed': [42, 0, 42, 0]})
    baseline = pd.DataFrame({
        'examId': [1, 2, 3, 4, 1], 'seriesNo': 5, 'aclDiagnosis': [0, 2, 0, 1, 0],
        'fold': 'fold_0', 'part': ['val'] * 4 + ['train'], 'iou': [0.4, 0.4, 0.2, 0.5, 0.99],
    })
    return folds_df, volumes_df, baseline


def test_summarize_cv_averages_seeds_before_estimating():
    folds_df, volumes_df, baseline = make_cv_tables()

    summary = cv.summarize_cv(folds_df, volumes_df, baseline, n_boot=500).set_index('class')

    assert summary.index.tolist() == ['all', '0', '1', '2']
    row = summary.loc['all']
    assert row['n'] == 4  # Volumes, not volume x seed
    assert row['iou'] == pytest.approx(0.6)
    assert row['ci_lo'] <= 0.6 <= row['ci_hi']
    assert row['frac_ge_05'] == pytest.approx(0.75)
    # Per-fold means 0.6 and 0.6; per-seed OOF means 0.5 and 0.7
    assert row['fold_std'] == pytest.approx(0.0)
    assert row['seed_std'] == pytest.approx(np.std([0.5, 0.7], ddof=1))
    # Paired against the val rows of M0 only: 0.6 - mean(0.4, 0.4, 0.2, 0.5)
    assert row['delta_m0'] == pytest.approx(0.6 - 0.375)
    assert summary.loc['0', 'iou'] == pytest.approx(0.4)
    assert summary.loc['2', 'n'] == 1


def test_summarize_cv_rejects_volume_without_m0():
    folds_df, volumes_df, baseline = make_cv_tables()
    with pytest.raises(ValueError, match='M0'):
        cv.summarize_cv(folds_df, volumes_df, baseline[baseline['examId'] != 4])


def test_refit_params_uses_medians():
    folds_df = pd.DataFrame({
        'run': list('abcd'), 'fold': 'fold_0', 'seed': [1, 2, 3, 4],
        'best_epoch': [3, 4, 6, 9], 'tau': [0.3, 0.4, 0.5, 0.5],
    })
    params = cv.refit_params(folds_df)

    # best_epoch is 0-based: epochs trained are 4, 5, 7, 10 -> median 6
    assert params['epochs'] == 6
    assert params['tau'] == pytest.approx(0.45)
    assert [r['best_epoch'] for r in params['runs']] == [3, 4, 6, 9]


def test_refit_params_rounds_up():
    folds_df = pd.DataFrame({'run': list('ab'), 'fold': 'f', 'seed': [1, 2],
                             'best_epoch': [3, 4], 'tau': [0.5, 0.5]})
    assert cv.refit_params(folds_df)['epochs'] == 5  # median(4, 5) = 4.5


@pytest.fixture
def cv_setup(run_cfg, tmp_path, monkeypatch):
    reports = tmp_path / 'reports'
    reports.mkdir()
    monkeypatch.setattr(cv, 'RUNS_DIR', tmp_path / 'cv')
    monkeypatch.setattr(cv, 'REPORTS_DIR', reports)
    monkeypatch.setattr(cv, 'REFIT_CFG', tmp_path / 'm1_refit.yaml')

    _, vols = load_data(run_cfg)
    vols[KEY + ['aclDiagnosis']].assign(fold=vols['split'], part='val', iou=0.1).to_csv(
        reports / 'baseline_cv_volumes.csv', index=False,
    )

    config = tmp_path / 'm1_best.yaml'
    config.write_text(yaml.safe_dump({**run_cfg, 'epochs': 1}))
    argv = ['cv', '--config', str(config), '--folds', 'fold_0', 'fold_1', '--seeds', '0', '1',
            '--device', 'cpu']
    monkeypatch.setattr(sys, 'argv', argv)
    return tmp_path


def test_main_writes_reports_and_refit_config(cv_setup, capsys):
    cv.main()

    reports = cv_setup / 'reports'
    assert len(pd.read_csv(reports / 'cv_folds.csv')) == 4
    cv_volumes = pd.read_csv(reports / 'cv_volumes.csv')
    baseline = pd.read_csv(reports / 'baseline_cv_volumes.csv')
    assert set(baseline.columns) <= set(cv_volumes.columns)
    assert (cv_volumes['part'] == 'val').all()
    summary = pd.read_csv(reports / 'cv_summary.csv')
    assert summary.loc[summary['class'] == 'all', 'n'].item() == len(baseline)

    text = (cv_setup / 'm1_refit.yaml').read_text()
    refit = yaml.safe_load(text)
    assert refit['epochs'] == 1
    assert refit['val_fold'] is None
    assert 0.3 <= refit['tau'] <= 0.7  # Median of taus drawn from run_cfg['taus']
    assert '# median of the tau' in text
    assert '=== fold_1_s1' in capsys.readouterr().out


def test_main_resumes_without_retraining(cv_setup, monkeypatch):
    cv.main()

    def fail(*args, **kwargs):
        raise AssertionError("finished runs must not be retrained")

    monkeypatch.setattr(ablate, 'run', fail)
    cv.main()


def test_main_without_config_points_to_tune(cv_setup, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['cv', '--config', str(cv_setup / 'missing.yaml')])
    with pytest.raises(FileNotFoundError, match='scripts.tune'):
        cv.main()
