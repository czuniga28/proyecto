import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import aggregation_ablation as agg
from src.eval.aggregate import XY_METHODS, Z_METHODS
from src.train.loop import load_data, volume_boxes


def perfect_preds(slices: pd.DataFrame) -> pd.DataFrame:
    """preds_val.csv-like table whose slices reproduce the ROI exactly."""
    preds = slices[['examId', 'seriesNo', 'slice_idx', 'aclDiagnosis', 'contains_acl']].copy()
    preds['p'] = slices['contains_acl'].astype(float)
    boxes = slices.groupby(['examId', 'seriesNo'])[['x0', 'y0', 'x1', 'y1']].transform('max')
    preds[['px0', 'py0', 'px1', 'py1']] = boxes.to_numpy()
    return preds


@pytest.fixture
def val(run_cfg):
    slices, volumes = load_data(run_cfg)
    val = slices[slices['split'] == run_cfg['val_fold']].reset_index(drop=True)
    gt = volume_boxes(volumes.merge(val[['examId', 'seriesNo']].drop_duplicates()))
    return perfect_preds(val), gt


def test_every_combination_is_scored_against_the_base(val, run_cfg):
    preds, gt = val

    table, curves = agg.ablate_aggregation(preds, gt, np.asarray(run_cfg['taus']), n_boot=200)

    assert len(table) == len(Z_METHODS) * len(XY_METHODS)
    base = table[(table['z_method'] == 'kadane') & (table['xy_method'] == 'wmedian')].iloc[0]
    assert base['iou'] == pytest.approx(1.0)
    assert base['delta'] == 0 and not base['ci_excludes_0']
    assert {'iou_z', 'iou_xy', 'frac_ge_05', 'boot_lo', 'boot_hi'} <= set(table.columns)
    assert len(curves) == len(table) * len(run_cfg['taus'])


def test_load_runs_rejects_the_same_volume_twice(tmp_path, val, run_cfg):
    preds, _ = val
    for name in ['a', 'b']:
        (tmp_path / name).mkdir()
        preds.to_csv(tmp_path / name / 'preds_val.csv', index=False)
        (tmp_path / name / 'config.yaml').write_text(yaml.safe_dump(run_cfg))

    with pytest.raises(ValueError, match='several runs'):
        agg.load_runs([tmp_path / 'a', tmp_path / 'b'])


def test_main_writes_table_curves_and_figure(tmp_path, val, run_cfg, monkeypatch):
    preds, _ = val
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    preds.to_csv(run_dir / 'preds_val.csv', index=False)
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(run_cfg))
    monkeypatch.setattr(agg, 'REPORTS_DIR', tmp_path / 'reports')
    monkeypatch.setattr(agg, 'FIGS_DIR', tmp_path / 'figs')
    monkeypatch.setattr(sys, 'argv', ['agg', '--run', str(run_dir), '--tag', 't', '--n-boot', '100'])

    agg.main()

    table = pd.read_csv(tmp_path / 'reports' / 'aggregation_ablation_t.csv')
    assert table['iou'].max() == pytest.approx(1.0)
    assert (tmp_path / 'reports' / 'tau_curves_t.csv').exists()
    assert (tmp_path / 'figs' / 'tau_curve_t.png').exists()
