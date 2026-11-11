import hashlib
import json
import math
import random

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from conftest import RUN_DEPTH, RUN_SIDE
from torch.utils.data import DataLoader

from src.data.slices import build_slice_index
from src.models.net import SliceNet
from src.train import loop
from src.train.dataset import SliceDataset
from src.train.loop import (
    evaluate,
    get_device,
    load_data,
    predict,
    run,
    set_seed,
    train_one_epoch,
    volume_boxes,
    write_provenance,
)

SIZE = 64
DEPTH = 6
TAUS = np.array([0.3, 0.5, 0.7])


def make_volumes() -> pd.DataFrame:
    return pd.DataFrame([
        {'examId': 1, 'seriesNo': 5, 'aclDiagnosis': 0, 'roiX': 10, 'roiY': 20, 'roiZ': 1,
         'roiWidth': 20, 'roiHeight': 30, 'roiDepth': 3},
        {'examId': 2, 'seriesNo': 7, 'aclDiagnosis': 2, 'roiX': 30, 'roiY': 5, 'roiZ': 2,
         'roiWidth': 25, 'roiHeight': 15, 'roiDepth': 2},
    ]).assign(depth=DEPTH, split='fold_0')


@pytest.fixture
def slices(tmp_path) -> pd.DataFrame:
    (tmp_path / 'volumes').mkdir()
    rng = np.random.default_rng(0)
    for exam, series in [(1, 5), (2, 7)]:
        vol = rng.standard_normal((DEPTH, SIZE, SIZE)).astype(np.float16)
        np.save(tmp_path / 'volumes' / f'{exam}-{series}.npy', vol)
    return build_slice_index(make_volumes())


@pytest.fixture
def loader(tmp_path, slices) -> DataLoader:
    return DataLoader(SliceDataset(slices, tmp_path, size=SIZE), batch_size=5)


def perfect_preds(volumes: pd.DataFrame) -> pd.DataFrame:
    """Per-slice predictions that reproduce the ROI exactly."""
    rows = []
    for v in volumes.itertuples():
        box = [v.roiX, v.roiY, v.roiX + v.roiWidth, v.roiY + v.roiHeight]
        for z in range(v.depth):
            inside = v.roiZ <= z < v.roiZ + v.roiDepth
            rows.append([v.examId, v.seriesNo, z, v.aclDiagnosis, inside, float(inside), *box])
    return pd.DataFrame(rows, columns=[
        'examId', 'seriesNo', 'slice_idx', 'aclDiagnosis', 'contains_acl', 'p',
        'px0', 'py0', 'px1', 'py1',
    ])


def test_set_seed_is_reproducible():
    set_seed(3)
    a = torch.rand(4), np.random.rand(), random.random()
    set_seed(3)
    b = torch.rand(4), np.random.rand(), random.random()
    assert torch.equal(a[0], b[0]) and a[1:] == b[1:]


def test_get_device():
    assert get_device('cpu') == torch.device('cpu')
    assert get_device(None).type in {'mps', 'cpu'}


def test_load_data_merges_splits_and_manifest(tmp_path):
    interim, processed = tmp_path / 'interim', tmp_path / 'processed'
    interim.mkdir()
    processed.mkdir()
    volumes = make_volumes()
    volumes.drop(columns='split').to_csv(interim / 'manifest.csv', index=False)
    volumes[['examId', 'seriesNo', 'aclDiagnosis', 'split']].to_csv(interim / 'splits.csv', index=False)
    build_slice_index(volumes).to_csv(processed / 'slices.csv', index=False)

    cfg = {'data': {'interim_dir': 'interim', 'processed_dir': 'processed'}}
    slices, vols = load_data(cfg, root=tmp_path)

    assert len(slices) == 2 * DEPTH
    assert {'split', 'roiX', 'roiDepth'} <= set(vols.columns)
    assert len(vols) == 2


def test_train_one_epoch_updates_model(loader):
    set_seed(0)
    model = SliceNet(pretrained=False)
    before = model.box_head[1].weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    losses = train_one_epoch(model, loader, optimizer, 2.0, 1.0, torch.device('cpu'))

    assert set(losses) == {'cls', 'box'}
    assert all(math.isfinite(v) for v in losses.values())
    assert not torch.equal(before, model.box_head[1].weight)


def test_predict_columns_order_and_range(loader, slices):
    model = SliceNet(pretrained=False)
    preds = predict(model, loader, slices, torch.device('cpu'), width=SIZE, height=SIZE)

    assert preds.columns.tolist() == [
        'examId', 'seriesNo', 'slice_idx', 'aclDiagnosis', 'contains_acl',
        'p', 'px0', 'py0', 'px1', 'py1',
    ]
    pd.testing.assert_frame_equal(
        preds[['examId', 'seriesNo', 'slice_idx']], slices[['examId', 'seriesNo', 'slice_idx']],
    )
    assert preds['p'].between(0, 1).all()
    boxes = preds[['px0', 'py0', 'px1', 'py1']].to_numpy()
    assert (boxes >= 0).all() and (boxes <= SIZE).all()


def test_evaluate_perfect_predictions():
    volumes = make_volumes()
    result = evaluate(perfect_preds(volumes), volume_boxes(volumes), TAUS)

    assert result['iou'] == pytest.approx(1.0)
    assert result['ious'] == pytest.approx([1.0, 1.0])
    assert all(v == pytest.approx(1.0) for k, v in result.items() if k.startswith('iou_class_'))
    assert result['auprc'] == pytest.approx(1.0)
    assert result['f1'] == pytest.approx(1.0)
    assert result['tau'] in TAUS


def test_evaluate_uses_given_tau():
    volumes = make_volumes()
    assert evaluate(perfect_preds(volumes), volume_boxes(volumes), TAUS, tau=0.42)['tau'] == 0.42


def test_evaluate_ious_follow_gt_order():
    volumes = make_volumes()
    preds = perfect_preds(volumes)
    # Ruin only exam 2's box, then pass gt in reverse order
    preds.loc[preds['examId'] == 2, ['px0', 'px1']] = [0, 1]
    result = evaluate(preds, volume_boxes(volumes.iloc[::-1]), TAUS, tau=0.5)
    assert result['ious'][0] < 0.1
    assert result['ious'][1] == pytest.approx(1.0)


def test_evaluate_rejects_gt_without_predictions():
    volumes = make_volumes()
    preds = perfect_preds(volumes)
    with pytest.raises(ValueError, match='no predictions'):
        evaluate(preds[preds['examId'] == 1], volume_boxes(volumes), TAUS, tau=0.5)


def test_run_smoke_writes_outputs(run_cfg, tmp_path):
    out = tmp_path / 'run'
    metrics = run(run_cfg, out, max_slices=16, device='cpu')

    base = {'best_epoch', 'tau', 'iou', 'auprc', 'f1', 'pos_weight', 'seconds_per_epoch'}
    assert base <= set(metrics)
    assert all(k.startswith('iou_class_') for k in set(metrics) - base)
    assert json.loads((out / 'metrics.json').read_text()) == pytest.approx(metrics)
    assert pd.read_csv(out / 'history.csv')['epoch'].tolist() == [0, 1]

    checkpoint = torch.load(out / 'best.pt', weights_only=False)
    assert checkpoint['epoch'] == metrics['best_epoch']
    assert checkpoint['tau'] == metrics['tau']

    volumes = pd.read_csv(out / 'val_volumes.csv')
    assert volumes['iou'].mean() == pytest.approx(metrics['iou'])

    # Smoke test validates on 2 whole volumes of val_fold only
    preds = pd.read_csv(out / 'preds_val.csv')
    assert preds.groupby(['examId', 'seriesNo']).size().tolist() == [RUN_DEPTH, RUN_DEPTH]
    assert set(preds['examId']) <= set(load_data(run_cfg)[1].query("split == 'fold_1'")['examId'])


def test_run_same_seed_same_history(run_cfg, tmp_path):
    histories = []
    for name in ['a', 'b']:
        run(run_cfg, tmp_path / name, max_slices=16, device='cpu')
        histories.append(pd.read_csv(tmp_path / name / 'history.csv').drop(columns='seconds'))
    pd.testing.assert_frame_equal(*histories)


def test_predict_returns_every_val_slice(run_cfg):
    slices, _ = load_data(run_cfg)
    val = slices[slices['split'] == 'fold_1'].reset_index(drop=True)
    ds = SliceDataset(val, run_cfg['data']['processed_dir'], size=32)
    model = SliceNet(pretrained=False)

    preds = predict(model, DataLoader(ds, batch_size=5), val, torch.device('cpu'),
                    width=RUN_SIDE, height=RUN_SIDE)

    assert len(preds) == len(val)
    pd.testing.assert_frame_equal(
        preds[['examId', 'seriesNo', 'slice_idx']], val[['examId', 'seriesNo', 'slice_idx']],
    )
    boxes = preds[['px0', 'py0', 'px1', 'py1']].to_numpy()
    assert (boxes >= 0).all() and (boxes <= RUN_SIDE).all()


def test_run_stops_early(run_cfg, tmp_path, monkeypatch):
    real_evaluate = loop.evaluate
    scores = iter([0.5, 0.4, 0.3, 0.2, 0.1])

    def worse_every_epoch(*args, **kwargs):
        return {**real_evaluate(*args, **kwargs), 'iou': next(scores)}

    monkeypatch.setattr(loop, 'evaluate', worse_every_epoch)
    run_cfg.update(epochs=5, patience=2)
    metrics = run(run_cfg, tmp_path / 'run', max_slices=16, device='cpu')

    assert metrics['best_epoch'] == 0
    assert len(pd.read_csv(tmp_path / 'run' / 'history.csv')) == 3


def test_run_prunes(run_cfg, tmp_path):
    optuna = pytest.importorskip('optuna')

    class AlwaysPrune:
        def __init__(self):
            self.reported = []

        def report(self, value, step):
            self.reported.append(step)

        def should_prune(self):
            return True

    trial = AlwaysPrune()
    with pytest.raises(optuna.TrialPruned):
        run(run_cfg, tmp_path / 'run', trial=trial, max_slices=16, device='cpu')
    assert trial.reported == [0]


def test_write_provenance_hashes_data(run_cfg, tmp_path):
    out = tmp_path / 'prov'
    write_provenance(out, run_cfg)

    hashes = json.loads((out / 'data_hashes.json').read_text())
    expected = hashlib.sha256((tmp_path / 'interim' / 'splits.csv').read_bytes()).hexdigest()
    assert hashes['splits.csv'] == expected
    assert set(hashes) == {'slices.csv', 'splits.csv', 'manifest.csv'}
    assert yaml.safe_load((out / 'config.yaml').read_text()) == run_cfg
    assert 'commit:' in (out / 'env.txt').read_text()
