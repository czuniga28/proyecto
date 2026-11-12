import hashlib
import json
import random
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.eval.aggregate import aggregate_predictions, select_tau
from src.eval.metrics import (
    BOX3D_COLUMNS,
    iou_3d,
    roi_to_box3d,
    slice_classification_metrics,
    summarize_iou,
)
from src.models.net import SliceNet
from src.train.dataset import SliceDataset, denormalize_boxes
from src.train.losses import detection_loss
from src.train.sampler import EpochSampler, pos_weight, train_slices

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KEY = ['examId', 'seriesNo']
SLICE_COLUMNS = ['examId', 'seriesNo', 'slice_idx', 'aclDiagnosis', 'contains_acl']


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # warn_only: some MPS/CPU ops have no deterministic version and would raise
    torch.use_deterministic_algorithms(True, warn_only=True)
    # MPS has no deterministic index_put (backward of out['box'][mask]); it warns on
    # every step. Runs are therefore not bit-identical on MPS: reproducibility is
    # checked as |delta IoU| < 0.01 instead
    warnings.filterwarnings('ignore', message='index_put_with_accumulate_mps')


def get_device(name: str | None = None) -> torch.device:
    """Return the named device, or MPS when available and CPU otherwise."""
    if name is not None:
        return torch.device(name)
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def load_data(cfg: dict, root: Path = PROJECT_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the slice index and the per-volume table.

    Args:
        cfg: Parsed configs/prep.yaml.
        root: Directory the paths in cfg are relative to.

    Returns:
        (slices, volumes). volumes is splits.csv merged with manifest.csv, so it
        has the split and the ROI of every volume.
    """
    interim_dir = root / cfg['data']['interim_dir']
    processed_dir = root / cfg['data']['processed_dir']

    slices = pd.read_csv(processed_dir / 'slices.csv')
    manifest = pd.read_csv(interim_dir / 'manifest.csv')
    splits = pd.read_csv(interim_dir / 'splits.csv')
    volumes = splits[KEY + ['split']].merge(manifest, on=KEY, validate='one_to_one')
    if not len(volumes) == len(splits) == len(manifest):
        raise ValueError("manifest.csv and splits.csv disagree")
    return slices, volumes


def volume_boxes(volumes: pd.DataFrame) -> pd.DataFrame:
    """Ground truth for evaluate: examId, seriesNo, aclDiagnosis and BOX3D_COLUMNS."""
    gt = volumes[KEY + ['aclDiagnosis']].reset_index(drop=True)
    gt[BOX3D_COLUMNS] = roi_to_box3d(volumes)
    return gt


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pos_weight: float,
    box_weight: float,
    device: torch.device,
    box_terms: tuple[str, ...] = ('l1', 'giou'),
) -> dict[str, float]:
    """
    Run one pass over loader, updating model. box_terms is passed to detection_loss.

    Returns:
        Mean 'cls' and 'box' loss over the epoch, each batch weighted by its size.
    """
    model.train()
    sums = {'cls': 0.0, 'box': 0.0}
    n = 0
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(batch['image'])
        loss, parts = detection_loss(out, batch, pos_weight, box_weight, box_terms)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        size = len(batch['label'])
        for k in sums:
            sums[k] += parts[k] * size
        n += size
    return {k: v / n for k, v in sums.items()}


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    slices: pd.DataFrame,
    device: torch.device,
    width: int = 320,
    height: int = 320,
) -> pd.DataFrame:
    """
    Per-slice predictions in the pixel coordinates aggregate.py expects.

    Args:
        model: SliceNet.
        loader: Loader over a SliceDataset built from slices.
        slices: The same table the dataset was built from; batch['index'] are
            positions in it.
        device: Device the model is on.
        width, height: Size of the original (padded) slice, to denormalize boxes.

    Returns:
        One row per slice: examId, seriesNo, slice_idx, aclDiagnosis,
        contains_acl, p, px0, py0, px1, py1.
    """
    model.eval()
    indices, probs, boxes = [], [], []
    for batch in loader:
        out = model(batch['image'].to(device))
        indices.append(batch['index'].cpu().numpy())
        probs.append(torch.sigmoid(out['logit']).cpu().numpy())
        # Clamped only here: in training it would zero the gradient at the border
        boxes.append(out['box'].clamp(0, 1).cpu().numpy())

    idx = np.concatenate(indices)
    preds = slices.iloc[idx][SLICE_COLUMNS].reset_index(drop=True)
    preds['p'] = np.concatenate(probs)
    preds[['px0', 'py0', 'px1', 'py1']] = denormalize_boxes(np.concatenate(boxes), width, height)
    return preds


def evaluate(
    preds: pd.DataFrame,
    gt: pd.DataFrame,
    taus: np.ndarray,
    tau: float | None = None,
    z_method: str = 'kadane',
    xy_method: str = 'wmedian',
) -> dict:
    """
    Volume-level IoU and slice-level classification metrics.

    Args:
        preds: Output of predict for one split.
        gt: Output of volume_boxes for the volumes of that split.
        taus: Candidate thresholds, used only when tau is None.
        tau: Fixed threshold, e.g. the one chosen on validation when scoring test.
        z_method, xy_method: Aggregation rules, see src/eval/aggregate.py.

    Returns:
        Dict with 'tau', mean 'iou', 'auprc', 'f1', 'ious' (an array with one
        IoU per row of gt, in gt's order) and 'iou_class_<c>', the mean IoU of
        each aclDiagnosis present in gt.
    """
    if tau is None:
        tau, _ = select_tau(preds, gt, taus, z_method, xy_method)

    pred_boxes = aggregate_predictions(preds, tau, z_method, xy_method)
    m = gt[KEY + BOX3D_COLUMNS].merge(
        pred_boxes, on=KEY, how='left', suffixes=('_gt', ''), validate='one_to_one',
    )
    if m[BOX3D_COLUMNS].isna().any().any():
        raise ValueError("some ground-truth volumes have no predictions")
    ious = iou_3d(m[BOX3D_COLUMNS].to_numpy(), m[[f'{c}_gt' for c in BOX3D_COLUMNS]].to_numpy())

    # Per class: the global mean can hide a failure on the ~6 % of ruptures
    summary = summarize_iou(ious, gt['aclDiagnosis'].to_numpy())
    by_class = {f'iou_class_{c}': float(summary.loc[c, 'mean']) for c in summary.index if c != 'all'}

    cls = slice_classification_metrics(preds['contains_acl'], preds['p'])
    return {'tau': float(tau), 'iou': float(ious.mean()), **by_class, **cls, 'ious': ious}


def _subsample(
    train: pd.DataFrame, val: pd.DataFrame, max_slices: int, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Smoke-test subsets: max_slices training slices, half positive, and 2 whole val volumes."""
    rng = np.random.default_rng(seed)
    contains_acl = train['contains_acl'].to_numpy(dtype=bool)
    pos, neg = np.flatnonzero(contains_acl), np.flatnonzero(~contains_acl)
    n_pos = min(max_slices // 2, len(pos))
    n_neg = min(max_slices - n_pos, len(neg))
    keep = np.sort(np.concatenate([
        rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False),
    ]))

    # Whole volumes only: aggregate_predictions rejects a volume with missing slices
    first_two = val[KEY].drop_duplicates().head(2)
    val = val.merge(first_two, on=KEY)
    return train.iloc[keep].reset_index(drop=True), val.reset_index(drop=True)


def run(
    cfg: dict,
    out_dir: Path,
    trial=None,
    max_slices: int | None = None,
    device: str | None = None,
) -> dict:
    """
    Train on every development fold but cfg['val_fold'] and validate on it.

    Writes to out_dir: history.csv (one row per epoch); best.pt, preds_val.csv
    and val_volumes.csv (one IoU per volume), all from the epoch with the best
    val IoU; and metrics.json.
    Provenance is left to the caller (write_provenance).

    Args:
        cfg: Parsed configs/train/m1.yaml (or another config with the same keys).
        out_dir: Directory for the outputs; created if missing.
        trial: Optuna trial to report the val IoU to after every epoch, or None.
        max_slices: If given, a smoke test on that many training slices (half
            positive) and 2 whole validation volumes.
        device: Device name, or None to pick MPS when available.

    Returns:
        Dict with best_epoch, tau, iou, iou_class_<c> for each class in
        validation, auprc, f1 (all from the best epoch), pos_weight and
        seconds_per_epoch.

    Raises:
        optuna.TrialPruned: If trial says the run should be pruned.
    """
    if cfg['val_fold'] is None:
        raise ValueError("run needs a val_fold to select tau and the best epoch")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg['seed'])
    dev = get_device(device)
    slices, volumes = load_data(cfg)

    train = train_slices(slices, cfg['val_fold'])
    val = slices[slices['split'] == cfg['val_fold']].reset_index(drop=True)
    if max_slices is not None:
        train, val = _subsample(train, val, max_slices, cfg['seed'])
    gt = volume_boxes(volumes.merge(val[KEY].drop_duplicates(), on=KEY))

    processed_dir = PROJECT_ROOT / cfg['data']['processed_dir']
    size = cfg['image_size']
    # .get: keys added for the ablations; their defaults reproduce the earlier runs
    context = cfg.get('context', 1)
    box_terms = tuple(cfg.get('box_terms', ('l1', 'giou')))
    aggregation = cfg.get('aggregation', {})
    train_ds = SliceDataset(
        train, processed_dir, size=size, augment=True, max_shift=cfg['max_shift'], context=context,
    )
    val_ds = SliceDataset(val, processed_dir, size=size, context=context)
    sampler = EpochSampler(train, cfg['seed'], **cfg['sampler'])
    loader_kwargs = {'batch_size': cfg['batch_size'], 'num_workers': cfg['num_workers']}
    train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    pw = pos_weight(train, sampler.sample())

    model = SliceNet(**cfg['model']).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    history, best, bad_epochs = [], None, 0
    for epoch in range(cfg['epochs']):
        start = time.perf_counter()
        sampler.set_epoch(epoch)
        lr = optimizer.param_groups[0]['lr']
        losses = train_one_epoch(model, train_loader, optimizer, pw, cfg['box_weight'], dev, box_terms)
        scheduler.step()

        preds = predict(model, val_loader, val, dev, cfg['data']['width'], cfg['data']['height'])
        result = evaluate(preds, gt, np.asarray(cfg['taus']), **aggregation)
        seconds = time.perf_counter() - start

        scores = {k: v for k, v in result.items() if k != 'ious'}
        history.append({
            'epoch': epoch, 'lr': lr,
            'train_cls': losses['cls'], 'train_box': losses['box'],
            **{f'val_{k}': v for k, v in scores.items()},
            'seconds': seconds,
        })
        pd.DataFrame(history).to_csv(out_dir / 'history.csv', index=False)

        if best is None or result['iou'] > best['iou']:
            best = {'best_epoch': epoch, **scores}
            torch.save(
                {'model': model.state_dict(), 'epoch': epoch, 'tau': result['tau'], 'cfg': cfg},
                out_dir / 'best.pt',
            )
            preds.to_csv(out_dir / 'preds_val.csv', index=False)
            gt[KEY + ['aclDiagnosis']].assign(iou=result['ious']).to_csv(
                out_dir / 'val_volumes.csv', index=False,
            )
            bad_epochs = 0
        else:
            bad_epochs += 1

        if trial is not None:
            trial.report(result['iou'], epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

        if bad_epochs >= cfg['patience']:
            break

    metrics = {
        **best,
        'pos_weight': pw,
        'seconds_per_epoch': float(np.mean([h['seconds'] for h in history])),
    }
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    return metrics


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _command(args: list[str]) -> str:
    try:
        out = subprocess.run(args, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        return f"unavailable: {e}"


def write_provenance(out_dir: Path, cfg: dict, root: Path = PROJECT_ROOT) -> None:
    """
    Record what produced a run: config.yaml, env.txt (commit, uncommitted
    changes, packages) and data_hashes.json (sha256 of the data tables).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))

    env = [
        f"commit: {_command(['git', 'rev-parse', 'HEAD'])}",
        "git status --porcelain:",
        _command(['git', 'status', '--porcelain']) or "(clean)",
        f"python: {sys.version}",
        "pip freeze:",
        _command([sys.executable, '-m', 'pip', 'freeze']),
    ]
    (out_dir / 'env.txt').write_text('\n'.join(env) + '\n')

    interim_dir = root / cfg['data']['interim_dir']
    processed_dir = root / cfg['data']['processed_dir']
    files = {
        'slices.csv': processed_dir / 'slices.csv',
        'splits.csv': interim_dir / 'splits.csv',
        'manifest.csv': interim_dir / 'manifest.csv',
    }
    hashes = {name: _sha256(path) for name, path in files.items()}
    (out_dir / 'data_hashes.json').write_text(json.dumps(hashes, indent=2))
