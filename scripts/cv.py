import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.ablate import run_variant
from src.eval.aggregate import select_tau
from src.eval.bootstrap import bootstrap_mean, paired_bootstrap
from src.train.loop import KEY, PROJECT_ROOT, load_data, volume_boxes

RUNS_DIR = PROJECT_ROOT / 'runs' / 'cv'
REPORTS_DIR = PROJECT_ROOT / 'reports' / 'fase3'
REFIT_CFG = PROJECT_ROOT / 'configs' / 'train' / 'm1_refit.yaml'
DEV_FOLDS = [f'fold_{k}' for k in range(5)]
DEFAULT_SEEDS = [42, 0]
CLASSES = ['all', 0, 1, 2]


def run_name(fold: str, seed: int) -> str:
    return f'{fold}_s{seed}'


def cv_configs(cfg: dict, folds: list[str], seeds: list[int]) -> dict[str, dict]:
    """
    One config per (fold, seed), in fold -> seed order.

    Raises:
        ValueError: If folds contains 'test': the test split is never used to
            validate, or its score would no longer be an unbiased estimate.
    """
    if 'test' in folds:
        raise ValueError("cv must not validate on 'test'; pass development folds only")
    configs = {}
    for fold in folds:
        for seed in seeds:
            c = copy.deepcopy(cfg)
            c['val_fold'], c['seed'] = fold, seed
            configs[run_name(fold, seed)] = c
    return configs


def collect(names: list[str], runs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read the finished runs.

    Returns:
        (folds_df, volumes_df). folds_df has one row per run: run, fold, seed and
        every key of metrics.json. volumes_df stacks the val_volumes.csv files
        with KEY, aclDiagnosis, fold, seed and iou.

    Raises:
        FileNotFoundError: If a run has no metrics.json yet.
    """
    rows, volumes = [], []
    for name in names:
        run_dir = Path(runs_dir) / name
        if not (run_dir / 'metrics.json').exists():
            raise FileNotFoundError(f"run {name} has no metrics.json in {run_dir}")
        fold, seed = name.rsplit('_s', 1)
        metrics = json.loads((run_dir / 'metrics.json').read_text())
        rows.append({'run': name, 'fold': fold, 'seed': int(seed), **metrics})
        v = pd.read_csv(run_dir / 'val_volumes.csv').assign(fold=fold, seed=int(seed))
        volumes.append(v[KEY + ['aclDiagnosis', 'fold', 'seed', 'iou']])
    return pd.DataFrame(rows), pd.concat(volumes, ignore_index=True)


def _std(x: pd.Series) -> float:
    return float(x.std(ddof=1)) if len(x) >= 2 else np.nan


def summarize_cv(
    folds_df: pd.DataFrame,
    volumes_df: pd.DataFrame,
    baseline_volumes: pd.DataFrame,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """
    Out-of-fold IoU of M1 per class, with its uncertainty and the paired gain over M0.

    Each volume's IoU is first averaged over seeds, so it is one observation in
    the bootstrap and the same patient is never counted twice.

    Args:
        folds_df, volumes_df: Output of collect.
        baseline_volumes: baseline_cv_volumes.csv (M0); only part == 'val' is used.
        n_boot: Bootstrap resamples.

    Returns:
        One row per class ('all', 0, 1, 2 when present) with n, iou, ci_lo,
        ci_hi, frac_ge_05, fold_std (between the per-fold means), seed_std
        (between the per-seed OOF means) and delta_m0, m0_lo, m0_hi.

    Raises:
        ValueError: If volumes_df misses a run of folds_df, or if an M1 volume
            has no M0 counterpart.
    """
    runs = set(zip(folds_df['fold'], folds_df['seed']))
    if set(zip(volumes_df['fold'], volumes_df['seed'])) != runs:
        raise ValueError("volumes_df and folds_df cover different runs")

    per_volume = volumes_df.groupby(KEY + ['aclDiagnosis', 'fold'], as_index=False)['iou'].mean()

    m0 = baseline_volumes.loc[baseline_volumes['part'] == 'val', KEY + ['iou']]
    paired = per_volume.merge(m0, on=KEY, how='left', suffixes=('', '_m0'), validate='one_to_one')
    missing = paired['iou_m0'].isna()
    if missing.any():
        raise ValueError(f"{missing.sum()} M1 volumes have no M0 counterpart, e.g. "
                         f"{paired.loc[missing, KEY].iloc[0].tolist()}")

    rows = []
    for c in CLASSES:
        in_class = np.ones(len(paired), dtype=bool) if c == 'all' else (paired['aclDiagnosis'] == c).to_numpy()
        if not in_class.any():
            continue
        g = paired[in_class]
        raw = volumes_df if c == 'all' else volumes_df[volumes_df['aclDiagnosis'] == c]

        ci = bootstrap_mean(g['iou'].to_numpy(), n=n_boot)
        gain = paired_bootstrap(g['iou'].to_numpy(), g['iou_m0'].to_numpy(), n=n_boot)
        rows.append({
            'class': str(c),
            'n': len(g),
            'iou': ci['mean'], 'ci_lo': ci['lo'], 'ci_hi': ci['hi'],
            'frac_ge_05': float((g['iou'] >= 0.5).mean()),
            'fold_std': _std(g.groupby('fold')['iou'].mean()),
            'seed_std': _std(raw.groupby('seed')['iou'].mean()),
            'delta_m0': gain['delta'], 'm0_lo': gain['lo'], 'm0_hi': gain['hi'],
        })
    return pd.DataFrame(rows)


def pooled_taus(
    folds_df: pd.DataFrame, runs_dir: Path, gt: pd.DataFrame, cfg: dict,
) -> tuple[dict[int, float], pd.DataFrame]:
    """
    One tau per seed, chosen by select_tau on the OOF predictions of every fold.

    Each run's own tau is picked on ~150 volumes of a flat curve, so it is
    noisy; pooling the 5 folds of a seed gives every development volume once.

    Args:
        folds_df: Output of collect.
        runs_dir: Directory with one subdirectory per run, holding preds_val.csv.
        gt: volume_boxes of the development volumes.
        cfg: Config with 'taus' and, optionally, 'aggregation'.

    Returns:
        ({seed: tau}, curves) where curves has seed, tau and mean_iou.
    """
    taus, curves = {}, []
    for seed, runs in folds_df.groupby('seed', sort=False):
        pooled = pd.concat(
            [pd.read_csv(Path(runs_dir) / name / 'preds_val.csv') for name in runs['run']],
            ignore_index=True,
        )
        tau, curve = select_tau(pooled, gt, np.asarray(cfg['taus']), **cfg.get('aggregation', {}))
        taus[int(seed)] = tau
        curves.append(curve.reset_index().assign(seed=int(seed)))
    return taus, pd.concat(curves, ignore_index=True)[['seed', 'tau', 'mean_iou']]


def refit_params(folds_df: pd.DataFrame, seed_taus: dict[int, float]) -> dict:
    """
    Epochs and tau for the phase 4 refit on the whole development set, where
    there is no validation left for early stopping or for choosing tau.

    best_epoch is 0-based, so the run trained best_epoch + 1 epochs to reach it.

    Returns:
        {'refit_epochs': ceil(median(best_epoch + 1)),
         'tau': median of the pooled per-seed taus, 'seed_taus': seed_taus,
         'tau_run_median': median of the per-run taus (diagnostic only),
         'runs': [{run, fold, seed, best_epoch, tau}, ...]}.
    """
    return {
        'refit_epochs': int(math.ceil(float(np.median(folds_df['best_epoch'] + 1)))),
        'tau': float(np.median(list(seed_taus.values()))),
        'seed_taus': seed_taus,
        'tau_run_median': float(np.median(folds_df['tau'])),
        'runs': folds_df[['run', 'fold', 'seed', 'best_epoch', 'tau']].to_dict('records'),
    }


def write_refit_config(cfg: dict, params: dict, path: Path, source: Path) -> None:
    """The CV config with refit_epochs and tau fixed, with comments on where each value came from."""
    refit = {}
    for key, value in copy.deepcopy(cfg).items():
        refit[key] = value
        if key == 'epochs':
            refit['refit_epochs'] = params['refit_epochs']
    # No validation in the refit; run() rejects None, so it cannot silently validate on fold_0
    refit['val_fold'] = None
    refit['tau'] = params['tau']

    runs = params['runs']
    epochs_list = ', '.join(str(r['best_epoch'] + 1) for r in runs)
    seed_taus = ', '.join(f"seed {s}: {t:g}" for s, t in params['seed_taus'].items())
    run_taus = ', '.join(f"{r['tau']:g}" for r in runs)
    comments = {
        'epochs:': "# Length of the cosine schedule (T_max), as in the CV runs",
        'refit_epochs:': (
            f"# Stop here: ceil(median) of the epochs to the best val IoU over {len(runs)} CV runs "
            f"({epochs_list}).\n# Trained with T_max = epochs, so the LR follows the same curve the CV "
            "validated;\n# compressing the cosine to refit_epochs would end at LR 0, an unvalidated trajectory"
        ),
        'tau:': (
            f"# Median over seeds of select_tau on the pooled OOF preds of all folds ({seed_taus}).\n"
            f"# Per-run taus, diagnostic only: {run_taus} (median {params['tau_run_median']:g})"
        ),
        'aggregation:': "# Kept after the aggregation ablation (aggregation_ablation_fold0.csv)",
    }
    lines = []
    for line in yaml.safe_dump(refit, sort_keys=False).splitlines():
        for key, comment in comments.items():
            if line.startswith(key):
                lines.append(comment)
        lines.append(line)

    header = [
        f"# Phase 4 refit config, written by scripts/cv.py from {source.name}.",
        "# Train on every development fold (val_fold: null) for refit_epochs epochs",
        "# and score test with this fixed `tau`; nothing here is tuned on test.",
    ]
    path.write_text('\n'.join(header + lines) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description="5-fold CV of M1 with one config, against M0.")
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs' / 'train' / 'm1.yaml')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--folds', nargs='+', default=DEV_FOLDS)
    parser.add_argument('--device', help="cpu or mps; default picks MPS when available")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    configs = cv_configs(cfg, args.folds, args.seeds)

    for name, c in configs.items():
        print(f"=== {name}", flush=True)
        metrics = run_variant(c, RUNS_DIR / name, args.device)
        print(f"    iou {metrics['iou']:.4f}", flush=True)

    folds_df, volumes_df = collect(list(configs), RUNS_DIR)
    baseline = pd.read_csv(REPORTS_DIR / 'baseline_cv_volumes.csv')
    summary = summarize_cv(folds_df, volumes_df, baseline)
    _, volumes = load_data(cfg)
    seed_taus, tau_curves = pooled_taus(folds_df, RUNS_DIR, volume_boxes(volumes), cfg)
    params = refit_params(folds_df, seed_taus)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    folds_df.to_csv(REPORTS_DIR / 'cv_folds.csv', index=False)
    # Same columns as baseline_cv_volumes.csv (plus seed) so the two can be stacked
    volumes_df.assign(part='val')[KEY + ['aclDiagnosis', 'fold', 'seed', 'part', 'iou']].to_csv(
        REPORTS_DIR / 'cv_volumes.csv', index=False,
    )
    summary.to_csv(REPORTS_DIR / 'cv_summary.csv', index=False)
    # Pooled OOF tau curve per seed: the evidence that the chosen tau sits on a plateau
    tau_curves.to_csv(REPORTS_DIR / 'cv_tau_curves.csv', index=False)
    write_refit_config(cfg, params, REFIT_CFG, args.config)

    print("\nOOF IoU of M1 (seed-averaged per volume), 95% bootstrap CI, gain over M0:")
    print(summary.round(4).to_string(index=False))
    print(f"\nRefit: refit_epochs={params['refit_epochs']} of a {cfg['epochs']}-epoch cosine, "
          f"tau={params['tau']:g} (per seed {params['seed_taus']}; "
          f"per-run median {params['tau_run_median']:g})")
    print(f"Saved {REPORTS_DIR}/cv_*.csv and {REFIT_CFG}")


if __name__ == '__main__':
    main()
