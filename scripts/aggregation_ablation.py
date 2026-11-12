import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.eval.aggregate import XY_METHODS, Z_METHODS, aggregate_predictions, score_volumes, select_tau
from src.eval.bootstrap import paired_bootstrap
from src.eval.metrics import summarize_iou
from src.train.loop import KEY, PROJECT_ROOT, load_data, volume_boxes

REPORTS_DIR = PROJECT_ROOT / 'reports' / 'fase3'
FIGS_DIR = PROJECT_ROOT / 'docs' / 'entregable3' / 'figs'
BASE = ('kadane', 'wmedian')


def load_runs(run_dirs: list[Path]) -> tuple[pd.DataFrame, dict]:
    """
    Concatenate preds_val.csv of several runs and return the first run's config.

    Several runs make sense only when they validate different volumes, as the
    folds of a CV do (out-of-fold predictions).

    Raises:
        ValueError: If a slice appears in more than one run.
    """
    cfg = yaml.safe_load((Path(run_dirs[0]) / 'config.yaml').read_text())
    preds = pd.concat([pd.read_csv(Path(d) / 'preds_val.csv') for d in run_dirs], ignore_index=True)
    if preds.duplicated(KEY + ['slice_idx']).any():
        raise ValueError("the same volume appears in several runs; pass runs that validate different folds")
    return preds, cfg


def ablate_aggregation(
    preds: pd.DataFrame, gt: pd.DataFrame, taus: np.ndarray, n_boot: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score every z_method x xy_method combination, each with its own best tau.

    Args:
        preds: Per-slice predictions, as in aggregate_predictions.
        gt: Output of volume_boxes for the volumes in preds.
        taus: Candidate thresholds.
        n_boot: Bootstrap resamples for the comparison with BASE.

    Returns:
        (table, curves). table has one row per combination with tau, global and
        per-class IoU, iou_z, iou_xy and a paired bootstrap of the difference
        with BASE; curves has the mean IoU of every combination at every tau.
    """
    rows, curves, ious = [], [], {}
    for z in Z_METHODS:
        for xy in XY_METHODS:
            tau, curve = select_tau(preds, gt, taus, z, xy)
            scored = score_volumes(aggregate_predictions(preds, tau, z, xy), gt)
            summary = summarize_iou(scored['iou'], scored['aclDiagnosis'])
            ious[(z, xy)] = scored['iou'].to_numpy()
            rows.append({
                'z_method': z, 'xy_method': xy, 'tau': tau,
                'iou': summary.loc['all', 'mean'],
                **{f'iou_class_{c}': summary.loc[c, 'mean'] for c in summary.index if c != 'all'},
                'frac_ge_05': summary.loc['all', 'frac_ge_05'],
                'iou_z': scored['iou_z'].mean(), 'iou_xy': scored['iou_xy'].mean(),
            })
            curves.append(curve.reset_index().assign(z_method=z, xy_method=xy))

    table = pd.DataFrame(rows)
    # Same volumes in the same (gt) order for every combination: a paired comparison
    boot = [paired_bootstrap(ious[(z, xy)], ious[BASE], n=n_boot)
            for z, xy in zip(table['z_method'], table['xy_method'])]
    table['delta'] = [b['delta'] for b in boot]
    table['boot_lo'] = [b['lo'] for b in boot]
    table['boot_hi'] = [b['hi'] for b in boot]
    table['ci_excludes_0'] = (table['boot_lo'] > 0) | (table['boot_hi'] < 0)
    return table.sort_values('iou', ascending=False, ignore_index=True), pd.concat(curves, ignore_index=True)


def plot_tau_curves(table: pd.DataFrame, curves: pd.DataFrame, path: Path) -> None:
    """IoU against tau for each z_method, using its best xy_method."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for z in Z_METHODS:
        best_xy = table.loc[table['z_method'] == z].iloc[0]['xy_method']
        c = curves[(curves['z_method'] == z) & (curves['xy_method'] == best_xy)]
        ax.plot(c['tau'], c['mean_iou'], marker='o', label=f"{z} + {best_xy}")
    ax.set_xlabel('τ')
    ax.set_ylabel('IoU 3D media (validación)')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare slice-to-volume aggregation rules (E1, E2).")
    parser.add_argument('--run', type=Path, nargs='+', required=True,
                        help="Run directories with preds_val.csv; several = out-of-fold predictions of a CV")
    parser.add_argument('--tag', default='fold0', help="Suffix of the output files, e.g. fold0 or oof")
    parser.add_argument('--n-boot', type=int, default=2000)
    args = parser.parse_args()

    preds, cfg = load_runs(args.run)
    _, volumes = load_data(cfg)
    gt = volume_boxes(volumes.merge(preds[KEY].drop_duplicates(), on=KEY))

    table, curves = ablate_aggregation(preds, gt, np.asarray(cfg['taus']), args.n_boot)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS_DIR / f'aggregation_ablation_{args.tag}.csv', index=False)
    curves.to_csv(REPORTS_DIR / f'tau_curves_{args.tag}.csv', index=False)
    plot_tau_curves(table, curves, FIGS_DIR / f'tau_curve_{args.tag}.png')

    print(f"{len(gt)} volumes, base = {BASE[0]} + {BASE[1]}\n")
    print(table.round(4).to_string(index=False))
    print(f"\nSaved to {REPORTS_DIR} and {FIGS_DIR}")


if __name__ == '__main__':
    main()
