import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.eval.bootstrap import paired_bootstrap
from src.train.loop import KEY, PROJECT_ROOT, run, write_provenance

OUT_DIR = PROJECT_ROOT / 'runs' / 'ablations'
SUMMARY_CSV = PROJECT_ROOT / 'reports' / 'fase3' / 'ablations.csv'
BASE = 'base'
SEED_PREFIX = 'seed_'


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """
    Return a copy of cfg with dotted keys replaced, e.g. {'model.pretrained': False}.

    Raises:
        KeyError: If a key path does not already exist in cfg, so that a typo
            fails at once instead of training hours on the unchanged config.
    """
    cfg = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        *parents, leaf = dotted.split('.')
        node = cfg
        for part in parents:
            if not isinstance(node.get(part), dict):
                raise KeyError(f"{dotted!r}: {part!r} is not a section of the base config")
            node = node[part]
        if leaf not in node:
            raise KeyError(f"{dotted!r}: {leaf!r} is not a key of the base config")
        node[leaf] = value
    return cfg


def run_variant(cfg: dict, out: Path, device: str | None) -> dict:
    """Train one variant, or read its metrics if a previous call already finished it."""
    done = out / 'metrics.json'
    if done.exists():
        return json.loads(done.read_text())
    write_provenance(out, cfg)
    return run(cfg, out, device=device)


def summarize(ablations: dict[str, dict], out_dir: Path) -> pd.DataFrame:
    """
    One row per finished variant, compared with the base.

    Two yardsticks, because they measure different noise:
      seed_sigma: std of the IoU of base and seed_* (training noise). A row
          counts as a real change only if |delta| > 2 * seed_sigma.
      boot_lo, boot_hi: paired bootstrap of the per-volume IoU against base
          (which volumes happen to be in fold_0).

    Args:
        ablations: Variant name -> overrides, in report order.
        out_dir: Directory with one subdirectory per variant.

    Returns:
        The table; variants without metrics.json yet are left out.
    """
    rows, per_volume = [], {}
    for name, overrides in ablations.items():
        run_dir = out_dir / name
        if not (run_dir / 'metrics.json').exists():
            continue
        metrics = json.loads((run_dir / 'metrics.json').read_text())
        rows.append({'variant': name, 'overrides': json.dumps(overrides or {}), **metrics})
        if (run_dir / 'val_volumes.csv').exists():
            per_volume[name] = pd.read_csv(run_dir / 'val_volumes.csv')

    table = pd.DataFrame(rows)
    if table.empty or BASE not in set(table['variant']):
        return table

    base = table.set_index('variant').loc[BASE]
    table['delta'] = table['iou'] - base['iou']
    if 'iou_class_2' in table:
        table['delta_class_2'] = table['iou_class_2'] - base['iou_class_2']

    is_seed = (table['variant'] == BASE) | table['variant'].str.startswith(SEED_PREFIX)
    seed_ious = table.loc[is_seed, 'iou']
    sigma = seed_ious.std(ddof=1) if len(seed_ious) >= 2 else np.nan
    table['seed_sigma'] = sigma
    # NaN until two seeds exist; never claimed for the seed rows themselves
    table['beyond_2sigma'] = (table['delta'].abs() > 2 * sigma).where(~is_seed & np.isfinite(sigma))

    lo, hi = [], []
    for name in table['variant']:
        if name == BASE or name not in per_volume or BASE not in per_volume:
            lo.append(np.nan)
            hi.append(np.nan)
            continue
        m = per_volume[name].merge(per_volume[BASE], on=KEY, suffixes=('', '_base'), validate='one_to_one')
        ci = paired_bootstrap(m['iou'].to_numpy(), m['iou_base'].to_numpy())
        lo.append(ci['lo'])
        hi.append(ci['hi'])
    table['boot_lo'], table['boot_hi'] = lo, hi
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-factor-at-a-time ablations on one fold.")
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs' / 'train' / 'm1.yaml')
    parser.add_argument('--ablations', type=Path, default=PROJECT_ROOT / 'configs' / 'ablations.yaml')
    parser.add_argument('--only', nargs='+', help="Run only these variants (the summary still lists all finished ones)")
    parser.add_argument('--device', help="cpu or mps; default picks MPS when available")
    args = parser.parse_args()

    base_cfg = yaml.safe_load(args.config.read_text())
    ablations = {name: overrides or {} for name, overrides in yaml.safe_load(args.ablations.read_text()).items()}
    if BASE not in ablations:
        raise ValueError(f"{args.ablations} must define a {BASE!r} variant to compare against")
    unknown = set(args.only or []) - set(ablations)
    if unknown:
        raise ValueError(f"Unknown variants {sorted(unknown)}. Defined: {list(ablations)}")

    # Build every config up front: a bad key fails now, not after hours of training
    configs = {name: apply_overrides(base_cfg, overrides) for name, overrides in ablations.items()}

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    for name, cfg in configs.items():
        if args.only and name not in args.only:
            continue
        print(f"=== {name}: {ablations[name] or 'base config'}", flush=True)
        metrics = run_variant(cfg, OUT_DIR / name, args.device)
        print(f"    iou {metrics['iou']:.4f}", flush=True)
        # Rewritten after every variant so a stopped run still leaves a report
        summarize(ablations, OUT_DIR).to_csv(SUMMARY_CSV, index=False)

    table = summarize(ablations, OUT_DIR)
    table.to_csv(SUMMARY_CSV, index=False)
    shown = [c for c in ['variant', 'iou', 'iou_class_2', 'delta', 'delta_class_2',
                         'beyond_2sigma', 'boot_lo', 'boot_hi', 'best_epoch', 'seconds_per_epoch']
             if c in table]
    print(table[shown].round(4).to_string(index=False))
    if 'seed_sigma' in table:
        print(f"\nseed sigma: {table['seed_sigma'].iloc[0]:.4f}")
    print(f"Saved {SUMMARY_CSV}")


if __name__ == '__main__':
    main()
