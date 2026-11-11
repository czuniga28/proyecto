import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from src.train.loop import PROJECT_ROOT, run, write_provenance


def short_commit() -> str:
    try:
        out = subprocess.run(
            ['git', 'rev-parse', '--short=7', 'HEAD'],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'nogit'


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SliceNet on one validation fold.")
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs' / 'train' / 'm1.yaml')
    parser.add_argument('--val-fold', help="Overrides val_fold in the config")
    parser.add_argument('--seed', type=int, help="Overrides seed in the config")
    parser.add_argument('--device', help="cpu or mps; default picks MPS when available")
    parser.add_argument('--max-slices', type=int, help="Smoke test on this many training slices")
    parser.add_argument('--out', type=Path, help="Default: runs/<YYYYmmdd-HHMM>_<val_fold>_s<seed>_<commit7>/")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.val_fold is not None:
        cfg['val_fold'] = args.val_fold
    if args.seed is not None:
        cfg['seed'] = args.seed

    out = args.out
    if out is None:
        stamp = datetime.now().strftime('%Y%m%d-%H%M')
        out = PROJECT_ROOT / 'runs' / f"{stamp}_{cfg['val_fold']}_s{cfg['seed']}_{short_commit()}"

    write_provenance(out, cfg)
    metrics = run(cfg, out, max_slices=args.max_slices, device=args.device)
    print(json.dumps(metrics, indent=2))
    print(f"Saved to {out}")


if __name__ == '__main__':
    main()
