import argparse
import copy
from pathlib import Path
from optuna.study import MaxTrialsCallback

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optuna
import yaml
from optuna.trial import TrialState

from src.train.loop import PROJECT_ROOT, run, write_provenance

OPTUNA_DIR = PROJECT_ROOT / 'runs' / 'optuna'
TRIALS_CSV = PROJECT_ROOT / 'reports' / 'fase3' / 'optuna_trials.csv'
FIGS_DIR = PROJECT_ROOT / 'docs' / 'entregable3' / 'figs'
BEST_CFG = PROJECT_ROOT / 'configs' / 'train' / 'm1_best.yaml'
STUDY_NAME = 'm1'


def suggest(trial: optuna.Trial, base_cfg: dict) -> dict:
    """Return a copy of base_cfg with the hyperparameters suggested by trial."""
    cfg = copy.deepcopy(base_cfg)
    cfg['lr'] = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    cfg['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    cfg['model']['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)
    cfg['box_weight'] = trial.suggest_float('box_weight', 0.5, 5.0, log=True)
    cfg['max_shift'] = trial.suggest_categorical('max_shift', [0, 10, 20, 30])
    cfg['sampler']['hard_frac'] = trial.suggest_float('hard_frac', 0.25, 0.75)
    # Shorter than a final run so 20 trials fit; the pruner cuts the bad ones earlier
    cfg['epochs'] = 15
    cfg['patience'] = 3
    return cfg


def objective(trial: optuna.Trial, base_cfg: dict, device: str | None = None) -> float:
    cfg = suggest(trial, base_cfg)
    out = OPTUNA_DIR / f'trial_{trial.number}'
    write_provenance(out, cfg)
    metrics = run(cfg, out, trial=trial, device=device)
    # Kept in the study so optuna_trials.csv shows per-class IoU, AUPRC, etc.
    for k, v in metrics.items():
        trial.set_user_attr(k, v)
    return metrics['iou']


def save_figure(ax, path: Path) -> None:
    ax.figure.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(ax.figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune M1 hyperparameters with Optuna.")
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs' / 'train' / 'm1.yaml')
    parser.add_argument('--n-trials', type=int, default=20)
    parser.add_argument('--timeout', type=float, help="Seconds; stops starting new trials after it")
    parser.add_argument('--device', help="cpu or mps; default picks MPS when available")
    args = parser.parse_args()

    base_cfg = yaml.safe_load(args.config.read_text())

    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        storage=f'sqlite:///{OPTUNA_DIR / "m1.db"}',
        study_name=STUDY_NAME,
        load_if_exists=True,
    )
    study.optimize(lambda t: objective(t, base_cfg, args.device),
    timeout=args.timeout,
    callbacks=[MaxTrialsCallback(args.n_trials,
                                states=(TrialState.COMPLETE,
                                TrialState.PRUNED))],)

    TRIALS_CSV.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(TRIALS_CSV, index=False)

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(optuna.visualization.matplotlib.plot_optimization_history(study),
                FIGS_DIR / 'optuna_history.png')
    n_complete = len(study.get_trials(states=[TrialState.COMPLETE]))
    if n_complete >= 2:
        save_figure(optuna.visualization.matplotlib.plot_param_importances(study),
                    FIGS_DIR / 'optuna_importance.png')
    else:
        print(f"Skipped parameter importance: needs 2 complete trials, has {n_complete}")

    # FixedTrial replays the best params through suggest, so m1_best.yaml has
    # exactly the config the best trial ran with
    best_cfg = suggest(optuna.trial.FixedTrial(study.best_params), base_cfg)
    BEST_CFG.write_text(yaml.safe_dump(best_cfg, sort_keys=False))

    print(f"Best trial {study.best_trial.number}: IoU {study.best_value:.4f}")
    print(f"Params: {study.best_params}")
    print(f"Saved {TRIALS_CSV}, figures in {FIGS_DIR} and {BEST_CFG}")


if __name__ == '__main__':
    main()
