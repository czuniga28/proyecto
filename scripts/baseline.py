import yaml
import pandas as pd
from pathlib import Path
from src.eval.baseline import cv_fixed_box
from src.eval.metrics import summarize_iou

def main():
    project_root = Path(__file__).resolve().parents[1]
    with open(project_root / "configs" / "prep.yaml", "r") as f:
        config = yaml.safe_load(f)

    interim_dir = project_root / config['data']['interim_dir']
    n_cv_folds = config['split']['n_cv_folds']
    out_dir = project_root / "reports" / "fase3"
    out_dir.mkdir(parents=True, exist_ok=True)

    # splits.csv is the frozen partition; the ROI lives in the manifest
    key = ['examId', 'seriesNo']
    manifest = pd.read_csv(interim_dir / "manifest.csv")
    splits = pd.read_csv(interim_dir / "splits.csv")
    volumes = splits[key + ['split']].merge(manifest, on=key, validate='one_to_one')
    assert len(volumes) == len(splits) == len(manifest), "manifest and splits disagree"

    per_volume = cv_fixed_box(volumes, n_cv_folds)
    per_volume.to_csv(out_dir / "baseline_cv_volumes.csv", index=False)

    # One summary per (fold, part): the unit the course reports as mean ± std over folds
    rows = []
    for (fold, part), g in per_volume.groupby(['fold', 'part']):
        s = summarize_iou(g['iou'], g['aclDiagnosis'])
        rows.append(s.rename_axis('clase').reset_index().assign(fold=fold, part=part))
    per_fold = pd.concat(rows, ignore_index=True)
    per_fold['clase'] = per_fold['clase'].astype(str)
    per_fold.to_csv(out_dir / "baseline_cv_folds.csv", index=False)

    print("\nMean IoU per fold (val):")
    print(per_fold[per_fold.part == 'val'].pivot(index='fold', columns='clase', values='mean').round(4))

    across = (per_fold.groupby(['part', 'clase'])['mean'].agg(['mean', 'std']).round(4))
    print("\nMean IoU across folds (mean ± std of the per-fold means):")
    print(across)

    # Q: overfitting gap; a fixed box should show almost none
    gap = across.loc['train', 'mean'] - across.loc['val', 'mean']
    print("\nTrain - val gap per class:")
    print(gap.round(4))

    # Out-of-fold: every development volume validated exactly once
    oof = per_volume[per_volume.part == 'val']
    print(f"\nOut-of-fold summary ({len(oof)} volumes):")
    print(summarize_iou(oof['iou'], oof['aclDiagnosis']).round(4))

    print(f"\nSaved to {out_dir}")

if __name__ == "__main__":
    main()
