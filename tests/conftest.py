from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.io import processed_volume_file
from src.data.slices import build_slice_index


def make_manifest(n_exams: int = 120, seed: int = 0) -> pd.DataFrame:
    """
    Synthetic manifest shaped like KneeMRI: ~75/19/6 classes, a few exams with
    two series, 320x320 volumes of 30 slices and an in-bounds ROI.
    """
    rng = np.random.default_rng(seed)
    labels = rng.choice([0, 1, 2], size=n_exams, p=[0.75, 0.19, 0.06])
    rows = []
    for exam, label in enumerate(labels):
        n_series = 2 if exam % 15 == 0 else 1
        for series in range(n_series):
            rows.append({
                'examId': 1000 + exam, 'seriesNo': 5 + series, 'aclDiagnosis': int(label),
                'kneeLR': exam % 2, 'roiX': 110, 'roiY': 105, 'roiZ': 13,
                'roiHeight': 90, 'roiWidth': 90, 'roiDepth': 3,
                'volumeFilename': f"{1000 + exam}-{5 + series}.pck",
                'path': f"{1000 + exam}-{5 + series}.pck", 'exists': True,
                'depth': 30, 'height': 320, 'width': 320, 'dtype': 'uint16',
                'intensity_min': 0, 'intensity_max': 2400,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def manifest() -> pd.DataFrame:
    return make_manifest()


RUN_SIDE = 64
RUN_DEPTH = 8


def make_run_files(root: Path, n_exams: int = 6) -> dict:
    """
    Small processed dataset on disk for run(): make_manifest shrunk to
    RUN_DEPTH x RUN_SIDE x RUN_SIDE volumes, exams alternating between fold_0
    and fold_1, plus a CPU-sized cfg pointing at it with val_fold = fold_1.
    """
    interim, processed = root / 'interim', root / 'processed'
    interim.mkdir()
    (processed / 'volumes').mkdir(parents=True)

    manifest = make_manifest(n_exams=n_exams).assign(
        depth=RUN_DEPTH, height=RUN_SIDE, width=RUN_SIDE,
        roiX=20, roiY=15, roiZ=3, roiWidth=24, roiHeight=30, roiDepth=2,
    )
    splits = manifest[['examId', 'seriesNo', 'aclDiagnosis']].assign(
        split=np.where(manifest['examId'] % 2 == 0, 'fold_0', 'fold_1'),
    )
    rng = np.random.default_rng(0)
    for v in manifest.itertuples():
        vol = rng.standard_normal((RUN_DEPTH, RUN_SIDE, RUN_SIDE)).astype(np.float16)
        np.save(processed / processed_volume_file(v.examId, v.seriesNo), vol)

    manifest.to_csv(interim / 'manifest.csv', index=False)
    splits.to_csv(interim / 'splits.csv', index=False)
    build_slice_index(splits.merge(manifest)).to_csv(processed / 'slices.csv', index=False)

    return {
        'seed': 0,
        'data': {'interim_dir': str(interim), 'processed_dir': str(processed),
                 'width': RUN_SIDE, 'height': RUN_SIDE},
        'val_fold': 'fold_1',
        'image_size': 32, 'batch_size': 4, 'num_workers': 0, 'max_shift': 2,
        'sampler': {'neg_per_pos': 1.0, 'hard_frac': 0.5, 'hard_dist': 1},
        'model': {'pretrained': False, 'dropout': 0.2},
        'lr': 0.001, 'weight_decay': 0.0001, 'epochs': 2, 'patience': 5,
        'box_weight': 1.0, 'taus': [0.3, 0.5, 0.7],
    }


@pytest.fixture
def run_cfg(tmp_path) -> dict:
    return make_run_files(tmp_path)
