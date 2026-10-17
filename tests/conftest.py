import numpy as np
import pandas as pd
import pytest


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
