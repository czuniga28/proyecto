from pathlib import Path
import pickle
import numpy as np
import pandas as pd

def load_volume(path: str | Path) -> np.ndarray:
    """Load a raw (D, H, W) volume from a .pck file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def processed_volume_file(exam_id, series_no):
    """
    Path of a processed volume, relative to processed_dir.

    Works on scalars or on pandas Series (vectorized), so the pipeline that
    writes the files and the slice index that points to them share one rule.
    """
    if isinstance(exam_id, pd.Series):
        return "volumes/" + exam_id.astype(str) + "-" + series_no.astype(str) + ".npy"
    return f"volumes/{exam_id}-{series_no}.npy"
