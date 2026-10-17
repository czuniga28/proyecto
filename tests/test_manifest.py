import pickle

import numpy as np
import pandas as pd
import pytest

from src.data.manifest import build_manifest

COLUMNS = ['examId', 'seriesNo', 'aclDiagnosis', 'kneeLR', 'roiX', 'roiY', 'roiZ',
           'roiHeight', 'roiWidth', 'roiDepth', 'volumeFilename']


def write_dataset(tmp_path, rows, files):
    """Write a metadata.csv and the given {filename: array} volumes under tmp_path."""
    csv = tmp_path / "metadata.csv"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(csv, index=False)
    vol_dir = tmp_path / "volumes"
    (vol_dir / "sub").mkdir(parents=True)
    for name, arr in files.items():
        with open(vol_dir / "sub" / name, "wb") as f:
            pickle.dump(arr, f)
    return csv, vol_dir


def test_reads_shape_dtype_and_intensity(tmp_path):
    arr = np.zeros((4, 6, 5), dtype=np.uint16)
    arr[1, 2, 3] = 900
    csv, vol_dir = write_dataset(tmp_path, [[1, 5, 0, 1, 0, 0, 0, 2, 2, 1, "1-5.pck"]], {"1-5.pck": arr})

    m = build_manifest(csv, vol_dir)

    row = m.iloc[0]
    assert row['exists']
    assert row['path'] == "sub/1-5.pck"  # relative to volumes_dir, found recursively
    assert (row['depth'], row['height'], row['width']) == (4, 6, 5)
    assert row['dtype'] == 'uint16'
    assert (row['intensity_min'], row['intensity_max']) == (0, 900)


def test_missing_file_is_recorded_not_dropped(tmp_path):
    rows = [[1, 5, 0, 1, 0, 0, 0, 2, 2, 1, "1-5.pck"], [2, 5, 1, 0, 0, 0, 0, 2, 2, 1, "2-5.pck"]]
    csv, vol_dir = write_dataset(tmp_path, rows, {"1-5.pck": np.ones((2, 3, 3), np.uint16)})

    m = build_manifest(csv, vol_dir)

    assert len(m) == 2
    assert m['exists'].tolist() == [True, False]
    assert pd.isna(m.loc[1, 'depth'])
    assert str(m['depth'].dtype) == 'Int64'


def test_duplicate_key_fails_fast(tmp_path):
    row = [1, 5, 0, 1, 0, 0, 0, 2, 2, 1, "1-5.pck"]
    csv, vol_dir = write_dataset(tmp_path, [row, row], {})

    with pytest.raises(ValueError, match="Duplicate"):
        build_manifest(csv, vol_dir)
