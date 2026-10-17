from pathlib import Path
import pandas as pd
from src.data.io import load_volume

def build_manifest(metadata_csv: str | Path, volumes_dir: str | Path) -> pd.DataFrame:
    """
    Build a manifest dataframe from the dataset metadata and volume files.

    Reads metadata_csv, checks for file existence under volumes_dir, and loads
    the shape, dtype, and intensity stats from each .pck file. Enforces unique
    (examId, seriesNo) pairs. Paths are stored relative to volumes_dir.
    """
    raw_dir_path = Path(volumes_dir)

    data = pd.read_csv(metadata_csv)

    # Enforce the key (Fail fast before processing files)
    if data.duplicated(subset=['examId', 'seriesNo']).any():
        raise ValueError("Duplicate (examId, seriesNo) pair found in the dataset.")

    file_lookup = {
        p.name: str(p.relative_to(raw_dir_path)) for p in raw_dir_path.rglob('*.pck')
    }

    data['path'] = data['volumeFilename'].map(file_lookup)
    data['exists'] = data['path'].notna()

    # Read shape, dtype, and intensity stats from each file
    depths, heights, widths, dtypes = [], [], [], []
    mins, maxs = [], []

    for _, row in data.iterrows():
        if row['exists']:
            # Using raw_dir_path to get the absolute path to the file
            arr = load_volume(raw_dir_path / row['path'])
            d, h, w = arr.shape

            depths.append(d)
            heights.append(h)
            widths.append(w)
            dtypes.append(str(arr.dtype))
            mins.append(arr.min())
            maxs.append(arr.max())
        else:
            depths.append(pd.NA)
            heights.append(pd.NA)
            widths.append(pd.NA)
            dtypes.append(pd.NA)
            mins.append(pd.NA)
            maxs.append(pd.NA)
            
    # Assign and cast integer columns to 'Int64' to support pd.NA safely
    data['depth'] = pd.Series(depths, dtype='Int64')
    data['height'] = pd.Series(heights, dtype='Int64')
    data['width'] = pd.Series(widths, dtype='Int64')
    data['dtype'] = dtypes
    data['intensity_min'] = pd.Series(mins, dtype='Int64')
    data['intensity_max'] = pd.Series(maxs, dtype='Int64')

    return data
