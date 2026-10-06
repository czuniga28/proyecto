from lib2to3.fixes import fix_throw
from lib2to3.fixes import fix_throw
from lib2to3.fixes import fix_throw
from lib2to3.fixes import fix_throw
from pathlib import Path
import pandas as pd
import pickle

# Build manifest to work at and dont touch raw data
def build_manifest(raw_data: str | Path) -> pd.DataFrame:
    """
    Build a manifest dataframe from the raw data directory.

    Reads the metadata.csv, checks for file existence, and loads
    the shape and dtype from each .pck file. Enforces unique
    (examId, seriesNo) pairs.
    """
    # Path to raw data folder
    raw_dir_path = Path(raw_data)

    # path to raw csv file from dataset
    csv_path = raw_dir_path / "metadata.csv"
    data = pd.read_csv(csv_path)

    # Enforce the key (Fail fast before processing files)
    if data.duplicated(subset=['examId', 'seriesNo']).any():
        raise ValueError("Duplicate (examId, seriesNo) pair found in the dataset.")

    file_lookup = {
        p.name: str(p.relative_to(raw_dir_path)) for p in raw_dir_path.rglob('*.pck')
    }

    data['path'] = data['volumeFilename'].map(file_lookup)
    data['exists'] = data['path'].notna()

    # Read shape and dtype from each file
    depths, heights, widths, dtypes = [], [], [], []

    for _, row in data.iterrows():
        if row['exists']:
            # Using raw_dir_path to get the absolute path to the file
            file_path = raw_dir_path / row['path']
            with open(file_path, "rb") as f:
                arr = pickle.load(f)
                d, h, w = arr.shape
            
                depths.append(d)
                heights.append(h)
                widths.append(w)
                dtypes.append(str(arr.dtype))
        else:
            depths.append(pd.NA)
            heights.append(pd.NA)
            widths.append(pd.NA)
            dtypes.append(pd.NA)
            
        # Assign and cast integer columns to 'Int64' to support pd.NA safely
        data['depth'] = pd.Series(depths, dtype='Int64')
        data['height'] = pd.Series(heights, dtype='Int64')
        data['width'] = pd.Series(widths, dtype='Int64')
        data['dtype'] = dtypes

    return data
