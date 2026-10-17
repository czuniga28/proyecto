import numpy as np
import pandas as pd

from src.data.io import processed_volume_file

def build_slice_index(splits_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand a per-volume table into one row per slice.

    A slice contains the ACL when roiZ <= slice_idx < roiZ + roiDepth (0-indexed,
    end-exclusive, the same convention the QA bounds check uses).

    dist_to_roi is the number of slices to the nearest ROI slice (0 inside it), so
    hard negatives can be sampled near the ROI. The 2D box (x0, y0, x1, y1) uses
    roiX as the column and roiY as the row, is end-exclusive (x1 = roiX + roiWidth,
    y1 = roiY + roiHeight), and is empty on negative slices.

    Args:
        splits_df: Per-volume DataFrame with examId, seriesNo, depth, roiX, roiY,
            roiZ, roiWidth, roiHeight, roiDepth, aclDiagnosis and split.

    Returns:
        A DataFrame with columns ['examId', 'seriesNo', 'slice_idx', 'split',
        'aclDiagnosis', 'contains_acl', 'dist_to_roi', 'x0', 'y0', 'x1', 'y1',
        'volume_file'].
    """
    vols = splits_df.reset_index(drop=True)
    depths = vols['depth'].astype(int).to_numpy()

    # Repeat each volume row once per slice, then number slices 0..depth-1 within it
    slices = vols.loc[np.repeat(vols.index, depths)].reset_index(drop=True)
    slices['slice_idx'] = np.concatenate([np.arange(d) for d in depths])

    slices['contains_acl'] = (
        (slices['slice_idx'] >= slices['roiZ'])
        & (slices['slice_idx'] < slices['roiZ'] + slices['roiDepth'])
    )

    # Distance above the ROI, below it, or 0 inside: at most one of the two is positive
    above = slices['roiZ'] - slices['slice_idx']
    below = slices['slice_idx'] - (slices['roiZ'] + slices['roiDepth'] - 1)
    slices['dist_to_roi'] = np.maximum(np.maximum(above, below), 0)

    # Int64 keeps the coordinates as integers while allowing NA on negative slices
    is_pos = slices['contains_acl']
    slices['x0'] = slices['roiX'].where(is_pos).astype('Int64')
    slices['y0'] = slices['roiY'].where(is_pos).astype('Int64')
    slices['x1'] = (slices['roiX'] + slices['roiWidth']).where(is_pos).astype('Int64')
    slices['y1'] = (slices['roiY'] + slices['roiHeight']).where(is_pos).astype('Int64')

    slices['volume_file'] = processed_volume_file(slices['examId'], slices['seriesNo'])

    # Self-checks: a bug in the slice condition must fail here, not in Phase 3
    assert len(slices) == vols['depth'].sum(), "slice count != sum of depth"
    assert slices['contains_acl'].sum() == vols['roiDepth'].sum(), "positive count != sum of roiDepth"
    key = ['examId', 'seriesNo']
    pos_per_volume = slices.groupby(key)['contains_acl'].sum()
    expected = vols.set_index(key)['roiDepth']
    assert pos_per_volume.sort_index().equals(expected.sort_index().astype(pos_per_volume.dtype)), \
        "some volume does not have exactly roiDepth positive slices"

    return slices[[
        'examId', 'seriesNo', 'slice_idx', 'split', 'aclDiagnosis', 'contains_acl',
        'dist_to_roi', 'x0', 'y0', 'x1', 'y1', 'volume_file',
    ]]
