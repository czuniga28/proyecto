import pandas as pd

def run_qa_checks(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run quality assurance checks on the dataset manifest.
    
    This is a pure function that evaluates the manifest dataframe for potential issues
    and returns a dataframe detailing any failures.
    
    Args:
        manifest_df: DataFrame containing the dataset manifest.
        
    Returns:
        A DataFrame containing the QA failures with columns:
        ['examId', 'seriesNo', 'check', 'detail']
        If no failures are found, an empty DataFrame with these columns is returned.
    """
    # Initialize an empty list to collect failure records
    failures = []
    
    # aclDiagnosis must be 0, 1, or 2
    invalid_acl = manifest_df[~manifest_df['aclDiagnosis'].isin([0, 1, 2])]
    for _, row in invalid_acl.iterrows():
        failures.append({
            'examId': row['examId'],
            'seriesNo': row['seriesNo'],
            'check': 'label_validity',
            'detail': f"Invalid aclDiagnosis: {row['aclDiagnosis']}"
        })
        
    # kneeLR must be 0 or 1
    invalid_knee = manifest_df[~manifest_df['kneeLR'].isin([0, 1])]
    for _, row in invalid_knee.iterrows():
        failures.append({
            'examId': row['examId'],
            'seriesNo': row['seriesNo'],
            'check': 'label_validity',
            'detail': f"Invalid kneeLR: {row['kneeLR']}"
        })
    
    # ROI size is positive
    roi_cols = ['roiWidth', 'roiHeight', 'roiDepth']
    # Check if any of the ROI dimensions are <= 0 for each row
    invalid_roi = manifest_df[manifest_df[roi_cols].le(0).any(axis=1)]
    for _, row in invalid_roi.iterrows():
        failures.append({
            'examId': row['examId'],
            'seriesNo': row['seriesNo'],
            'check': 'roi_size',
            'detail': f"Invalid ROI size (W,H,D): ({row['roiWidth']}, {row['roiHeight']}, {row['roiDepth']})"
        })
        
    # Volume files must exist on disk
    missing_files = manifest_df[~manifest_df['exists']]
    for _, row in missing_files.iterrows():
        failures.append({
            'examId': row['examId'],
            'seriesNo': row['seriesNo'],
            'check': 'file_exists',
            'detail': 'file missing'
        })

    # ROI inside the volume
    # Filter only rows where the volume exists, as we need its shape
    existing_vols = manifest_df[manifest_df['exists'] == True]
    
    axes = [
        ('x', 'roiX', 'roiWidth', 'width'),
        ('y', 'roiY', 'roiHeight', 'height'),
        ('z', 'roiZ', 'roiDepth', 'depth')
    ]
    
    for axis_name, start_col, size_col, dim_col in axes:
        # Check start < 0
        invalid_start = existing_vols[existing_vols[start_col] < 0]
        for _, row in invalid_start.iterrows():
            failures.append({
                'examId': row['examId'],
                'seriesNo': row['seriesNo'],
                'check': 'roi_bounds',
                'detail': f"{axis_name}: {row[start_col]} < 0"
            })
            
        # Check start + size > dimension
        out_of_bounds = existing_vols[(existing_vols[start_col] + existing_vols[size_col]) > existing_vols[dim_col]]
        for _, row in out_of_bounds.iterrows():
            failures.append({
                'examId': row['examId'],
                'seriesNo': row['seriesNo'],
                'check': 'roi_bounds',
                'detail': f"{axis_name}: {row[start_col]}+{row[size_col]} > {row[dim_col]}"
            })

    # dtype must be uint16 for correct normalization later
    invalid_dtype = existing_vols[existing_vols['dtype'] != 'uint16']
    for _, row in invalid_dtype.iterrows():
        failures.append({
            'examId': row['examId'],
            'seriesNo': row['seriesNo'],
            'check': 'dtype',
            'detail': f"Invalid dtype: {row['dtype']} (expected uint16)"
        })
        
        
    # Volume should not be all zeros
    all_zeros = existing_vols[existing_vols['intensity_max'] == 0]
    for _, row in all_zeros.iterrows():
        failures.append({
            'examId': row['examId'],
            'seriesNo': row['seriesNo'],
            'check': 'intensity',
            'detail': "Volume is all zeros (max intensity = 0)"
        })
        
    # Max intensity should be <= 4095 (12-bit data limit)
    too_bright = existing_vols[existing_vols['intensity_max'] > 4095]
    for _, row in too_bright.iterrows():
        failures.append({
            'examId': row['examId'],
            'seriesNo': row['seriesNo'],
            'check': 'intensity',
            'detail': f"Max intensity too high: {row['intensity_max']} > 4095"
        })
    
    # Return failures as a DataFrame
    return pd.DataFrame(failures, columns=['examId', 'seriesNo', 'check', 'detail'])
