import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def assign_splits(
    manifest: pd.DataFrame,
    seed: int,
    n_test_splits: int = 6,
    n_cv_folds: int = 5,
) -> pd.DataFrame:
    """
    Assign every volume to 'test' or to a CV fold 'fold_0'..'fold_{n_cv_folds-1}'.

    Both splits use StratifiedGroupKFold: stratified by aclDiagnosis and grouped
    by examId, so series from the same exam never land in different splits.
    The test set is one of n_test_splits SGKF folds (1/n_test_splits of the data).
    Returns a copy of the manifest with an added 'split' column.
    """
    missing = (~manifest['exists']).sum()
    if missing:
        raise ValueError(f"{missing} volumes in the manifest are missing on disk.")

    # SGKF returns positions; reset so positions and index labels match for .loc
    df = manifest.copy().reset_index(drop=True)
    y = df['aclDiagnosis']
    groups = df['examId']

    # 1. Test set: one SGKF fold out of n_test_splits
    test_splitter = StratifiedGroupKFold(n_splits=n_test_splits, shuffle=True, random_state=seed)
    train_validation_indices, test_indices = next(test_splitter.split(df, y, groups=groups))

    df['split'] = 'unassigned'
    df.loc[test_indices, 'split'] = 'test'

    # 2. CV folds on the remaining development data
    train_validation_df = df.iloc[train_validation_indices]

    cross_validation_splitter = StratifiedGroupKFold(n_splits=n_cv_folds, shuffle=True, random_state=seed)
    cv_splits = cross_validation_splitter.split(
        train_validation_df,
        train_validation_df['aclDiagnosis'],
        groups=train_validation_df['examId'],
    )

    for fold, (_, fold_validation_indices_local) in enumerate(cv_splits):
        # Local positions in train_validation_df -> labels in df
        fold_validation_indices_original = train_validation_df.iloc[fold_validation_indices_local].index
        df.loc[fold_validation_indices_original, 'split'] = f'fold_{fold}'

    verify_splits(df, n_cv_folds)

    return df


def verify_splits(df: pd.DataFrame, n_cv_folds: int) -> None:
    print("\nRunning verification checks...")

    # a) No examId appears on both sides (i.e., an examId is mapped to strictly 1 split)
    exam_split_counts = df.groupby('examId')['split'].nunique()
    if (exam_split_counts > 1).any():
        violators = exam_split_counts[exam_split_counts > 1].index.tolist()
        raise ValueError(f"Leakage detected! These examIds appear in multiple splits: {violators}")
    print("Leakage check passed: No examId appears in multiple splits.")

    # b) Class proportions per part are close to the totals
    total_props = df['aclDiagnosis'].value_counts(normalize=True)
    print("\nTotal dataset class proportions:")
    print(total_props.to_string())

    print("\nClass proportions per split:")
    split_props = df.groupby('split')['aclDiagnosis'].value_counts(normalize=True).unstack(fill_value=0)
    print(split_props.to_string())

    # c) Every volume is assigned exactly once
    if (df['split'] == 'unassigned').any():
        raise ValueError("Some volumes were not assigned to any split.")

    expected_splits = {'test'} | {f'fold_{i}' for i in range(n_cv_folds)}
    actual_splits = set(df['split'].unique())
    if actual_splits != expected_splits:
        raise ValueError(f"Unexpected splits found. Expected {expected_splits}, got {actual_splits}")

    print("\n - Assignment check passed: Every volume is assigned exactly once.")
