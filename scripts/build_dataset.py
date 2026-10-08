import yaml
import pandas as pd
from pathlib import Path
from src.data.manifest import build_manifest
from src.data.split import assign_splits

def main():
    # Enforce that paths are relative to the project root
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "prep.yaml"
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    metadata_csv = project_root / config['data']['metadata_csv']
    volumes_dir = project_root / config['data']['volumes_dir']
    interim_dir = project_root / config['data']['interim_dir']
    seed = config['seed']
    n_test_splits = config['split']['n_test_splits']
    n_cv_folds = config['split']['n_cv_folds']

    interim_dir.mkdir(parents=True, exist_ok=True)

    # 1. Manifest: build, save, summarize
    print(f"Building manifest from {volumes_dir}...")
    manifest_df = build_manifest(metadata_csv, volumes_dir)

    out_path = interim_dir / "manifest.csv"
    manifest_df.to_csv(out_path, index=False)
    print(f"Saved manifest to {out_path}\n")

    total_rows = len(manifest_df)
    existing_rows = manifest_df['exists'].sum()
    
    print("--- Summary ---")
    print(f"Total rows: {total_rows}")
    print(f"Files existing on disk: {existing_rows}")
    
    # crosstab keeps every class visible even when none are missing
    print("\nFiles on disk by aclDiagnosis class:")
    print(pd.crosstab(manifest_df['aclDiagnosis'], manifest_df['exists']))

    # 2. Split: test set + CV folds, grouped by examId and stratified by class
    splits_df = assign_splits(manifest_df, seed, n_test_splits, n_cv_folds)

    splits_path = interim_dir / "splits.csv"
    splits_df[['examId', 'seriesNo', 'volumeFilename', 'aclDiagnosis', 'split']].to_csv(splits_path, index=False)
    print(f"\nSaved splits to {splits_path}")

    print("\nVolumes per split by aclDiagnosis class:")
    print(pd.crosstab(splits_df['split'], splits_df['aclDiagnosis'], margins=True))

if __name__ == "__main__":
    main()
