import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.data.io import load_volume, processed_volume_file
from src.data.manifest import build_manifest
from src.data.split import assign_splits
from src.data.qa import run_qa_checks
from src.data.slices import build_slice_index
from src.data.transforms import normalize_volume, pad_volume

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
    processed_dir = project_root / config['data']['processed_dir']
    clip_low_pct = config['preprocess']['clip_low_pct']
    clip_high_pct = config['preprocess']['clip_high_pct']
    pad_height = config['preprocess']['pad_height']
    pad_width = config['preprocess']['pad_width']
    storage_dtype = config['preprocess']['storage_dtype']
    seed = config['seed']
    n_test_splits = config['split']['n_test_splits']
    n_cv_folds = config['split']['n_cv_folds']
    max_prop_diff = config['split']['max_prop_diff']

    interim_dir.mkdir(parents=True, exist_ok=True)

    # Manifest: build, save, summarize
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

    # QA Checks
    print("\nRunning QA checks...")
    qa_df = run_qa_checks(manifest_df)
    qa_path = interim_dir / "qa_report.csv"
    qa_df.to_csv(qa_path, index=False)
    
    if not qa_df.empty:
        print(f"\nQA Failed! Found {len(qa_df)} issues:")
        print(qa_df)
        raise SystemExit(f"\nPipeline stopped: QA checks failed. See {qa_path} for details.")
    print("QA passed! (0 issues found)")

    # Split: test set + CV folds, grouped by examId and stratified by class
    splits_df = assign_splits(manifest_df, seed, n_test_splits, n_cv_folds, max_prop_diff)

    splits_path = interim_dir / "splits.csv"
    splits_df[['examId', 'seriesNo', 'volumeFilename', 'aclDiagnosis', 'split']].to_csv(splits_path, index=False)
    print(f"\nSaved splits to {splits_path}")

    print("\nVolumes per split by aclDiagnosis class:")
    print(pd.crosstab(splits_df['split'], splits_df['aclDiagnosis'], margins=True))

    # Process: normalize, pad, cast, and save one .npy per volume, named by the key
    volumes_out_dir = processed_dir / "volumes"
    volumes_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing volumes into {volumes_out_dir}...")
    for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df)):
        raw = load_volume(volumes_dir / row['path'])
        vol = normalize_volume(raw, clip_low_pct, clip_high_pct)
        vol = pad_volume(vol, pad_height, pad_width)
        vol = vol.astype(storage_dtype)
        np.save(processed_dir / processed_volume_file(row['examId'], row['seriesNo']), vol)

    # Slice index: one row per slice, labelled by whether it contains the ACL
    slices_df = build_slice_index(splits_df)
    slices_path = processed_dir / "slices.csv"
    slices_df.to_csv(slices_path, index=False)
    print(f"\nSaved slice index ({len(slices_df)} slices) to {slices_path}")

    # Q evidence: the positive-slice ratio should be similar in every split
    print("\nSlices per split by contains_acl:")
    print(pd.crosstab(slices_df['split'], slices_df['contains_acl'], margins=True))
    print("\nPositive ratio per split:")
    print(pd.crosstab(slices_df['split'], slices_df['contains_acl'], normalize='index', margins=True).round(4))

    positives = slices_df[slices_df['contains_acl']]
    print("\nPositive slices per split by aclDiagnosis:")
    print(pd.crosstab(positives['split'], positives['aclDiagnosis'], margins=True))

if __name__ == "__main__":
    main()
