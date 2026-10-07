import yaml
import pandas as pd
from pathlib import Path
from src.data.manifest import build_manifest

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

    # 1. Build the manifest
    print(f"Building manifest from {volumes_dir}...")
    manifest_df = build_manifest(metadata_csv, volumes_dir)
    
    # 2. Ensure interim directory exists
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Save the manifest
    out_path = interim_dir / "manifest.csv"
    manifest_df.to_csv(out_path, index=False)
    print(f"Saved manifest to {out_path}\n")
    
    # 4. Print summary
    total_rows = len(manifest_df)
    existing_rows = manifest_df['exists'].sum()
    
    print("--- Summary ---")
    print(f"Total rows: {total_rows}")
    print(f"Files existing on disk: {existing_rows}")
    
    # crosstab keeps every class visible even when none are missing
    print("\nFiles on disk by aclDiagnosis class:")
    print(pd.crosstab(manifest_df['aclDiagnosis'], manifest_df['exists']))

if __name__ == "__main__":
    main()
