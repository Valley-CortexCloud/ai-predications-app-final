#!/usr/bin/env python3
"""
Fast production inference using cached snapshots.

Flow:
1. Load latest snapshot (feature_matrix.parquet)
2. Build labels (production mode, no feature computation)
3. Apply ranker (generate predictions)
4. Save to datasets/predictions_YYYY-MM-DD.csv

Target runtime: < 5 minutes

Usage:
    python scripts/production_inference.py [--snapshot-dir DIR] [--output-dir DIR]
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
SNAPSHOTS_DIR = ROOT / "data" / "snapshots"
DATASETS_DIR = ROOT / "datasets"
MODEL_DIR = ROOT / "model"
EARNINGS_FILE = ROOT / "data" / "earnings.csv"


def find_latest_snapshot(snapshots_dir: Path) -> Path:
    """Find the most recent snapshot directory"""
    snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
    
    if not snapshot_dirs:
        raise FileNotFoundError(f"No snapshot directories found in {snapshots_dir}")
    
    # Sort by directory name (YYYY-MM-DD format sorts correctly)
    latest = sorted(snapshot_dirs)[-1]
    return latest


def run_command(cmd: str, description: str):
    """Run shell command and handle errors"""
    print(f"\n{'='*60}")
    print(description)
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"✓ {description} completed")


def production_inference(snapshot_dir: Path, output_dir: Path, model_dir: Path) -> Path:
    """
    Run production inference using snapshot.
    
    Returns:
        Path to predictions file
    """
    print(f"{'='*60}")
    print("Production Inference Pipeline")
    print(f"{'='*60}")
    print(f"Snapshot: {snapshot_dir.name}")
    print(f"Output: {output_dir}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load snapshot metadata
    import json
    metadata_path = snapshot_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    snapshot_date = metadata['snapshot_date']
    print(f"\nSnapshot Info:")
    print(f"  Date: {snapshot_date}")
    print(f"  Symbols: {metadata['symbol_count']}")
    print(f"  Features: {metadata['feature_count']}")
    print(f"  Git commit: {metadata['git_commit'][:8] if metadata['git_commit'] != 'unknown' else 'unknown'}")
    
    # Paths
    feature_matrix_path = snapshot_dir / "feature_matrix.parquet"
    today_features_path = output_dir / f"features_{snapshot_date}.parquet"
    predictions_path = output_dir / f"predictions_{snapshot_date}.csv"
    
    # Step 1: Load feature matrix from snapshot
    print(f"\n{'='*60}")
    print("Step 1: Loading feature matrix from snapshot")
    print(f"{'='*60}")
    
    df = pd.read_parquet(feature_matrix_path)
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Add 'date' column if not present (for build_labels compatibility)
    if 'date' not in df.columns and 'Date' not in df.columns:
        df['date'] = snapshot_date
        print(f"✓ Added 'date' column: {snapshot_date}")
    
    # Save as intermediate features file for build_labels
    df.to_parquet(today_features_path, compression='zstd', index=False)
    print(f"✓ Saved features to {today_features_path}")
    
    # Step 2: Build labels (production-only mode, no feature computation)
    # Note: build_labels_final.py already has --production-only flag that filters to latest date
    build_labels_cmd = (
        f"python3 scripts/build_labels_final.py "
        f"--input {today_features_path} "
        f"--output {today_features_path} "
        f"--earnings-file {EARNINGS_FILE} "
        f"--production-only"
    )
    
    run_command(build_labels_cmd, "Step 2: Building labels (production mode)")
    
    # Step 3: Apply ranker to generate predictions
    apply_ranker_cmd = (
        f"python3 scripts/apply_ranker.py "
        f"--dataset {today_features_path} "
        f"--model-dir {model_dir} "
        f"--out-dir {output_dir}"
    )
    
    run_command(apply_ranker_cmd, "Step 3: Applying ranker")
    
    # Step 4: Verify predictions were created
    print(f"\n{'='*60}")
    print("Step 4: Verifying predictions")
    print(f"{'='*60}")
    
    # Check for predictions.csv (default name from apply_ranker)
    default_predictions = output_dir / "predictions.csv"
    
    if default_predictions.exists():
        # Rename to dated version
        pred_df = pd.read_csv(default_predictions)
        pred_df.to_csv(predictions_path, index=False)
        print(f"✓ Predictions saved: {predictions_path}")
        print(f"  Rows: {len(pred_df)}")
        print(f"  Columns: {list(pred_df.columns)}")
        
        # Show top 10
        if 'pred' in pred_df.columns:
            top10 = pred_df.nlargest(10, 'pred')
            print(f"\nTop 10 predictions:")
            print(top10[['symbol', 'pred']].to_string(index=False))
    else:
        print(f"❌ Predictions file not found: {default_predictions}")
        sys.exit(1)
    
    # Final summary
    print(f"\n{'='*60}")
    print("✅ Production Inference Complete")
    print(f"{'='*60}")
    print(f"Snapshot date: {snapshot_date}")
    print(f"Predictions: {predictions_path}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    return predictions_path


def parse_args():
    parser = argparse.ArgumentParser(description='Fast production inference using snapshots')
    parser.add_argument('--snapshot-dir', type=str, default=None,
                       help='Specific snapshot directory (default: latest)')
    parser.add_argument('--snapshots-root', type=str, default=str(SNAPSHOTS_DIR),
                       help='Root directory containing snapshots')
    parser.add_argument('--output-dir', type=str, default=str(DATASETS_DIR),
                       help='Output directory for predictions')
    parser.add_argument('--model-dir', type=str, default=str(MODEL_DIR),
                       help='Directory containing trained model')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine snapshot directory
    if args.snapshot_dir:
        snapshot_dir = Path(args.snapshot_dir)
    else:
        snapshots_root = Path(args.snapshots_root)
        if not snapshots_root.exists():
            print(f"❌ Snapshots directory not found: {snapshots_root}")
            print(f"   Run: python scripts/create_snapshot.py")
            return 1
        
        try:
            snapshot_dir = find_latest_snapshot(snapshots_root)
            print(f"Using latest snapshot: {snapshot_dir.name}\n")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return 1
    
    if not snapshot_dir.exists():
        print(f"❌ Snapshot directory not found: {snapshot_dir}")
        return 1
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return 1
    
    # Run inference
    try:
        predictions_path = production_inference(snapshot_dir, output_dir, model_dir)
        return 0
    except Exception as e:
        print(f"\n❌ Production inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
