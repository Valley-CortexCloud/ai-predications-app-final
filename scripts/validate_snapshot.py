#!/usr/bin/env python3
"""
Validate snapshot is fresh and complete.

Checks:
- Snapshot exists
- Data age < 7 days
- Symbol count >= 400
- Feature count >= 100
- No all-NaN columns
- Date is T-1 (yesterday) or recent

Usage:
    python scripts/validate_snapshot.py [--snapshot-dir DIR] [--max-age-days N]
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
SNAPSHOTS_DIR = ROOT / "data" / "snapshots"

# Validation thresholds
MIN_SYMBOLS = 400
MIN_FEATURES = 100
MAX_AGE_DAYS = 7


def find_latest_snapshot(snapshots_dir: Path) -> Path:
    """Find the most recent snapshot directory"""
    snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
    
    if not snapshot_dirs:
        raise FileNotFoundError(f"No snapshot directories found in {snapshots_dir}")
    
    # Sort by directory name (YYYY-MM-DD format sorts correctly)
    latest = sorted(snapshot_dirs)[-1]
    return latest


def validate_snapshot(snapshot_dir: Path, max_age_days: int = MAX_AGE_DAYS) -> bool:
    """
    Validate snapshot is fresh and complete.
    
    Returns:
        True if valid, False otherwise
    """
    print(f"{'='*60}")
    print(f"Validating Snapshot: {snapshot_dir.name}")
    print(f"{'='*60}")
    
    all_checks_passed = True
    
    # Check 1: Snapshot directory exists
    if not snapshot_dir.exists():
        print(f"❌ Snapshot directory does not exist: {snapshot_dir}")
        return False
    
    print(f"✓ Snapshot directory exists")
    
    # Check 2: Required files exist
    required_files = ['feature_matrix.parquet', 'universe.csv', 'metadata.json']
    for filename in required_files:
        file_path = snapshot_dir / filename
        if not file_path.exists():
            print(f"❌ Required file missing: {filename}")
            all_checks_passed = False
        else:
            print(f"✓ {filename} exists")
    
    if not all_checks_passed:
        return False
    
    # Load metadata
    metadata_path = snapshot_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\n{'='*60}")
    print("Metadata:")
    print(f"{'='*60}")
    print(f"Snapshot date: {metadata['snapshot_date']}")
    print(f"Created at: {metadata['created_at']}")
    print(f"Git commit: {metadata['git_commit'][:8] if metadata['git_commit'] != 'unknown' else 'unknown'}")
    print(f"Symbol count: {metadata['symbol_count']}")
    print(f"Feature count: {metadata['feature_count']}")
    
    # Check 3: Data age
    snapshot_date = pd.to_datetime(metadata['snapshot_date'])
    current_date = pd.Timestamp.now().normalize()
    age_days = (current_date - snapshot_date).days
    
    print(f"\n{'='*60}")
    print("Freshness Check:")
    print(f"{'='*60}")
    print(f"Snapshot date: {snapshot_date.date()}")
    print(f"Current date: {current_date.date()}")
    print(f"Age: {age_days} days")
    
    if age_days > max_age_days:
        print(f"❌ Data is too old ({age_days} days > {max_age_days} days threshold)")
        all_checks_passed = False
    elif age_days < -1:
        print(f"❌ Data is in the future ({age_days} days)")
        all_checks_passed = False
    else:
        print(f"✓ Data is fresh (within {max_age_days} days)")
    
    # Check 4: Symbol count
    print(f"\n{'='*60}")
    print("Completeness Check:")
    print(f"{'='*60}")
    
    symbol_count = metadata['symbol_count']
    print(f"Symbol count: {symbol_count}")
    
    if symbol_count < MIN_SYMBOLS:
        print(f"❌ Too few symbols ({symbol_count} < {MIN_SYMBOLS} threshold)")
        all_checks_passed = False
    else:
        print(f"✓ Sufficient symbols ({symbol_count} >= {MIN_SYMBOLS})")
    
    # Check 5: Feature count
    feature_count = metadata['feature_count']
    print(f"Feature count: {feature_count}")
    
    if feature_count < MIN_FEATURES:
        print(f"❌ Too few features ({feature_count} < {MIN_FEATURES} threshold)")
        all_checks_passed = False
    else:
        print(f"✓ Sufficient features ({feature_count} >= {MIN_FEATURES})")
    
    # Check 6: All-NaN columns
    all_nan_columns = metadata.get('all_nan_columns', [])
    print(f"All-NaN columns: {len(all_nan_columns)}")
    
    if all_nan_columns:
        print(f"⚠️  WARNING: {len(all_nan_columns)} all-NaN columns found:")
        for col in all_nan_columns[:10]:
            print(f"   - {col}")
        if len(all_nan_columns) > 10:
            print(f"   ... and {len(all_nan_columns) - 10} more")
    else:
        print(f"✓ No all-NaN columns")
    
    # Check 7: Load and validate feature matrix
    print(f"\n{'='*60}")
    print("Data Integrity Check:")
    print(f"{'='*60}")
    
    try:
        feature_matrix_path = snapshot_dir / "feature_matrix.parquet"
        df = pd.read_parquet(feature_matrix_path)
        
        print(f"Feature matrix shape: {df.shape}")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        
        # Check for required columns
        if 'symbol' not in df.columns:
            print(f"❌ Missing 'symbol' column")
            all_checks_passed = False
        else:
            print(f"✓ 'symbol' column present")
            print(f"  Unique symbols: {df['symbol'].nunique()}")
        
        # Check for critical features
        critical_features = ['Close', 'Adj Close', 'Volume']
        missing_critical = [f for f in critical_features if f not in df.columns]
        
        if missing_critical:
            print(f"⚠️  WARNING: Missing critical features: {missing_critical}")
        else:
            print(f"✓ All critical features present")
        
        # Check for NaN percentage (sample first 1000 rows for efficiency)
        sample_size = min(1000, len(df))
        df_sample = df.head(sample_size)
        nan_pct = (df_sample.isna().sum().sum() / (len(df_sample) * len(df_sample.columns))) * 100
        print(f"Overall NaN percentage (sample of {sample_size} rows): {nan_pct:.2f}%")
        
        if nan_pct > 50:
            print(f"⚠️  WARNING: High NaN percentage ({nan_pct:.2f}%)")
        else:
            print(f"✓ Acceptable NaN percentage")
        
    except Exception as e:
        print(f"❌ Error loading feature matrix: {e}")
        all_checks_passed = False
    
    # Final verdict
    print(f"\n{'='*60}")
    if all_checks_passed:
        print("✅ VALIDATION PASSED")
        print(f"{'='*60}")
        print(f"Snapshot {snapshot_dir.name} is valid and ready for production use")
    else:
        print("❌ VALIDATION FAILED")
        print(f"{'='*60}")
        print(f"Snapshot {snapshot_dir.name} has issues that need to be resolved")
    print(f"{'='*60}")
    
    return all_checks_passed


def parse_args():
    parser = argparse.ArgumentParser(description='Validate feature snapshot')
    parser.add_argument('--snapshot-dir', type=str, default=None,
                       help='Specific snapshot directory to validate (default: latest)')
    parser.add_argument('--snapshots-root', type=str, default=str(SNAPSHOTS_DIR),
                       help='Root directory containing snapshots')
    parser.add_argument('--max-age-days', type=int, default=MAX_AGE_DAYS,
                       help=f'Maximum age in days (default: {MAX_AGE_DAYS})')
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
            return 1
        
        try:
            snapshot_dir = find_latest_snapshot(snapshots_root)
            print(f"Using latest snapshot: {snapshot_dir.name}\n")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return 1
    
    # Validate
    is_valid = validate_snapshot(snapshot_dir, max_age_days=args.max_age_days)
    
    return 0 if is_valid else 1


if __name__ == '__main__':
    sys.exit(main())
