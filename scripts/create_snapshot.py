#!/usr/bin/env python3
"""
Create point-in-time snapshot of all ticker features for production inference.

Purpose:
- Load latest date from all *_features_enhanced.parquet files
- Create single feature matrix parquet (all tickers, one date)
- Save to data/snapshots/YYYY-MM-DD/ with metadata

Usage:
    python scripts/create_snapshot.py [--features-dir DIR] [--output-dir DIR]
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent
TICKER_CACHE_DIR = ROOT / "data_cache" / "10y_ticker_features"
SNAPSHOTS_DIR = ROOT / "data" / "snapshots"


def get_git_commit_hash() -> str:
    """Get current git commit hash"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"Warning: Could not get git commit hash: {e}")
        return "unknown"


def compute_file_checksum(file_path: Path) -> str:
    """Compute SHA256 checksum of file"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def normalize_df_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe index to tz-naive dates"""
    if 'Date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).set_index('Date')
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')
    
    if df.index.tz is not None:
        try:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        except Exception:
            df.index = df.index.tz_localize(None)
    
    df.index = df.index.normalize()
    df.index.name = 'Date'
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def find_enhanced_feature_files(features_dir: Path) -> List[Path]:
    """Find all *_features_enhanced.parquet files"""
    files = list(features_dir.glob("*_features_enhanced.parquet"))
    return sorted(files)


def load_ticker_latest_date(file_path: Path, verbose: bool = False) -> Optional[pd.DataFrame]:
    """Load ticker enhanced features and return latest date row"""
    try:
        ticker = file_path.stem.split('_')[0]
        df = pd.read_parquet(file_path)
        df = normalize_df_index(df)
        
        if len(df) == 0:
            if verbose:
                print(f"  ⚠️  {ticker}: Empty file")
            return None
        
        # Get latest date
        latest_date = df.index.max()
        latest_row = df.loc[[latest_date]].copy()
        
        # Add ticker symbol as column
        latest_row['symbol'] = ticker
        
        if verbose:
            print(f"  ✓ {ticker}: {latest_date.date()} ({len(df.columns)} features)")
        
        return latest_row
        
    except Exception as e:
        if verbose:
            print(f"  ✗ {file_path.name}: Error - {e}")
        return None


def create_snapshot(features_dir: Path, output_dir: Path, verbose: bool = True) -> Optional[Path]:
    """
    Create snapshot of latest features for all tickers.
    
    Returns:
        Path to snapshot directory, or None on failure
    """
    print(f"{'='*60}")
    print("Creating Feature Snapshot")
    print(f"{'='*60}")
    
    # Find all enhanced feature files
    files = find_enhanced_feature_files(features_dir)
    
    if not files:
        print(f"❌ No enhanced feature files found in {features_dir}")
        return None
    
    print(f"Found {len(files)} enhanced feature files")
    
    # Load latest date from each ticker
    print("\nLoading latest date from each ticker...")
    ticker_rows = []
    
    for i, file_path in enumerate(files, 1):
        if verbose and i % 50 == 0:
            print(f"  [{i}/{len(files)}] Processing...")
        
        row = load_ticker_latest_date(file_path, verbose=False)
        if row is not None:
            ticker_rows.append(row)
    
    if not ticker_rows:
        print(f"❌ No valid ticker data loaded")
        return None
    
    print(f"✓ Loaded {len(ticker_rows)} tickers")
    
    # Combine all tickers into single dataframe
    print("\nCombining tickers...")
    feature_matrix = pd.concat(ticker_rows, axis=0, ignore_index=True)
    
    # Get snapshot date (should be consistent across all tickers)
    dates = pd.to_datetime([df.index[0] for df in ticker_rows if len(df) > 0])
    snapshot_date = dates[0] if len(dates) > 0 else pd.Timestamp.now()
    snapshot_date_str = snapshot_date.strftime("%Y-%m-%d")
    
    # Check date consistency
    unique_dates = pd.Series(dates).unique()
    if len(unique_dates) > 1:
        print(f"⚠️  WARNING: Multiple dates in snapshot: {[d.strftime('%Y-%m-%d') for d in unique_dates[:5]]}")
        print(f"   Using most common date: {snapshot_date_str}")
    
    print(f"\n📅 Snapshot Date: {snapshot_date_str}")
    print(f"   Symbols: {len(feature_matrix)}")
    print(f"   Features: {len(feature_matrix.columns)}")
    
    # Create snapshot directory
    snapshot_dir = output_dir / snapshot_date_str
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    # Save feature matrix
    feature_matrix_path = snapshot_dir / "feature_matrix.parquet"
    feature_matrix.to_parquet(feature_matrix_path, compression='zstd', index=False)
    print(f"\n✓ Saved feature matrix: {feature_matrix_path}")
    
    # Save universe (list of symbols)
    universe_path = snapshot_dir / "universe.csv"
    universe_df = pd.DataFrame({'symbol': sorted(feature_matrix['symbol'].unique())})
    universe_df.to_csv(universe_path, index=False)
    print(f"✓ Saved universe: {universe_path} ({len(universe_df)} symbols)")
    
    # Check for all-NaN columns
    nan_cols = []
    for col in feature_matrix.columns:
        if feature_matrix[col].isna().all():
            nan_cols.append(col)
    
    if nan_cols:
        print(f"\n⚠️  WARNING: {len(nan_cols)} all-NaN columns: {nan_cols[:10]}")
    
    # Create metadata
    metadata = {
        "snapshot_date": snapshot_date_str,
        "created_at": datetime.now().isoformat(),
        "git_commit": get_git_commit_hash(),
        "symbol_count": len(universe_df),
        "feature_count": len(feature_matrix.columns),
        "all_nan_columns": nan_cols,
        "file_checksums": {
            "feature_matrix.parquet": compute_file_checksum(feature_matrix_path),
            "universe.csv": compute_file_checksum(universe_path)
        }
    }
    
    # Save metadata
    metadata_path = snapshot_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Snapshot created successfully!")
    print(f"{'='*60}")
    print(f"Location: {snapshot_dir}")
    print(f"Date: {snapshot_date_str}")
    print(f"Symbols: {len(universe_df)}")
    print(f"Features: {len(feature_matrix.columns)}")
    print(f"Git commit: {metadata['git_commit'][:8]}")
    print(f"{'='*60}")
    
    return snapshot_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Create point-in-time feature snapshot')
    parser.add_argument('--features-dir', type=str, default=str(TICKER_CACHE_DIR),
                       help='Directory containing enhanced feature files')
    parser.add_argument('--output-dir', type=str, default=str(SNAPSHOTS_DIR),
                       help='Output directory for snapshots')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    return parser.parse_args()


def main():
    args = parse_args()
    
    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    
    if not features_dir.exists():
        print(f"❌ Features directory not found: {features_dir}")
        return 1
    
    snapshot_dir = create_snapshot(features_dir, output_dir, verbose=args.verbose)
    
    if snapshot_dir is None:
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
