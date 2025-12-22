#!/usr/bin/env python3
"""
Maintenance utilities for data pipeline.

Usage:
    # Clean old snapshots (keep last 12)
    python scripts/maintenance.py --clean-snapshots --keep 12
    
    # Show disk usage
    python scripts/maintenance.py --disk-usage
    
    # Verify data integrity
    python scripts/maintenance.py --verify-integrity
"""

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).parent.parent
SNAPSHOTS_DIR = ROOT / "data" / "snapshots"
DATA_CACHE_DIR = ROOT / "data_cache"


def clean_old_snapshots(keep_count: int = 12, dry_run: bool = False):
    """
    Remove old snapshots, keeping only the most recent N.
    
    Args:
        keep_count: Number of snapshots to keep (default: 12 = 3 months)
        dry_run: If True, only show what would be deleted
    """
    print(f"{'='*60}")
    print(f"Cleaning Old Snapshots (keep {keep_count} most recent)")
    print(f"{'='*60}")
    
    if not SNAPSHOTS_DIR.exists():
        print(f"❌ Snapshots directory not found: {SNAPSHOTS_DIR}")
        return
    
    # Find all snapshot directories
    snapshot_dirs = sorted([d for d in SNAPSHOTS_DIR.iterdir() if d.is_dir()])
    
    print(f"Found {len(snapshot_dirs)} snapshots")
    
    if len(snapshot_dirs) <= keep_count:
        print(f"✓ No cleanup needed (have {len(snapshot_dirs)}, keeping {keep_count})")
        return
    
    # Determine which to delete
    to_delete = snapshot_dirs[:-keep_count]
    to_keep = snapshot_dirs[-keep_count:]
    
    print(f"\nKeeping {len(to_keep)} snapshots:")
    for d in to_keep:
        print(f"  ✓ {d.name}")
    
    print(f"\nDeleting {len(to_delete)} old snapshots:")
    for d in to_delete:
        print(f"  × {d.name}")
    
    if dry_run:
        print(f"\n(Dry run - no files deleted)")
        return
    
    # Delete old snapshots
    import shutil
    for d in to_delete:
        try:
            shutil.rmtree(d)
            print(f"  ✓ Deleted {d.name}")
        except Exception as e:
            print(f"  ✗ Error deleting {d.name}: {e}")
    
    print(f"\n✅ Cleanup complete")


def show_disk_usage():
    """Show disk usage breakdown"""
    print(f"{'='*60}")
    print("Disk Usage")
    print(f"{'='*60}")
    
    import subprocess
    
    dirs = [
        ("Snapshots", SNAPSHOTS_DIR),
        ("Data Cache", DATA_CACHE_DIR),
        ("Total Repo", ROOT)
    ]
    
    for name, path in dirs:
        if path.exists():
            try:
                result = subprocess.run(
                    ["du", "-sh", str(path)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                size = result.stdout.split()[0]
                print(f"{name:20s}: {size}")
            except Exception as e:
                print(f"{name:20s}: Error - {e}")
        else:
            print(f"{name:20s}: Not found")
    
    print(f"{'='*60}")


def verify_data_integrity():
    """Verify data integrity (no duplicates, consistent columns)"""
    print(f"{'='*60}")
    print("Data Integrity Check")
    print(f"{'='*60}")
    
    # Check snapshots
    if SNAPSHOTS_DIR.exists():
        snapshot_dirs = sorted([d for d in SNAPSHOTS_DIR.iterdir() if d.is_dir()])
        print(f"\n📸 Snapshots: {len(snapshot_dirs)}")
        
        for snapshot_dir in snapshot_dirs[-3:]:  # Check last 3
            metadata_path = snapshot_dir / "metadata.json"
            feature_matrix_path = snapshot_dir / "feature_matrix.parquet"
            
            if metadata_path.exists() and feature_matrix_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                df = pd.read_parquet(feature_matrix_path)
                
                print(f"\n  {snapshot_dir.name}:")
                print(f"    Symbols: {len(df)} (metadata: {metadata['symbol_count']})")
                print(f"    Features: {len(df.columns)} (metadata: {metadata['feature_count']})")
                
                if len(df) != metadata['symbol_count']:
                    print(f"    ⚠️  WARNING: Symbol count mismatch!")
                else:
                    print(f"    ✓ Counts match")
    
    # Check data cache
    if DATA_CACHE_DIR.exists():
        ticker_cache = DATA_CACHE_DIR / "10y_ticker_features"
        if ticker_cache.exists():
            files = list(ticker_cache.glob("*_2y_adj.parquet"))
            print(f"\n💾 Raw ticker files: {len(files)}")
            
            # Check for duplicates in a sample
            if files:
                sample = files[:5]
                print(f"\n  Checking sample of {len(sample)} files for duplicates:")
                
                for file in sample:
                    try:
                        df = pd.read_parquet(file)
                        
                        # Normalize index
                        if not isinstance(df.index, pd.DatetimeIndex):
                            df.index = pd.to_datetime(df.index, errors='coerce')
                        
                        dup_count = df.index.duplicated().sum()
                        
                        if dup_count > 0:
                            print(f"    ✗ {file.name}: {dup_count} duplicates!")
                        else:
                            print(f"    ✓ {file.name}: no duplicates ({len(df)} rows)")
                    except Exception as e:
                        print(f"    ✗ {file.name}: Error - {e}")
    
    print(f"\n{'='*60}")
    print("✅ Integrity check complete")
    print(f"{'='*60}")


def parse_args():
    parser = argparse.ArgumentParser(description='Maintenance utilities')
    parser.add_argument('--clean-snapshots', action='store_true',
                       help='Remove old snapshots')
    parser.add_argument('--keep', type=int, default=12,
                       help='Number of snapshots to keep (default: 12)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run (show what would be deleted)')
    parser.add_argument('--disk-usage', action='store_true',
                       help='Show disk usage')
    parser.add_argument('--verify-integrity', action='store_true',
                       help='Verify data integrity')
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.clean_snapshots:
        clean_old_snapshots(keep_count=args.keep, dry_run=args.dry_run)
    
    if args.disk_usage:
        show_disk_usage()
    
    if args.verify_integrity:
        verify_data_integrity()
    
    if not any([args.clean_snapshots, args.disk_usage, args.verify_integrity]):
        print("No action specified. Use --help for options.")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
