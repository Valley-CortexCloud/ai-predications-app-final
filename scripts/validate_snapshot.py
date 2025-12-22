#!/usr/bin/env python3
"""
Validate snapshot freshness, completeness, and data quality.
Outputs clear pass/fail summary for production readiness.

Usage:
    python scripts/validate_snapshot.py [--snapshot-dir DIR] [--fail-fast]
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
SNAPSHOTS_DIR = ROOT / "data" / "snapshots"

# Validation thresholds
MIN_SYMBOLS = 400
MIN_FEATURES = 100
MAX_AGE_DAYS = 5  # Changed from 7 to 5 per spec


class ValidationTracker:
    """Track validation checks and generate summary"""
    def __init__(self):
        self.checks = []
        self.passed = 0
        self.failed = 0
    
    def check(self, name: str, condition: bool, error_msg: str = ""):
        """Register a validation check"""
        self.checks.append({
            'name': name,
            'passed': condition,
            'error': error_msg if not condition else None
        })
        if condition:
            self.passed += 1
            print(f"  ✓ {name}")
        else:
            self.failed += 1
            print(f"  ✗ {name}: {error_msg}")
    
    def summary(self) -> bool:
        """Print summary and return overall pass/fail"""
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        if self.failed == 0:
            print(f"✅ {self.passed}/{total} CHECKS PASSED - DATA VALIDATED!")
            print(f"{'='*60}")
            return True
        else:
            print(f"❌ {self.failed}/{total} CHECKS FAILED")
            print(f"{'='*60}")
            print(f"\nFailed checks:")
            for check in self.checks:
                if not check['passed']:
                    print(f"  • {check['name']}: {check['error']}")
            return False


def find_latest_snapshot(snapshots_dir: Path) -> Path:
    """Find the most recent snapshot directory"""
    snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
    
    if not snapshot_dirs:
        raise FileNotFoundError(f"No snapshot directories found in {snapshots_dir}")
    
    # Sort by directory name (YYYY-MM-DD format sorts correctly)
    latest = sorted(snapshot_dirs)[-1]
    return latest


def validate_snapshot(snapshot_dir: Path) -> bool:
    """
    Comprehensive snapshot validation with clear reporting.
    Returns True if all checks pass, False otherwise.
    """
    tracker = ValidationTracker()
    
    print(f"\n{'='*60}")
    print(f"SNAPSHOT VALIDATION: {snapshot_dir.name}")
    print(f"{'='*60}\n")
    
    # ========================================================================
    # SECTION 1: FILE EXISTENCE
    # ========================================================================
    print("📁 FILE EXISTENCE CHECKS")
    
    feature_matrix_path = snapshot_dir / "feature_matrix.parquet"
    tracker.check(
        "Feature matrix exists",
        feature_matrix_path.exists(),
        "feature_matrix.parquet not found"
    )
    
    universe_path = snapshot_dir / "universe.csv"
    tracker.check(
        "Universe file exists",
        universe_path.exists(),
        "universe.csv not found"
    )
    
    metadata_path = snapshot_dir / "metadata.json"
    tracker.check(
        "Metadata file exists",
        metadata_path.exists(),
        "metadata.json not found"
    )
    
    if not all([feature_matrix_path.exists(), universe_path.exists(), metadata_path.exists()]):
        return tracker.summary()
    
    # ========================================================================
    # SECTION 2: DATA FRESHNESS
    # ========================================================================
    print("\n🕐 DATA FRESHNESS CHECKS")
    
    metadata = json.load(open(metadata_path))
    
    # Check snapshot creation age
    created = pd.to_datetime(metadata['created_at'])
    age_hours = (pd.Timestamp.now() - created).total_seconds() / 3600
    tracker.check(
        "Snapshot age < 48 hours",
        age_hours <= 48,
        f"Snapshot is {age_hours:.1f} hours old (max: 48)"
    )
    
    # Check data date (should be T-1 or T-2 for weekend)
    data_date = pd.to_datetime(metadata.get('data_date') or metadata.get('snapshot_date'))
    now = pd.Timestamp.now().normalize()
    days_old = (now - data_date).days
    
    # Allow up to 5 days (for long weekends/holidays)
    tracker.check(
        "Data age < 5 days",
        days_old <= 5,
        f"Data is {days_old} days old (max: 5)"
    )
    
    # ========================================================================
    # SECTION 3: DATA COMPLETENESS
    # ========================================================================
    print("\n📊 DATA COMPLETENESS CHECKS")
    
    df = pd.read_parquet(feature_matrix_path)
    universe = pd.read_csv(universe_path)
    
    # Symbol count
    tracker.check(
        "Universe size >= 400 symbols",
        len(df) >= 400,
        f"Only {len(df)} symbols (expected >= 400)"
    )
    
    # Feature count
    tracker.check(
        "Feature count >= 100",
        len(df.columns) >= 100,
        f"Only {len(df.columns)} features (expected >= 100)"
    )
    
    # Universe consistency
    tracker.check(
        "Feature matrix matches universe",
        len(df) == len(universe),
        f"Feature matrix has {len(df)} rows but universe has {len(universe)}"
    )
    
    # ========================================================================
    # SECTION 4: FEATURE QUALITY
    # ========================================================================
    print("\n🔬 FEATURE QUALITY CHECKS")
    
    # NaN check
    nan_pct = df.isna().sum() / len(df)
    max_nan = nan_pct.max()
    tracker.check(
        "No features with >50% NaN",
        max_nan <= 0.5,
        f"Max NaN: {max_nan:.1%} (threshold: 50%)"
    )
    
    # Inf check
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_cols = [col for col in numeric_cols if np.isinf(df[col]).any()]
    tracker.check(
        "No infinite values",
        len(inf_cols) == 0,
        f"{len(inf_cols)} features contain Inf: {inf_cols[:3]}"
    )
    
    # Constant features check
    constant_cols = [col for col in numeric_cols if df[col].nunique() <= 1]
    tracker.check(
        "No constant features",
        len(constant_cols) == 0,
        f"{len(constant_cols)} features are constant: {constant_cols[:3]}"
    )
    
    # ========================================================================
    # SECTION 5: CRITICAL FEATURES
    # ========================================================================
    print("\n⚡ CRITICAL FEATURE CHECKS")
    
    # Check for essential features
    required_features = ['Close', 'Volume', 'Adj Close']
    missing_required = [f for f in required_features if f not in df.columns]
    tracker.check(
        "OHLCV features present",
        len(missing_required) == 0,
        f"Missing: {missing_required}"
    )
    
    # Validate beta range (if exists)
    if 'feat_beta_spy_126' in df.columns:
        beta = df['feat_beta_spy_126']
        tracker.check(
            "Beta values in range [-5, 5]",
            (beta.abs() <= 5).all(),
            f"Beta range: [{beta.min():.2f}, {beta.max():.2f}]"
        )
    
    # Validate volatility range (if exists)
    if 'volatility_20' in df.columns:
        vol = df['volatility_20']
        tracker.check(
            "Volatility values in range [0, 10]",
            ((vol >= 0) & (vol <= 10)).all(),
            f"Vol range: [{vol.min():.3f}, {vol.max():.3f}]"
        )
    
    # ========================================================================
    # SECTION 6: METADATA INTEGRITY
    # ========================================================================
    print("\n🔐 METADATA INTEGRITY CHECKS")
    
    tracker.check(
        "Git commit hash present",
        'git_commit' in metadata and len(metadata.get('git_commit', '')) > 0,
        "Missing git commit hash"
    )
    
    tracker.check(
        "Feature count matches metadata",
        metadata.get('feature_count', 0) == len(df.columns),
        f"Metadata says {metadata.get('feature_count')} features, actual: {len(df.columns)}"
    )
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    passed = tracker.summary()
    
    if passed:
        print(f"\n🚀 SNAPSHOT READY FOR PRODUCTION INFERENCE")
        print(f"   Symbols: {len(df)}")
        print(f"   Features: {len(df.columns)}")
        print(f"   Data date: {data_date.date()}")
        print(f"   Age: {days_old} days")
    else:
        print(f"\n⛔ SNAPSHOT VALIDATION FAILED - DO NOT USE FOR PRODUCTION")
    
    return passed


def parse_args():
    parser = argparse.ArgumentParser(description='Validate snapshot for production use')
    parser.add_argument('--snapshot-dir', type=str, default=None,
                       help='Specific snapshot directory to validate (default: latest)')
    parser.add_argument('--snapshots-root', type=str, default=str(SNAPSHOTS_DIR),
                       help='Root directory containing snapshots')
    parser.add_argument('--fail-fast', action='store_true',
                       help='Exit on first failure (default: run all checks)')
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
    
    if not snapshot_dir.exists():
        print(f"❌ Snapshot directory not found: {snapshot_dir}")
        return 1
    
    # Run validation
    passed = validate_snapshot(snapshot_dir)
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
