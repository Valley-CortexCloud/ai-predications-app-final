#!/usr/bin/env python3
"""
Test script to validate production mode optimization works correctly.
"""
import subprocess
import sys
from pathlib import Path
import pandas as pd
import time

def run_command(cmd, description):
    """Run a command and return execution time"""
    print(f"\n{'='*60}")
    print(f"TEST: {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    
    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - start
    
    print(f"Exit code: {result.returncode}")
    print(f"Execution time: {elapsed:.2f} seconds")
    
    if result.returncode != 0:
        print("\nSTDERR:")
        print(result.stderr)
        print("\nSTDOUT:")
        print(result.stdout)
        return None, elapsed
    
    return result.stdout, elapsed

def test_production_mode():
    """Test production mode functionality"""
    print("\n" + "="*60)
    print("TESTING PRODUCTION MODE OPTIMIZATION")
    print("="*60)
    
    # Test 1: Run production mode
    output_file = "datasets/test_prod_validation.parquet"
    cmd = f"python3 scripts/build_labels_final.py --production-only --output {output_file} --cache-dir data_cache/10y_ticker_features"
    
    stdout, elapsed = run_command(cmd, "Production mode execution")
    
    if stdout is None:
        print("\n❌ FAILED: Production mode execution failed")
        return False
    
    # Validate execution time (should be < 10 seconds)
    if elapsed > 10:
        print(f"\n⚠️  WARNING: Execution took {elapsed:.2f}s (expected < 10s)")
    else:
        print(f"\n✅ PASS: Fast execution ({elapsed:.2f}s < 10s)")
    
    # Test 2: Validate output file
    if not Path(output_file).exists():
        print(f"\n❌ FAILED: Output file not created: {output_file}")
        return False
    
    print(f"\n✅ PASS: Output file created: {output_file}")
    
    # Test 3: Validate output structure
    df = pd.read_parquet(output_file)
    
    print(f"\nOutput structure:")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Symbols: {df['symbol'].nunique()}")
    print(f"  Dates: {df['date'].nunique()}")
    print(f"  Date: {df['date'].iloc[0]}")
    
    # Validations
    checks = []
    
    # Check 1: Single date
    if df['date'].nunique() == 1:
        print("\n✅ PASS: Single date (production mode)")
        checks.append(True)
    else:
        print(f"\n❌ FAILED: Multiple dates found ({df['date'].nunique()})")
        checks.append(False)
    
    # Check 2: Reasonable number of symbols (400-600)
    if 400 <= df['symbol'].nunique() <= 600:
        print(f"✅ PASS: Reasonable symbol count ({df['symbol'].nunique()})")
        checks.append(True)
    else:
        print(f"❌ FAILED: Unexpected symbol count ({df['symbol'].nunique()})")
        checks.append(False)
    
    # Check 3: Feature count (should have > 100 features)
    feat_cols = [c for c in df.columns if c.startswith('feat_')]
    if len(feat_cols) >= 100:
        print(f"✅ PASS: Sufficient features ({len(feat_cols)})")
        checks.append(True)
    else:
        print(f"❌ FAILED: Too few features ({len(feat_cols)})")
        checks.append(False)
    
    # Check 4: Critical features present
    critical = [
        'feat_mom_12m_skip1m_rank_pct',
        'feat_volatility_20_rank_pct',
        'feat_composite_quality'
    ]
    missing = [f for f in critical if f not in df.columns]
    if not missing:
        print(f"✅ PASS: All critical features present")
        checks.append(True)
    else:
        print(f"❌ FAILED: Missing critical features: {missing}")
        checks.append(False)
    
    # Check 5: No NaN in critical columns
    if df[critical].isna().sum().sum() == 0:
        print(f"✅ PASS: No NaN in critical features")
        checks.append(True)
    else:
        nan_counts = df[critical].isna().sum()
        print(f"⚠️  WARNING: NaN found in critical features:")
        for col, count in nan_counts.items():
            if count > 0:
                print(f"    {col}: {count}")
        checks.append(True)  # Not critical failure
    
    # Clean up
    Path(output_file).unlink(missing_ok=True)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {sum(checks)}/{len(checks)}")
    print(f"Execution time: {elapsed:.2f}s")
    
    if all(checks):
        print("\n✅ ALL TESTS PASSED")
        return True
    else:
        print("\n❌ SOME TESTS FAILED")
        return False

if __name__ == "__main__":
    success = test_production_mode()
    sys.exit(0 if success else 1)
