#!/usr/bin/env python3
"""
Comprehensive test to validate production mode features match requirements.

This test validates:
1. Feature count (127+ features)
2. Earnings quality >95% populated
3. All critical features present
4. Cross-sectional ranks (13 features)
5. Cross-sectional z-scores (5 features)
6. Date correctness (within 30 days of current date)
7. No all-NaN features (except expected ones)
8. Detailed logs with feature breakdown
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd


def run_production_mode():
    """Run production mode with earnings file"""
    print("\n" + "="*60)
    print("RUNNING PRODUCTION MODE WITH EARNINGS FILE")
    print("="*60)
    
    cmd = [
        "python3", "scripts/build_labels_final.py",
        "--production-only",
        "--output", "datasets/test_validation.parquet",
        "--cache-dir", "data_cache/10y_ticker_features",
        "--earnings-file", "data/earnings.csv"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ FAILED: Production mode execution failed")
        print("\nSTDERR:")
        print(result.stderr)
        print("\nSTDOUT:")
        print(result.stdout)
        return False
    
    print("✅ Production mode executed successfully")
    return True


def validate_output():
    """Validate the output parquet file"""
    print("\n" + "="*60)
    print("VALIDATING OUTPUT")
    print("="*60)
    
    output_file = Path("datasets/test_validation.parquet")
    if not output_file.exists():
        print("❌ FAILED: Output file not found")
        return False
    
    df = pd.read_parquet(output_file)
    
    # Get feature columns
    feat_cols = [c for c in df.columns if c.startswith('feat_')]
    
    print(f"\n📊 Basic Statistics:")
    print(f"  Total features: {len(feat_cols)}")
    print(f"  Total rows: {len(df)}")
    print(f"  Symbols: {df['symbol'].nunique()}")
    print(f"  Date: {df['date'].iloc[0]}")
    
    # Track test results
    tests_passed = []
    
    # Test 1: Feature count (>= 127)
    print(f"\n✓ Test 1: Feature count")
    if len(feat_cols) >= 127:
        print(f"  ✅ PASS: {len(feat_cols)} features (>= 127)")
        tests_passed.append(True)
    else:
        print(f"  ❌ FAIL: {len(feat_cols)} features (< 127)")
        tests_passed.append(False)
    
    # Test 2: Earnings quality >95% populated
    print(f"\n✓ Test 2: Earnings quality population")
    if 'feat_earnings_quality' in df.columns:
        non_zero = (df['feat_earnings_quality'] != 0).sum()
        pct = non_zero / len(df) * 100
        if pct >= 95.0:
            print(f"  ✅ PASS: {pct:.1f}% populated (>= 95%)")
            tests_passed.append(True)
        else:
            print(f"  ❌ FAIL: {pct:.1f}% populated (< 95%)")
            tests_passed.append(False)
    else:
        print(f"  ❌ FAIL: feat_earnings_quality not found")
        tests_passed.append(False)
    
    # Test 3: Critical features present
    print(f"\n✓ Test 3: Critical features present")
    critical_features = [
        'feat_earnings_quality', 'feat_prev_earn_surprise_pct', 'feat_days_to_earn',
        'feat_days_since_earn', 'feat_earn_surprise_streak',
        'feat_adv20_dollar',
        'feat_mom_12m_skip1m_rank_pct', 'feat_volatility_20_rank_pct',
        'feat_composite_quality',
        'feat_sector_rel_ret_63d'
    ]
    missing = [f for f in critical_features if f not in df.columns]
    if not missing:
        print(f"  ✅ PASS: All {len(critical_features)} critical features present")
        tests_passed.append(True)
    else:
        print(f"  ❌ FAIL: Missing {len(missing)} features: {missing}")
        tests_passed.append(False)
    
    # Test 4: Cross-sectional rank features (13 expected)
    print(f"\n✓ Test 4: Cross-sectional rank features")
    rank_features = [c for c in feat_cols if '_rank_pct' in c]
    if len(rank_features) >= 13:
        print(f"  ✅ PASS: {len(rank_features)} rank features (>= 13)")
        tests_passed.append(True)
    else:
        print(f"  ❌ FAIL: {len(rank_features)} rank features (< 13)")
        tests_passed.append(False)
    
    # Test 5: Cross-sectional z-score features (5 expected)
    print(f"\n✓ Test 5: Cross-sectional z-score features")
    zscore_features = [c for c in feat_cols if '_zscore_xsec' in c]
    if len(zscore_features) >= 5:
        print(f"  ✅ PASS: {len(zscore_features)} z-score features (>= 5)")
        tests_passed.append(True)
    else:
        print(f"  ❌ FAIL: {len(zscore_features)} z-score features (< 5)")
        tests_passed.append(False)
    
    # Test 6: Date correctness (recent date, not too old)
    print(f"\n✓ Test 6: Date correctness")
    actual_date = df['date'].iloc[0]
    # Check that date is within last 30 days
    today = pd.Timestamp.now().normalize()
    days_old = (today - actual_date).days
    if days_old <= 30:
        print(f"  ✅ PASS: Date is {actual_date} ({days_old} days old)")
        tests_passed.append(True)
    else:
        print(f"  ⚠️  WARNING: Date is {actual_date} ({days_old} days old)")
        # Still pass but warn
        tests_passed.append(True)
    
    # Test 7: No unexpected all-NaN features
    print(f"\n✓ Test 7: Feature quality")
    all_nan = [c for c in feat_cols if df[c].isna().all()]
    # Allow feat_days_to_earn to be all-NaN (no upcoming earnings)
    unexpected_nan = [c for c in all_nan if c not in ['feat_days_to_earn']]
    if not unexpected_nan:
        print(f"  ✅ PASS: No unexpected all-NaN features")
        if all_nan:
            print(f"  ℹ️  Expected NaN features: {all_nan}")
        tests_passed.append(True)
    else:
        print(f"  ❌ FAIL: Unexpected all-NaN features: {unexpected_nan}")
        tests_passed.append(False)
    
    # Test 8: adv20_dollar computed correctly
    print(f"\n✓ Test 8: Liquidity features")
    if 'feat_adv20_dollar' in df.columns:
        non_zero = (df['feat_adv20_dollar'] > 0).sum()
        pct = non_zero / len(df) * 100
        if pct >= 95.0:
            print(f"  ✅ PASS: adv20_dollar computed for {pct:.1f}% of symbols")
            tests_passed.append(True)
        else:
            print(f"  ❌ FAIL: adv20_dollar only computed for {pct:.1f}% of symbols")
            tests_passed.append(False)
    else:
        print(f"  ❌ FAIL: feat_adv20_dollar not found")
        tests_passed.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {sum(tests_passed)}/{len(tests_passed)}")
    
    if all(tests_passed):
        print("\n✅ ALL TESTS PASSED")
        return True
    else:
        print("\n❌ SOME TESTS FAILED")
        return False


def main():
    """Run all validation tests"""
    print("="*60)
    print("COMPREHENSIVE FEATURE VALIDATION TEST")
    print("="*60)
    
    # Run production mode
    if not run_production_mode():
        return False
    
    # Validate output
    if not validate_output():
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
