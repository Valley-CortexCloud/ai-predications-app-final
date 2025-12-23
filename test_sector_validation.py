#!/usr/bin/env python3
"""
Test script to validate sector NaN bug fix and comprehensive validation.
Tests the new requirements:
1. Sector dummy variables are 0/1, no NaN
2. Each stock has exactly 1 sector assigned
3. Multi-stock validation shows diverse stocks
4. Cross-sectional features are validated
5. Sector features are computed correctly
"""
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def run_production_mode():
    """Run production mode and capture output"""
    print("\n" + "="*80)
    print("RUNNING PRODUCTION MODE WITH SECTOR VALIDATION")
    print("="*80)
    
    output_file = "datasets/test_sector_validation.parquet"
    cmd = [
        "python3", "scripts/build_labels_final.py",
        "--production-only",
        "--output", output_file,
        "--cache-dir", "data_cache/10y_ticker_features",
        "--earnings-file", "data/earnings.csv"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ FAILED: Production mode execution failed")
        print("STDERR:", result.stderr)
        print("STDOUT:", result.stdout)
        return None, result.stdout
    
    return output_file, result.stdout

def test_sector_dummy_fix(df):
    """Test 1: Sector dummy variables have no NaN"""
    print("\n" + "="*80)
    print("TEST 1: SECTOR DUMMY NaN FIX")
    print("="*80)
    
    sector_cols = [c for c in df.columns if c.startswith('feat_sector_code_')]
    
    if not sector_cols:
        print("❌ FAILED: No sector dummy columns found!")
        return False
    
    print(f"Found {len(sector_cols)} sector dummy columns")
    
    # Check for NaN
    nan_count = df[sector_cols].isna().sum().sum()
    
    if nan_count > 0:
        print(f"❌ FAILED: Found {nan_count} NaN values in sector dummies!")
        return False
    
    print(f"✅ PASS: No NaN values in sector dummies")
    
    # Check values are only 0 or 1
    for col in sector_cols:
        unique_vals = df[col].unique()
        if not all(v in [0.0, 1.0] for v in unique_vals):
            print(f"❌ FAILED: {col} has values other than 0/1: {unique_vals}")
            return False
    
    print(f"✅ PASS: All sector dummies are 0 or 1")
    
    return True

def test_single_sector_assignment(df):
    """Test 2: Each stock has exactly 1 sector"""
    print("\n" + "="*80)
    print("TEST 2: SINGLE SECTOR ASSIGNMENT")
    print("="*80)
    
    sector_cols = [c for c in df.columns if c.startswith('feat_sector_code_')]
    sector_sums = df[sector_cols].sum(axis=1)
    
    stocks_with_one = (sector_sums == 1).sum()
    stocks_with_multiple = (sector_sums > 1).sum()
    stocks_with_none = (sector_sums == 0).sum()
    
    print(f"Total stocks: {len(df)}")
    print(f"Exactly 1 sector: {stocks_with_one}")
    print(f"Multiple sectors: {stocks_with_multiple}")
    print(f"No sector: {stocks_with_none}")
    
    # Allow 1-2 stocks with no sector (edge cases)
    if stocks_with_none > 2:
        print(f"❌ FAILED: Too many stocks with no sector: {stocks_with_none}")
        return False
    
    if stocks_with_multiple > 0:
        print(f"❌ FAILED: {stocks_with_multiple} stocks have multiple sectors!")
        return False
    
    if stocks_with_one < len(df) - 2:
        print(f"❌ FAILED: Only {stocks_with_one}/{len(df)} stocks have exactly 1 sector")
        return False
    
    print(f"✅ PASS: {stocks_with_one}/{len(df)} stocks have exactly 1 sector")
    
    return True

def test_multi_stock_validation(output_text):
    """Test 3: Multi-stock validation in output"""
    print("\n" + "="*80)
    print("TEST 3: MULTI-STOCK VALIDATION")
    print("="*80)
    
    required_symbols = ['AAPL', 'NFLX', 'JPM', 'BAC', 'XOM', 'CVX', 'JNJ', 'UNH', 'TSLA', 'F']
    
    found_count = 0
    for symbol in required_symbols:
        if f"📈 {symbol} -" in output_text:
            found_count += 1
            print(f"✅ Found validation for {symbol}")
        else:
            print(f"⚠️  Validation for {symbol} not found (may not be in dataset)")
    
    if found_count < 5:
        print(f"❌ FAILED: Only {found_count} stocks validated (expected at least 5)")
        return False
    
    print(f"✅ PASS: {found_count} diverse stocks validated")
    
    return True

def test_cross_sectional_features(df):
    """Test 4: Cross-sectional features validation"""
    print("\n" + "="*80)
    print("TEST 4: CROSS-SECTIONAL FEATURES")
    print("="*80)
    
    # Check rank features
    rank_cols = [c for c in df.columns if c.endswith('_rank_pct')]
    
    if len(rank_cols) < 10:
        print(f"❌ FAILED: Only {len(rank_cols)} rank features (expected >= 10)")
        return False
    
    print(f"Found {len(rank_cols)} rank features")
    
    # Validate rank ranges
    issues = []
    for col in rank_cols:
        ranks = df[col]
        if ranks.min() < 0 or ranks.max() > 1:
            issues.append(f"{col}: range [{ranks.min():.4f}, {ranks.max():.4f}]")
        if abs(ranks.mean() - 0.5) > 0.2:
            issues.append(f"{col}: mean {ranks.mean():.4f} (far from 0.5)")
    
    if issues:
        print(f"❌ FAILED: Rank feature issues:")
        for issue in issues[:5]:
            print(f"   {issue}")
        return False
    
    print(f"✅ PASS: All rank features in valid range [0, 1] with mean ≈ 0.5")
    
    # Check z-score features
    zscore_cols = [c for c in df.columns if c.endswith('_zscore_xsec')]
    
    if len(zscore_cols) < 3:
        print(f"⚠️  WARNING: Only {len(zscore_cols)} z-score features (expected >= 3)")
    else:
        print(f"Found {len(zscore_cols)} z-score features")
        
        # Validate z-scores
        for col in zscore_cols:
            zscores = df[col]
            mean_ok = abs(zscores.mean()) < 0.3
            std_ok = abs(zscores.std() - 1.0) < 0.5
            
            if not (mean_ok and std_ok):
                print(f"⚠️  {col}: mean={zscores.mean():.4f}, std={zscores.std():.4f}")
    
    print(f"✅ PASS: Cross-sectional features validated")
    
    return True

def test_sector_feature_computation(df):
    """Test 5: Sector features are computed correctly"""
    print("\n" + "="*80)
    print("TEST 5: SECTOR FEATURE COMPUTATION")
    print("="*80)
    
    # Check for sector relative return features
    sector_rel_features = [c for c in df.columns if 'sector_rel_ret' in c and not c.endswith('_rank_pct')]
    
    if len(sector_rel_features) < 1:
        print(f"❌ FAILED: No sector relative return features found!")
        return False
    
    print(f"Found {len(sector_rel_features)} sector relative return features:")
    for feat in sector_rel_features:
        print(f"   - {feat}")
    
    # Validate sector relative returns
    for feat in sector_rel_features:
        values = df[feat]
        non_null = values.notna().sum()
        
        if non_null < len(df) * 0.95:
            print(f"⚠️  WARNING: {feat} has only {non_null}/{len(df)} non-null values")
        
        print(f"   {feat}: {non_null}/{len(df)} non-null, range [{values.min():.4f}, {values.max():.4f}]")
    
    # Validate per-sector computation
    sector_cols = [c for c in df.columns if c.startswith('feat_sector_code_')]
    
    if 'feat_sector_rel_ret_63d' in df.columns:
        print(f"\nPer-sector validation:")
        
        for col in sector_cols:
            sector_name = col.replace('feat_sector_code_', '')
            sector_stocks = df[df[col] == 1.0]
            
            if len(sector_stocks) > 0:
                rel_rets = sector_stocks['feat_sector_rel_ret_63d']
                non_null = rel_rets.notna().sum()
                
                if non_null > 0:
                    status = '✅' if non_null == len(sector_stocks) else '⚠️ '
                    print(f"   {status} {sector_name}: {non_null}/{len(sector_stocks)} stocks")
                else:
                    print(f"   ❌ {sector_name}: No relative returns!")
                    return False
    
    print(f"✅ PASS: Sector features computed correctly")
    
    return True

def test_feature_completeness(df):
    """Test 6: Feature completeness"""
    print("\n" + "="*80)
    print("TEST 6: FEATURE COMPLETENESS")
    print("="*80)
    
    feature_cols = [c for c in df.columns if c.startswith('feat_')]
    
    print(f"Total features: {len(feature_cols)}")
    
    if len(feature_cols) < 100:
        print(f"❌ FAILED: Only {len(feature_cols)} features (expected >= 100)")
        return False
    
    # Check for all-NaN features (except known ones)
    all_nan = []
    for col in feature_cols:
        if df[col].isna().all():
            all_nan.append(col)
    
    # feat_days_to_earn is expected to be all NaN (future looking)
    expected_nan = ['feat_days_to_earn']
    unexpected_nan = [c for c in all_nan if c not in expected_nan]
    
    if unexpected_nan:
        print(f"⚠️  WARNING: {len(unexpected_nan)} features are all NaN:")
        for feat in unexpected_nan[:5]:
            print(f"   - {feat}")
    
    # Check per-stock completeness
    nan_per_stock = df[feature_cols].isna().sum(axis=1)
    avg_populated = len(feature_cols) - nan_per_stock.mean()
    min_populated = len(feature_cols) - nan_per_stock.max()
    
    print(f"Per-stock coverage:")
    print(f"   Average: {avg_populated:.1f}/{len(feature_cols)} ({avg_populated/len(feature_cols)*100:.1f}%)")
    print(f"   Minimum: {min_populated:.0f}/{len(feature_cols)} ({min_populated/len(feature_cols)*100:.1f}%)")
    
    if avg_populated / len(feature_cols) < 0.90:
        print(f"❌ FAILED: Average completeness < 90%")
        return False
    
    print(f"✅ PASS: Feature completeness > 90%")
    
    return True

def test_value_sanity(df):
    """Test 7: Value sanity checks"""
    print("\n" + "="*80)
    print("TEST 7: VALUE SANITY CHECKS")
    print("="*80)
    
    checks = [
        ('feat_volatility_20', 0.001, 0.10, 'Daily volatility'),
        ('feat_rsi', 0, 100, 'RSI'),
        ('feat_beta_spy_126', -3, 6, 'Beta (allow some outliers)'),
    ]
    
    passed = 0
    for feat, min_val, max_val, desc in checks:
        if feat not in df.columns:
            print(f"⚠️  {desc}: feature not found")
            continue
        
        values = df[feat]
        outliers = ((values < min_val) | (values > max_val)).sum()
        outlier_pct = outliers / len(values) * 100
        
        if outlier_pct > 5:
            print(f"⚠️  {desc}: {outlier_pct:.1f}% outliers")
        else:
            print(f"✅ {desc}: {outlier_pct:.1f}% outliers")
            passed += 1
    
    if passed >= 2:
        print(f"✅ PASS: Most value sanity checks passed")
        return True
    else:
        print(f"❌ FAILED: Too many sanity check failures")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("SECTOR VALIDATION TEST SUITE")
    print("="*80)
    
    # Run production mode
    output_file, output_text = run_production_mode()
    
    if output_file is None:
        print("\n❌ OVERALL: FAILED - Could not run production mode")
        return False
    
    # Load output
    df = pd.read_parquet(output_file)
    
    print(f"\nLoaded {len(df)} rows with {len(df.columns)} columns")
    
    # Run all tests
    tests = [
        ("Sector Dummy NaN Fix", lambda: test_sector_dummy_fix(df)),
        ("Single Sector Assignment", lambda: test_single_sector_assignment(df)),
        ("Multi-Stock Validation", lambda: test_multi_stock_validation(output_text)),
        ("Cross-Sectional Features", lambda: test_cross_sectional_features(df)),
        ("Sector Feature Computation", lambda: test_sector_feature_computation(df)),
        ("Feature Completeness", lambda: test_feature_completeness(df)),
        ("Value Sanity", lambda: test_value_sanity(df)),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name}: EXCEPTION - {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    # Clean up
    Path(output_file).unlink(missing_ok=True)
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED")
        return True
    else:
        print("\n❌ SOME TESTS FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
