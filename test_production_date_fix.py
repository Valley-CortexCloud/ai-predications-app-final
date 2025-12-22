#!/usr/bin/env python3
"""
Test to validate that production mode correctly reloads data from cache
when the actual latest date is significantly newer than the training latest date.
"""
import sys
import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_production_date_reload():
    """Test that production mode reloads from cache for fresh data"""
    print("\n" + "="*60)
    print("TEST: Production mode date reload from cache")
    print("="*60)
    
    # Run build_labels_final.py with production-only flag
    output_file = "/tmp/test_production_date.parquet"
    cmd = [
        "python3", "scripts/build_labels_final.py",
        "--tickers", "AAPL,MSFT,GOOGL,AMZN,NVDA",
        "--production-only",
        "--output", output_file
    ]
    
    print(f"\n📝 Running command:")
    print(f"   {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
    
    # Check if command succeeded
    if result.returncode != 0:
        print("\n❌ Command failed!")
        print("STDERR:", result.stderr[-1000:])
        return False
    
    # Parse output to check for key messages
    output = result.stdout + result.stderr
    
    print("\n📊 Checking output logs...")
    
    # Check 1: Should detect date difference
    if "Training latest date (with 63d forward):" in output:
        print("   ✓ Detected training latest date")
        # Extract training date
        for line in output.split('\n'):
            if "Training latest date" in line:
                print(f"     {line.strip()}")
                break
    else:
        print("   ❌ Missing training latest date message")
        return False
    
    # Check 2: Should detect actual latest date from cache
    if "Actual latest date (from cache):" in output:
        print("   ✓ Detected actual latest date from cache")
        for line in output.split('\n'):
            if "Actual latest date" in line:
                print(f"     {line.strip()}")
                break
    else:
        print("   ❌ Missing actual latest date message")
        return False
    
    # Check 3: Should warn about fresh data
    if "WARNING:" in output and "days of fresh data available" in output:
        print("   ✓ Warning about fresh data detected")
        for line in output.split('\n'):
            if "WARNING:" in line:
                print(f"     {line.strip()}")
                break
    else:
        print("   ❌ Missing fresh data warning")
        return False
    
    # Check 4: Should reload tickers
    if "Re-processing all tickers for date:" in output:
        print("   ✓ Reloading tickers for fresh date")
    else:
        print("   ❌ Missing ticker reload message")
        return False
    
    # Check 5: Should recompute cross-sectional features
    if "Computing cross-sectional rank features" in output:
        print("   ✓ Recomputing cross-sectional features")
    else:
        print("   ❌ Missing cross-sectional feature computation")
        return False
    
    # Check 6: Should recompute composite quality
    if "Re-computing composite quality for production date" in output:
        print("   ✓ Recomputing composite quality")
    else:
        print("   ❌ Missing composite quality computation")
        return False
    
    # Verify output file
    print("\n📁 Checking output file...")
    if not Path(output_file).exists():
        print("   ❌ Output file not created")
        return False
    
    df = pd.read_parquet(output_file)
    print(f"   ✓ Output file created: {len(df)} rows, {df['symbol'].nunique()} symbols")
    
    # Check 7: All rows should be from the same date
    if df['date'].nunique() != 1:
        print(f"   ❌ Multiple dates in output: {df['date'].nunique()}")
        return False
    print(f"   ✓ Single date in output: {df['date'].iloc[0]}")
    
    # Check 8: Date should be recent (2025-12-22)
    latest_date = df['date'].max()
    expected_date = pd.Timestamp('2025-12-22')
    
    if latest_date == expected_date:
        print(f"   ✅ Latest date is correct: {latest_date}")
    else:
        # Allow for some flexibility in case cache is updated
        days_diff = abs((latest_date - expected_date).days)
        if days_diff <= 3:
            print(f"   ✓ Latest date is close: {latest_date} (within {days_diff} days)")
        else:
            print(f"   ❌ Latest date is wrong: {latest_date} (expected ~{expected_date})")
            return False
    
    # Check 9: Should have cross-sectional features
    rank_features = [c for c in df.columns if '_rank_pct' in c]
    if len(rank_features) > 0:
        print(f"   ✓ Cross-sectional rank features present: {len(rank_features)}")
    else:
        print("   ❌ Missing cross-sectional rank features")
        return False
    
    # Check 10: Should have composite quality
    if 'feat_composite_quality' in df.columns:
        quality_stats = df['feat_composite_quality'].describe()
        print(f"   ✓ Composite quality present: mean={quality_stats['mean']:.3f}")
    else:
        print("   ❌ Missing composite quality feature")
        return False
    
    print("\n✅ All checks passed!")
    return True


def test_non_production_mode_unchanged():
    """Test that non-production mode is not affected by changes"""
    print("\n" + "="*60)
    print("TEST: Non-production mode unchanged")
    print("="*60)
    
    output_file = "/tmp/test_non_production.parquet"
    cmd = [
        "python3", "scripts/build_labels_final.py",
        "--tickers", "AAPL,MSFT",
        "--output", output_file
    ]
    
    print(f"\n📝 Running command (without --production-only):")
    print(f"   {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print("\n❌ Command failed!")
        print("STDERR:", result.stderr[-500:])
        return False
    
    output = result.stdout + result.stderr
    
    # Should NOT have production mode messages
    if "PRODUCTION MODE" in output:
        print("   ❌ Production mode messages found (should not be present)")
        return False
    
    print("   ✓ No production mode messages (as expected)")
    
    # Verify output file
    if not Path(output_file).exists():
        print("   ❌ Output file not created")
        return False
    
    df = pd.read_parquet(output_file)
    print(f"   ✓ Output file created: {len(df)} rows, {df['symbol'].nunique()} symbols")
    
    # Should have multiple dates (full history)
    if df['date'].nunique() > 1:
        print(f"   ✓ Multiple dates in output: {df['date'].nunique()} dates")
        print(f"     Date range: {df['date'].min()} to {df['date'].max()}")
    else:
        print(f"   ❌ Only one date in output (expected multiple for non-production mode)")
        return False
    
    print("\n✅ Non-production mode test passed!")
    return True


def run_all_tests():
    """Run all validation tests"""
    print("\n" + "="*60)
    print("RUNNING PRODUCTION MODE DATE FIX VALIDATION TESTS")
    print("="*60)
    
    tests = [
        ("Production date reload", test_production_date_reload),
        ("Non-production mode unchanged", test_non_production_mode_unchanged),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"Running: {test_name}")
            print(f"{'='*60}")
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ Test FAILED with exception: {test_name}")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed > 0:
        print("\n❌ Some tests FAILED")
        return False
    else:
        print("\n✅ All tests PASSED")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
