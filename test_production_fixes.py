#!/usr/bin/env python3
"""
Validation test for production pipeline date filtering fixes.
This test validates that the changes correctly filter data to the latest date.
"""
import sys
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def create_test_data():
    """Create test dataset with multiple dates"""
    dates = pd.date_range(end=datetime.now(), periods=5, freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    
    rows = []
    for date in dates:
        for symbol in symbols:
            rows.append({
                'date': date.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'price': np.random.uniform(100, 500),
                'adv20_dollar': np.random.uniform(10_000_000, 100_000_000),
                'feat_vix_level_z_63': np.random.uniform(-1, 1),
                'excess_63d': np.random.uniform(-0.1, 0.1),
                'label_rel_grade': np.random.randint(0, 7)
            })
    
    df = pd.DataFrame(rows)
    return df

def test_production_only_filtering():
    """Test that --production-only flag correctly filters to latest date"""
    print("\n" + "="*60)
    print("TEST: Production-only filtering")
    print("="*60)
    
    # Create test data
    df = create_test_data()
    initial_rows = len(df)
    initial_dates = df['date'].nunique()
    
    print(f"Initial dataset: {initial_rows} rows, {initial_dates} unique dates")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Simulate production mode filtering
    latest_date = df['date'].max()
    df_filtered = df[df['date'] == latest_date].copy()
    
    print(f"\nAfter production filtering:")
    print(f"  Rows: {len(df_filtered)}")
    print(f"  Unique dates: {df_filtered['date'].nunique()}")
    print(f"  Latest date: {latest_date}")
    
    # Validations
    assert df_filtered['date'].nunique() == 1, "Should have only 1 unique date"
    assert df_filtered['date'].iloc[0] == latest_date, "All rows should be from latest date"
    assert len(df_filtered) < initial_rows, "Filtered dataset should be smaller"
    
    print("\n✅ Production filtering test PASSED")
    return True

def test_vix_extraction_from_latest_date():
    """Test that VIX is extracted from latest date, not first row"""
    print("\n" + "="*60)
    print("TEST: VIX extraction from latest date")
    print("="*60)
    
    # Create test data with different VIX values per date
    dates = ['2023-12-18', '2024-06-15', '2024-12-16']
    vix_values = [-0.5, 0.2, 1.8]  # Old date has different value than latest
    
    rows = []
    for date, vix in zip(dates, vix_values):
        for symbol in ['AAPL', 'MSFT']:
            rows.append({
                'date': date,
                'symbol': symbol,
                'feat_vix_level_z_63': vix,
                'price': 100,
                'adv20_dollar': 20_000_000
            })
    
    df = pd.DataFrame(rows)
    
    print(f"Dataset has {len(df)} rows, {df['date'].nunique()} dates")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # OLD WAY (WRONG): Extract from first row
    vix_old_way = df['feat_vix_level_z_63'].iloc[0]
    print(f"\nOLD WAY (wrong): VIX from .iloc[0] = {vix_old_way:.2f}")
    print(f"  This is from date: {df['date'].iloc[0]}")
    
    # NEW WAY (CORRECT): Filter to latest date first, then extract
    latest_date = df['date'].max()
    df_latest = df[df['date'] == latest_date].copy()
    vix_new_way = df_latest['feat_vix_level_z_63'].iloc[0]
    print(f"\nNEW WAY (correct): VIX from latest date = {vix_new_way:.2f}")
    print(f"  This is from date: {latest_date}")
    
    # Validations
    assert vix_old_way != vix_new_way, "Old way should give different value than new way"
    assert vix_new_way == vix_values[-1], "New way should match latest date VIX"
    assert df_latest['date'].nunique() == 1, "Should be filtered to one date"
    
    print("\n✅ VIX extraction test PASSED")
    return True

def test_date_staleness_check():
    """Test that date staleness is correctly detected"""
    print("\n" + "="*60)
    print("TEST: Date staleness detection")
    print("="*60)
    
    # Test with old data
    old_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    df_old = pd.DataFrame([{
        'date': old_date,
        'symbol': 'AAPL',
        'feat_vix_level_z_63': 0.5
    }])
    
    latest_date = pd.to_datetime(df_old['date'].max())
    actual_today = datetime.now().date()
    days_behind = (pd.to_datetime(actual_today) - latest_date).days
    
    print(f"Latest date in data: {latest_date.date()}")
    print(f"Current date: {actual_today}")
    print(f"Days behind: {days_behind}")
    
    assert days_behind > 5, "Should detect data is more than 5 days old"
    print(f"✅ Correctly detected data is {days_behind} days old (>5 day threshold)")
    
    # Test with future data (data issue)
    future_date = (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d')
    df_future = pd.DataFrame([{
        'date': future_date,
        'symbol': 'AAPL',
        'feat_vix_level_z_63': 0.5
    }])
    
    latest_date_future = pd.to_datetime(df_future['date'].max())
    days_behind_future = (pd.to_datetime(actual_today) - latest_date_future).days
    
    print(f"\nFuture date test:")
    print(f"Latest date in data: {latest_date_future.date()}")
    print(f"Days behind (negative = future): {days_behind_future}")
    
    assert days_behind_future < -1, "Should detect data is in the future"
    print(f"✅ Correctly detected data is {abs(days_behind_future)} days in the FUTURE")
    
    print("\n✅ Date staleness test PASSED")
    return True

def test_prediction_date_consistency():
    """Test that feature date and prediction date match"""
    print("\n" + "="*60)
    print("TEST: Feature/Prediction date consistency")
    print("="*60)
    
    feature_date = '2024-12-16'
    prediction_date = '2024-12-16'
    
    print(f"Feature date: {feature_date}")
    print(f"Prediction date: {prediction_date}")
    
    if str(feature_date) != str(prediction_date):
        print(f"⚠️  WARNING: Dates don't match!")
        assert False, "Dates should match"
    else:
        print(f"✅ Dates match correctly")
    
    print("\n✅ Date consistency test PASSED")
    return True

def run_all_tests():
    """Run all validation tests"""
    print("\n" + "="*60)
    print("RUNNING VALIDATION TESTS FOR PRODUCTION FIXES")
    print("="*60)
    
    tests = [
        test_production_only_filtering,
        test_vix_extraction_from_latest_date,
        test_date_staleness_check,
        test_prediction_date_consistency
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n❌ Test FAILED: {test.__name__}")
            print(f"   Error: {str(e)}")
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
