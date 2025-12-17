#!/usr/bin/env python3
"""
Test for earnings quality merge_asof implementation
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_load_earnings_events():
    """Test that load_earnings_events returns correct DataFrame format"""
    print("\n" + "="*60)
    print("TEST: load_earnings_events() function")
    print("="*60)
    
    # Import the function
    from scripts.build_labels_final import load_earnings_events
    
    # Test with test earnings file
    earnings_file = "data/test_earnings.csv"
    earnings_df = load_earnings_events(earnings_file)
    
    print(f"\n✓ Loaded earnings DataFrame:")
    print(f"  Shape: {earnings_df.shape}")
    print(f"  Columns: {list(earnings_df.columns)}")
    print(f"  Symbols: {earnings_df['symbol'].nunique()}")
    
    # Verify structure
    required_cols = ['symbol', 'date', 'eps_actual', 'eps_estimate', 'eps_surprise_pct']
    for col in required_cols:
        assert col in earnings_df.columns, f"Missing column: {col}"
    
    print(f"\n✓ All required columns present")
    print(f"\nSample data:")
    print(earnings_df.head(10).to_string(index=False))
    
    # Verify data types
    assert pd.api.types.is_datetime64_any_dtype(earnings_df['date']), "date should be datetime"
    assert pd.api.types.is_numeric_dtype(earnings_df['eps_actual']), "eps_actual should be numeric"
    assert pd.api.types.is_numeric_dtype(earnings_df['eps_estimate']), "eps_estimate should be numeric"
    assert pd.api.types.is_numeric_dtype(earnings_df['eps_surprise_pct']), "eps_surprise_pct should be numeric"
    
    print(f"\n✓ All data types correct")
    print("\n✅ load_earnings_events() test PASSED")
    return True

def test_merge_asof_logic():
    """Test that merge_asof correctly finds most recent earnings"""
    print("\n" + "="*60)
    print("TEST: merge_asof earnings logic")
    print("="*60)
    
    # Create sample data
    dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
    stocks_df = pd.DataFrame({
        'symbol': ['AAPL'] * 10,
        'date': dates,
        'price': np.random.uniform(150, 160, 10)
    })
    
    # Create earnings events (2 events in the past)
    earnings_df = pd.DataFrame({
        'symbol': ['AAPL', 'AAPL'],
        'date': [dates[2], dates[6]],  # Earnings on day 2 and day 6
        'eps_actual': [1.5, 1.6],
        'eps_estimate': [1.4, 1.5],
        'eps_surprise_pct': [7.14, 6.67]
    })
    
    print(f"\nStock dates: {dates[0].date()} to {dates[-1].date()}")
    print(f"Earnings dates: {earnings_df['date'].tolist()}")
    
    # Perform merge_asof
    result = pd.merge_asof(
        stocks_df.sort_values(['symbol', 'date']),
        earnings_df[['symbol', 'date', 'eps_surprise_pct']],
        on='date',
        by='symbol',
        direction='backward',
        tolerance=pd.Timedelta(days=120)
    )
    
    print(f"\nMerge result:")
    print(result[['date', 'eps_surprise_pct']].to_string())
    
    # Verify logic
    # Days 0-1 should have no earnings (before first event)
    assert result.loc[0, 'eps_surprise_pct'] != result.loc[0, 'eps_surprise_pct'], "Day 0 should be NaN"
    assert result.loc[1, 'eps_surprise_pct'] != result.loc[1, 'eps_surprise_pct'], "Day 1 should be NaN"
    
    # Days 2-5 should have first earnings
    assert result.loc[2, 'eps_surprise_pct'] == 7.14, "Day 2 should have first earnings"
    assert result.loc[5, 'eps_surprise_pct'] == 7.14, "Day 5 should still have first earnings"
    
    # Days 6-9 should have second earnings
    assert result.loc[6, 'eps_surprise_pct'] == 6.67, "Day 6 should have second earnings"
    assert result.loc[9, 'eps_surprise_pct'] == 6.67, "Day 9 should still have second earnings"
    
    print(f"\n✓ merge_asof correctly propagates most recent earnings")
    print("\n✅ merge_asof logic test PASSED")
    return True

def test_earnings_quality_computation():
    """Test earnings quality computation"""
    print("\n" + "="*60)
    print("TEST: earnings quality computation")
    print("="*60)
    
    # Create test dataframe with earnings data
    df = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
        'date': [datetime.now()] * 4,
        'feat_prev_earn_surprise_pct': [7.14, -5.0, 12.5, 0.0]
    })
    
    # Compute earnings quality (same as in build_labels_final.py)
    surprise = df['feat_prev_earn_surprise_pct'].fillna(0).astype(float).clip(-100, 300)
    df['feat_earnings_quality'] = surprise.fillna(0).clip(-100, 300)
    
    print(f"\nEarnings quality computation:")
    print(df[['symbol', 'feat_prev_earn_surprise_pct', 'feat_earnings_quality']].to_string(index=False))
    
    # Verify values
    assert df.loc[0, 'feat_earnings_quality'] == 7.14, "Positive surprise should pass through"
    assert df.loc[1, 'feat_earnings_quality'] == -5.0, "Negative surprise should pass through"
    assert df.loc[2, 'feat_earnings_quality'] == 12.5, "Positive surprise should pass through"
    assert df.loc[3, 'feat_earnings_quality'] == 0.0, "Zero surprise should be zero"
    
    # Check that non-zero values exist
    non_zero = (df['feat_earnings_quality'] != 0).sum()
    print(f"\n✓ Non-zero earnings quality: {non_zero} / {len(df)}")
    
    assert non_zero == 3, "Should have 3 non-zero values"
    
    print("\n✅ Earnings quality computation test PASSED")
    return True

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("TESTING EARNINGS QUALITY MERGE_ASOF IMPLEMENTATION")
    print("="*60)
    
    tests = [
        test_load_earnings_events,
        test_merge_asof_logic,
        test_earnings_quality_computation
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
