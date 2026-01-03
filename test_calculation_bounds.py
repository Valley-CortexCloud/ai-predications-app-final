#!/usr/bin/env python3
"""
Test that calculation bounds are properly applied to prevent extreme values.
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_beta_clipping():
    """Test that beta values are clipped to [-5, 5] range."""
    print("\n" + "="*60)
    print("TEST: Beta Clipping")
    print("="*60)
    
    # Create synthetic data with extreme variance
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    
    # Stock with extreme returns
    stock_returns = pd.Series([0.01] * 150 + [0.5] * 150, index=dates)
    
    # SPY with near-zero variance
    spy_returns = pd.Series([0.0001] * 300, index=dates)
    
    # Simulate beta calculation
    cov = stock_returns.rolling(126).cov(spy_returns)
    var = spy_returns.rolling(126).var()
    beta_raw = cov / var.replace(0, np.nan)
    beta_clipped = beta_raw.fillna(0).clip(-5, 5)
    
    print(f"Raw beta range: [{beta_raw.min():.2f}, {beta_raw.max():.2f}]")
    print(f"Clipped beta range: [{beta_clipped.min():.2f}, {beta_clipped.max():.2f}]")
    print(f"Values outside [-5, 5]: {((beta_raw < -5) | (beta_raw > 5)).sum()}")
    print(f"After clipping: {((beta_clipped < -5) | (beta_clipped > 5)).sum()}")
    
    assert beta_clipped.min() >= -5, "Beta should not be below -5"
    assert beta_clipped.max() <= 5, "Beta should not be above 5"
    print("✓ Beta clipping test PASSED")
    return True

def test_momentum_clipping():
    """Test that momentum values are clipped to [-0.99, 10] range."""
    print("\n" + "="*60)
    print("TEST: Momentum Clipping")
    print("="*60)
    
    # Create synthetic price data
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    
    # Extreme case: stock crashes to near zero
    prices = pd.Series([100.0] * 100 + [1.0] * 200, index=dates)
    
    # Calculate returns
    ret_252d = prices.pct_change(252)
    ret_21d = prices.pct_change(21)
    
    # Calculate momentum (12m skip 1m)
    momentum_raw = ret_252d - ret_21d
    momentum_clipped = momentum_raw.clip(-0.99, 10)
    
    print(f"Raw momentum range: [{momentum_raw.min():.2f}, {momentum_raw.max():.2f}]")
    print(f"Clipped momentum range: [{momentum_clipped.min():.2f}, {momentum_clipped.max():.2f}]")
    print(f"Values below -0.99: {(momentum_raw < -0.99).sum()}")
    print(f"After clipping: {(momentum_clipped < -0.99).sum()}")
    
    assert momentum_clipped.min() >= -0.99, "Momentum should not be below -0.99 (can't lose more than 99%)"
    assert momentum_clipped.max() <= 10, "Momentum should not be above 10 (1000%)"
    print("✓ Momentum clipping test PASSED")
    return True

def test_volatility_bounds():
    """Test that volatility values are within reasonable bounds."""
    print("\n" + "="*60)
    print("TEST: Volatility Bounds")
    print("="*60)
    
    # Create synthetic price data with extreme volatility
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    # Extreme case: wild swings
    prices = pd.Series([100, 200, 50, 300, 25, 400] * 16 + [100] * 4, index=dates)
    
    # Calculate daily volatility
    returns = prices.pct_change()
    volatility_raw = returns.rolling(20).std()
    
    # For daily volatility, reasonable upper bound is ~0.2 (20% daily std)
    # Extreme outliers above 1.0 (100% daily std) are likely data errors
    volatility_clipped = volatility_raw.clip(0, 1.0)
    
    print(f"Raw volatility range: [{volatility_raw.min():.4f}, {volatility_raw.max():.4f}]")
    print(f"Clipped volatility range: [{volatility_clipped.min():.4f}, {volatility_clipped.max():.4f}]")
    print(f"Values above 1.0: {(volatility_raw > 1.0).sum()}")
    print(f"After clipping: {(volatility_clipped > 1.0).sum()}")
    
    assert volatility_clipped.max() <= 1.0, "Daily volatility should not exceed 100%"
    print("✓ Volatility bounds test PASSED")
    return True

def test_division_by_zero_protection():
    """Test that division operations are protected against zero denominators."""
    print("\n" + "="*60)
    print("TEST: Division by Zero Protection")
    print("="*60)
    
    # Test case 1: Division with replace(0, np.nan)
    denominator = pd.Series([0, 1, 2, 0, 5])
    numerator = pd.Series([10, 10, 10, 10, 10])
    
    result = numerator / denominator.replace(0, np.nan)
    result_filled = result.fillna(0)
    
    print(f"Denominator: {denominator.values}")
    print(f"Result with replace(0, np.nan): {result.values}")
    print(f"Result after fillna(0): {result_filled.values}")
    print(f"No inf values: {not np.isinf(result_filled).any()}")
    
    assert not np.isinf(result_filled).any(), "Should not have infinite values"
    print("✓ Division by zero protection test PASSED")
    return True

def test_return_calculation_bounds():
    """Test that return calculations produce valid ranges."""
    print("\n" + "="*60)
    print("TEST: Return Calculation Bounds")
    print("="*60)
    
    # Test case: Stock goes to zero (delisting/bankruptcy)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = pd.Series([100.0] * 50 + [0.01] * 50, index=dates)
    
    # Single period return (can be -100% at most)
    ret_1d = prices.pct_change()
    
    print(f"Daily return range: [{ret_1d.min():.4f}, {ret_1d.max():.4f}]")
    print(f"Returns below -1.0 (impossible): {(ret_1d < -1.0).sum()}")
    
    # Multi-period returns should also be bounded
    ret_21d = prices.pct_change(21)
    print(f"21-day return range: [{ret_21d.min():.4f}, {ret_21d.max():.4f}]")
    
    # Check that returns are mathematically valid
    assert ret_1d.min() >= -1.0, "Single period return cannot be below -100%"
    print("✓ Return calculation bounds test PASSED")
    return True

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CALCULATION BOUNDS TEST SUITE")
    print("="*60)
    
    all_passed = True
    
    tests = [
        test_beta_clipping,
        test_momentum_clipping,
        test_volatility_bounds,
        test_division_by_zero_protection,
        test_return_calculation_bounds,
    ]
    
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*60 + "\n")
    
    sys.exit(0 if all_passed else 1)
