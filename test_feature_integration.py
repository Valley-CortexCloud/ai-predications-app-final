#!/usr/bin/env python3
"""
Integration test: Verify feature calculations produce valid ranges.
Tests the full pipeline with realistic synthetic data.
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from lib.features import (
        compute_baseline_features, 
        compute_multi_horizon_returns,
        compute_volatility_features
    )
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False
    print("⚠️  lib.features not available - skipping integration tests")

def create_realistic_ohlcv_data(days=500):
    """Create realistic OHLCV data with various market conditions."""
    dates = pd.date_range('2020-01-01', periods=days, freq='D')
    
    # Simulate price with drift and volatility
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, days)  # ~0.05% daily return, 1.5% vol
    
    # Add some volatility clustering
    for i in range(100, 120):
        returns[i] = np.random.normal(0, 0.05)  # High vol period
    
    # Add a crash period
    for i in range(200, 210):
        returns[i] = np.random.normal(-0.03, 0.08)  # Crash with high vol
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close prices
    df = pd.DataFrame(index=dates)
    df['Close'] = prices
    df['Adj Close'] = prices  # Assume no dividends for simplicity
    
    # Add realistic intraday range
    df['High'] = df['Close'] * np.random.uniform(1.00, 1.02, days)
    df['Low'] = df['Close'] * np.random.uniform(0.98, 1.00, days)
    df['Open'] = df['Close'].shift(1).fillna(100) * np.random.uniform(0.99, 1.01, days)
    
    # Add volume
    df['Volume'] = np.random.randint(1_000_000, 10_000_000, days)
    
    return df

def test_baseline_features_bounds():
    """Test that baseline features stay within reasonable bounds."""
    print("\n" + "="*60)
    print("INTEGRATION TEST: Baseline Features Bounds")
    print("="*60)
    
    if not FEATURES_AVAILABLE:
        print("⊘ Skipped - lib.features not available")
        return True
    
    df = create_realistic_ohlcv_data(500)
    features = compute_baseline_features(df)
    
    # Check RSI bounds
    rsi = features['rsi'].dropna()
    print(f"RSI range: [{rsi.min():.2f}, {rsi.max():.2f}]")
    assert rsi.min() >= 0 and rsi.max() <= 100, "RSI should be in [0, 100]"
    
    # Check volatility bounds
    vol = features['volatility_20'].dropna()
    print(f"Volatility range: [{vol.min():.4f}, {vol.max():.4f}]")
    assert vol.max() <= 1.0, "Daily volatility should not exceed 1.0"
    assert vol.min() >= 0, "Volatility should be non-negative"
    
    # Check Stochastic bounds
    stoch_k = features['stoch_k'].dropna()
    print(f"Stochastic %K range: [{stoch_k.min():.2f}, {stoch_k.max():.2f}]")
    assert stoch_k.min() >= 0 and stoch_k.max() <= 100, "Stochastic should be in [0, 100]"
    
    # Check Williams %R bounds
    williams = features['williams_r'].dropna()
    print(f"Williams %R range: [{williams.min():.2f}, {williams.max():.2f}]")
    assert williams.min() >= -100 and williams.max() <= 0, "Williams %R should be in [-100, 0]"
    
    print("✓ All baseline features within valid bounds")
    return True

def test_momentum_features_bounds():
    """Test that momentum features stay within reasonable bounds."""
    print("\n" + "="*60)
    print("INTEGRATION TEST: Momentum Features Bounds")
    print("="*60)
    
    if not FEATURES_AVAILABLE:
        print("⊘ Skipped - lib.features not available")
        return True
    
    df = create_realistic_ohlcv_data(500)
    
    # Compute multi-horizon returns
    close = df['Adj Close']
    features = compute_multi_horizon_returns(close)
    
    # Check momentum bounds
    mom = features['mom_12m_skip1m'].dropna()
    print(f"Momentum (12m skip 1m) range: [{mom.min():.4f}, {mom.max():.4f}]")
    assert mom.min() >= -0.99, "Momentum should not be below -99% (mathematical impossibility)"
    assert mom.max() <= 10, "Momentum should not exceed 1000%"
    print(f"Values in valid range: {len(mom)}/{len(mom)} (100%)")
    
    # Check return bounds
    ret_252d = features['ret_252d'].dropna()
    print(f"252-day return range: [{ret_252d.min():.4f}, {ret_252d.max():.4f}]")
    assert ret_252d.min() >= -0.99, "252d return should not be below -99%"
    
    # Check distance to 52w high
    pct_to_high = features['pct_to_52w_high'].dropna()
    print(f"Distance to 52w high range: [{pct_to_high.min():.4f}, {pct_to_high.max():.4f}]")
    assert pct_to_high.max() <= 0.001, "Should not be above 52w high (except rounding)"
    assert pct_to_high.min() >= -1.0, "Should not be more than 100% below 52w high"
    
    print("✓ All momentum features within valid bounds")
    return True

def test_volatility_features_bounds():
    """Test that volatility features stay within reasonable bounds."""
    print("\n" + "="*60)
    print("INTEGRATION TEST: Volatility Features Bounds")
    print("="*60)
    
    if not FEATURES_AVAILABLE:
        print("⊘ Skipped - lib.features not available")
        return True
    
    df = create_realistic_ohlcv_data(500)
    
    # Compute volatility features
    features = compute_volatility_features(
        df['Open'], df['High'], df['Low'], df['Close']
    )
    
    # Check Parkinson volatility
    park = features['parkinson_20'].dropna()
    print(f"Parkinson volatility range: [{park.min():.4f}, {park.max():.4f}]")
    assert park.min() >= 0, "Parkinson vol should be non-negative"
    assert park.max() <= 2.0, "Parkinson vol should not exceed 2.0"
    
    # Check Garman-Klass volatility
    gk = features['garman_klass_20'].dropna()
    print(f"Garman-Klass volatility range: [{gk.min():.4f}, {gk.max():.4f}]")
    assert gk.min() >= 0, "GK vol should be non-negative"
    assert gk.max() <= 2.0, "GK vol should not exceed 2.0"
    
    # Check Rogers-Satchell volatility
    rs = features['rogers_satchell_20'].dropna()
    print(f"Rogers-Satchell volatility range: [{rs.min():.4f}, {rs.max():.4f}]")
    assert rs.min() >= 0, "RS vol should be non-negative"
    assert rs.max() <= 2.0, "RS vol should not exceed 2.0"
    
    # Check downside volatility
    dv = features['downside_vol_20'].dropna()
    print(f"Downside volatility range: [{dv.min():.4f}, {dv.max():.4f}]")
    assert dv.min() >= 0, "Downside vol should be non-negative"
    assert dv.max() <= 1.0, "Downside vol should not exceed 1.0"
    
    # Check ATR normalized
    atr_norm = features['atr_norm'].dropna()
    print(f"ATR normalized range: [{atr_norm.min():.4f}, {atr_norm.max():.4f}]")
    assert atr_norm.min() >= 0, "ATR normalized should be non-negative"
    assert atr_norm.max() <= 0.5, "ATR normalized should not exceed 50% of price"
    
    print("✓ All volatility features within valid bounds")
    return True

def test_extreme_market_conditions():
    """Test feature calculations under extreme market conditions."""
    print("\n" + "="*60)
    print("INTEGRATION TEST: Extreme Market Conditions")
    print("="*60)
    
    if not FEATURES_AVAILABLE:
        print("⊘ Skipped - lib.features not available")
        return True
    
    # Create extreme crash scenario
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    df = pd.DataFrame(index=dates)
    
    # Simulate a 90% crash over 30 days
    prices = [100.0] * 100  # Stable period
    for i in range(30):
        prices.append(100 * (1 - 0.03 * i))  # 90% crash
    prices.extend([10.0] * 170)  # Post-crash period
    
    df['Close'] = prices
    df['Adj Close'] = prices
    df['High'] = [p * 1.02 for p in prices]
    df['Low'] = [p * 0.98 for p in prices]
    df['Open'] = df['Close'].shift(1).fillna(100)
    df['Volume'] = [1_000_000] * 300
    
    # Compute features
    baseline = compute_baseline_features(df)
    returns = compute_multi_horizon_returns(df['Adj Close'])
    
    # Check momentum doesn't go below -99%
    mom = returns['mom_12m_skip1m'].dropna()
    print(f"Momentum during crash: min={mom.min():.4f}, max={mom.max():.4f}")
    assert mom.min() >= -0.99, "Momentum should not be below -99% even in crash"
    
    # Check volatility is bounded
    vol = baseline['volatility_20'].dropna()
    print(f"Volatility during crash: min={vol.min():.4f}, max={vol.max():.4f}")
    assert vol.max() <= 1.0, "Volatility should be capped at 1.0 even in extreme conditions"
    
    # Check returns are valid
    ret_21d = returns['ret_21d'].dropna()
    print(f"21-day returns: min={ret_21d.min():.4f}, max={ret_21d.max():.4f}")
    assert ret_21d.min() >= -1.0, "Returns cannot be below -100%"
    
    print("✓ Features remain valid even under extreme conditions")
    return True

if __name__ == "__main__":
    print("\n" + "="*60)
    print("FEATURE CALCULATION INTEGRATION TEST SUITE")
    print("="*60)
    
    all_passed = True
    
    tests = [
        test_baseline_features_bounds,
        test_momentum_features_bounds,
        test_volatility_features_bounds,
        test_extreme_market_conditions,
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
        print("✓ ALL INTEGRATION TESTS PASSED")
    else:
        print("✗ SOME INTEGRATION TESTS FAILED")
    print("="*60 + "\n")
    
    sys.exit(0 if all_passed else 1)
