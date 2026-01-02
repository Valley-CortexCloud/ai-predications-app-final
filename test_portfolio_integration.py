#!/usr/bin/env python3
"""
Basic integration test for Portfolio Intelligence Engine

Tests the dry-run functionality of all three core scripts.
"""

import os
import sys
import tempfile
from pathlib import Path
import pandas as pd
import json

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

def test_portfolio_tracker():
    """Test portfolio_tracker.py basic operations"""
    print("=" * 60)
    print("Testing Portfolio Tracker")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker_path = Path(tmpdir) / "tracker.csv"
        
        # Test init
        from portfolio_tracker import init_tracker
        init_tracker(tracker_path)
        assert tracker_path.exists(), "Tracker should be created"
        print("✓ Tracker initialization works")
        
        # Add test data
        test_data = pd.DataFrame([
            {
                'symbol': 'AAPL',
                'entry_date': '2025-12-01',
                'entry_price': 180.50,
                'shares': 10,
                'current_price': 185.25,
                'pnl_pct': 2.6,
                'days_held': 32,
                'last_exit_score': 0
            }
        ])
        test_data.to_csv(tracker_path, index=False)
        
        # Test report generation
        from portfolio_tracker import generate_report
        generate_report(tracker_path)
        print("✓ Report generation works")
    
    print()
    return True


def test_portfolio_validator_dry_run():
    """Test portfolio_validator.py in dry-run mode"""
    print("=" * 60)
    print("Testing Portfolio Validator (Dry Run)")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker_path = Path(tmpdir) / "tracker.csv"
        output_dir = Path(tmpdir)
        
        # Create test tracker
        test_data = pd.DataFrame([
            {
                'symbol': 'AAPL',
                'entry_date': '2025-12-01',
                'entry_price': 180.50,
                'shares': 10,
                'current_price': 185.25,
                'pnl_pct': 2.6,
                'days_held': 32,
                'last_exit_score': 0
            },
            {
                'symbol': 'MSFT',
                'entry_date': '2025-11-20',
                'entry_price': 415.00,
                'shares': 5,
                'current_price': 425.50,
                'pnl_pct': 2.5,
                'days_held': 43,
                'last_exit_score': 0
            }
        ])
        test_data.to_csv(tracker_path, index=False)
        
        # Run validator in dry-run mode
        from portfolio_validator import validate_portfolio
        output_path = validate_portfolio(
            tracker_path,
            Path('datasets'),
            output_dir,
            dry_run=True
        )
        
        assert output_path.exists(), "Proposed file should be created"
        
        # Verify output format
        proposed = pd.read_csv(output_path)
        expected_cols = ['symbol', 'action', 'exit_score', 'days_held', 'reason', 'evidence', 'replacement']
        assert all(col in proposed.columns for col in expected_cols), "Missing expected columns"
        assert len(proposed) == 2, "Should have 2 holdings analyzed"
        print("✓ Validator dry-run works")
        print(f"✓ Generated proposed file with {len(proposed)} recommendations")
    
    print()
    return True


def test_trade_executor():
    """Test trade_executor.py order generation"""
    print("=" * 60)
    print("Testing Trade Executor")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker_path = Path(tmpdir) / "tracker.csv"
        confirmed_path = Path(tmpdir) / "confirmed_test.csv"
        output_dir = Path(tmpdir)
        
        # Create test tracker
        tracker_data = pd.DataFrame([
            {
                'symbol': 'AAPL',
                'entry_date': '2025-12-01',
                'entry_price': 180.50,
                'shares': 10,
                'current_price': 185.25,
                'pnl_pct': 2.6,
                'days_held': 32,
                'last_exit_score': 0
            }
        ])
        tracker_data.to_csv(tracker_path, index=False)
        
        # Create confirmed actions
        confirmed_data = pd.DataFrame([
            {
                'symbol': 'AAPL',
                'action': 'SELL',
                'exit_score': 82,
                'days_held': 32,
                'reason': 'Earnings miss + technical breakdown',
                'evidence': 'Q4 miss by 8%',
                'replacement': 'NVDA'
            },
            {
                'symbol': 'NVDA',
                'action': 'BUY',
                'exit_score': 0,
                'days_held': 0,
                'reason': 'Fresh signal - Rank #1 Strong Buy',
                'evidence': 'AI capex acceleration',
                'replacement': None
            }
        ])
        confirmed_data.to_csv(confirmed_path, index=False)
        
        # Generate orders
        from trade_executor import execute_trades
        orders_path = execute_trades(
            confirmed_path,
            tracker_path,
            output_dir,
            portfolio_value=100000.0,
            auto=False,
            paper=True
        )
        
        assert orders_path.exists(), "Orders file should be created"
        
        # Verify order format
        with open(orders_path) as f:
            orders = json.load(f)
        
        assert 'orders' in orders, "Should have orders key"
        assert orders['order_count'] == 2, "Should have 2 orders"
        assert orders['sell_count'] == 1, "Should have 1 sell"
        assert orders['buy_count'] == 1, "Should have 1 buy"
        
        # Check order details
        sell_order = [o for o in orders['orders'] if o['side'] == 'sell'][0]
        buy_order = [o for o in orders['orders'] if o['side'] == 'buy'][0]
        
        assert sell_order['symbol'] == 'AAPL', "Sell order should be for AAPL"
        assert sell_order['qty'] == 10, "Sell order should be for 10 shares"
        
        assert buy_order['symbol'] == 'NVDA', "Buy order should be for NVDA"
        assert buy_order['notional'] == 21000.0, "Buy order notional should be $21k (Strong Buy = 1.4x)"
        
        print("✓ Trade executor works")
        print(f"✓ Generated {orders['order_count']} orders")
    
    print()
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Portfolio Intelligence Engine - Integration Tests")
    print("=" * 60 + "\n")
    
    tests = [
        test_portfolio_tracker,
        test_portfolio_validator_dry_run,
        test_trade_executor
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
