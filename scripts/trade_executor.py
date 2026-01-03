#!/usr/bin/env python3
"""
Trade Executor - Alpaca Order Generation & Submission

Generates and optionally submits orders based on confirmed portfolio changes.

Usage:
    # V1 (Manual) - Generate orders JSON
    python scripts/trade_executor.py --confirmed confirmed_2024-01-15.csv --paper
    
    # V2 (Auto) - Submit directly to Alpaca
    python scripts/trade_executor.py --confirmed confirmed_2024-01-15.csv --auto --paper
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

MAX_POSITION_SIZE = 0.15  # 15% max per name

CONVICTION_MULTIPLIERS = {
    'Strong Buy': 1.4,
    'Buy': 1.1,
    'Hold': 0.8,
    'Avoid': 0.0
}


# ============================================================================
# Position Sizing
# ============================================================================

def calculate_position_size(
    conviction: str,
    portfolio_value: float,
    max_position_size: float = MAX_POSITION_SIZE
) -> float:
    """
    Calculate position size with conviction weighting
    
    Args:
        conviction: 'Strong Buy', 'Buy', 'Hold', or 'Avoid'
        portfolio_value: Total portfolio value
        max_position_size: Maximum % per position
        
    Returns:
        Dollar amount to invest
    """
    base_size = portfolio_value * max_position_size
    multiplier = CONVICTION_MULTIPLIERS.get(conviction, 1.0)
    
    return base_size * multiplier


def extract_conviction_from_reason(reason: str) -> str:
    """
    Extract conviction level from Grok reason string
    
    Looks for patterns like "Strong Buy" or "Buy" in the reason
    """
    reason_upper = reason.upper()
    
    if 'STRONG BUY' in reason_upper:
        return 'Strong Buy'
    elif 'BUY' in reason_upper:
        return 'Buy'
    elif 'HOLD' in reason_upper:
        return 'Hold'
    elif 'AVOID' in reason_upper:
        return 'Avoid'
    
    # Default to Buy for new positions
    return 'Buy'


# ============================================================================
# Order Generation
# ============================================================================

def generate_orders(
    confirmed_df: pd.DataFrame,
    tracker_df: pd.DataFrame,
    portfolio_value: float,
    paper: bool = True
) -> Dict:
    """
    Generate order specifications from confirmed actions
    
    Returns dict with orders list and metadata
    """
    orders = []
    
    # Process SELL orders first (free up capital)
    sells = confirmed_df[confirmed_df['proposed_action'] == 'SELL']
    
    for _, row in sells.iterrows():
        symbol = row['symbol']
        
        # Get shares from tracker
        tracker_row = tracker_df[tracker_df['symbol'] == symbol]
        if tracker_row.empty:
            print(f"⚠️  Cannot sell {symbol} - not in tracker")
            continue
        
        shares = float(tracker_row.iloc[0]['shares'])
        
        order = {
            'symbol': symbol,
            'side': 'sell',
            'type': 'market',
            'qty': shares,
            'time_in_force': 'day',
            'reason': row.get('reason', 'Exit signal')
        }
        
        orders.append(order)
        print(f"📤 SELL: {symbol} - {shares} shares")
    
    # Process BUY orders
    buys = confirmed_df[confirmed_df['proposed_action'] == 'BUY']
    
    for _, row in buys.iterrows():
        symbol = row['symbol']
        reason = row.get('reason', '')
        
        # Extract conviction from reason
        conviction = extract_conviction_from_reason(reason)
        
        # Calculate notional amount
        notional = calculate_position_size(conviction, portfolio_value)
        
        order = {
            'symbol': symbol,
            'side': 'buy',
            'type': 'market',
            'notional': notional,  # Use notional for fractional shares
            'time_in_force': 'day',
            'conviction': conviction,
            'reason': reason
        }
        
        orders.append(order)
        print(f"📥 BUY: {symbol} - ${notional:,.0f} ({conviction})")
    
    # Build complete order spec
    order_spec = {
        'generated_at': datetime.now().isoformat(),
        'mode': 'paper' if paper else 'live',
        'portfolio_value': portfolio_value,
        'order_count': len(orders),
        'sell_count': len(sells),
        'buy_count': len(buys),
        'orders': orders
    }
    
    return order_spec


def save_orders_json(order_spec: Dict, output_dir: Path, date_str: Optional[str] = None) -> Path:
    """Save orders to JSON file"""
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    output_path = output_dir / f"orders_{date_str}.json"
    
    with open(output_path, 'w') as f:
        json.dump(order_spec, f, indent=2)
    
    return output_path


# ============================================================================
# Alpaca Submission (V2)
# ============================================================================

def submit_to_alpaca(
    order_spec: Dict,
    api_key: str,
    api_secret: str,
    paper: bool = True
) -> List[Dict]:
    """
    Submit orders to Alpaca API
    
    Requires alpaca-py package
    
    Returns list of fill results
    """
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
    except ImportError:
        raise ImportError(
            "alpaca-py not installed. Run: pip install alpaca-py>=0.21.0"
        )
    
    print(f"\n🔌 Connecting to Alpaca {'Paper' if paper else 'Live'} API...")
    
    client = TradingClient(api_key, api_secret, paper=paper)
    
    # Verify account
    account = client.get_account()
    print(f"✅ Connected - Account: {account.account_number}")
    print(f"   Buying Power: ${float(account.buying_power):,.2f}")
    
    fills = []
    
    for order in order_spec['orders']:
        symbol = order['symbol']
        side = OrderSide.BUY if order['side'] == 'buy' else OrderSide.SELL
        
        print(f"\n📝 Submitting {order['side'].upper()} {symbol}...")
        
        try:
            if order['side'] == 'sell':
                # Market sell with qty
                request = MarketOrderRequest(
                    symbol=symbol,
                    qty=order['qty'],
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
            else:
                # Market buy with notional (fractional shares)
                request = MarketOrderRequest(
                    symbol=symbol,
                    notional=order['notional'],
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
            
            # Submit order
            alpaca_order = client.submit_order(request)
            
            fill = {
                'symbol': symbol,
                'side': order['side'],
                'order_id': str(alpaca_order.id),
                'status': alpaca_order.status.value,
                'submitted_at': alpaca_order.submitted_at.isoformat() if alpaca_order.submitted_at else None,
                'qty': order.get('qty'),
                'notional': order.get('notional'),
                'reason': order.get('reason', '')
            }
            
            fills.append(fill)
            print(f"   ✅ Order submitted - ID: {alpaca_order.id}")
            print(f"   Status: {alpaca_order.status.value}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            fills.append({
                'symbol': symbol,
                'side': order['side'],
                'error': str(e)
            })
    
    return fills


# ============================================================================
# Main Execution Logic
# ============================================================================

def execute_trades(
    confirmed_file: Path,
    tracker_file: Path,
    output_dir: Path,
    portfolio_value: float,
    auto: bool = False,
    paper: bool = True,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None
) -> Path:
    """
    Main trade execution workflow
    
    Returns path to orders JSON file
    """
    print(f"{'='*60}")
    print("Trade Executor - Order Generation & Submission")
    print(f"{'='*60}")
    print(f"Mode: {'AUTO (Alpaca)' if auto else 'MANUAL (JSON only)'}")
    print(f"Account: {'Paper Trading' if paper else 'LIVE TRADING'}")
    print(f"Portfolio Value: ${portfolio_value:,.2f}")
    
    # Load confirmed actions
    if not confirmed_file.exists():
        raise FileNotFoundError(f"Confirmed file not found: {confirmed_file}")
    
    confirmed_df = pd.read_csv(confirmed_file)
    print(f"\n📋 Loaded {len(confirmed_df)} confirmed actions")
    
    # Load tracker
    if not tracker_file.exists():
        print(f"⚠️  Tracker not found - using empty tracker")
        tracker_df = pd.DataFrame(columns=['symbol', 'shares'])
    else:
        tracker_df = pd.read_csv(tracker_file)
    
    # Generate orders
    print(f"\n🔨 Generating orders...")
    order_spec = generate_orders(confirmed_df, tracker_df, portfolio_value, paper)
    
    # Extract date from confirmed filename
    date_str = None
    if confirmed_file.stem.startswith('confirmed_'):
        date_str = confirmed_file.stem.split('_')[1]
    
    # Save orders JSON
    orders_path = save_orders_json(order_spec, output_dir, date_str)
    print(f"\n💾 Orders saved: {orders_path}")
    
    # Auto-submit if requested
    if auto:
        print(f"\n{'='*60}")
        print("SUBMITTING TO ALPACA")
        print(f"{'='*60}")
        
        import os
        api_key = api_key or os.getenv('ALPACA_API_KEY')
        api_secret = api_secret or os.getenv('ALPACA_API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError(
                "Alpaca credentials not found. "
                "Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables"
            )
        
        fills = submit_to_alpaca(order_spec, api_key, api_secret, paper)
        
        # Save fill results
        fills_path = output_dir / f"fills_{date_str or datetime.now().strftime('%Y-%m-%d')}.json"
        with open(fills_path, 'w') as f:
            json.dump({'fills': fills}, f, indent=2)
        
        print(f"\n💾 Fill results saved: {fills_path}")
        print(f"\n✅ {len(fills)} orders submitted!")
        print(f"\n📋 Next steps:")
        print(f"   1. Monitor fills in Alpaca dashboard")
        print(f"   2. Run: python scripts/portfolio_tracker.py --update-fills {fills_path}")
        print(f"   3. Run: python scripts/portfolio_tracker.py --sync")
    else:
        print(f"\n📋 Next steps (Manual Mode):")
        print(f"   1. Review {orders_path}")
        print(f"   2. Submit orders manually in Alpaca dashboard")
        print(f"   3. Export fills to JSON")
        print(f"   4. Run: python scripts/portfolio_tracker.py --update-fills <fills.json>")
    
    return orders_path


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Trade Executor - Generate and submit orders to Alpaca"
    )
    parser.add_argument(
        '--confirmed',
        type=Path,
        required=True,
        help='Path to confirmed_YYYY-MM-DD.csv file'
    )
    parser.add_argument(
        '--tracker',
        type=Path,
        default=Path('data/portfolio/tracker.csv'),
        help='Path to tracker CSV (default: data/portfolio/tracker.csv)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/portfolio'),
        help='Output directory for orders JSON'
    )
    parser.add_argument(
        '--portfolio-value',
        type=float,
        default=100000.0,
        help='Total portfolio value for position sizing (default: $100,000)'
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Auto-submit orders to Alpaca (V2 mode)'
    )
    parser.add_argument(
        '--paper',
        action='store_true',
        default=True,
        help='Use paper trading account (default: True)'
    )
    parser.add_argument(
        '--live',
        action='store_true',
        help='Use LIVE trading account (DANGEROUS - requires explicit flag)'
    )
    
    args = parser.parse_args()
    
    # Safety check for live trading
    if args.live:
        paper = False
        print("\n" + "="*60)
        print("⚠️  LIVE TRADING MODE ENABLED ⚠️")
        print("="*60)
        response = input("Are you SURE you want to trade with real money? (type 'YES'): ")
        if response != 'YES':
            print("Aborted.")
            return 1
    else:
        paper = True
    
    try:
        orders_path = execute_trades(
            args.confirmed,
            args.tracker,
            args.output_dir,
            args.portfolio_value,
            args.auto,
            paper
        )
        
        print(f"\n✅ Trade execution complete!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
