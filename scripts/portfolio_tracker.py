#!/usr/bin/env python3
"""
Portfolio Tracker - State Management Utilities

Manages portfolio state, syncs prices, calculates P&L, and generates reports.

Usage:
    python scripts/portfolio_tracker.py --init                    # Initialize empty tracker
    python scripts/portfolio_tracker.py --sync                    # Sync prices and calculate P&L
    python scripts/portfolio_tracker.py --report                  # Generate performance report
    python scripts/portfolio_tracker.py --update-fills <file>     # Update from Alpaca fills
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


TRACKER_PATH = Path('data/portfolio/tracker.csv')


# ============================================================================
# Core Functions
# ============================================================================

def init_tracker(tracker_path: Path = TRACKER_PATH) -> None:
    """Initialize empty portfolio tracker"""
    print(f"🔧 Initializing tracker at {tracker_path}")
    
    # Create directory if needed
    tracker_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create empty tracker with proper columns
    df = pd.DataFrame(columns=[
        'symbol', 'entry_date', 'entry_price', 'shares', 
        'current_price', 'pnl_pct', 'days_held', 'last_exit_score'
    ])
    
    df.to_csv(tracker_path, index=False)
    print(f"✅ Tracker initialized (empty)")


def sync_prices(tracker_path: Path = TRACKER_PATH) -> pd.DataFrame:
    """
    Sync current prices from yfinance and calculate P&L
    
    Returns updated DataFrame
    """
    print(f"📊 Syncing prices from yfinance...")
    
    if not tracker_path.exists():
        raise FileNotFoundError(f"Tracker not found: {tracker_path}")
    
    tracker = pd.read_csv(tracker_path)
    
    if tracker.empty:
        print("📭 Portfolio is empty")
        return tracker
    
    # Ensure entry_date is datetime
    tracker['entry_date'] = pd.to_datetime(tracker['entry_date'])
    
    # Fetch current prices
    symbols = tracker['symbol'].tolist()
    print(f"   Fetching prices for {len(symbols)} symbols...")
    
    for i, row in tracker.iterrows():
        symbol = row['symbol']
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d')
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                tracker.at[i, 'current_price'] = current_price
                
                # Calculate P&L
                entry_price = row['entry_price']
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                tracker.at[i, 'pnl_pct'] = pnl_pct
                
                # Calculate days held
                days_held = (datetime.now() - row['entry_date']).days
                tracker.at[i, 'days_held'] = days_held
                
                print(f"   ✓ {symbol}: ${current_price:.2f} ({pnl_pct:+.1f}%)")
            else:
                print(f"   ⚠️  {symbol}: No data available")
        
        except Exception as e:
            print(f"   ❌ {symbol}: Error fetching price - {e}")
    
    # Save updated tracker
    tracker.to_csv(tracker_path, index=False)
    print(f"\n💾 Tracker updated: {tracker_path}")
    
    return tracker


def update_from_fills(fills_file: Path, tracker_path: Path = TRACKER_PATH) -> pd.DataFrame:
    """
    Update tracker with fills from Alpaca orders JSON
    
    Fills file format:
    {
        "fills": [
            {
                "symbol": "AAPL",
                "side": "buy",
                "qty": 10,
                "price": 150.25,
                "timestamp": "2024-01-15T10:30:00Z"
            },
            ...
        ]
    }
    """
    print(f"📥 Updating tracker from fills: {fills_file}")
    
    if not fills_file.exists():
        raise FileNotFoundError(f"Fills file not found: {fills_file}")
    
    with open(fills_file) as f:
        data = json.load(f)
    
    fills = data.get('fills', [])
    if not fills:
        print("⚠️  No fills found in file")
        return pd.read_csv(tracker_path) if tracker_path.exists() else pd.DataFrame()
    
    # Load tracker
    if tracker_path.exists():
        tracker = pd.read_csv(tracker_path)
        tracker['entry_date'] = pd.to_datetime(tracker['entry_date'])
    else:
        tracker = pd.DataFrame(columns=[
            'symbol', 'entry_date', 'entry_price', 'shares',
            'current_price', 'pnl_pct', 'days_held', 'last_exit_score'
        ])
    
    # Process fills
    for fill in fills:
        symbol = fill['symbol']
        side = fill['side'].lower()
        qty = float(fill['qty'])
        price = float(fill['price'])
        timestamp = pd.to_datetime(fill['timestamp'])
        
        if side == 'buy':
            # Add or increase position
            existing = tracker[tracker['symbol'] == symbol]
            
            if existing.empty:
                # New position
                new_row = {
                    'symbol': symbol,
                    'entry_date': timestamp.strftime('%Y-%m-%d'),
                    'entry_price': price,
                    'shares': qty,
                    'current_price': price,
                    'pnl_pct': 0.0,
                    'days_held': 0,
                    'last_exit_score': 0
                }
                tracker = pd.concat([tracker, pd.DataFrame([new_row])], ignore_index=True)
                print(f"   ✓ New position: {symbol} - {qty} shares @ ${price:.2f}")
            else:
                # Average up
                idx = existing.index[0]
                old_shares = tracker.at[idx, 'shares']
                old_price = tracker.at[idx, 'entry_price']
                
                new_shares = old_shares + qty
                new_avg_price = ((old_shares * old_price) + (qty * price)) / new_shares
                
                tracker.at[idx, 'shares'] = new_shares
                tracker.at[idx, 'entry_price'] = new_avg_price
                print(f"   ✓ Added to {symbol}: +{qty} shares @ ${price:.2f} (avg: ${new_avg_price:.2f})")
        
        elif side == 'sell':
            # Reduce or close position
            existing = tracker[tracker['symbol'] == symbol]
            
            if existing.empty:
                print(f"   ⚠️  Cannot sell {symbol} - no position found")
                continue
            
            idx = existing.index[0]
            old_shares = tracker.at[idx, 'shares']
            
            if qty >= old_shares:
                # Close position completely
                tracker = tracker.drop(idx)
                print(f"   ✓ Closed position: {symbol} - sold {qty} shares @ ${price:.2f}")
            else:
                # Partial exit
                tracker.at[idx, 'shares'] = old_shares - qty
                print(f"   ✓ Reduced {symbol}: -{qty} shares @ ${price:.2f} ({old_shares - qty} remaining)")
    
    # Save updated tracker
    tracker.to_csv(tracker_path, index=False)
    print(f"\n💾 Tracker updated: {tracker_path}")
    
    return tracker


def generate_report(tracker_path: Path = TRACKER_PATH) -> None:
    """Generate performance report"""
    print(f"{'='*60}")
    print("Portfolio Performance Report")
    print(f"{'='*60}")
    
    if not tracker_path.exists():
        print("⚠️  Tracker not found")
        return
    
    tracker = pd.read_csv(tracker_path)
    
    if tracker.empty:
        print("📭 Portfolio is empty")
        return
    
    # Ensure numeric types
    tracker['shares'] = pd.to_numeric(tracker['shares'], errors='coerce')
    tracker['entry_price'] = pd.to_numeric(tracker['entry_price'], errors='coerce')
    tracker['current_price'] = pd.to_numeric(tracker['current_price'], errors='coerce')
    tracker['pnl_pct'] = pd.to_numeric(tracker['pnl_pct'], errors='coerce')
    tracker['days_held'] = pd.to_numeric(tracker['days_held'], errors='coerce')
    
    # Calculate position values
    tracker['position_value'] = tracker['shares'] * tracker['current_price']
    tracker['cost_basis'] = tracker['shares'] * tracker['entry_price']
    tracker['pnl_dollar'] = tracker['position_value'] - tracker['cost_basis']
    
    # Portfolio totals
    total_value = tracker['position_value'].sum()
    total_cost = tracker['cost_basis'].sum()
    total_pnl = tracker['pnl_dollar'].sum()
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
    
    # Display holdings
    print(f"\nHoldings ({len(tracker)} positions):")
    print(f"{'-'*60}")
    
    display_cols = ['symbol', 'shares', 'entry_price', 'current_price', 'pnl_pct', 'position_value', 'days_held']
    display = tracker[display_cols].copy()
    display['shares'] = display['shares'].apply(lambda x: f"{x:.2f}")
    display['entry_price'] = display['entry_price'].apply(lambda x: f"${x:.2f}")
    display['current_price'] = display['current_price'].apply(lambda x: f"${x:.2f}")
    display['pnl_pct'] = display['pnl_pct'].apply(lambda x: f"{x:+.1f}%")
    display['position_value'] = display['position_value'].apply(lambda x: f"${x:,.0f}")
    
    print(display.to_string(index=False))
    
    print(f"\n{'-'*60}")
    print(f"Portfolio Summary:")
    print(f"{'-'*60}")
    print(f"Total Value:        ${total_value:,.2f}")
    print(f"Total Cost Basis:   ${total_cost:,.2f}")
    print(f"Total P&L:          ${total_pnl:,.2f} ({total_pnl_pct:+.1f}%)")
    print(f"Average Days Held:  {tracker['days_held'].mean():.0f} days")
    
    # Winners and losers
    winners = tracker[tracker['pnl_pct'] > 0]
    losers = tracker[tracker['pnl_pct'] < 0]
    
    print(f"\nWinners: {len(winners)} / Losers: {len(losers)}")
    
    if not winners.empty:
        best = winners.nlargest(3, 'pnl_pct')
        print(f"\nTop 3 Winners:")
        for _, row in best.iterrows():
            print(f"  {row['symbol']}: {row['pnl_pct']:+.1f}%")
    
    if not losers.empty:
        worst = losers.nsmallest(3, 'pnl_pct')
        print(f"\nTop 3 Losers:")
        for _, row in worst.iterrows():
            print(f"  {row['symbol']}: {row['pnl_pct']:+.1f}%")
    
    print(f"{'='*60}")


def sync_from_alpaca(api_key: Optional[str] = None, 
                     api_secret: Optional[str] = None,
                     paper: bool = True,
                     tracker_path: Path = TRACKER_PATH) -> pd.DataFrame:
    """
    Sync portfolio state from Alpaca API
    
    Requires alpaca-py package
    """
    try:
        from alpaca.trading.client import TradingClient
    except ImportError:
        raise ImportError("alpaca-py not installed. Run: pip install alpaca-py")
    
    import os
    
    api_key = api_key or os.getenv('ALPACA_API_KEY')
    api_secret = api_secret or os.getenv('ALPACA_API_SECRET')
    
    if not api_key or not api_secret:
        raise ValueError("Alpaca credentials not found. Set ALPACA_API_KEY and ALPACA_API_SECRET")
    
    print(f"🔌 Syncing from Alpaca {'Paper' if paper else 'Live'} API...")
    
    client = TradingClient(api_key, api_secret, paper=paper)
    
    # Get all positions
    positions = client.get_all_positions()
    
    if not positions:
        print("📭 No positions in Alpaca account")
        # Create empty tracker
        tracker = pd.DataFrame(columns=[
            'symbol', 'entry_date', 'entry_price', 'shares',
            'current_price', 'pnl_pct', 'days_held', 'last_exit_score'
        ])
        tracker.to_csv(tracker_path, index=False)
        return tracker
    
    # Build tracker from positions
    rows = []
    for pos in positions:
        pnl_pct = float(pos.unrealized_plpc) * 100
        
        rows.append({
            'symbol': pos.symbol,
            'entry_date': 'unknown',  # Alpaca doesn't provide entry date easily
            'entry_price': float(pos.avg_entry_price),
            'shares': float(pos.qty),
            'current_price': float(pos.current_price),
            'pnl_pct': pnl_pct,
            'days_held': 0,  # Cannot calculate without entry date
            'last_exit_score': 0
        })
    
    tracker = pd.DataFrame(rows)
    tracker.to_csv(tracker_path, index=False)
    
    print(f"✅ Synced {len(tracker)} positions from Alpaca")
    print(tracker[['symbol', 'shares', 'entry_price', 'current_price', 'pnl_pct']].to_string(index=False))
    
    return tracker


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Portfolio Tracker - State management utilities"
    )
    parser.add_argument(
        '--tracker',
        type=Path,
        default=TRACKER_PATH,
        help='Path to tracker CSV (default: data/portfolio/tracker.csv)'
    )
    
    # Actions (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        '--init',
        action='store_true',
        help='Initialize empty tracker'
    )
    action_group.add_argument(
        '--sync',
        action='store_true',
        help='Sync current prices and calculate P&L'
    )
    action_group.add_argument(
        '--report',
        action='store_true',
        help='Generate performance report'
    )
    action_group.add_argument(
        '--update-fills',
        type=Path,
        metavar='FILE',
        help='Update from Alpaca fills JSON file'
    )
    action_group.add_argument(
        '--sync-alpaca',
        action='store_true',
        help='Sync from Alpaca API (requires alpaca-py)'
    )
    
    # Alpaca options
    parser.add_argument(
        '--paper',
        action='store_true',
        default=True,
        help='Use Alpaca paper trading account (default: True)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.init:
            init_tracker(args.tracker)
        
        elif args.sync:
            sync_prices(args.tracker)
            print("\n💡 Tip: Run --report to see portfolio summary")
        
        elif args.report:
            generate_report(args.tracker)
        
        elif args.update_fills:
            update_from_fills(args.update_fills, args.tracker)
            print("\n💡 Tip: Run --sync to update current prices")
        
        elif args.sync_alpaca:
            sync_from_alpaca(paper=args.paper, tracker_path=args.tracker)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
