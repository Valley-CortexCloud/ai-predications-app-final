#!/usr/bin/env python3
"""
Earnings staleness validator.
Identifies tickers that need earnings data refresh based on:
1. Last earnings date is too old (> 90 days)
2. Insufficient quarterly coverage (< 6 quarters in last 2 years)
3. New tickers in universe but missing from earnings.csv
4. Random sampling for rolling freshness checks
"""
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import random
import sys

def identify_stale_tickers(
    earnings_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    stale_days: int = 90,
    min_quarters: int = 6,
    random_sample_pct: float = 0.10
) -> list:
    """
    Returns list of dicts with ticker info that needs refreshing.
    
    Each dict contains:
    - symbol (str): The ticker symbol
    - reason (str): One of 'new_ticker', 'stale_date', 'low_coverage', 'random_refresh'
    - priority (int): 1=highest (new), 2=high (stale), 3=medium (low coverage), 4=low (random)
    - last_date (str, optional): For stale_date reason, the last earnings date
    - quarters (int, optional): For low_coverage reason, number of recent quarters
    
    Staleness criteria:
    1. max_earnings_date < today - stale_days
    2. < min_quarters in last 2 years
    3. In universe but not in earnings.csv (new tickers)
    4. Random sample of fresh tickers for rolling checks
    """
    today = pd.Timestamp.today().normalize()
    cutoff = today - pd.Timedelta(days=stale_days)
    two_years_ago = today - pd.Timedelta(days=730)
    
    stale = []
    fresh = []
    
    universe_symbols = set(universe_df['symbol'].unique())
    earnings_symbols = set(earnings_df['symbol'].unique()) if not earnings_df.empty else set()
    
    # Check each symbol in universe
    for symbol in universe_symbols:
        if symbol not in earnings_symbols:
            # New ticker - needs full fetch
            stale.append({
                'symbol': symbol,
                'reason': 'new_ticker',
                'priority': 1  # High priority
            })
            continue
        
        sym_data = earnings_df[earnings_df['symbol'] == symbol].copy()
        sym_data['earnings_date'] = pd.to_datetime(sym_data['earnings_date'])
        
        max_date = sym_data['earnings_date'].max()
        recent_data = sym_data[sym_data['earnings_date'] >= two_years_ago]
        recent_quarters = len(recent_data)
        
        if max_date < cutoff:
            stale.append({
                'symbol': symbol,
                'reason': 'stale_date',
                'last_date': max_date.strftime('%Y-%m-%d'),
                'priority': 2  # Medium-high priority
            })
        elif recent_quarters < min_quarters:
            stale.append({
                'symbol': symbol,
                'reason': 'low_coverage',
                'quarters': recent_quarters,
                'priority': 3  # Medium priority
            })
        else:
            fresh.append(symbol)
    
    # Add random sample of fresh tickers for rolling freshness
    if fresh and random_sample_pct > 0:
        sample_size = max(1, int(len(fresh) * random_sample_pct))
        sampled = random.sample(fresh, min(sample_size, len(fresh)))
        for symbol in sampled:
            stale.append({
                'symbol': symbol,
                'reason': 'random_refresh',
                'priority': 4  # Low priority
            })
    
    # Sort by priority (highest priority first)
    stale.sort(key=lambda x: x['priority'])
    
    return stale

def main():
    parser = argparse.ArgumentParser(description="Validate earnings data staleness")
    parser.add_argument("--earnings", type=str, required=True, help="Path to earnings.csv")
    parser.add_argument("--universe", type=str, required=True, help="Path to ticker_universe.csv")
    parser.add_argument("--stale-days", type=int, default=90, help="Days threshold for staleness (default: 90)")
    parser.add_argument("--min-quarters", type=int, default=6, help="Minimum quarters required in last 2 years (default: 6)")
    parser.add_argument("--random-sample-pct", type=float, default=0.10, help="Percentage of fresh tickers to randomly sample (default: 0.10)")
    parser.add_argument("--output", type=str, required=True, help="Output file for stale ticker list")
    parser.add_argument("--max-tickers", type=int, default=0, help="Maximum tickers to output (0=all)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    # Load data
    earnings_df = pd.DataFrame()
    if Path(args.earnings).exists():
        earnings_df = pd.read_csv(args.earnings)
        print(f"✓ Loaded earnings: {len(earnings_df):,} rows, {earnings_df['symbol'].nunique()} symbols")
    else:
        print(f"⚠️ No earnings file found at {args.earnings}")
    
    universe_df = pd.read_csv(args.universe)
    print(f"✓ Loaded universe: {len(universe_df)} symbols")
    
    # Find stale tickers
    stale = identify_stale_tickers(
        earnings_df,
        universe_df,
        stale_days=args.stale_days,
        min_quarters=args.min_quarters,
        random_sample_pct=args.random_sample_pct
    )
    
    print(f"\n📊 Staleness Analysis:")
    print(f"   Total universe: {len(universe_df)}")
    print(f"   Stale tickers:  {len(stale)}")
    
    # Count by reason
    reasons = {}
    for t in stale:
        r = t['reason']
        reasons[r] = reasons.get(r, 0) + 1
    for reason, count in sorted(reasons.items()):
        print(f"   - {reason}: {count}")
    
    # Apply max limit if specified
    if args.max_tickers > 0 and len(stale) > args.max_tickers:
        stale = stale[:args.max_tickers]
        print(f"\n⚠️ Limited to {args.max_tickers} tickers")
    
    # Write output
    if stale:
        with open(args.output, 'w') as f:
            for t in stale:
                f.write(f"{t['symbol']}\n")
        print(f"\n✅ Wrote {len(stale)} tickers to {args.output}")
    else:
        # Create empty file
        Path(args.output).touch()
        print(f"\n✅ No stale tickers found - created empty {args.output}")
    
    if args.verbose:
        print(f"\nStale tickers:")
        for t in stale[:20]:  # Show first 20
            print(f"   {t['symbol']}: {t['reason']}")
        if len(stale) > 20:
            print(f"   ... and {len(stale) - 20} more")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
