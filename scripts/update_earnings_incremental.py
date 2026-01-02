#!/usr/bin/env python3
"""
Incremental earnings calendar updater using existing build_earnings_calendar.py. 
Loads existing earnings.csv, fetches recent events (Yahoo-first!), merges, dedupes, saves.

New batch processing features:
- --batch-size: Process N tickers per run (prevents timeouts)
- --batch-offset: Starting offset in symbol list (for sequential batches)
- --stale-list: File with stale tickers to process (one per line)
- --stale-only: Only process tickers from stale list

Note: Rate limiting is handled internally by build_earnings_calendar (especially for AlphaVantage).
Batching is the primary mechanism to prevent GitHub Actions timeouts.
"""
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sys
import random 

script_dir = Path(__file__).parent  # scripts/data/
scripts_root = script_dir.parent     # scripts/
sys.path.insert(0, str(scripts_root))


# Import from your existing script
from build_earnings_calendar import (
    build_earnings_calendar,
    load_symbols,
    parse_provider_order
)

def main():
    ap = argparse.ArgumentParser(description="Incremental earnings calendar updater (Yahoo-based)")
    ap.add_argument("--existing", type=str, required=True, help="Path to existing earnings.csv")
    ap.add_argument("--output", type=str, required=True, help="Output path for updated CSV")
    ap.add_argument("--lookback-days", type=int, default=90, help="Days to look back for updates (default: 90)")
    ap.add_argument("--symbols-file", type=str, default=None, help="Symbol CSV (default: from cache)")
    ap.add_argument("--provider-order", type=str, default="yahoo,finnhub,fmp,alphavantage", 
                    help="Provider priority (default: yahoo,finnhub,fmp,alphavantage)")
    ap.add_argument("--max-symbols", type=int, default=0, help="Limit symbols to process (0=all, default: 0)")
    ap.add_argument("--verbose", action="store_true")
    
    # Batch processing arguments
    ap.add_argument("--batch-size", type=int, default=0, help="Number of tickers to process per batch (0=all)")
    ap.add_argument("--batch-offset", type=int, default=0, help="Starting offset in symbol list (default: 0)")
    ap.add_argument("--stale-list", type=str, default=None, help="Path to file with stale tickers (one per line)")
    ap.add_argument("--stale-only", action="store_true", help="Only process tickers from stale list")
    
    args = ap.parse_args()
    
    # Load existing data
    existing_df = None
    if Path(args.existing).exists():
        existing_df = pd.read_csv(args.existing)
        existing_df['earnings_date'] = pd.to_datetime(existing_df['earnings_date'])
        print(f"✓ Loaded existing earnings:  {len(existing_df):,} rows")
        print(f"  Date range: {existing_df['earnings_date'].min().date()} to {existing_df['earnings_date'].max().date()}")
        print(f"  Symbols: {existing_df['symbol'].nunique()}")
    else:
        print(f"ℹ️  No existing file found at {args.existing}")
        print(f"   Creating fresh dataset from scratch")
    
    # Determine date range for incremental update
    end_date = datetime.now().date()
    
    if existing_df is not None and len(existing_df) > 0:
        # Incremental mode:  fetch last N days + some overlap
        last_event = existing_df['earnings_date'].max().date()
        lookback_start = end_date - timedelta(days=args.lookback_days)
        start_date = min(last_event - timedelta(days=7), lookback_start)  # 7-day overlap to catch updates
        mode = "INCREMENTAL"
    else: 
        # Fresh build:  go back 2 years
        start_date = end_date - timedelta(days=730)
        mode = "FULL BUILD"
    
    print(f"\n{'='*60}")
    print(f"MODE: {mode}")
    print(f"{'='*60}")
    print(f"Date range: {start_date} to {end_date} ({(end_date - start_date).days} days)")
    
    # Load symbols
    all_symbols = load_symbols(args.symbols_file)
    
    # Apply stale list filter if provided
    if args.stale_list and Path(args.stale_list).exists():
        with open(args.stale_list, 'r') as f:
            stale_symbols = [line.strip() for line in f if line.strip()]
        print(f"✓ Loaded {len(stale_symbols)} stale tickers from {args.stale_list}")
        
        if args.stale_only:
            # Only process stale symbols
            symbols = [s for s in all_symbols if s in stale_symbols]
            print(f"✓ Filtered to {len(symbols)} symbols from stale list")
        else:
            # Prioritize stale symbols but include others
            stale_set = set(stale_symbols)
            stale_in_universe = [s for s in all_symbols if s in stale_set]
            non_stale = [s for s in all_symbols if s not in stale_set]
            random.shuffle(non_stale)
            symbols = stale_in_universe + non_stale
            print(f"✓ Prioritized {len(stale_in_universe)} stale symbols, {len(non_stale)} others")
    else:
        symbols = all_symbols
        random.shuffle(symbols)
        print(f"✓ Loaded {len(symbols)} symbols from {args.symbols_file} (randomized order)")
    
    # Apply batch processing if specified
    if args.batch_size > 0:
        batch_end = args.batch_offset + args.batch_size
        original_count = len(symbols)
        symbols = symbols[args.batch_offset:batch_end]
        print(f"✓ Batch processing: offset {args.batch_offset}, size {args.batch_size}")
        print(f"  Processing symbols {args.batch_offset+1} to {min(batch_end, original_count)} of {original_count}")
    elif args.max_symbols > 0:
        symbols = symbols[:args.max_symbols]
        print(f"⚠️  Limited to {args.max_symbols} symbols for testing")
    
    if len(symbols) == 0:
        print(f"⚠️  No symbols to process after filtering/batching")
        return 0
    
    print(f"Processing {len(symbols)} symbols")
    print(f"Provider order: {args.provider_order}")
    print(f"{'='*60}\n")
    
    # Fetch new/updated earnings using YOUR existing function
    new_df = build_earnings_calendar(
        symbols=symbols,
        start=start_date,
        end=end_date,
        use_nasdaq_fallback=False,  # Yahoo is primary
        dedupe_window_days=2,
        normalize_to="bdays",
        provider_order=parse_provider_order(args.provider_order),
        disable_yahoo=False,  # ← YAHOO IS ENABLED! 
        require_eps=False,  # Keep all events (EPS may come later)
        av_min_interval=12.5,
        av_retry_cooldown=65.0,
        av_max_retries=1,
        verbose=args.verbose
    )
    
    print(f"\n✓ Fetched {len(new_df):,} earnings events")
    
    if len(new_df) == 0:
        print(f"⚠️  No new earnings data fetched!")
        if existing_df is not None: 
            print(f"   Keeping existing file unchanged")
            return 0
        else:
            print(f"   Nothing to save")
            return 1
    
    # Merge with existing
    if existing_df is not None and len(existing_df) > 0:
        print(f"\n📊 Merging with existing data...")
        
        # Combine datasets
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Convert date columns
        combined['earnings_date'] = pd.to_datetime(combined['earnings_date'])
        combined['trading_date'] = pd.to_datetime(combined['trading_date'])
        
        # Dedupe strategy:  prefer higher quality sources for same symbol+date
        source_rank = {
            'yahoo': 4,  # Yahoo is best for dates (free!)
            'finnhub': 3,
            'fmp':  2,
            'alphavantage': 1,
            'nasdaq': 0
        }
        
        # Sort by symbol, date, and source quality
        combined['_source_rank'] = combined['source']. map(source_rank).fillna(-1)
        combined = combined. sort_values(
            ['symbol', 'earnings_date', '_source_rank'],
            ascending=[True, True, False]
        )
        
        # Keep best source for each symbol+date combination
        combined = combined. drop_duplicates(
            subset=['symbol', 'earnings_date'],
            keep='first'  # Keeps highest-ranked source
        )
        
        combined = combined.drop(columns=['_source_rank'])
        
        final_df = combined
        
        print(f"   Existing rows:      {len(existing_df):,}")
        print(f"   New rows fetched:  {len(new_df):,}")
        print(f"   After merge+dedupe: {len(final_df):,}")
        print(f"   Net change:        {len(final_df) - len(existing_df):+,} rows")
    else:
        final_df = new_df
        print(f"\n   Created fresh dataset:  {len(final_df):,} rows")
    
    # Convert back to string format for CSV
    final_df['earnings_date'] = pd.to_datetime(final_df['earnings_date']).dt.strftime('%Y-%m-%d')
    final_df['trading_date'] = pd.to_datetime(final_df['trading_date']).dt.strftime('%Y-%m-%d')
    
    # Sort and save
    final_df = final_df.sort_values(['symbol', 'earnings_date']).reset_index(drop=True)
    final_df.to_csv(args.output, index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ SAVED TO: {args.output}")
    print(f"{'='*60}")
    print(f"Total rows:    {len(final_df):,}")
    print(f"Symbols:       {final_df['symbol'].nunique()}")
    print(f"Date range:    {final_df['earnings_date'].min()} to {final_df['earnings_date'].max()}")
    print(f"Sources:       {final_df['source'].value_counts().to_dict()}")
    print(f"{'='*60}")

if __name__ == "__main__":
    sys.exit(main())
