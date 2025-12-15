#!/usr/bin/env python3
"""Fast vectorized feature augmentation with multiprocessing support.

Usage:
    python scripts/augment_caches_fast.py [--processes N] [--overwrite] [--limit N]
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
from multiprocessing import Pool, cpu_count
from functools import partial

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.features import compute_all_features
from lib.market_data import fetch_spy_history, fetch_sector_etf_history, get_sector_etf_symbol
from lib.sector import get_ticker_sector
from pathlib import Path

# Hardcoded cache dirs (matches your structure)
ROOT = Path(__file__).parent.parent
TICKER_CACHE_DIR = ROOT / "data_cache" / "10y_ticker_features"
ETF_CACHE_DIR = ROOT / "data_cache" / "_etf_cache"

# Required OHLCV columns for validation
REQUIRED_OHLCV_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']


def _normalize_daily_index(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Make index tz-naive date-only, deduped and sorted. Accepts df with index or 'Date' column."""
    if df is None or len(df) == 0:
        return df
    # If 'Date' is a column and index isn't a DatetimeIndex, prefer setting it as index
    if not isinstance(df.index, pd.DatetimeIndex) and 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']).set_index('Date')
        except Exception:
            # Fall back to coercing the existing index
            pass
    # Coerce index to datetime if not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')
    # Drop timezone info without changing the day
    if df.index.tz is not None:
        try:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        except Exception:
            df.index = df.index.tz_localize(None)
    # Normalize to day boundary (00:00) preserving the calendar date
    df.index = df.index.normalize()
    df.index.name = 'Date'
    # Deduplicate on Date and sort
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def process_ticker_file(file_path: Path, overwrite: bool = False, 
                        spy_df: pd.DataFrame = None, verbose: bool = True) -> dict:
    """Process a single ticker cache file and augment with features.
    
    Args:
        file_path: Path to ticker cache file
        overwrite: Whether to overwrite existing features file
        spy_df: SPY DataFrame for market-relative features
        verbose: Print progress messages
        
    Returns:
        Dict with processing stats
    """
    ticker = file_path.stem.split('_')[0]  # Extract ticker from filename
    out_file = file_path.with_name(file_path.stem + '_features.parquet')
    
    # Check if already exists
    if out_file.exists() and not overwrite:
        if verbose:
            print(f"  Skipping {ticker}: features file exists (use --overwrite to replace)")
        return {'ticker': ticker, 'status': 'skipped', 'reason': 'exists'}
    
    try:
        # Load ticker data
        df = pd.read_parquet(file_path)
        df = _normalize_daily_index(df)

        if df is None or df.empty or len(df) < 50:
            if verbose:
                n = 0 if df is None else len(df)
                print(f"  Skipping {ticker}: insufficient data ({n} rows)")
            return {'ticker': ticker, 'status': 'skipped', 'reason': 'insufficient_data'}
        
        # Get sector for this ticker
        sector = get_ticker_sector(ticker, max_retries=2)
        sector_df = None
        
        if sector:
            etf_symbol = get_sector_etf_symbol(sector)
            sector_df = fetch_sector_etf_history(etf_symbol, period='2y', max_retries=2)
            sector_df = _normalize_daily_index(sector_df)
        
        # Normalize SPY once per call (already pre-normalized in main, but safe to reapply)
        spy_df_local = _normalize_daily_index(spy_df.copy()) if spy_df is not None else None

        # Compute all features
        features = compute_all_features(df, market_df=spy_df_local, sector_df=sector_df)
        
        # Join original OHLCV with features
        # Reset index to ensure alignment by row position
        df_reset = df.reset_index()
        features_reset = features.reset_index(drop=True)
        
        # Combine
        out_df = pd.concat([df_reset, features_reset], axis=1)
        
        # Remove duplicate columns if any
        out_df = out_df.loc[:, ~out_df.columns.duplicated()]
        
        # Optionally downcast to float32 to save space
        float_cols = out_df.select_dtypes(include=['float64']).columns
        out_df[float_cols] = out_df[float_cols].astype('float32')
        
        # Debug logging: Show what columns are being saved
        if verbose:
            print(f"  Columns being saved: {list(out_df.columns)[:10]}... (total: {len(out_df.columns)})")
        
        # Defensive check: Ensure OHLCV columns exist before saving
        missing_cols = [col for col in REQUIRED_OHLCV_COLUMNS if col not in out_df.columns]
        if missing_cols:
            if verbose:
                print(f"  Warning: Missing OHLCV columns: {missing_cols}")
            return {
                'ticker': ticker,
                'status': 'error',
                'error': f"Missing required columns: {missing_cols}"
            }
        
        # Write to parquet with compression, preserving Date column but not as index
        # This ensures Date is explicitly saved as a column for proper round-tripping
        out_df.to_parquet(out_file, compression='zstd', index=False)
        
        if verbose:
            print(f"  ✓ {ticker}: {len(out_df)} rows, {len(features.columns)} features")
        
        return {
            'ticker': ticker,
            'status': 'success',
            'rows': len(out_df),
            'features': len(features.columns)
        }
        
    except Exception as e:
        if verbose:
            print(f"  ✗ {ticker}: Error - {e}")
        return {
            'ticker': ticker,
            'status': 'error',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Fast vectorized feature augmentation')
    parser.add_argument('--processes', type=int, default=None,
                       help='Number of parallel processes (default: CPU count)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing feature files')
    parser.add_argument('--limit', type=int, default=0,
                       help='Limit number of files to process (0 = all)')
    parser.add_argument('--tickers', type=str, default=None,
                       help='Comma-separated list of tickers to process (default: all)')
    parser.add_argument('--cache-dir', type=str, default=str(TICKER_CACHE_DIR),
                       help='Directory to search for cache files (default: from data_paths)')
    args = parser.parse_args()
    
    # Set number of processes
    n_processes = args.processes if args.processes else max(1, cpu_count() - 1)
    
    print(f"Fast feature augmentation")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Processes: {n_processes}")
    print(f"Overwrite: {args.overwrite}")
    
    # Find all ticker cache files
    cache_path = Path(args.cache_dir)
    if not cache_path.exists():
        print(f"Error: Cache directory {cache_path} does not exist")
        return 1
    
    # Find files matching pattern *_2y_*.parquet (but not *_features.parquet)
    all_files = sorted([
        f for f in cache_path.glob('*_2y_*.parquet')
        if '_features' not in f.stem
    ])
    
    # Filter by ticker list if provided
    if args.tickers:
        ticker_list = [t.strip().upper() for t in args.tickers.split(',')]
        all_files = [f for f in all_files if f.stem.split('_')[0] in ticker_list]
    
    # Apply limit
    if args.limit > 0:
        all_files = all_files[:args.limit]
    
    if not all_files:
        print("No cache files found to process")
        return 0
    
    print(f"Found {len(all_files)} files to process")
    
    # Fetch SPY data once (shared across all tickers)
    print("\nFetching SPY data for market-relative features...")
    spy_df = fetch_spy_history(period='2y')
    spy_df = _normalize_daily_index(spy_df)
    if spy_df is None:
        print("Warning: Could not fetch SPY data, skipping market-relative features")
    else:
        print(f"Loaded SPY data: {len(spy_df)} rows (normalized)")

    # Process files
    print(f"\nProcessing {len(all_files)} tickers...")
    start_time = time.time()
    
    if n_processes == 1:
        # Single-threaded for debugging
        results = []
        for i, file_path in enumerate(all_files, 1):
            print(f"\n[{i}/{len(all_files)}] Processing {file_path.stem}...")
            result = process_ticker_file(file_path, overwrite=args.overwrite, 
                                        spy_df=spy_df, verbose=True)
            results.append(result)
    else:
        # Multi-process
        process_func = partial(process_ticker_file, overwrite=args.overwrite,
                              spy_df=spy_df, verbose=False)
        
        with Pool(processes=n_processes) as pool:
            results = []
            for i, result in enumerate(pool.imap(process_func, all_files), 1):
                results.append(result)
                ticker = result['ticker']
                status = result['status']
                
                if status == 'success':
                    print(f"[{i}/{len(all_files)}] ✓ {ticker}: {result['rows']} rows, {result['features']} features")
                elif status == 'skipped':
                    print(f"[{i}/{len(all_files)}] ⊘ {ticker}: {result.get('reason', 'skipped')}")
                else:
                    print(f"[{i}/{len(all_files)}] ✗ {ticker}: {result.get('error', 'unknown error')}")
    
    elapsed = time.time() - start_time
    
    # Summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')
    error_count = sum(1 for r in results if r['status'] == 'error')
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total files: {len(all_files)}")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(all_files):.2f}s per ticker)")
    print(f"{'='*60}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
