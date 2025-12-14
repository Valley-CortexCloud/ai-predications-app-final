#!/usr/bin/env python3
"""Sector mapping utilities with caching."""
import yfinance as yf
import json
from pathlib import Path
from typing import Optional
import time

# Default sector cache path
SECTOR_CACHE_PATH = Path('data_cache/sector_map.json')


def load_sector_cache() -> dict:
    """Load sector mapping cache from disk."""
    if SECTOR_CACHE_PATH.exists():
        try:
            with open(SECTOR_CACHE_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_sector_cache(cache: dict):
    """Save sector mapping cache to disk."""
    SECTOR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(SECTOR_CACHE_PATH, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save sector cache: {e}")


def get_ticker_sector(ticker: str, max_retries: int = 3) -> Optional[str]:
    """Get sector for a ticker using yfinance with caching.
    
    Args:
        ticker: Stock ticker symbol
        max_retries: Maximum retry attempts
        
    Returns:
        Sector name string or None if unavailable
    """
    # Load cache
    cache = load_sector_cache()
    
    # Check cache
    if ticker in cache:
        return cache[ticker]
    
    # Fetch from yfinance
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            sector = info.get('sector')
            
            if sector:
                # Update cache
                cache[ticker] = sector
                save_sector_cache(cache)
                return sector
            else:
                # Cache None result to avoid repeated lookups
                cache[ticker] = None
                save_sector_cache(cache)
                return None
                
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to fetch sector for {ticker} after {max_retries} attempts: {e}")
                # Cache None to avoid repeated failures
                cache[ticker] = None
                save_sector_cache(cache)
                return None
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return None


def get_sector_batch(tickers: list, verbose: bool = False) -> dict:
    """Get sectors for multiple tickers efficiently.
    
    Args:
        tickers: List of ticker symbols
        verbose: Print progress
        
    Returns:
        Dictionary mapping ticker to sector (or None)
    """
    result = {}
    cache = load_sector_cache()
    
    # Separate cached and uncached
    uncached = []
    for ticker in tickers:
        if ticker in cache:
            result[ticker] = cache[ticker]
        else:
            uncached.append(ticker)
    
    # Fetch uncached
    if uncached:
        if verbose:
            print(f"Fetching sectors for {len(uncached)} tickers...")
        
        for i, ticker in enumerate(uncached):
            if verbose and i % 10 == 0:
                print(f"  Progress: {i}/{len(uncached)}")
            
            sector = get_ticker_sector(ticker)
            result[ticker] = sector
            
            # Small delay to avoid rate limits
            if i < len(uncached) - 1:
                time.sleep(0.1)
    
    return result
