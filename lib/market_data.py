#!/usr/bin/env python3
"""Market data utilities for fetching and caching SPY and sector ETF histories."""
import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import Optional
import time

# Sector ETF mapping
SECTOR_ETFS = {
    'Technology': 'XLK',
    'Consumer Cyclical': 'XLY',
    'Consumer Defensive': 'XLP',
    'Energy': 'XLE',
    'Financial Services': 'XLF',
    'Financials': 'XLF',
    'Healthcare': 'XLV',
    'Industrials': 'XLI',
    'Basic Materials': 'XLB',
    'Real Estate': 'XLRE',
    'Utilities': 'XLU',
    'Communication Services': 'XLC',
    'Communication': 'XLC',
}

# Default cache directory
ETF_CACHE_DIR = Path('data_cache/_etf_cache')


def get_etf_cache_dir() -> Path:
    """Get or create the ETF cache directory."""
    ETF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return ETF_CACHE_DIR


def fetch_spy_history(period: str = '5y', max_retries: int = 3) -> Optional[pd.DataFrame]:
    """Fetch SPY history with caching.
    
    Args:
        period: History period (e.g., '5y', '10y')
        max_retries: Maximum retry attempts
        
    Returns:
        DataFrame with SPY OHLCV data or None on failure
    """
    cache_dir = get_etf_cache_dir()
    cache_file = cache_dir / f'SPY_{period}_adj.parquet'
    
    # Try to load from cache
    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            # Check if cache is recent (less than 1 day old for daily updates)
            if not df.empty and (pd.Timestamp.now() - df.index[-1]) < pd.Timedelta(days=1):
                return df
        except Exception:
            pass
    
    # Fetch from yfinance
    for attempt in range(max_retries):
        try:
            spy = yf.Ticker('SPY')
            hist = spy.history(period=period, auto_adjust=True)
            if hist is not None and len(hist) > 0:
                hist.to_parquet(cache_file, compression='zstd')
                return hist
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to fetch SPY after {max_retries} attempts: {e}")
                return None
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return None


def fetch_sector_etf_history(etf_symbol: str, period: str = '5y', max_retries: int = 3) -> Optional[pd.DataFrame]:
    """Fetch sector ETF history with caching.
    
    Args:
        etf_symbol: ETF ticker symbol (e.g., 'XLK', 'XLF')
        period: History period
        max_retries: Maximum retry attempts
        
    Returns:
        DataFrame with ETF OHLCV data or None on failure
    """
    cache_dir = get_etf_cache_dir()
    cache_file = cache_dir / f'{etf_symbol}_{period}_adj.parquet'
    
    # Try to load from cache
    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            if not df.empty and (pd.Timestamp.now() - df.index[-1]) < pd.Timedelta(days=1):
                return df
        except Exception:
            pass
    
    # Fetch from yfinance
    for attempt in range(max_retries):
        try:
            etf = yf.Ticker(etf_symbol)
            hist = etf.history(period=period, auto_adjust=True)
            if hist is not None and len(hist) > 0:
                hist.to_parquet(cache_file, compression='zstd')
                return hist
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to fetch {etf_symbol} after {max_retries} attempts: {e}")
                return None
            time.sleep(2 ** attempt)
    
    return None


def get_sector_etf_symbol(sector_name: str) -> str:
    """Map sector name to ETF symbol, defaulting to SPY for market."""
    return SECTOR_ETFS.get(sector_name, 'SPY')
