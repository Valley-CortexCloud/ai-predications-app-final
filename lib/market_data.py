#!/usr/bin/env python3
"""Market data utilities for loading SPY and sector ETF histories from cache."""
import pandas as pd
from pathlib import Path
from typing import Optional

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
    """Get the ETF cache directory."""
    return ETF_CACHE_DIR


def fetch_spy_history(period: str = '2y', max_retries: int = 3) -> Optional[pd.DataFrame]:
    """Load SPY history from existing cache files.
    
    Args:
        period: History period (e.g., '2y', '5y', '10y')
        max_retries:  Ignored (kept for backwards compatibility)
        
    Returns: 
        DataFrame with SPY OHLCV data or None if not found
    """
    cache_dir = get_etf_cache_dir()
    
    # Try multiple filename patterns (raw, adj, or no suffix)
    possible_files = [
        cache_dir / f'SPY_{period}_raw.parquet',
        cache_dir / f'SPY_{period}_adj.parquet',
        cache_dir / f'SPY_{period}.parquet',
    ]
    
    for cache_file in possible_files: 
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    print(f"  Loaded SPY from:  {cache_file. name}")
                    return df
            except Exception as e:
                print(f"  Warning: Failed to load {cache_file.name}: {e}")
                continue
    
    print(f"  ⚠️  SPY not found in {cache_dir} (tried {period}_raw. parquet, {period}_adj. parquet)")
    return None


def fetch_sector_etf_history(etf_symbol: str, period:  str = '2y', max_retries: int = 3) -> Optional[pd.DataFrame]: 
    """Load sector ETF history from existing cache files.
    
    Args:
        etf_symbol: ETF ticker symbol (e. g., 'XLK', 'XLF')
        period: History period
        max_retries: Ignored (kept for backwards compatibility)
        
    Returns:
        DataFrame with ETF OHLCV data or None if not found
    """
    cache_dir = get_etf_cache_dir()
    
    # Try multiple filename patterns
    possible_files = [
        cache_dir / f'{etf_symbol}_{period}_raw. parquet',
        cache_dir / f'{etf_symbol}_{period}_adj.parquet',
        cache_dir / f'{etf_symbol}_{period}. parquet',
    ]
    
    for cache_file in possible_files:
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    return df
            except Exception as e: 
                print(f"  Warning: Failed to load {cache_file.name}: {e}")
                continue
    
    # Silently return None if not found (sector ETFs are optional)
    return None


def get_sector_etf_symbol(sector_name: str) -> str:
    """Map sector name to ETF symbol, defaulting to SPY for market."""
    return SECTOR_ETFS.get(sector_name, 'SPY')
