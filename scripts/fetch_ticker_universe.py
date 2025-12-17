#!/usr/bin/env python3
"""
Fetch ticker universe from authoritative sources (S&P 500 + NASDAQ 100).
Outputs:  config/ticker_universe.csv with columns: symbol, name, exchange

Sources:
  - Wikipedia S&P 500
  - Wikipedia NASDAQ 100
"""
import pandas as pd
import requests
from pathlib import Path
from typing import List, Set
import sys

# ✅ ADD USER-AGENT HEADERS
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def fetch_sp500() -> pd.DataFrame:
    """Fetch S&P 500 from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        # ✅ Pass headers to read_html via requests
        tables = pd.read_html(url, storage_options={'User-Agent': HEADERS['User-Agent']})
        df = tables[0]. copy()
        
        # Normalize symbols
        df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
        df["Symbol"] = df["Symbol"]. replace({
            "BRK.B": "BRK-B",
            "BF.B": "BF-B"
        })
        
        # Standardize columns
        df = df.rename(columns={
            "Symbol": "symbol",
            "Security": "name",
        })
        
        df["exchange"] = "NYSE/NASDAQ"
        df["source"] = "SP500"
        
        return df[["symbol", "name", "exchange", "source"]]
    
    except Exception as e:
        print(f"⚠️  Failed to fetch S&P 500: {e}")
        return pd.DataFrame(columns=["symbol", "name", "exchange", "source"])


def fetch_nasdaq100() -> pd.DataFrame:
    """Fetch NASDAQ 100 from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        # ✅ Pass headers to read_html
        tables = pd.read_html(url, storage_options={'User-Agent': HEADERS['User-Agent']})
        df = tables[4].copy()  # Usually table 4
        
        df["Ticker"] = df["Ticker"]. astype(str).str.upper().str.strip()
        
        df = df.rename(columns={
            "Ticker": "symbol",
            "Company": "name"
        })
        
        df["exchange"] = "NASDAQ"
        df["source"] = "NASDAQ100"
        
        return df[["symbol", "name", "exchange", "source"]]
    
    except Exception as e:
        print(f"⚠️  Failed to fetch NASDAQ 100: {e}")
        return pd.DataFrame(columns=["symbol", "name", "exchange", "source"])


def main():
    print("🔍 Fetching ticker universe from authoritative sources...")
    print("="*60)
    
    # Fetch from multiple sources
    sp500_df = fetch_sp500()
    nasdaq100_df = fetch_nasdaq100()
    
    print(f"✓ S&P 500: {len(sp500_df)} tickers")
    print(f"✓ NASDAQ 100: {len(nasdaq100_df)} tickers")
    
    # ✅ EXIT IF NO DATA FETCHED
    if len(sp500_df) == 0 and len(nasdaq100_df) == 0:
        print("\n❌ FATAL: Could not fetch any ticker data!")
        print("   Wikipedia may be blocking requests.")
        print("   Check network connectivity or try manual fetch.")
        return 1
    
    # Combine and deduplicate (S&P 500 takes priority)
    combined = pd.concat([sp500_df, nasdaq100_df], ignore_index=True)
    
    # Dedupe:  keep first occurrence (S&P 500 prioritized)
    combined = combined. drop_duplicates(subset=["symbol"], keep="first")
    
    # Sort alphabetically
    combined = combined.sort_values("symbol").reset_index(drop=True)
    
    # Filter out ETFs and special cases
    exclude = {"SPY", "QQQ", "DIA", "IWM"}
    combined = combined[~combined["symbol"].isin(exclude)]
    
    print(f"\n📊 Final Universe:")
    print(f"   Total tickers: {len(combined)}")
    print(f"   Sources: {combined['source'].value_counts().to_dict()}")
    
    # Save
    output_path = Path("config/ticker_universe.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved to: {output_path}")
    print(f"\n📋 Sample (first 10):")
    print(combined.head(10).to_string(index=False))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
