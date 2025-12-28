#!/usr/bin/env python3
"""
Fetch ticker universe from authoritative sources (S&P 500 + Russell 1000).
Outputs: config/ticker_universe.csv with columns: symbol, name, exchange, source

Sources:
  - Wikipedia S&P 500
  - iShares IWB holdings CSV (exact Russell 1000 replicate)
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
        tables = pd.read_html(requests.get(url, headers=HEADERS).text)
        df = tables[0].copy()
        
        df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
        df["Symbol"] = df["Symbol"].replace({
            "BRK.B": "BRK-B",
            "BF.B": "BF-B"
        })
        
        df = df.rename(columns={
            "Symbol": "symbol",
            "Security": "name",
        })
        
        df["exchange"] = "NYSE/NASDAQ"
        df["source"] = "SP500"
        
        return df[["symbol", "name", "exchange", "source"]]
    
    except Exception as e:
        print(f"⚠️ Failed to fetch S&P 500: {e}")
        return pd.DataFrame(columns=["symbol", "name", "exchange", "source"])

def fetch_russell1000() -> pd.DataFrame:
    """Fetch Russell 1000 from iShares IWB holdings CSV."""
    url = "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
    try:
        # Download CSV
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        
        # Read with pandas, skip metadata rows (holdings start after ~10 rows with "-" or after header)
        df = pd.read_csv(pd.compat.StringIO(response.text))
        
        # Find actual holdings: rows with valid Ticker (not NaN or "-")
        df = df[pd.to_numeric(df.get('Shares', 0), errors='coerce') > 0]  # Rough filter on shares
        
        if 'Ticker' not in df.columns or 'Name' not in df.columns:
            raise ValueError("CSV format changed - missing Ticker/Name columns")
        
        df = df.rename(columns={
            "Ticker": "symbol",
            "Name": "name"
        })
        
        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
        df = df[df["symbol"] != "-"]  # Remove cash/etc.
        
        df["exchange"] = "NYSE/NASDAQ"
        df["source"] = "Russell1000"
        
        return df[["symbol", "name", "exchange", "source"]].drop_duplicates(subset=["symbol"])
    
    except Exception as e:
        print(f"⚠️ Failed to fetch Russell 1000 (IWB holdings): {e}")
        print("   Try manual download from: https://www.ishares.com/us/products/239707/ishares-russell-1000-etf")
        return pd.DataFrame(columns=["symbol", "name", "exchange", "source"])

def main():
    print("🔍 Fetching ticker universe from authoritative sources...")
    print("="*60)
    
    # Fetch from multiple sources
    sp500_df = fetch_sp500()
    russell1000_df = fetch_russell1000()
    
    print(f"✓ S&P 500: {len(sp500_df)} tickers")
    print(f"✓ Russell 1000: {len(russell1000_df)} tickers")
    
    # ✅ EXIT IF NO DATA FETCHED
    if len(sp500_df) == 0 and len(russell1000_df) == 0:
        print("\n❌ FATAL: Could not fetch any ticker data!")
        return 1
    
    # Combine: S&P 500 first for priority in dedupe
    combined = pd.concat([sp500_df, russell1000_df], ignore_index=True)
    
    # Dedupe: keep first (S&P 500 prioritized)
    combined = combined.drop_duplicates(subset=["symbol"], keep="first")
    
    # Sort alphabetically
    combined = combined.sort_values("symbol").reset_index(drop=True)
    
    # Filter out major ETFs
    exclude = {"SPY", "QQQ", "DIA", "IWM", "IWB"}  # Added IWB just in case
    combined = combined[~combined["symbol"].isin(exclude)]
    
    print(f"\n📊 Final Universe:")
    print(f" Total tickers: {len(combined)}")
    print(f" Sources: {combined['source'].value_counts().to_dict()}")
    
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
