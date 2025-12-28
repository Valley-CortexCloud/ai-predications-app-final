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
import sys
from io import StringIO  # Fixed: import from io (pandas.compat removed long ago)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def fetch_sp500() -> pd.DataFrame:
    """Fetch S&P 500 from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        # Fixed warning: wrap response.text in StringIO
        tables = pd.read_html(StringIO(requests.get(url, headers=HEADERS).text))
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
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        
        # iShares CSVs have ~10 metadata rows before holdings (e.g., "iShares Russell 1000 ETF", dates, then "----" line, then header)
        # Skip rows until we see the header line containing "Ticker"
        lines = response.text.splitlines()
        skip_rows = 0
        for i, line in enumerate(lines):
            if "Ticker" in line and "Name" in line:
                skip_rows = i
                break
        
        if skip_rows == 0:
            raise ValueError("Could not find holdings header (Ticker, Name) in CSV")
        
        # Read from the header line onward
        df = pd.read_csv(StringIO(response.text), skiprows=skip_rows)

        df = df.dropna(subset=['Ticker', 'Name'])  # Drop any rows missing core cols
        df = df[df['Ticker'] != '-']  # Remove cash/derivatives
        df = df[df['Ticker'].str.len() <= 6]  # Tickers usually 1-5 chars, catches bad rows
        df = df[pd.to_numeric(df['Weight (%)'], errors='coerce') > 0]  # Must have positive weight

        if 'Ticker' not in df.columns or 'Name' not in df.columns:
            raise ValueError("CSV format missing Ticker/Name columns")
        
        df = df.rename(columns={
            "Ticker": "symbol",
            "Name": "name"
        })
        
        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
        df = df[df["symbol"] != "-"]  # Remove cash/placeholders
        df = df[df["symbol"].str.len() <= 5]  # Basic ticker length filter
        
        df["exchange"] = "NYSE/NASDAQ"
        df["source"] = "Russell1000"
        
        return df[["symbol", "name", "exchange", "source"]].drop_duplicates(subset=["symbol"])
    
    except Exception as e:
        print(f"⚠️ Failed to fetch Russell 1000 (IWB holdings): {e}")
        print("   This URL sometimes returns 403 - try adding better headers or manual download.")
        print("   Manual: https://www.ishares.com/us/products/239707/ishares-russell-1000-etf → Download Holdings")
        return pd.DataFrame(columns=["symbol", "name", "exchange", "source"])

def main():
    print("🔍 Fetching ticker universe from authoritative sources...")
    print("="*60)
    
    sp500_df = fetch_sp500()
    russell1000_df = fetch_russell1000()
    
    print(f"✓ S&P 500: {len(sp500_df)} tickers")
    print(f"✓ Russell 1000: {len(russell1000_df)} tickers")
    
    if len(sp500_df) == 0 and len(russell1000_df) == 0:
        print("\n❌ FATAL: Could not fetch any ticker data!")
        return 1
    
    # Combine: S&P 500 first (priority in dedupe)
    combined = pd.concat([sp500_df, russell1000_df], ignore_index=True)
    
    combined = combined.drop_duplicates(subset=["symbol"], keep="first")
    
    combined = combined.sort_values("symbol").reset_index(drop=True)
    
    exclude = {"SPY", "QQQ", "DIA", "IWM", "IWB"}
    combined = combined[~combined["symbol"].isin(exclude)]
    
    print(f"\n📊 Final Universe:")
    print(f" Total tickers: {len(combined)}")
    print(f" Sources: {combined['source'].value_counts().to_dict()}")
    
    output_path = Path("config/ticker_universe.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved to: {output_path}")
    print(f"\n📋 Sample (first 10):")
    print(combined.head(10).to_string(index=False))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
