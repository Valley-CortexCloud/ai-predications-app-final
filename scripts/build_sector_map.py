#!/usr/bin/env python3
"""
Build a sector map (symbol -> SPDR ETF) for your current universe.

Priority (strict, no correlation overrides):
1) Official GICS for S&P 500 from Wikipedia (SP500 table)
2) Yahoo sector -> SPDR (yfinance), with optional conservative heuristics (off by default)
3) FinanceDatabase sector normalized to GICS-like names
4) Correlation fallback to SPDR sector ETFs ONLY when all above are missing

NEVER override an existing mapping (GICS/Yahoo/FinanceDB) with correlation.

Outputs CSV columns:
  symbol, sector_etf, source, sector_name, industry, corr_top_etf, corr_top_value

Usage (recommended):
  pip install yfinance requests-cache
  python3 scripts/build_sector_map.py --out configs/sector_map.csv --cache-requests

Options:
  --tickers AAPL,MSFT       Comma-separated tickers to map (overrides cache scan)
  --symbols-file path.csv   CSV with a 'symbol' column (overrides cache scan if --tickers not set)
  --cache-root /path        Override data_paths.CACHE_DIR for scanning and correlation fallback
  --no-yfinance             Disable Yahoo lookups
  --with-heuristics         Apply tiny conservative heuristics on top of Yahoo only
  --lookback 252            Correlation lookback (used only for last-resort fallback)
  --min-overlap-days 60     Minimum overlap days for correlation
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import sys
import requests

# Project default CACHE_DIR (used if --cache-root not provided)
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_paths import CACHE_DIR  # your data_cache root

SECTOR_ETFS = ["XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY"]

GICS_TO_SPDR = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

FINANCEDB_NORM = {
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Technology": "Information Technology",
    "Financial Services": "Financials",
    "Healthcare": "Health Care",
    "Basic Materials": "Materials",
}

def try_imports():
    yf = None
    cache = None
    try:
        import yfinance as yf_mod
        yf = yf_mod
    except Exception:
        yf = None
    try:
        import requests_cache
        cache = requests_cache
    except Exception:
        cache = None
    return yf, cache

def yahoo_symbol_norm(sym: str) -> str:
    # Normalize e.g., BRK.B -> BRK-B, BF.B -> BF-B
    if "." in sym and sym.upper() not in ("GOOG","GOOGL"):
        return sym.replace(".", "-")
    return sym

def parse_tickers_arg(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    toks = [t.strip().upper() for t in str(raw).split(",")]
    return [t for t in toks if t]

def list_universe_symbols(cache_root: Path) -> List[str]:
    files = sorted(cache_root.rglob("*_10y_*.parquet"))
    syms: List[str] = []
    for fp in files:
        if not fp.is_file() or fp.name.endswith("_features.parquet"):
            continue
        sym = fp.stem.split("_")[0].upper()
        if sym == "SPY" or sym in SECTOR_ETFS:
            continue
        syms.append(sym)
    return sorted(set(syms))

def load_parquet_indexed(fp: Path) -> pd.DataFrame:
    df = pd.read_parquet(fp)
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.set_index("Date")
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is not None:
            df.index = df.index.tz_convert("America/New_York").tz_localize(None)
        df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "Date"
    return df.sort_index()

def load_close_series(df: pd.DataFrame) -> pd.Series:
    for col in ("Adj Close", "AdjClose", "Close"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").astype(float)
            s.name = "close"
            return s
    raise ValueError("No Close/Adj Close in DF")

def daily_return(s: pd.Series) -> pd.Series:
    return s.pct_change()

def find_cache_for_ticker_anywhere(ticker: str, cache_root: Path) -> Optional[Path]:
    cand = list(cache_root.rglob(f"{ticker}_*10y_*.parquet"))
    if cand:
        return sorted(cand)[0]
    cand = list(cache_root.rglob(f"{ticker}_*.parquet"))
    if cand:
        return sorted(cand)[0]
    return None

def load_sector_etf_returns(cache_root: Path) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for sec in SECTOR_ETFS:
        fp = find_cache_for_ticker_anywhere(sec, cache_root)
        if not fp:
            continue
        df = load_parquet_indexed(fp)
        close = load_close_series(df)
        out[sec] = daily_return(close)
    if not out:
        raise SystemExit(f"No sector ETF caches found under {cache_root}; ensure XLB..XLY exist.")
    return out

def corr_top_sector(sym: str, sector_rets: Dict[str, pd.Series], cache_root: Path, lookback=252, min_overlap_days=60) -> Tuple[str, float]:
    fp = find_cache_for_ticker_anywhere(sym, cache_root)
    if not fp:
        return "", float("nan")
    df = load_parquet_indexed(fp)
    close = load_close_series(df)
    r_sym = daily_return(close)
    end = r_sym.dropna().index.max()
    if end is None:
        return "", float("nan")
    start = end - pd.Timedelta(days=lookback*2)
    r_sym_w = r_sym.loc[r_sym.index >= start]
    best = ("", -9e9)
    for sec, r_sec in sector_rets.items():
        both = pd.concat([r_sym_w, r_sec.reindex_like(r_sym_w)], axis=1).dropna()
        if len(both) < min_overlap_days:
            continue
        c = both.iloc[-lookback:].corr().iloc[0,1] if len(both) >= lookback else both.corr().iloc[0,1]
        if pd.notna(c) and c > best[1]:
            best = (sec, float(c))
    return best

def fetch_sp500_gics() -> Dict[str,str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url)
        sp500 = tables[0]
        sp500["Symbol"] = sp500["Symbol"].astype(str).str.upper().replace({"BRK.B":"BRK-B","BF.B":"BF-B"})
        return dict(zip(sp500["Symbol"], sp500["GICS Sector"]))
    except Exception:
        return {}

def fetch_financedb_sectors() -> Dict[str,str]:
    url = "https://raw.githubusercontent.com/JerBouma/FinanceDatabase/master/Database/Equities/equities.json"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        out = {}
        for ticker, info in data.items():
            t = str(ticker).upper()
            sector = info.get("sector")
            if not sector:
                continue
            sector = FINANCEDB_NORM.get(sector, sector)
            out[t] = sector
        return out
    except Exception:
        return {}

def fetch_yahoo(yf_mod, sym: str) -> Tuple[Optional[str], Optional[str]]:
    if yf_mod is None:
        return None, None
    try:
        t = yf_mod.Ticker(sym)
        info = t.get_info() if hasattr(t, "get_info") else t.info
        sector = info.get("sector")
        industry = info.get("industry")
        # Normalize to GICS names
        if sector == "Technology": sector = "Information Technology"
        elif sector == "Healthcare": sector = "Health Care"
        elif sector == "Financial Services": sector = "Financials"
        elif sector == "Consumer Cyclical": sector = "Consumer Discretionary"
        elif sector == "Consumer Defensive": sector = "Consumer Staples"
        elif sector == "Basic Materials": sector = "Materials"
        return sector, industry
    except Exception:
        return None, None

def heuristics_on_yahoo(spdr: Optional[str], sector: Optional[str], industry: Optional[str], symbol: str, enabled: bool) -> Tuple[Optional[str], str]:
    if not enabled or not spdr:
        return spdr, "yahoo"
    s = f"{sector or ''} {industry or ''} {symbol}".lower()
    # Only a few safe rules
    if "coal" in s and "mining" in s: return "XLE", "yahoo+heur"
    if "mining" in s and "coal" not in s: return "XLB", "yahoo+heur"
    if any(k in s for k in ["restaurant","e-commerce","ecommerce","online retail","home improvement"]): return "XLY", "yahoo+heur"
    if "utility" in s: return "XLU", "yahoo+heur"
    if any(k in s for k in ["bank","insurance","brokerage","asset management","capital market"]): return "XLF", "yahoo+heur"
    if "reit" in s or "real estate" in s: return "XLRE", "yahoo+heur"
    if "semiconductor" in s or "chip" in s: return "XLK", "yahoo+heur"
    if "auto" in s and ("part" in s or "component" in s): return "XLI", "yahoo+heur"
    return spdr, "yahoo"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="configs/sector_map.csv")
    ap.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers to map (overrides cache scan)")
    ap.add_argument("--symbols-file", type=str, default="config/ticker_universe.csv", help="CSV with 'symbol' column (overrides cache scan if --tickers not set)")
    ap.add_argument("--cache-root", type=str, default=None, help="Override data_paths.CACHE_DIR for scanning and correlation fallback")
    ap.add_argument("--no-yfinance", action="store_true")
    ap.add_argument("--cache-requests", action="store_true")
    ap.add_argument("--with-heuristics", action="store_true", help="Conservative heuristics applied only on top of Yahoo")
    ap.add_argument("--lookback", type=int, default=252)
    ap.add_argument("--min-overlap-days", type=int, default=60)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve cache root
    cache_root = Path(args.cache_root) if args.cache_root else Path(CACHE_DIR)

    yf_mod, req_cache = try_imports()
    if args.no_yfinance:
        yf_mod = None
    elif yf_mod is not None and req_cache is not None and args.cache_requests:
        # optional HTTP caching
        sess = req_cache.CachedSession("yahoo_cache", expire_after=86400)
        try:
            yf_mod.shared._requests = sess  # type: ignore
            print("Enabled requests-cache for Yahoo lookups.")
        except Exception:
            pass

    # Resolve symbols: --tickers > --symbols-file > cache scan
    if args.tickers:
        symbols = parse_tickers_arg(args.tickers)
        print(f"Using {len(symbols)} symbols from --tickers")
    elif args.symbols_file:
        df_sym = pd.read_csv(args.symbols_file)
        col = "symbol" if "symbol" in df_sym.columns else df_sym.columns[0]
        symbols = (df_sym[col].astype(str).str.upper().str.strip().tolist())
        # Dedup while preserving order
        seen = set(); symbols = [s for s in symbols if (s not in seen and not seen.add(s) and s != "SPY")]
        print(f"Using {len(symbols)} symbols from --symbols-file")
    else:
        symbols = list_universe_symbols(cache_root)
        print(f"Found {len(symbols)} symbols under {cache_root}")

    sp500 = fetch_sp500_gics()
    financedb = fetch_financedb_sectors()
    sector_rets = load_sector_etf_returns(cache_root)

    rows = []
    for i, sym in enumerate(symbols, 1):
        chosen_spdr = ""
        source = ""
        sector_name = ""
        industry = ""

        # 1) SP500 GICS
        gics = sp500.get(sym)
        if gics and gics in GICS_TO_SPDR:
            chosen_spdr = GICS_TO_SPDR[gics]
            source = "gics_sp500"
            sector_name = gics
        else:
            # 2) Yahoo
            ysec = yind = None
            if yf_mod is not None:
                ysym = yahoo_symbol_norm(sym)
                ysec, yind = fetch_yahoo(yf_mod, ysym)
            if ysec and ysec in GICS_TO_SPDR:
                sp = GICS_TO_SPDR[ysec]
                sp, src = heuristics_on_yahoo(sp, ysec, yind, sym, enabled=args.with_heuristics)
                chosen_spdr = sp or ""
                source = src
                sector_name = ysec or ""
                industry = yind or ""
            else:
                # 3) FinanceDatabase
                fsec = financedb.get(sym)
                if fsec and fsec in GICS_TO_SPDR:
                    chosen_spdr = GICS_TO_SPDR[fsec]
                    source = "financedb"
                    sector_name = fsec
                else:
                    # 4) Correlation fallback (only if everything else missing)
                    top_etf, top_val = corr_top_sector(sym, sector_rets, cache_root, lookback=args.lookback, min_overlap_days=args.min_overlap_days)
                    chosen_spdr = top_etf or ""
                    source = "corr_fallback" if chosen_spdr else ""
                    sector_name = ""
                    industry = ""

        # For logging: compute top corr anyway (optional)
        top_etf_log, top_val_log = corr_top_sector(sym, sector_rets, cache_root, lookback=args.lookback, min_overlap_days=args.min_overlap_days)

        rows.append({
            "symbol": sym,
            "sector_etf": chosen_spdr,
            "source": source,
            "sector_name": sector_name,
            "industry": industry,
            "corr_top_etf": top_etf_log,
            "corr_top_value": round(top_val_log, 6) if isinstance(top_val_log, float) and not np.isnan(top_val_log) else "",
        })

        if i % 50 == 0 or i == len(symbols):
            print(f"Processed {i}/{len(symbols)} symbols...")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote sector map for {len(df)} symbols to {out_path}")
    counts = df["source"].value_counts(dropna=False).to_dict()
    print("Counts by source:", counts)

if __name__ == "__main__":
    main()
