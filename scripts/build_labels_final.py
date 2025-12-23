#!/usr/bin/env python3
"""
Build 3-month (63 trading days) market-excess labels ONLY.
Merges precomputed features from *_features.parquet (shifted T-1 to avoid leakage).

v3.1 (FIXED):
- ONLY computes labels (excess_63d, label_rel_grade)
- Merges precomputed features from augment_caches_fast + enhance_features
- CRITICAL FIX: Features are shifted by 1 day (use T-1 data to predict T→T+63)
- Adds earnings calendar context (days_to_earn, prev_surprise, etc.)
- Fixed: Double "_features" in filename lookup
- Fixed: Earnings streak bounds checking
"""
import argparse
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import sys
import logging

# Add repo root for data_paths import
sys.path.insert(0, str(Path(__file__).parent.parent))
from pathlib import Path

# ADD at top
ROOT = Path(__file__).parent.parent
TICKER_CACHE_DIR = ROOT / "data_cache" / "10y_ticker_features"
ETF_CACHE_DIR = ROOT / "data_cache" / "_etf_cache"

# ---------------- Config ----------------
HORIZON = 63
BENCHMARK = "SPY"
MIN_PRICE = 5.0
MIN_ADV20_DOLLAR = 2_000_000.0
NY_TZ = "America/New_York"

# Production mode config
SAMPLE_TICKER_COUNT = 5  # Number of tickers to sample when checking latest date
MAX_STALE_DAYS = 5  # Maximum days before considering data stale
EXCLUDE_COLUMNS = {"Open", "High", "Low", "Close", "Adj Close", "Volume", "Date", "date", "symbol"}  # Columns to exclude from features

# ---------------- CLI ----------------

def parse_args():
    ap = argparse.ArgumentParser(description="Build market-excess labels from precomputed features (v3.1).")
    ap.add_argument("--output", type=str, default="datasets/train_excess63d_graded.parquet")
    ap.add_argument("--tickers", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0)
    # Grading
    ap.add_argument("--edges", type=str, default="0.0,0.60,0.75,0.85,0.92,0.97,0.99,1.0")
    ap.add_argument("--gains", type=str, default="0,1,2,4,8,16,32")
    # Context
    ap.add_argument("--earnings-file", type=str, default=None, help="Earnings CSV for calendar features")
    ap.add_argument("--exclude-earnings-pre", type=int, default=0)
    ap.add_argument("--exclude-earnings-post", type=int, default=0)
    # Paths
    ap.add_argument("--cache-dir", type=str, default=str(TICKER_CACHE_DIR))
    ap.add_argument("--spy-file", type=str, default=None, help="Direct path to SPY Parquet")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--production-only", action="store_true", 
                    help="Only output most recent date for production inference (not full history)")
    return ap.parse_args()

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("build_labels.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# ---------------- Time helpers ----------------

def normalize_index_to_ny_dates(idx: pd.Index) -> pd.DatetimeIndex:
    if pd.api.types.is_numeric_dtype(idx):
        idx = pd.to_datetime(idx, unit='ms', errors="coerce")
    else:
        idx = pd.to_datetime(idx, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(NY_TZ).tz_localize(None)
    idx = pd.to_datetime(idx).normalize()
    idx = idx[~idx.duplicated(keep="last")]
    return pd.DatetimeIndex(idx)

def normalize_df_index_to_ny(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")
    else:
        df.index = pd.to_datetime(df.index, errors="coerce")
    
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]
    
    idx = normalize_index_to_ny_dates(df.index)
    df.index = idx
    df.index.name = "Date"
    return df.sort_index()

def to_ny_date_col(dates_like) -> pd.Series:
    if isinstance(dates_like, pd.DatetimeIndex):
        idx = dates_like
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(NY_TZ).tz_localize(None)
        idx = pd.DatetimeIndex(pd.to_datetime(idx, errors="coerce")).normalize()
        return pd.Series(idx)
    
    s = pd.Series(dates_like)
    s = pd.to_datetime(s, errors="coerce")
    try:
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            s = s.dt.tz_convert(NY_TZ).dt.tz_localize(None)
    except Exception:
        pass
    return s.dt.normalize()

# ---------------- IO helpers ----------------

def find_cache_for_ticker(ticker: str, cache_dir: str) -> Optional[Path]:
    """
    Find raw OHLCV file for ticker. Searches:
    1. cache_dir/{ticker}_*.parquet (for stock files)
    2. cache_dir/../_etf_cache/{ticker}_*.parquet (for ETFs/indexes)
    
    Returns path to RAW file (not *_features.parquet)
    """
    p = Path(cache_dir)
    
    # Try main cache directory first (for stock files)
    cand = list(p.glob(f"{ticker}_*2y_*.parquet")) + list(p.glob(f"{ticker}_*.parquet"))
    cand = [f for f in cand if not f.name.endswith("_features.parquet") and "_features_enhanced" not in f.name]
    if cand:
        return sorted(cand)[0]
    
    # Try _etf_cache in PARENT directory (go up one level if we're in a ticker features directory)
    cache_root = p.parent if p.name in ["2y_ticker_features", "10y_ticker_features"] else p
    etf_dir = cache_root / '_etf_cache'
    
    if etf_dir.exists():
        cand = list(etf_dir.glob(f"{ticker}_*2y_*.parquet")) + list(etf_dir.glob(f"{ticker}_*.parquet"))
        cand = [f for f in cand if not f.name.endswith("_features.parquet") and "_features_enhanced" not in f.name]
        if cand:
            return sorted(cand)[0]
    
    # Special handling for tickers with special chars (^VIX, BTC-USD)
    if ticker.upper() in ["^VIX", "VIX", "^BTC-USD", "BTC-USD", "TLT", "SPY"]:
        # Normalize ticker name (remove special chars)
        norm = ticker.replace("^", "").replace("-", "").upper()
        
        # Search both directories
        for search_dir in [p, etf_dir]:
            if search_dir.exists():
                # Use rglob to search recursively
                cand = list(search_dir.rglob(f"*{norm}*2y*.parquet"))
                cand = [f for f in cand if not f.name.endswith("_features.parquet") and "_features_enhanced" not in f.name]
                if cand:
                    return sorted(cand)[0]
    
    return None

def list_training_files(limit: int, tickers: Optional[List[str]], cache_dir: str) -> List[Path]:
    root = Path(cache_dir)
    files: List[Path] = []
    
    if tickers:
        for t in tickers:
            fp = find_cache_for_ticker(t, cache_dir)
            if fp:
                files.append(fp)
    else:
        files = list(root.glob("*_2y_*.parquet")) + list(root.glob("*.parquet"))
        files = [f for f in files if f.is_file() and not f.name.endswith("_features.parquet") and "_features_enhanced" not in f.name]
        files = [f for f in files if not f.name.startswith("SPY_")]
    
    if limit > 0:
        files = files[:limit]
    
    return files

def load_parquet_indexed(fp: Path) -> pd.DataFrame:
    df = pd.read_parquet(fp)
    return normalize_df_index_to_ny(df)

# ---------------- Returns ----------------

def forward_return(s: pd.Series, H: int) -> pd.Series:
    """Forward return from date T: (price[T+H] / price[T]) - 1"""
    s = s.copy()
    s.index = normalize_index_to_ny_dates(s.index)
    return s.shift(-H) / s - 1.0

# ---------------- Earnings (calendar only) ----------------

def load_earnings_events(file_path: Optional[str]) -> Dict[str, pd.DataFrame]:
    """Load earnings calendar; returns dict[symbol -> DataFrame(reaction_date, eps_surprise_pct)]"""
    if not file_path:
        return {}
    
    df = pd.read_csv(file_path)
    
    # Diagnostics
    print(f"\n📅 Earnings Calendar Loaded:")
    print(f"  Total rows: {len(df)}")
    print(f"  Symbols: {df['symbol'].nunique()}")
    print(f"  Date range: {df['earnings_date'].min()} to {df['earnings_date'].max()}")
    
    if 'eps_actual' in df.columns and 'eps_estimate' in df.columns:
        has_both = df[['eps_actual', 'eps_estimate']].notna().all(axis=1).sum()
        print(f"  Rows with both EPS actual & estimate: {has_both} / {len(df)} ({has_both/len(df)*100:.1f}%)")
    
    if df.empty:
        return {}
    
    cm = {c.lower(): c for c in df.columns}
    sym_col = cm.get("symbol") or cm.get("ticker")
    if not sym_col:
        return {}
    
    earn_col = cm.get("earnings_date") or cm.get("reportdate")
    sup_col = cm.get("eps_surprise_pct") or cm.get("surprise_pct")
    
    df["_symbol"] = df[sym_col].astype(str).str.upper()
    df["_earnings_date"] = to_ny_date_col(df[earn_col]) if earn_col else pd.NaT
    df["_eps_surprise_pct"] = pd.to_numeric(df[sup_col], errors="coerce") if sup_col else np.nan
    
    df = df.dropna(subset=["_earnings_date"])
    df = df.sort_values(["_symbol", "_earnings_date"]).drop_duplicates(["_symbol", "_earnings_date"], keep="last")
    
    out: Dict[str, pd.DataFrame] = {}
    for sym, g in df.groupby("_symbol"):
        out[sym] = g[["_earnings_date", "_eps_surprise_pct"]].rename(columns={
            "_earnings_date": "earnings_date",
            "_eps_surprise_pct": "eps_surprise_pct"
        }).reset_index(drop=True)
    
    logging.info(f"Loaded {sum(len(v) for v in out.values()):,} earnings events across {len(out)} symbols")
    return out

def add_earnings_calendar_features(dates: pd.DatetimeIndex, symbol: str, 
                                   earn_events: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Add calendar-based earnings features (days_to_earn, prev_surprise, etc.)"""
    out = pd.DataFrame({"date": dates})
    out["feat_days_to_earn"] = np.nan
    out["feat_days_since_earn"] = np.nan
    out["feat_is_preearn_3"] = 0.0
    out["feat_is_postearn_3"] = 0.0
    out["feat_is_postearn_10"] = 0.0
    out["feat_prev_earn_surprise_pct"] = np.nan
    out["feat_earn_surprise_streak"] = 0.0
    
    if symbol not in earn_events or earn_events[symbol].empty:
        return out
    
    ev = earn_events[symbol]
    ev_dates = pd.DatetimeIndex(ev["earnings_date"])
    ev_sup = ev["eps_surprise_pct"].values
    
    d_values = dates.values
    ev_values = ev_dates.values
    
    # Find next/prev event
    idx_next = np.searchsorted(ev_values, d_values, side="left")
    idx_prev = idx_next - 1
    
    has_next = idx_next < len(ev_values)
    has_prev = idx_prev >= 0
    
    # Days to next earnings
    next_dates = np.where(has_next, ev_values[np.minimum(idx_next, len(ev_values)-1)], np.datetime64("NaT"))
    days_to = (pd.to_datetime(next_dates) - pd.to_datetime(d_values)).days
    out["feat_days_to_earn"] = np.where(has_next, days_to, np.nan)
    
    # Days since last earnings
    prev_dates = np.where(has_prev, ev_values[np.maximum(idx_prev, 0)], np.datetime64("NaT"))
    days_since = (pd.to_datetime(d_values) - pd.to_datetime(prev_dates)).days
    out["feat_days_since_earn"] = np.where(has_prev, days_since, np.nan)
    
    # Flags
    out["feat_is_preearn_3"] = ((out["feat_days_to_earn"] >= 0) & (out["feat_days_to_earn"] <= 3)).astype(float)
    out["feat_is_postearn_3"] = ((out["feat_days_since_earn"] >= 0) & (out["feat_days_since_earn"] <= 3)).astype(float)
    out["feat_is_postearn_10"] = ((out["feat_days_since_earn"] >= 0) & (out["feat_days_since_earn"] <= 10)).astype(float)
    
    # Previous surprise (leakage-safe: only past events)
    prev_sup = np.where(has_prev, ev_sup[np.maximum(idx_prev, 0)], np.nan)
    out["feat_prev_earn_surprise_pct"] = np.where(has_prev, prev_sup, np.nan)
    
    # Streak of positive surprises (last 4 events including current) - FIXED BOUNDS CHECKING
    streak = np.zeros(len(out))
    for i in range(len(out)):
        if has_prev[i]:
            p = int(idx_prev[i])
            if 0 <= p < len(ev_sup):  # Bounds check
                # Look back up to 4 events (including p)
                start = max(0, p - 3)
                end = min(p + 1, len(ev_sup))
                if start < end:
                    streak[i] = float((ev_sup[start:end] > 0).sum())
    out["feat_earn_surprise_streak"] = streak
    
    return out

# ============ CROSS-SECTIONAL FEATURES ============

def add_cross_sectional_ranks(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Computing cross-sectional rank features...")
    
    rank_features = [
        'feat_volatility_20', 'feat_atr_norm', 'feat_idio_vol_63',
        'feat_parkinson_20', 'feat_garman_klass_20',
        'feat_beta_spy_126', 'feat_beta_spy_252',
        'feat_mom_12m_skip1m', 'feat_ret_63d',
        'feat_avg_volume_20', 'feat_adv20_dollar',  # Include adv20_dollar with feat_ prefix!
        'feat_rsi', 'feat_breakout_strength_20d',
        'feat_sector_rel_ret_63d'  # Total: 14 features
    ]
    rank_features = [f for f in rank_features if f in df.columns]
    
    if not rank_features:
        logging.warning("No rank features found!")
        return df

    # THE BULLETPROOF WAY
    rank_df = df.groupby('date')[rank_features].rank(pct=True, method='average')
    rank_df = rank_df.add_suffix('_rank_pct')
    
    # This line will NEVER fail
    df = df.join(rank_df, how='left')
    
    logging.info(f"Added {len(rank_features)} cross-sectional rank features")
    return df


def add_cross_sectional_z_scores(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Computing cross-sectional z-score features...")
    
    z_features = ['feat_volatility_20', 'feat_mom_12m_skip1m', 'feat_ret_63d',
                  'feat_beta_spy_126', 'feat_adv20_dollar']  # Include adv20_dollar with feat_ prefix!
    z_features = [f for f in z_features if f in df.columns]
    
    if z_features:
        z_df = df.groupby('date')[z_features].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        z_df = z_df.add_suffix('_zscore_xsec')
        df = df.join(z_df, how='left')
    
    logging.info(f"Added {len(z_features)} z-score features")
    return df

# ---------------- Labels ----------------

def graded_labels_per_date(df: pd.DataFrame, excess_col: str, 
                          edges: List[float], gains: List[float]) -> pd.Series:
    """Assign graded relevance labels based on percentile bins per date"""
    def per_date(g: pd.DataFrame) -> pd.Series:
        if len(g) == 0:
            return pd.Series(dtype=float)
        q = g[excess_col].rank(pct=True, method="average")
        idx = np.digitize(q.values, bins=edges, right=False) - 1
        idx = np.clip(idx, 0, len(gains) - 1)
        return pd.Series([gains[i] for i in idx], index=g.index, dtype=float)
    
    return df.groupby("date", group_keys=False).apply(per_date)

# ---------------- Main ----------------

def main():
    args = parse_args()
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # ============================================================
    # PRODUCTION MODE: Load ONLY latest row per ticker (early exit)
    # ============================================================
    if args.production_only:
        logging.info("=" * 60)
        logging.info("🎯 PRODUCTION MODE: Loading latest rows only")
        logging.info("=" * 60)
        
        # Get list of enhanced feature files
        cache_path = Path(args.cache_dir)
        feature_files = list(cache_path.glob("*_features_enhanced.parquet"))
        
        if not feature_files:
            # Fallback to regular features
            feature_files = list(cache_path.glob("*_features.parquet"))
            feature_files = [f for f in feature_files if "_enhanced" not in f.name]
        
        if not feature_files:
            raise RuntimeError(f"No feature files found in {args.cache_dir}")
        
        logging.info(f"Found {len(feature_files)} enhanced feature files")
        
        # Load ONLY latest row from each ticker
        production_rows = []
        
        for ticker_file in feature_files:
            try:
                ticker = ticker_file.stem.split('_')[0].upper()
                
                # Skip SPY
                if ticker == "SPY":
                    continue
                
                # Load enhanced features
                df = pd.read_parquet(ticker_file)
                
                if len(df) == 0:
                    continue
                
                # Get ONLY the latest row (last row = most recent date)
                latest_row = df.iloc[-1:].copy()
                
                # Reset index to get Date as column
                latest_row = latest_row.reset_index()
                
                # Add symbol
                latest_row['symbol'] = ticker
                
                production_rows.append(latest_row)
                
            except Exception as e:
                logging.warning(f"Could not load {ticker_file.name}: {e}")
                continue
        
        if not production_rows:
            raise RuntimeError("No production data loaded!")
        
        # Combine: N tickers × 1 row = N rows total
        df_all = pd.concat(production_rows, ignore_index=True)
        
        # Normalize date column name
        if 'Date' in df_all.columns:
            df_all = df_all.rename(columns={'Date': 'date'})
        elif 'index' in df_all.columns:
            df_all = df_all.rename(columns={'index': 'date'})
        
        # Ensure date column exists
        if 'date' not in df_all.columns:
            raise RuntimeError("No date column found in production data!")
        
        df_all['date'] = pd.to_datetime(df_all['date']).dt.normalize()
        
        # Validate we have data
        if len(df_all) == 0:
            raise RuntimeError("No production data after loading!")
        
        # Store production date for logging
        production_date = df_all['date'].iloc[0]
        n_symbols = df_all['symbol'].nunique()
        
        print(f"✓ Loaded {len(df_all)} symbols for production date: {production_date}")
        
        # ============================================================
        # ADD: Compute liquidity features (adv20_dollar)
        # ============================================================
        
        logging.info("Computing liquidity features (adv20_dollar)...")
        
        for idx, row in df_all.iterrows():
            symbol = row['symbol']
            ticker_upper = symbol.upper()
            
            # Find raw OHLCV file
            raw_fp = find_cache_for_ticker(ticker_upper, args.cache_dir)
            
            if raw_fp and raw_fp.exists():
                try:
                    df_raw = load_parquet_indexed(raw_fp)
                    
                    if "Close" in df_raw.columns and "Volume" in df_raw.columns:
                        close = pd.to_numeric(df_raw["Close"], errors="coerce")
                        vol = pd.to_numeric(df_raw["Volume"], errors="coerce").fillna(0)
                        
                        # Compute 20-day average dollar volume
                        dollar_vol = (close * vol).rolling(20, min_periods=1).mean()
                        
                        # Get value for production date
                        prod_date = row['date']
                        if prod_date in dollar_vol.index:
                            df_all.loc[idx, 'adv20_dollar'] = dollar_vol.loc[prod_date]
                        else:
                            # Use most recent value
                            df_all.loc[idx, 'adv20_dollar'] = dollar_vol.iloc[-1]
                            
                except Exception as e:
                    logging.warning(f"Could not compute adv20_dollar for {symbol}: {e}")
                    df_all.loc[idx, 'adv20_dollar'] = 0.0
            else:
                df_all.loc[idx, 'adv20_dollar'] = 0.0
        
        logging.info(f"✅ Computed adv20_dollar for {(df_all['adv20_dollar'] > 0).sum()}/{len(df_all)} symbols")
        
        # ============================================================
        # ADD: Merge earnings calendar features
        # ============================================================
        
        if args.earnings_file:
            logging.info("Merging earnings calendar features...")
            
            # Load earnings events
            earn_events = load_earnings_events(args.earnings_file)
            
            # Initialize earnings columns
            df_all["feat_days_to_earn"] = np.nan
            df_all["feat_days_since_earn"] = np.nan
            df_all["feat_is_preearn_3"] = 0.0
            df_all["feat_is_postearn_3"] = 0.0
            df_all["feat_is_postearn_10"] = 0.0
            df_all["feat_prev_earn_surprise_pct"] = np.nan
            df_all["feat_earn_surprise_streak"] = 0.0
            
            # Merge earnings features for each symbol
            for symbol in df_all['symbol'].unique():
                symbol_mask = df_all['symbol'] == symbol
                symbol_dates = pd.DatetimeIndex(df_all.loc[symbol_mask, 'date'])
                
                # Get earnings features for this symbol
                earn_feat = add_earnings_calendar_features(symbol_dates, symbol, earn_events)
                
                # Merge into df_all
                for col in earn_feat.columns:
                    if col != 'date':
                        df_all.loc[symbol_mask, col] = earn_feat[col].values
            
            # Compute earnings quality (streak × surprise)
            logging.info("🔧 Computing earnings quality...")
            
            if 'feat_earn_surprise_streak' in df_all.columns and 'feat_prev_earn_surprise_pct' in df_all.columns:
                streak = df_all['feat_earn_surprise_streak'].fillna(0).astype(float)
                surprise = df_all['feat_prev_earn_surprise_pct'].fillna(0).astype(float).clip(-100, 300)
                
                df_all['feat_earnings_quality'] = (streak * surprise).fillna(0).clip(-500, 1500)
                
                non_zero = (df_all['feat_earnings_quality'] != 0).sum()
                logging.info(f"✅ Earnings quality: {non_zero:,} / {len(df_all):,} non-zero ({non_zero/len(df_all)*100:.1f}%)")
                
                # Sample for verification
                if non_zero > 0:
                    sample = df_all[df_all['feat_earnings_quality'] != 0][
                        ['symbol', 'date', 'feat_earn_surprise_streak', 'feat_prev_earn_surprise_pct', 'feat_earnings_quality']
                    ].head(5)
                    logging.info(f"Sample non-zero earnings quality:\n{sample}")
            else:
                logging.warning("⚠️ Earnings columns not found, setting earnings quality to 0")
                df_all['feat_earnings_quality'] = 0.0
        else:
            logging.warning("⚠️ No earnings file provided, skipping earnings features")
            df_all['feat_earnings_quality'] = 0.0
            df_all["feat_days_to_earn"] = np.nan
            df_all["feat_days_since_earn"] = np.nan
            df_all["feat_is_preearn_3"] = 0.0
            df_all["feat_is_postearn_3"] = 0.0
            df_all["feat_is_postearn_10"] = 0.0
            df_all["feat_prev_earn_surprise_pct"] = np.nan
            df_all["feat_earn_surprise_streak"] = 0.0
        
        # Remove OHLCV columns that shouldn't be features (before renaming)
        for col in list(df_all.columns):
            if col in EXCLUDE_COLUMNS and col not in ['symbol', 'date']:
                df_all = df_all.drop(columns=[col])
        
        # Ensure feat_ prefix on all features (exclude metadata)
        rename_map = {}
        for col in df_all.columns:
            if col not in ['symbol', 'date'] and not col.startswith('feat_'):
                rename_map[col] = f'feat_{col}'
        
        if rename_map:
            df_all.rename(columns=rename_map, inplace=True)
        
        # Remove duplicate columns (keep first occurrence)
        df_all = df_all.loc[:, ~df_all.columns.duplicated()]
        
        # Compute cross-sectional features (AFTER adding earnings/liquidity)
        logging.info(f"Computing cross-sectional rank features (across {n_symbols} symbols)...")
        df_all = add_cross_sectional_ranks(df_all)
        n_rank = len([c for c in df_all.columns if c.startswith('feat_') and 'rank' in c])
        logging.info(f"Added {n_rank} cross-sectional rank features")
        
        logging.info("Computing cross-sectional z-score features...")
        df_all = add_cross_sectional_z_scores(df_all)
        n_zscore = len([c for c in df_all.columns if c.startswith('feat_') and 'zscore' in c])
        logging.info(f"Added {n_zscore} z-score features")
        
        logging.info("🔧 Computing composite quality...")
        
        # Compute composite quality for this date
        quality_parts = []
        
        if 'feat_low_vol_raw' in df_all.columns:
            vol_max = df_all['feat_low_vol_raw'].max()
            if vol_max > 1e-8:  # Avoid division by zero
                vol_score = 1.0 - (df_all['feat_low_vol_raw'] / vol_max)
                quality_parts.append(vol_score.fillna(0.5))
        
        if 'feat_in_uptrend' in df_all.columns:
            quality_parts.append(df_all['feat_in_uptrend'])
        
        if 'feat_earnings_quality' in df_all.columns:
            earn_score = ((df_all['feat_earnings_quality'] + 500) / 2000).clip(0, 1)
            quality_parts.append(earn_score.fillna(0.5))
        
        if quality_parts:
            df_all['feat_composite_quality'] = pd.concat(quality_parts, axis=1).mean(axis=1)
        else:
            df_all['feat_composite_quality'] = 0.5
        
        logging.info(f"✅ Composite quality: [{df_all['feat_composite_quality'].min():.3f}, {df_all['feat_composite_quality'].max():.3f}], mean={df_all['feat_composite_quality'].mean():.3f}")
        
        # ============================================================
        # COMPREHENSIVE FEATURE VALIDATION
        # ============================================================
        
        print("\n" + "=" * 60)
        print("ENHANCED FEATURE DEBUGGING")
        print("=" * 60)
        
        feature_cols = [c for c in df_all.columns if c.startswith('feat_')]
        
        print(f"Total features: {len(feature_cols)}")
        print(f"Total rows: {len(df_all)}")
        print(f"Date range: {df_all['date'].min()} to {df_all['date'].max()}")
        print(f"Symbols: {df_all['symbol'].nunique()}")
        
        # ============================================================
        # Feature category breakdown
        # ============================================================
        
        print(f"\n📋 FEATURE BREAKDOWN BY CATEGORY:")
        
        # Count by category
        technical_features = [c for c in feature_cols if any(x in c for x in ['rsi', 'macd', 'atr', 'adx', 'stoch', 'williams', 'mfi', 'cci', 'bb_', 'ema', 'sma'])]
        momentum_features = [c for c in feature_cols if 'mom' in c or 'ret_' in c]
        volatility_features = [c for c in feature_cols if 'vol' in c or 'atr' in c or 'parkinson' in c or 'garman' in c]
        volume_features = [c for c in feature_cols if 'volume' in c or 'obv' in c or 'vpt' in c or 'adv' in c]
        earnings_features = [c for c in feature_cols if 'earn' in c]
        sector_features = [c for c in feature_cols if 'sector' in c]
        cross_sectional_features = [c for c in feature_cols if '_rank_pct' in c or '_zscore_xsec' in c]
        quality_features = [c for c in feature_cols if 'quality' in c or 'composite' in c]
        
        print(f"  Technical indicators:    {len(technical_features)}")
        print(f"  Momentum/Returns:        {len(momentum_features)}")
        print(f"  Volatility:              {len(volatility_features)}")
        print(f"  Volume:                  {len(volume_features)}")
        print(f"  Earnings:                {len(earnings_features)}")
        print(f"  Sector:                  {len(sector_features)}")
        print(f"  Cross-sectional:         {len(cross_sectional_features)}")
        print(f"  Quality:                 {len(quality_features)}")
        
        # ============================================================
        # Check for critical features
        # ============================================================
        
        print(f"\n🔍 CRITICAL FEATURE VALIDATION:")
        
        critical_features = {
            'Earnings': ['feat_earnings_quality', 'feat_prev_earn_surprise_pct', 'feat_days_to_earn'],
            'Liquidity': ['feat_adv20_dollar'],
            'Cross-sectional': ['feat_mom_12m_skip1m_rank_pct', 'feat_volatility_20_rank_pct', 'feat_ret_63d_zscore_xsec'],
            'Quality': ['feat_composite_quality'],
            'Sector': ['feat_sector_rel_ret_63d'],
        }
        
        all_critical_present = True
        
        for category, features in critical_features.items():
            missing = [f for f in features if f not in df_all.columns]
            if missing:
                print(f"  ❌ {category}: MISSING {missing}")
                all_critical_present = False
            else:
                print(f"  ✅ {category}: All present ({len(features)} features)")
        
        if not all_critical_present:
            logging.warning("⚠️ Some critical features are missing!")
        
        # ============================================================
        # Feature coverage statistics
        # ============================================================
        
        print(f"\n📊 FEATURE COVERAGE STATISTICS:")
        
        key_features = [
            'feat_earnings_quality',
            'feat_prev_earn_surprise_pct',
            'feat_days_to_earn',
            'feat_days_since_earn',
            'feat_adv20_dollar',
            'feat_mom_12m_skip1m',
            'feat_composite_quality',
            'feat_volatility_20',
            'feat_rsi',
        ]
        
        for feat in key_features:
            if feat in df_all.columns:
                vals = df_all[feat]
                non_null = vals.notna().sum()
                non_zero = (vals != 0).sum()
                
                print(f"  {feat}:")
                print(f"    Non-null: {non_null}/{len(vals)} ({non_null/len(vals)*100:.1f}%)")
                print(f"    Non-zero: {non_zero}/{len(vals)} ({non_zero/len(vals)*100:.1f}%)")
                if non_null > 0:
                    print(f"    Range: [{vals.min():.4f}, {vals.max():.4f}]")
                    print(f"    Mean: {vals.mean():.4f}, Std: {vals.std():.4f}")
            else:
                print(f"  ❌ {feat}: MISSING!")
        
        # ============================================================
        # ALL FEATURE COLUMNS (detailed list)
        # ============================================================
        
        print(f"\n📋 ALL FEATURE COLUMNS ({len(feature_cols)} total):")
        
        for i, feat in enumerate(sorted(feature_cols), 1):
            vals = df_all[feat]
            sample_val = vals.iloc[0] if len(vals) > 0 else np.nan
            non_zero = (vals != 0).sum()
            
            print(f"  {i:3d}. {feat:50s} (sample={sample_val:>10.4f}, non-zero={non_zero}/{len(vals)})")
        
        # ============================================================
        # Problematic features
        # ============================================================
        
        print(f"\n⚠️ PROBLEMATIC FEATURES:")
        
        all_zero = [c for c in feature_cols if (df_all[c] == 0).all()]
        all_nan = [c for c in feature_cols if df_all[c].isna().all()]
        constant = [c for c in feature_cols if df_all[c].nunique() == 1]
        
        if all_zero:
            print(f"  All-zero features ({len(all_zero)}): {all_zero[:5]}")
        else:
            print(f"  ✓ No all-zero features")
        
        if all_nan:
            print(f"  All-NaN features ({len(all_nan)}): {all_nan}")
        else:
            print(f"  ✓ No all-NaN features")
        
        if constant:
            print(f"  Constant features ({len(constant)}): {[(c, df_all[c].iloc[0]) for c in constant[:5]]}")
        
        # ============================================================
        # Feature count validation
        # ============================================================
        
        expected_feature_count = 127
        actual_feature_count = len(feature_cols)
        
        print(f"\n🎯 FEATURE COUNT VALIDATION:")
        print(f"  Expected: {expected_feature_count}")
        print(f"  Actual:   {actual_feature_count}")
        
        if actual_feature_count < expected_feature_count:
            missing_count = expected_feature_count - actual_feature_count
            print(f"  ❌ MISSING {missing_count} FEATURES!")
            logging.error(f"Feature count mismatch: expected {expected_feature_count}, got {actual_feature_count}")
        else:
            print(f"  ✅ Feature count matches!")
        
        print("=" * 60)
        
        # Save
        df_all.to_parquet(out_path, compression='zstd', index=False)
        logging.info(f"✓ Wrote {len(df_all):,} rows to {out_path}")
        logging.info(f"Features: {len([c for c in df_all.columns if c.startswith('feat_')])}, Symbols: {n_symbols}")
        logging.info(f"Date: {production_date}")
        
        return  # Exit early - no need for training pipeline
    
    # ============================================================
    # TRAINING MODE: Process all historical data
    # ============================================================
    
    # Parse grading scheme
    edges = [float(x) for x in args.edges.split(",") if x.strip()]
    gains = [float(x) for x in args.gains.split(",") if x.strip()]
    if len(edges) != len(gains) + 1:
        raise SystemExit(f"--edges must have len = len(--gains)+1")
    
    # Get tickers
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    files = list_training_files(args.limit, tickers, args.cache_dir)
    if not files:
        raise RuntimeError(f"No training files found in {args.cache_dir}")
    
    logging.info(f"Found {len(files)} ticker files")
    
    # Load benchmark (SPY) for excess returns
    spy_fp = None
    if args.spy_file:
        spy_fp = Path(args.spy_file)
    if spy_fp is None or not spy_fp.exists():
        spy_fp = find_cache_for_ticker("SPY", args.cache_dir)
    
    if not spy_fp or not spy_fp.exists():
        raise SystemExit(f"SPY benchmark not found in {args.cache_dir}")
    
    spy_df = load_parquet_indexed(spy_fp)
    # Use Adj Close for returns (or Close if Adj Close missing)
    if "Adj Close" in spy_df.columns:
        spy_close = spy_df["Adj Close"]
    elif "Close" in spy_df.columns:
        spy_close = spy_df["Close"]
    else:
        raise SystemExit("SPY file missing Close/Adj Close column")
    
    spy_close = pd.to_numeric(spy_close, errors="coerce").dropna()
    bench_fwd = forward_return(spy_close, HORIZON).dropna()
    bench_df = pd.DataFrame({"date": to_ny_date_col(bench_fwd.index), "bench_ret_63d": bench_fwd.values})
    logging.info(f"Loaded SPY benchmark: {len(bench_df):,} forward returns")
    
    # Load earnings calendar
    earn_events = load_earnings_events(args.earnings_file) if args.earnings_file else {}
    
    # Process each ticker
    rows = []
    for fp in files:
        try:
            ticker = fp.stem.split("_")[0].upper()
            if ticker == "SPY":
                continue
            
            # Load raw price file
            df_raw = load_parquet_indexed(fp)
            
            # Get Adj Close for label calculation
            if "Adj Close" in df_raw.columns:
                close = df_raw["Adj Close"]
            elif "Close" in df_raw.columns:
                close = df_raw["Close"]
            else:
                logging.warning(f"Skipping {ticker}: no Close/Adj Close column")
                continue
            
            close = pd.to_numeric(close, errors="coerce").dropna()
            
            # Compute forward returns (labels)
            fwd = forward_return(close, HORIZON).dropna()
            if len(fwd) < 50:
                logging.info(f"Skipping {ticker}: insufficient data ({len(fwd)} rows)")
                continue
            
            # Base table: label dates
            base = pd.DataFrame({
                "date": to_ny_date_col(fwd.index),
                "symbol": ticker
            })
            
            # Add current price (for liquidity filter)
            base["price"] = close.reindex(pd.DatetimeIndex(base["date"])).values
            
            # Compute ADV20 for liquidity filter
            if "Volume" in df_raw.columns:
                vol = pd.to_numeric(df_raw["Volume"], errors="coerce").fillna(0)
                dollar = (close * vol).rolling(20, min_periods=1).mean()
                base["adv20_dollar"] = dollar.reindex(pd.DatetimeIndex(base["date"])).values
            else:
                base["adv20_dollar"] = 0.0
            
            # Merge benchmark returns
            base = base.merge(bench_df, on="date", how="left")
            base["bench_ret_63d"] = base["bench_ret_63d"].fillna(0.0)
            
            # Compute excess return (stock forward return - benchmark forward return)
            base["excess_63d"] = fwd.reindex(pd.DatetimeIndex(base["date"])).values - base["bench_ret_63d"].values
            
            # ===== FIXED: Load features from T-1 (shift by 1 day) =====
            # Look for enhanced features by ticker name (handles both raw and feature files in cache_dir)
            ticker_upper = ticker.upper()
            
            # Try enhanced version first (most common case)
            feat_fp = fp.parent / f"{ticker_upper}_2y_raw_features_enhanced.parquet"
            
            if not feat_fp.exists():
                # Fallback: glob for any enhanced features file for this ticker
                candidates = list(fp.parent.glob(f"{ticker_upper}_*_features_enhanced.parquet"))
                if candidates:
                    feat_fp = sorted(candidates)[0]
                else:
                    # Fallback to non-enhanced
                    feat_fp = fp.parent / f"{ticker_upper}_2y_raw_features.parquet"
                    if not feat_fp.exists():
                        # Glob for any features file
                        candidates = list(fp.parent.glob(f"{ticker_upper}_*_features.parquet"))
                        # Exclude "features_enhanced" files (already checked above)
                        candidates = [c for c in candidates if "_enhanced" not in c.name]
                        if candidates:
                            feat_fp = sorted(candidates)[0]
                        else:
                            feat_fp = None
            
            if feat_fp and feat_fp.exists():
                fdf = pd.read_parquet(feat_fp)
                
                # Normalize feature file index
                if "Date" in fdf.columns:
                    fdf["Date"] = pd.to_datetime(fdf["Date"], errors="coerce")
                    fdf = fdf.set_index("Date")
                else:
                    fdf.index = pd.to_datetime(fdf.index, errors="coerce")
                
                fdf.index = normalize_index_to_ny_dates(fdf.index)
                fdf = fdf.sort_index()
                
                # Select only numeric columns (features)
                # Only keep actual feature columns (exclude OHLCV if they snuck in)
                exclude = {"Open", "High", "Low", "Close", "Adj Close", "Volume", "Date", "date", "symbol"}
                num_cols = [c for c in fdf.columns 
                            if pd.api.types.is_numeric_dtype(fdf[c]) and c not in exclude]

                if num_cols:
                    fdf = fdf[num_cols].copy()
                    
                    # CRITICAL: Only add "feat_" prefix if it's NOT already there
                    new_names = []
                    for c in fdf.columns:
                        if c.startswith("feat_"):
                            new_names.append(c)
                        else:
                            new_names.append(f"feat_{c}")
                    fdf.columns = new_names
                    
                    # Remove duplicates (in case both raw + enhanced had same feature)
                    fdf = fdf.loc[:, ~fdf.columns.duplicated()]
                    # SHIFT FEATURES BY 1 DAY (use T-1 features for T→T+63 label)
                    fdf = fdf.shift(1)
                    
                    # Reset index for merge
                    fdf = fdf.reset_index().rename(columns={"Date": "date"})
                    fdf["date"] = to_ny_date_col(fdf["date"])
                    
                    # Deduplicate (keep last if multiple rows per date)
                    fdf = fdf.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
                    
                    # Merge features into base
                    base = base.merge(fdf, on="date", how="left")
                    
                    logging.info(f"{ticker}: merged {len(num_cols)} features (shifted T-1)")
                else:
                    logging.warning(f"{ticker}: features file has no numeric columns")
            else:
                logging.warning(f"{ticker}: features file not found (tried {fp.parent}/{ticker_upper}_*_features*.parquet)")
            
            # Add earnings calendar features
            if earn_events:
                earn_feat = add_earnings_calendar_features(pd.DatetimeIndex(base["date"]), ticker, earn_events)
                base = base.merge(earn_feat, on="date", how="left")
            
            rows.append(base)
            
        except Exception as e:
            logging.error(f"Error processing {fp.name}: {e}", exc_info=args.debug)
    
    if not rows:
        raise RuntimeError("No rows generated")
    
    # Combine all tickers
    df_all = pd.concat(rows, ignore_index=True)
    df_all["date"] = to_ny_date_col(df_all["date"])
    df_all = df_all.sort_values(["date", "symbol"]).reset_index(drop=True)
    
    logging.info(f"Combined dataset: {len(df_all):,} rows, {df_all['symbol'].nunique()} symbols")
    
    # ===== ADD CROSS-SECTIONAL FEATURES (AFTER COMBINING ALL STOCKS) =====
    df_all = add_cross_sectional_ranks(df_all)
    df_all = add_cross_sectional_z_scores(df_all)
    # ============================================================
    # COMPUTE EARNINGS QUALITY (uses merged earnings features)
    # ============================================================
    
    logging.info("🔧 Computing earnings quality...")
    
    if 'feat_earn_surprise_streak' in df_all.columns and 'feat_prev_earn_surprise_pct' in df_all.columns:
        # Convert to float and handle NaN
        streak = df_all['feat_earn_surprise_streak'].fillna(0).astype(float)
        surprise = df_all['feat_prev_earn_surprise_pct'].fillna(0).astype(float).clip(-100, 300)
        
        # Compute quality score
        df_all['feat_earnings_quality'] = (streak * surprise).fillna(0).clip(-500, 1500)
        
        # Log results
        non_zero = (df_all['feat_earnings_quality'] != 0).sum()
        logging.info(f"✅ Earnings quality: {non_zero:,} / {len(df_all):,} non-zero ({non_zero/len(df_all)*100:.1f}%)")
        
        # Sample for verification
        if non_zero > 0:
            sample = df_all[df_all['feat_earnings_quality'] != 0][['symbol', 'date', 'feat_earn_surprise_streak', 'feat_prev_earn_surprise_pct', 'feat_earnings_quality']].head(5)
            logging.info(f"Sample non-zero earnings quality:\n{sample}")
    else:
        logging.warning("⚠️  Earnings columns not found, setting earnings quality to 0")
        df_all['feat_earnings_quality'] = 0.0
    
    # ============================================================
    # COMPUTE COMPOSITE QUALITY (cross-sectional, per-date)
    # ============================================================
    
    logging.info("🔧 Computing composite quality...")
    
    quality_parts = []
    
    # 1. Low volatility score (per-date normalization)
    if 'feat_low_vol_raw' in df_all.columns:
        vol_max = df_all.groupby('date')['feat_low_vol_raw'].transform('max')
        vol_score = 1.0 - (df_all['feat_low_vol_raw'] / vol_max.replace(0, np.nan))
        quality_parts.append(vol_score.fillna(0.5))
        logging.info("   ✓ Low vol score added")
    
    # 2.  Uptrend (binary: 0 or 1)
    if 'feat_in_uptrend' in df_all.columns:
        quality_parts.append(df_all['feat_in_uptrend'])
        uptrend_pct = (df_all['feat_in_uptrend'] == 1).sum() / len(df_all) * 100
        logging.info(f"   ✓ Uptrend added ({uptrend_pct:.1f}% in uptrend)")
    
    # 3. Earnings quality (scaled to 0-1)
    if 'feat_earnings_quality' in df_all.columns:
        # Map [-500, 1500] → [0, 1]
        earn_score = ((df_all['feat_earnings_quality'] + 500) / 2000).clip(0, 1)
        quality_parts.append(earn_score.fillna(0.5))
        logging.info("   ✓ Earnings score added")
    
    # Final: average all components
    if quality_parts:
        df_all['feat_composite_quality'] = pd.concat(quality_parts, axis=1).mean(axis=1)
        logging.info(f"✅ Composite quality: [{df_all['feat_composite_quality'].min():.3f}, {df_all['feat_composite_quality'].max():.3f}], mean={df_all['feat_composite_quality'].mean():.3f}")
    else:
        df_all['feat_composite_quality'] = 0.5
        logging.warning("⚠️  No quality components found, defaulting to 0.5")
    
    logging.info("✅ Quality features computed\n")

    # Optional: exclude earnings windows
    if args.earnings_file and (args.exclude_earnings_pre > 0 or args.exclude_earnings_post > 0):
        before = len(df_all)
        mask = pd.Series(True, index=df_all.index)
        if "feat_days_to_earn" in df_all.columns and args.exclude_earnings_pre > 0:
            mask &= ~((df_all["feat_days_to_earn"] >= 0) & (df_all["feat_days_to_earn"] <= args.exclude_earnings_pre))
        if "feat_days_since_earn" in df_all.columns and args.exclude_earnings_post > 0:
            mask &= ~((df_all["feat_days_since_earn"] >= 0) & (df_all["feat_days_since_earn"] <= args.exclude_earnings_post))
        df_all = df_all.loc[mask].copy()
        logging.info(f"Excluded {before - len(df_all):,} rows around earnings dates")
    
    # Liquidity filter
    before_ct = len(df_all)
    df_all = df_all[(df_all["price"] >= MIN_PRICE) & (df_all["adv20_dollar"] >= MIN_ADV20_DOLLAR)]
    logging.info(f"Liquidity filter: kept {len(df_all):,}/{before_ct:,} rows")
    
    # Keep only rows with valid labels
    df_all = df_all.dropna(subset=["excess_63d"])
    
    # Fill NaNs in features (from shift or missing data)
    feat_cols = [c for c in df_all.columns if c.startswith("feat_")]
    if feat_cols:
        for col in feat_cols:
            df_all[col] = df_all[col].fillna(0)
        logging.info(f"Filled NaNs in {len(feat_cols)} feature columns")
    else:
        logging.warning("No feature columns found! Ensure *_features.parquet files exist.")
    
    # Compute graded labels
    df_all["label_rel_grade"] = graded_labels_per_date(df_all, "excess_63d", edges, gains)
    
    # Deduplicate by (date, symbol)
    df_all = df_all.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"], keep="last")
    
    # ===== ENHANCED FEATURE DEBUGGING =====
    print("\n" + "=" * 60)
    print("ENHANCED FEATURE DEBUGGING")
    print("=" * 60)

    feature_cols = [c for c in df_all.columns if c.startswith('feat_')]
    print(f"Total features: {len(feature_cols)}")
    print(f"Total rows: {len(df_all)}")
    print(f"Date range: {df_all['date'].min()} to {df_all['date'].max()}")
    print(f"Symbols: {df_all['symbol'].nunique()}")

    # Show ALL feature columns (sorted) so we can see naming patterns
    print(f"\n📋 ALL FEATURE COLUMNS ({len(feature_cols)} total):")
    for i, col in enumerate(sorted(feature_cols), 1):
        sample_val = df_all[col].iloc[0] if len(df_all) > 0 else np.nan
        non_zero = (df_all[col] != 0).sum()
        print(f"  {i:3d}. {col:50s} (sample={sample_val:.4f}, non-zero={non_zero}/{len(df_all)})")

    # Search for specific feature patterns
    print(f"\n🔍 FEATURE PATTERN SEARCH:")

    # Mom 12m skip1m variations
    mom_variations = [c for c in df_all.columns if 'mom' in c.lower() and '12' in c]
    print(f"\n  12-month momentum features ({len(mom_variations)}):")
    for col in mom_variations:
        print(f"    - {col}")
    if not mom_variations:
        print(f"    ⚠️  NONE FOUND!")

    # RSI variations
    rsi_variations = [c for c in df_all.columns if 'rsi' in c.lower()]
    print(f"\n  RSI features ({len(rsi_variations)}):")
    for col in rsi_variations:
        print(f"    - {col}")
    if not rsi_variations:
        print(f"    ⚠️  NONE FOUND!")

    # Sector relative features
    sector_variations = [c for c in df_all.columns if 'sector' in c.lower()]
    print(f"\n  Sector features ({len(sector_variations)}):")
    for col in sector_variations:
        print(f"    - {col}")
    if not sector_variations:
        print(f"    ⚠️  NONE FOUND!")

    # Check for problematic features
    print(f"\n⚠️  PROBLEMATIC FEATURES:")

    zero_features = [c for c in feature_cols if (df_all[c] == 0).all()]
    if zero_features:
        print(f"\n  Features that are ALL ZERO ({len(zero_features)}):")
        for feat in zero_features[:20]:
            print(f"    - {feat}")
        if len(zero_features) > 20:
            print(f"    ... and {len(zero_features) - 20} more")
    else:
        print(f"\n  ✓ No all-zero features")

    nan_features = [c for c in feature_cols if df_all[c].isna().all()]
    if nan_features:
        print(f"\n  Features that are ALL NaN ({len(nan_features)}):")
        for feat in nan_features[:20]:
            print(f"    - {feat}")
        if len(nan_features) > 20:
            print(f"    ... and {len(nan_features) - 20} more")
    else:
        print(f"\n  ✓ No all-NaN features")

    constant_features = []
    for c in feature_cols:
        if df_all[c].nunique() == 1:
            constant_features.append((c, df_all[c].iloc[0]))
    if constant_features:
        print(f"\n  Features with only ONE unique value ({len(constant_features)}):")
        for feat, val in constant_features[:20]:
            print(f"    - {feat} = {val}")
        if len(constant_features) > 20:
            print(f"    ... and {len(constant_features) - 20} more")
    else:
        print(f"\n  ✓ No constant features")

    # CORRECTED: Check for features WITH proper feat_ prefix
    print(f"\n📊 KEY FEATURE STATISTICS (corrected names):")
    key_features = [
        'feat_vix_level_z_63',
        'feat_beta_spy_126', 
        'feat_earnings_quality',
        'feat_mom_12m_skip1m',  # ← CORRECTED: Added feat_ prefix
        'feat_high_vol_regime',
        'feat_sector_rel_ret_21d',
        'feat_sector_rel_ret_63d',  # ← Alternative
        'feat_rsi',  # ← CORRECTED: Added feat_ prefix
        'feat_idio_vol_63',
        'feat_low_vol_raw',
        'feat_parkinson_20',
        'feat_composite_quality'
    ]

    for feat in key_features:
        if feat in df_all.columns:
            vals = df_all[feat]
            null_count = vals.isna().sum()
            zero_count = (vals == 0).sum()
            print(f"  ✓ {feat}:")
            print(f"      min={vals.min():.6f}, max={vals.max():.6f}, mean={vals.mean():.6f}, std={vals.std():.6f}")
            print(f"      nulls={null_count} ({null_count/len(vals)*100:.1f}%), zeros={zero_count} ({zero_count/len(vals)*100:.1f}%)")
        else:
            print(f"  ✗ {feat}: MISSING!")

    # Check for missing critical features
    missing_critical = [f for f in key_features if f not in df_all.columns]

    if missing_critical:
        print(f"\n⚠️  CRITICAL: Missing {len(missing_critical)} key features!")
        print(f"   {missing_critical}")
        
        # Try to find similar named columns
        print(f"\n   Searching for similar column names:")
        for missing in missing_critical:
            # Strip feat_ prefix and search
            base_name = missing.replace('feat_', '')
            similar = [c for c in df_all.columns if base_name in c.lower()]
            if similar:
                print(f"     {missing} → Found similar: {similar}")
            else:
                print(f"     {missing} → No similar columns found")

    # Check sample of first ticker's enhanced file to see what's in there
    print(f"\n🔬 SAMPLE: Checking first ticker's enhanced file directly...")
    try:
        first_ticker = df_all['symbol'].iloc[0] if len(df_all) > 0 else None
        if first_ticker:
            cache_path = Path(args.cache_dir)
            enhanced_file = list(cache_path.glob(f"{first_ticker}_*_features_enhanced.parquet"))
            if enhanced_file:
                sample_enhanced = pd.read_parquet(enhanced_file[0])
                enhanced_cols = [c for c in sample_enhanced.columns if 'mom' in c.lower() or 'rsi' in c.lower() or 'sector' in c.lower()]
                print(f"  File: {enhanced_file[0].name}")
                print(f"  Mom/RSI/Sector columns in enhanced file:")
                for col in enhanced_cols[:20]:
                    print(f"    - {col}")
            else:
                print(f"  ⚠️  No enhanced file found for {first_ticker}")
    except Exception as e:
        print(f"  ⚠️  Error checking enhanced file: {e}")

    print("=" * 60)
    
    # Save
    df_all.to_parquet(out_path)
    logging.info(f"✓ Wrote {len(df_all):,} rows to {out_path}")
    logging.info(f"Features: {len(feat_cols)}, Symbols: {df_all['symbol'].nunique()}")
    logging.info(f"Date range: {df_all['date'].min()} → {df_all['date'].max()}")

if __name__ == "__main__":
    main()
