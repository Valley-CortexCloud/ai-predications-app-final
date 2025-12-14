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
    
    # Try _etf_cache in PARENT directory (go up one level if we're in 2y_ticker_features)
    cache_root = p.parent if p.name == "2y_ticker_features" else p
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
        'feat_avg_volume_20', 'adv20_dollar',
        'feat_rsi', 'feat_breakout_strength_20d',
        'feat_sector_rel_ret_63d'
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
                  'feat_beta_spy_126', 'adv20_dollar']
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
    
    # Save
    df_all.to_parquet(out_path)
    logging.info(f"✓ Wrote {len(df_all):,} rows to {out_path}")
    logging.info(f"Features: {len(feat_cols)}, Symbols: {df_all['symbol'].nunique()}")
    logging.info(f"Date range: {df_all['date'].min()} → {df_all['date'].max()}")

if __name__ == "__main__":
    main()
