#!/usr/bin/env python3
"""
ULTIMATE feature enhancement: combines market-relative + cross-sectional + interactions.

Processes per-ticker features files and adds:
1. Market-relative: SPY beta, idio vol, residual returns
2. Sector-relative: returns vs sector ETF
3. VIX regime: z-score, delta, high vol flag
4. Crypto: correlation, beta, residual returns
5. Rates: TLT beta
6. Cross-sectional ranks: percentile ranks within each date
7. Volatility interactions: vol*momentum, vol expansion, idio vol * low beta
8. Momentum quality: risk-adjusted, Sharpe-adjusted, consistency
9. Technical patterns: quiet before breakout, upside quality, gap signals
10. Acceleration: momentum accel, convexity, new high streaks, sector divergence
11. Liquidity: liquid momentum, liquidity rank

Input: data_cache/10y_ticker_features/{TICKER}_features.parquet
Output: data_cache/10y_ticker_features/{TICKER}_features_enhanced.parquet
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from typing import Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))
from pathlib import Path

# ADD at top
ROOT = Path(__file__).parent.parent
TICKER_CACHE_DIR = ROOT / "data_cache" / "10y_ticker_features"
ETF_CACHE_DIR = ROOT / "data_cache" / "_etf_cache"

SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
DEFAULT_CRYPTO_PROXY = "^BTC-USD"
HIGH_VOL_THRESHOLD = 1.5  # VIX z-score threshold for high volatility regime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("enhance_features.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def parse_args():
    ap = argparse.ArgumentParser(description="Ultimate feature enhancement (market + cross-sectional + interactions)")
    ap.add_argument("--features-dir", type=str, default=str(TICKER_CACHE_DIR))
    ap.add_argument("--cache-dir", type=str, default=str(TICKER_CACHE_DIR))
    ap.add_argument("--sector-map", type=str, default="config/sector_map.csv")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--processes", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tickers", type=str, default=None)
    return ap.parse_args()

def normalize_index_to_ny_dates(idx: pd.Index) -> pd.DatetimeIndex:
    if pd.api.types.is_numeric_dtype(idx):
        idx = pd.to_datetime(idx, unit='ms', errors="coerce")
    else:
        idx = pd.to_datetime(idx, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    idx = pd.to_datetime(idx).normalize()
    return pd.DatetimeIndex(idx[~idx.duplicated(keep="last")])

def normalize_df_index(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")
    else:
        df.index = pd.to_datetime(df.index, errors="coerce")
    
    df.index = normalize_index_to_ny_dates(df.index)
    df.index.name = "Date"
    return df.sort_index()

def find_cache_for_ticker(ticker: str, cache_dir: str) -> Optional[Path]:
    p = Path(cache_dir)
    cand = list(p.glob(f"{ticker}_*2y_*.parquet")) + list(p.glob(f"{ticker}_*.parquet"))
    cand = [f for f in cand if not f.name.endswith("_features.parquet")]
    if cand:
        return sorted(cand)[0]
    
    etf_dir = p.parent / '_etf_cache'
    if etf_dir.exists():
        cand = list(etf_dir.glob(f"{ticker}_*2y_*.parquet")) + list(etf_dir.glob(f"{ticker}_*.parquet"))
        cand = [f for f in cand if not f.name.endswith("_features.parquet")]
        if cand:
            return sorted(cand)[0]
    
    return None

def load_sector_map(csv_path: Optional[str]) -> Dict[str, str]:
    if not csv_path or not Path(csv_path).exists():
        return {}
    df = pd.read_csv(csv_path)
    if not {"symbol", "sector_etf"}.issubset(df.columns):
        return {}
    return {str(r["symbol"]).upper(): str(r["sector_etf"]).upper() for _, r in df.iterrows()}

# ============ MARKET-RELATIVE FEATURES ============

def add_market_relative_features(df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    """SPY beta, idio vol, residual returns"""
    if "ret_1d" not in df.columns:
        df["ret_1d"] = df["Adj Close"].pct_change()
    if "ret_1d" not in spy_df.columns:
        spy_df["ret_1d"] = spy_df["Adj Close"].pct_change()
    
    spy_ret = spy_df["ret_1d"].reindex(df.index, method="ffill").fillna(0)
    stock_ret = df["ret_1d"].fillna(0)
    
    def rolling_beta(x, y, window):
        cov = x.rolling(window).cov(y)
        var = y.rolling(window).var()
        return cov / var.replace(0, np.nan)
    
    df["feat_beta_spy_126"] = rolling_beta(stock_ret, spy_ret, 126).fillna(0)
    df["feat_beta_spy_252"] = rolling_beta(stock_ret, spy_ret, 252).fillna(0)
    
    def idio_vol(x, y, beta, window):
        resid = x - beta * y
        return resid.rolling(window).std()
    
    df["feat_idio_vol_63"] = idio_vol(stock_ret, spy_ret, df["feat_beta_spy_126"], 63).fillna(0)
    
    if "ret_21d" in df.columns:
        spy_ret_21 = spy_df["Adj Close"].pct_change(21).reindex(df.index, method="ffill").fillna(0)
        df["feat_resid_ret_21d"] = (df["ret_21d"].fillna(0) - df["feat_beta_spy_126"] * spy_ret_21).fillna(0)
    
    if "ret_63d" in df.columns:
        spy_ret_63 = spy_df["Adj Close"].pct_change(63).reindex(df.index, method="ffill").fillna(0)
        df["feat_resid_ret_63d"] = (df["ret_63d"].fillna(0) - df["feat_beta_spy_126"] * spy_ret_63).fillna(0)
    
    return df

def add_sector_relative_features(df: pd.DataFrame, sector_df: pd.DataFrame, sector_etf: str) -> pd.DataFrame:
    """Sector-relative returns"""
    if "ret_21d" not in df.columns or "Adj Close" not in sector_df.columns:
        return df
    
    sector_ret_21 = sector_df["Adj Close"].pct_change(21).reindex(df.index, method="ffill").fillna(0)
    sector_ret_63 = sector_df["Adj Close"].pct_change(63).reindex(df.index, method="ffill").fillna(0)
    
    df["feat_sector_rel_ret_21d"] = (df["ret_21d"].fillna(0) - sector_ret_21).fillna(0)
    df["feat_sector_rel_ret_63d"] = (df["ret_63d"].fillna(0) - sector_ret_63).fillna(0)
    df[f"feat_sector_code_{sector_etf}"] = 1.0
    
    return df

def add_vix_regime_features(df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """High-vol regime, VIX z-scores"""
    if "Close" not in vix_df.columns:
        print("⚠️  VIX DataFrame missing 'Close' column")
        return df
    
    # Check if VIX data is valid
    if vix_df["Close"].isna().all() or len(vix_df) == 0:
        print("⚠️  VIX data is empty or all NaN")
        return df
    
    vix_level = vix_df["Close"].reindex(df.index).ffill()
    
    # Debug: Check reindexing result
    if vix_level.isna().all():
        print(f"⚠️  VIX reindex failed - no overlap between VIX dates and feature dates")
        print(f"   VIX date range: {vix_df.index.min()} to {vix_df.index.max()}")
        print(f"   Feature date range: {df.index.min()} to {df.index.max()}")
        # Return df with zero features rather than breaking
        df["feat_vix_level_z_63"] = 0.0
        df["feat_high_vol_regime"] = 0
        return df
    
    # Calculate z-score with proper handling
    vix_mean = vix_level.rolling(63, min_periods=20).mean()
    vix_std = vix_level.rolling(63, min_periods=20).std()
    
    # Prevent division by zero
    vix_std_safe = vix_std.replace(0, np.nan)
    vix_z = (vix_level - vix_mean) / vix_std_safe
    
    # Forward fill first, then backfill, then fill remaining with 0
    df["feat_vix_level_z_63"] = vix_z.ffill().bfill().fillna(0)
    
    # VIX delta (21-day change)
    df["feat_vix_delta_21d"] = vix_level.pct_change(21).ffill().bfill().fillna(0)
    
    # High vol regime (z > HIGH_VOL_THRESHOLD)
    df["feat_high_vol_regime"] = (df["feat_vix_level_z_63"] > HIGH_VOL_THRESHOLD).astype(int)
    
    # Debug output
    vix_stats = df["feat_vix_level_z_63"]
    print(f"✓ VIX features computed: mean={vix_stats.mean():.4f}, std={vix_stats.std():.4f}, min={vix_stats.min():.4f}, max={vix_stats.max():.4f}")
    print(f"  High vol regime: {df['feat_high_vol_regime'].sum()} / {len(df)} rows ({df['feat_high_vol_regime'].sum()/len(df)*100:.1f}%)")
    
    return df

def add_crypto_features(df: pd.DataFrame, crypto_df: pd.DataFrame) -> pd.DataFrame:
    """Crypto correlation, beta, residual returns"""
    if "Adj Close" not in crypto_df.columns or "ret_1d" not in df.columns:
        return df
    
    crypto_ret = crypto_df["Adj Close"].pct_change().reindex(df.index, method="ffill").fillna(0)
    stock_ret = df["ret_1d"].fillna(0)
    
    df["feat_crypto_corr_63d"] = stock_ret.rolling(63).corr(crypto_ret).fillna(0)
    
    cov = stock_ret.rolling(63).cov(crypto_ret)
    var = crypto_ret.rolling(63).var()
    df["feat_crypto_beta_63d"] = (cov / var.replace(0, np.nan)).fillna(0)
    
    df["feat_crypto_ret_5d"] = crypto_df["Adj Close"].pct_change(5).reindex(df.index, method="ffill").fillna(0)
    df["feat_crypto_ret_21d"] = crypto_df["Adj Close"].pct_change(21).reindex(df.index, method="ffill").fillna(0)
    df["feat_crypto_ret_63d"] = crypto_df["Adj Close"].pct_change(63).reindex(df.index, method="ffill").fillna(0)
    
    if "ret_21d" in df.columns:
        crypto_ret_21 = crypto_df["Adj Close"].pct_change(21).reindex(df.index, method="ffill").fillna(0)
        df["feat_resid_ret_crypto_21d"] = (df["ret_21d"].fillna(0) - df["feat_crypto_beta_63d"] * crypto_ret_21).fillna(0)
    
    if "ret_63d" in df.columns:
        crypto_ret_63 = crypto_df["Adj Close"].pct_change(63).reindex(df.index, method="ffill").fillna(0)
        df["feat_resid_ret_crypto_63d"] = (df["ret_63d"].fillna(0) - df["feat_crypto_beta_63d"] * crypto_ret_63).fillna(0)
    
    return df

def add_rates_beta(df: pd.DataFrame, tlt_df: pd.DataFrame) -> pd.DataFrame:
    """TLT (rates) beta"""
    if "ret_1d" not in df.columns or "Adj Close" not in tlt_df.columns:
        return df
    
    tlt_ret = tlt_df["Adj Close"].pct_change().reindex(df.index, method="ffill").fillna(0)
    stock_ret = df["ret_1d"].fillna(0)
    
    cov = stock_ret.rolling(63).cov(tlt_ret)
    var = tlt_ret.rolling(63).var()
    df["feat_beta_rates_63"] = (cov / var.replace(0, np.nan)).fillna(0)
    
    return df

# ============ VOLATILITY INTERACTIONS ============

def add_volatility_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Vol*momentum, vol expansion, idio vol * low beta"""
    # Vol * momentum
    if 'atr_norm' in df.columns and 'mom_12m_skip1m' in df.columns:
        df['feat_vol_mom_interaction'] = df['atr_norm'] * df['mom_12m_skip1m']
    
    # Idio vol * low beta (idiosyncratic opportunities)
    if 'feat_idio_vol_63' in df.columns and 'feat_beta_spy_126' in df.columns:
        df['feat_idio_vol_low_beta'] = df['feat_idio_vol_63'] * (1 - df['feat_beta_spy_126'].abs())
    
    # Vol expansion (current vs 63d average)
    if 'volatility_20' in df.columns:
        vol_ma_63 = df['volatility_20'].rolling(63, min_periods=20).mean()
        df['feat_vol_expansion'] = (df['volatility_20'] / vol_ma_63.replace(0, np.nan) - 1.0).fillna(0).clip(-3, 3)
    
    return df

# ============ MOMENTUM QUALITY ============

def add_momentum_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Risk-adjusted, Sharpe-adjusted, consistency"""
    # Momentum consistency
    if 'up_day_frac_63' in df.columns and 'ret_63d' in df.columns:
        df['feat_mom_consistency_63'] = (df['up_day_frac_63'] * df['ret_63d'].abs()).fillna(0)
    
    # Risk-adjusted momentum
    if 'mom_12m_skip1m' in df.columns and 'feat_idio_vol_63' in df.columns:
        denom = df['feat_idio_vol_63'].replace(0, np.nan).fillna(0.01)
        df['feat_risk_adj_mom12m'] = (df['mom_12m_skip1m'] / np.clip(denom, 0.01, None)).fillna(0).clip(-10, 10)
    
    # Sharpe-adjusted momentum
    if 'ret_63d' in df.columns and 'ret_1d' in df.columns and 'mom_12m_skip1m' in df.columns:
        vol_63 = df['ret_1d'].rolling(63).std().replace(0, np.nan)
        sharpe_63 = (df['ret_63d'] / vol_63).clip(-5, 5).fillna(0)
        df['feat_sharpe_adj_mom63'] = (df['mom_12m_skip1m'] * sharpe_63).fillna(0)
    
    # Sharpe with idio vol
    if 'feat_idio_vol_63' in df.columns and 'ret_63d' in df.columns:
        denom = df['feat_idio_vol_63'].replace(0, np.nan).fillna(0.01)
        df['feat_sharpe_mom_idio'] = (df['ret_63d'] / denom).clip(-10, 10).fillna(0)
    
    # Breakout volume confirmation
    if 'breakout_strength_20d' in df.columns and 'rvol_z_60' in df.columns:
        df['feat_breakout_volume_confirm'] = (df['breakout_strength_20d'] * np.maximum(df['rvol_z_60'], 0)).fillna(0)
    
    return df

# ============ TECHNICAL PATTERNS ============

def add_technical_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """Quiet before breakout, upside quality, gap signals"""
    # Low vol → breakout
    if 'volatility_20' in df.columns and 'breakout_strength_20d' in df.columns:
        vol_rank = df['volatility_20'].rank(pct=True)
        df['feat_quiet_before_breakout'] = ((vol_rank < 0.3).astype(float) * np.maximum(df['breakout_strength_20d'], 0)).fillna(0)
    
    # Upside quality (return / downside vol)
    if 'ret_63d' in df.columns and 'downside_vol_20' in df.columns:
        df['feat_upside_quality'] = (df['ret_63d'] / np.clip(df['downside_vol_20'], 0.01, None)).fillna(0).clip(-10, 10)
    
    # Gap volume signal
    if 'gap_size' in df.columns and 'rvol_z_60' in df.columns:
        df['feat_gap_volume_signal'] = (df['gap_size'] * np.maximum(df['rvol_z_60'], 0)).fillna(0)
    
    return df

# ============ ACCELERATION FEATURES ============

def add_acceleration_features(df: pd.DataFrame) -> pd.DataFrame:
    """Momentum accel, convexity, new highs, MA cross, sector divergence, vol accel, VWAP"""
    # Momentum acceleration (2nd derivative)
    if 'ret_21d' in df.columns:
        df['feat_mom_accel_21d'] = df['ret_21d'].diff().fillna(0).clip(-1, 1)
    
    # Momentum convexity (short > long)
    if 'ret_5d' in df.columns and 'Close' in df.columns:
        ret_5 = df['Close'].pct_change(5)
        ret_10 = df['Close'].pct_change(10)
        df['feat_mom_convexity'] = (ret_5 / ret_10.replace(0, np.nan) - 1).fillna(0).clip(-2, 2)
    
    # New high streak
    if 'Close' in df.columns:
        high_5d = df['Close'].rolling(5).max()
        new_high = (df['Close'] >= high_5d * 0.999).astype(int)
        df['feat_new_high_streak_5d'] = new_high.rolling(21, min_periods=1).sum().fillna(0)
    
    # MA cross strength
    if 'sma_50' in df.columns and 'Close' in df.columns:
        sma_200 = df['Close'].rolling(200, min_periods=100).mean()
        df['feat_ma_cross_strength'] = (df['sma_50'] / sma_200.replace(0, np.nan) - 1).fillna(0).clip(-0.5, 0.5)
    
    # Volume acceleration
    if 'avg_volume_20' in df.columns:
        vol_5d = df['avg_volume_20'].rolling(5).mean()
        df['feat_volume_acceleration'] = (vol_5d / df['avg_volume_20'].replace(0, np.nan) - 1).fillna(0).clip(-3, 3)
    
    # Price vs VWAP
    if 'Close' in df.columns and 'Volume' in df.columns:
        price_vol = df['Close'] * df['Volume']
        vwap_num = price_vol.rolling(21).sum()
        vwap_den = df['Volume'].rolling(21).sum()
        vwap_21 = vwap_num / vwap_den.replace(0, np.nan)
        df['feat_price_vs_vwap_21d'] = (df['Close'] / vwap_21 - 1).fillna(0).clip(-0.3, 0.3)
    
    # Breakout ATR normalization
    if 'breakout_strength_20d' in df.columns and 'atr_norm' in df.columns:
        df['feat_breakout_atr_norm'] = (df['breakout_strength_20d'] / (df['atr_norm'] + 0.01)).fillna(0).clip(-10, 10)
    
    return df

# ============ REGIME ADAPTIVE ============

def add_regime_adaptive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Beta in high vol, defensive score"""
    # Beta behavior in high VIX
    if 'feat_beta_spy_126' in df.columns and 'feat_vix_level_z_63' in df.columns:
        df['feat_beta_in_high_vol'] = (df['feat_beta_spy_126'] * (df['feat_vix_level_z_63'] > 0).astype(float)).fillna(0)
        df['feat_low_beta_in_high_vol'] = ((1 - df['feat_beta_spy_126'].abs()) * (df['feat_vix_level_z_63'] > 0).astype(float)).fillna(0)
    
    # Defensive score (low beta, low vol)
    if 'feat_beta_spy_126' in df.columns and 'volatility_20' in df.columns:
        beta_inv = 1 - df['feat_beta_spy_126'].abs()
        vol_rank = df['volatility_20'].rank(pct=True)
        df['feat_defensive_score'] = (beta_inv * (1 - vol_rank)).fillna(0)
    
    return df

# ============ LIQUIDITY ============

def add_liquidity_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Liquid momentum"""
    if 'avg_volume_20' in df.columns and 'mom_12m_skip1m' in df.columns:
        liquidity_log = np.log1p(df['avg_volume_20'])
        liquidity_scaled = ((liquidity_log - liquidity_log.mean()) / (liquidity_log.std() + 1e-8)).fillna(0)
        df['feat_liquid_momentum'] = (liquidity_scaled * df['mom_12m_skip1m']).fillna(0)
    
    return df

# ============ THEME ACCELERATION ============

def add_theme_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """Theme accel = sector momentum change * stock momentum * (1 - rates beta)"""
    if 'feat_sector_rel_ret_63d' in df.columns and 'mom_12m_skip1m' in df.columns:
        sector_accel = df['feat_sector_rel_ret_63d'] - df['feat_sector_rel_ret_63d'].shift(63)
        rates_factor = 1.0
        if 'feat_beta_rates_63' in df.columns:
            rates_factor = 1 - df['feat_beta_rates_63'].abs()
        df['feat_theme_accel'] = (sector_accel * df['mom_12m_skip1m'] * rates_factor).clip(-1, 1).fillna(0)
    
    return df

# ============ WORKER ============

def process_ticker(ticker: str, features_dir: Path, cache_dir: Path,
                   sector_map: Dict[str, str], spy_df: pd.DataFrame,
                   vix_df: Optional[pd.DataFrame], crypto_df: Optional[pd.DataFrame],
                   tlt_df:  Optional[pd.DataFrame], overwrite: bool, etf_cache_dir: Path) -> dict:
    
    feat_path = features_dir / f"{ticker}_features.parquet"
    if not feat_path.exists():
        candidates = list(features_dir.glob(f"{ticker}_*_features.parquet"))
        if not candidates:
            return {"ticker": ticker, "status": "skip", "reason": "no_features_file"}
        feat_path = candidates[0]
    
    out_path = feat_path.with_name(feat_path.stem + "_enhanced.parquet")
    if out_path.exists() and not overwrite:
        return {"ticker": ticker, "status": "skip", "reason": "exists"}
    
    try:
        df = pd.read_parquet(feat_path)
        df = normalize_df_index(df)
        
        # Debug logging: Show what columns were loaded
        logging.info(f"{ticker}: Loaded {len(df)} rows, columns: {list(df.columns)[:10]}... (total: {len(df.columns)})")
        
        # Defensive check: Ensure 'Adj Close' column exists
        if 'Adj Close' not in df.columns:
            return {"ticker": ticker, "status": "error", "error": f"Missing 'Adj Close' column. Available columns (first 10): {list(df.columns)[:10]}..."}
        
        if len(df) < 126:
            return {"ticker": ticker, "status": "skip", "reason": "insufficient_data"}
        
        initial_cols = len(df.columns)
        
        # Market-relative
        df = add_market_relative_features(df, spy_df)
        
        # Sector-relative
        if ticker in sector_map:
            sector_etf = sector_map[ticker]
            etf_cache_dir = Path(cache_dir).parent / '_etf_cache'
            sector_fp = find_cache_for_ticker(sector_etf, str(etf_cache_dir))
            if sector_fp and sector_fp.exists():
                sector_df = pd.read_parquet(sector_fp)
                sector_df = normalize_df_index(sector_df)
                df = add_sector_relative_features(df, sector_df, sector_etf)
        
        # VIX regime
        if vix_df is not None:
            df = add_vix_regime_features(df, vix_df)
        
        # Crypto
        if crypto_df is not None:
            df = add_crypto_features(df, crypto_df)
        
        # Rates
        if tlt_df is not None:
            df = add_rates_beta(df, tlt_df)
        
        # Derived features
        df = add_volatility_interactions(df)
        df = add_momentum_quality_features(df)
        df = add_technical_pattern_features(df)
        df = add_acceleration_features(df)
        df = add_regime_adaptive_features(df)
        df = add_liquidity_quality_features(df)
        df = add_theme_acceleration(df)
        
        # Downcast to save space
        float_cols = df.select_dtypes(include=["float64"]).columns
        df[float_cols] = df[float_cols].astype("float32")
        
       # ===================================================================
        # FINAL ELITE FEATURES — SIMPLIFIED (NO COMPOSITE QUALITY HERE)
        # ===================================================================

        # 1. LOW VOLATILITY SCORE (raw value, no normalization yet)
        vol_cols = []
        if 'feat_idio_vol_63' in df.columns:
            vol_cols.append(df['feat_idio_vol_63'])
        elif 'idio_vol_63' in df.columns:
            vol_cols.append(df['idio_vol_63'])
        
        if 'feat_parkinson_20' in df.columns:
            vol_cols.append(df['feat_parkinson_20'])
        elif 'parkinson_20' in df.columns:
            vol_cols.append(df['parkinson_20'])
        elif 'volatility_20' in df.columns:
            vol_cols.append(df['volatility_20'])
        
        if len(vol_cols) >= 1:
            df['feat_low_vol_raw'] = pd.concat(vol_cols, axis=1). mean(axis=1)
        else:
            df['feat_low_vol_raw'] = 0.1  # neutral default

        # 2. EARNINGS QUALITY (placeholder - will be computed in build_labels)
        df['feat_earnings_quality'] = 0.0

        # 3. UPTREND FILTER (EMA50 > EMA200)
        ema50 = None
        ema200 = None
        
        for col in ['feat_ema_50', 'ema_50', 'sma_50']:
            if col in df.columns:
                ema50 = df[col]
                break
        
        for col in ['feat_ema_200', 'ema_200', 'sma_200']:
            if col in df.columns:
                ema200 = df[col]
                break
        
        if ema50 is None or ema200 is None:
            if 'Close' in df.columns:
                if ema50 is None:
                    ema50 = df['Close']. ewm(span=50, adjust=False).mean()
                if ema200 is None:
                    ema200 = df['Close'].ewm(span=200, adjust=False).mean()
        
        if ema50 is not None and ema200 is not None:
            df['feat_in_uptrend'] = (ema50 > ema200).astype(float)
        else:
            df['feat_in_uptrend'] = 0.5  # neutral

        # 4. COMPOSITE QUALITY (placeholder - will be computed in build_labels)
        df['feat_composite_quality'] = 0.5  # neutral placeholder

        # ===================================================================
        # END ELITE FEATURES
        # ===================================================================


        # Save
        df.to_parquet(out_path, compression="zstd")
        
        added_cols = len(df.columns) - initial_cols
        return {"ticker": ticker, "status": "success", "rows": len(df), "added_features": added_cols}
        
    except Exception as e:
        return {"ticker": ticker, "status": "error", "error": str(e)}

def main():
    args = parse_args()
    
    features_dir = Path(args.features_dir)
    if not features_dir.exists():
        raise SystemExit(f"Features directory not found: {features_dir}")
    
    cache_dir = Path(args.cache_dir)
    sector_map = load_sector_map(args.sector_map)
    
    # Load SPY (required)
    etf_cache_dir = ETF_CACHE_DIR

    all_spy_files = list(Path(etf_cache_dir).glob("SPY_*. parquet"))
    logging.info(f"🔍 All SPY files in {etf_cache_dir}:")
    for f in all_spy_files:
        spy_debug = pd.read_parquet(f)
        logging.info(f"  {f.name}: shape={spy_debug.shape}, columns={list(spy_debug.columns)}")

    spy_fp = find_cache_for_ticker("SPY", str(etf_cache_dir))
    logging.info(f"✓ Selected SPY file: {spy_fp}")
    spy_fp = find_cache_for_ticker("SPY", str(etf_cache_dir))
    if not spy_fp:
        raise SystemExit("SPY not found")
    spy_df = pd.read_parquet(spy_fp)
    logging.info(f"SPY raw columns: {list(spy_df.columns)}")
    spy_df = normalize_df_index(spy_df)
    logging.info(f"SPY normalized columns: {list(spy_df.columns)}")
    logging.info(f"Loaded SPY: {len(spy_df)} rows")
    
    # Load VIX (optional)
    vix_df = None
    vix_fp = find_cache_for_ticker("^VIX", str(etf_cache_dir)) or find_cache_for_ticker("VIX", str(etf_cache_dir))
    if vix_fp:
        vix_df = pd.read_parquet(vix_fp)
        vix_df = normalize_df_index(vix_df)
        logging.info(f"Loaded VIX: {len(vix_df)} rows")
    
    # Load crypto (optional)
    crypto_df = None
    crypto_fp = find_cache_for_ticker(DEFAULT_CRYPTO_PROXY, str(etf_cache_dir))
    if crypto_fp:
        crypto_df = pd.read_parquet(crypto_fp)
        crypto_df = normalize_df_index(crypto_df)
        logging.info(f"Loaded crypto: {len(crypto_df)} rows")
    
    # Load TLT (optional)
    tlt_df = None
    tlt_fp = find_cache_for_ticker("TLT", str(etf_cache_dir))
    if tlt_fp:
        tlt_df = pd.read_parquet(tlt_fp)
        tlt_df = normalize_df_index(tlt_df)
        logging.info(f"Loaded TLT: {len(tlt_df)} rows")
    
    # Find tickers
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        all_files = list(features_dir.glob("*_features.parquet"))
        tickers = list(set([f.stem.split("_")[0] for f in all_files]))
    
    if args.limit > 0:
        tickers = tickers[:args.limit]
    
    logging.info(f"Processing {len(tickers)} tickers with {args.processes} processes")
    
    # Process
    results = []
    with ProcessPoolExecutor(max_workers=args.processes) as executor:
        futures = {
            executor.submit(process_ticker, ticker, features_dir, cache_dir,
                          sector_map, spy_df, vix_df, crypto_df, tlt_df, args.overwrite, etf_cache_dir): ticker
            for ticker in tickers
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            
            ticker = result["ticker"]
            status = result["status"]
            
            if status == "success":
                print(f"[{i}/{len(tickers)}] ✓ {ticker}: +{result['added_features']} features")
            elif status == "skip":
                print(f"[{i}/{len(tickers)}] ⊘ {ticker}: {result.get('reason', 'skipped')}")
            else:
                print(f"[{i}/{len(tickers)}] ✗ {ticker}: {result.get('error', 'unknown')}")
    
    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skip")
    errors = sum(1 for r in results if r["status"] == "error")
    
    print(f"\n{'='*60}")
    print(f"Success: {success} | Skipped: {skipped} | Errors: {errors}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
