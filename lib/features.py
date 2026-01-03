#!/usr/bin/env python3
"""Vectorized feature computation library for technical indicators and advanced features."""
import numpy as np
import pandas as pd
from typing import Optional, Tuple

# ============ HELPER FUNCTIONS FOR ADJUSTED CLOSE ============

def _get_close_for_returns(df: pd.DataFrame) -> pd.Series:
    """Get adjusted close for return calculations (handles splits/dividends).
    
    Falls back to raw Close if Adj Close not available.
    """
    if 'Adj Close' in df.columns:
        return df['Adj Close']
    return df['Close']


def _get_close_for_levels(df: pd.DataFrame) -> pd.Series:
    """Get raw close for price-level features (support/resistance, etc.)."""
    return df['Close']


def compute_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute baseline technical indicators (existing features for parity).
    
    Args:
        df: DataFrame with OHLCV columns (Open, High, Low, Close, Adj Close, Volume)
        
    Returns:
        DataFrame with baseline feature columns
    """
    out = pd.DataFrame(index=df.index)
    
    # Use adjusted close for returns, raw close for price-level features
    close = _get_close_for_returns(df)
    high = df['High'] if 'High' in df.columns else close
    low = df['Low'] if 'Low' in df.columns else close
    volume = df['Volume'] if 'Volume' in df.columns else pd.Series(0, index=df.index)

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD and signal
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out['macd'] = ema12 - ema26
    out['macd_signal'] = out['macd'].ewm(span=9, adjust=False).mean()
    
    # SMA
    out['sma_10'] = close.rolling(10).mean()
    out['sma_50'] = close.rolling(50).mean()
    
    # EMA
    out['ema_10'] = close.ewm(span=10, adjust=False).mean()
    out['ema_50'] = close.ewm(span=50, adjust=False).mean()
    
    # Bollinger Bands (20, 2)
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    out['bb_upper'] = sma_20 + 2 * std_20
    out['bb_lower'] = sma_20 - 2 * std_20
    out['bb_width'] = out['bb_upper'] - out['bb_lower']
    
    # ATR (14)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out['atr_14'] = tr.rolling(14).mean()
    
    # OBV
    sign = np.sign(close.diff().fillna(0))
    out['obv'] = (sign * volume).cumsum()
    
    # Momentum and ROC
    out['momentum_10'] = close - close.shift(10)
    out['roc_10'] = close.pct_change(10)
    
    # Stochastic %K/%D (14, 3)
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    stoch_k = (close - low14) / (high14 - low14 + 1e-12) * 100
    out['stoch_k'] = stoch_k
    out['stoch_d'] = stoch_k.rolling(3).mean()
    
    # Williams %R (14)
    out['williams_r'] = (high14 - close) / (high14 - low14 + 1e-12) * -100
    
    # MFI (14)
    tp = (high + low + close) / 3
    mf = tp * volume
    pos = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg = mf.where(tp < tp.shift(1), 0).rolling(14).sum().abs()
    with np.errstate(divide='ignore', invalid='ignore'):
        mfr = pos / neg.replace(0, np.nan)
        out['mfi_14'] = 100 - (100 / (1 + mfr))
    
    # Volume and volatility
    out['avg_volume_20'] = volume.rolling(20).mean()
    # Daily volatility - clip to reasonable bounds (1.0 = 100% daily std is extreme)
    out['volatility_20'] = close.pct_change().rolling(20).std().clip(0, 1.0)
    
    # Simple returns
    out['ret_1d'] = close.pct_change()
    out['ret_5d'] = close.pct_change(5)
    out['ret_21d'] = close.pct_change(21)
    
    return out


def compute_multi_horizon_returns(close: pd.Series) -> pd.DataFrame:
    """Compute multi-horizon returns and momentum features.
    
    Args:
        close: Close price series
        
    Returns:
        DataFrame with return columns
    """
    out = pd.DataFrame(index=close.index)
    
    # Returns at different horizons
    out['ret_21d'] = close.pct_change(21)
    out['ret_63d'] = close.pct_change(63)
    out['ret_126d'] = close.pct_change(126)
    out['ret_252d'] = close.pct_change(252)
    
    # Log returns
    log_close = np.log(close)
    out['log_ret_63d'] = log_close - log_close.shift(63)
    out['log_ret_252d'] = log_close - log_close.shift(252)
    
    # 12-month momentum skip 1 month (clip to prevent impossible values)
    # Can't lose more than 99%, reasonable upper bound is 10x (1000%)
    out['mom_12m_skip1m'] = (out['ret_252d'] - out['ret_21d']).clip(-0.99, 10)
    
    # Distance to 52-week high/low
    rolling_max_252 = close.rolling(252).max()
    rolling_min_252 = close.rolling(252).min()
    out['pct_to_52w_high'] = (close / rolling_max_252) - 1
    out['pct_in_52w_range'] = (close - rolling_min_252) / (rolling_max_252 - rolling_min_252 + 1e-12)
    
    return out


def compute_trend_quality(close: pd.Series, window: int = 63) -> pd.DataFrame:
    """Compute trend quality and stability metrics.
    
    Args:
        close: Close price series
        window: Rolling window size
        
    Returns:
        DataFrame with trend quality columns
    """
    out = pd.DataFrame(index=close.index)
    
    # Trend R^2 via rolling correlation of log price with time
    log_close = np.log(close)
    
    def rolling_r2(series, w):
        """Compute rolling R^2 of series vs time index."""
        result = pd.Series(index=series.index, dtype=float)
        for i in range(w - 1, len(series)):
            y = series.iloc[i - w + 1:i + 1].values
            if len(y) < w or np.isnan(y).any():
                result.iloc[i] = np.nan
                continue
            x = np.arange(len(y))
            corr = np.corrcoef(x, y)[0, 1]
            result.iloc[i] = corr ** 2 if not np.isnan(corr) else np.nan
        return result
    
    out['trend_r2_63d'] = rolling_r2(log_close, window)
    
    # Up day fraction
    is_up = close.diff() > 0
    out['up_day_frac_21'] = is_up.rolling(21).mean()
    out['up_day_frac_63'] = is_up.rolling(63).mean()
    
    # Consecutive up/down days
    def max_consecutive(series, window):
        """Maximum consecutive True values in rolling window."""
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window - 1, len(series)):
            vals = series.iloc[i - window + 1:i + 1].values
            max_consec = 0
            current = 0
            for v in vals:
                if v:
                    current += 1
                    max_consec = max(max_consec, current)
                else:
                    current = 0
            result.iloc[i] = max_consec
        return result
    
    out['consec_up_max_21'] = max_consecutive(is_up, 21)
    out['consec_down_max_21'] = max_consecutive(~is_up, 21)
    
    return out


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.DataFrame:
    """Compute Average Directional Index (ADX) using Wilder smoothing.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        window: Lookback period
        
    Returns:
        DataFrame with ADX, +DI, -DI columns
    """
    out = pd.DataFrame(index=close.index)
    
    # True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)
    
    # Wilder smoothing (EMA with alpha = 1/window)
    alpha = 1.0 / window
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_di_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_di_smooth / (atr + 1e-12)
    minus_di = 100 * minus_di_smooth / (atr + 1e-12)
    
    # DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    
    out['adx_14'] = adx
    out['plus_di_14'] = plus_di
    out['minus_di_14'] = minus_di
    
    return out


def compute_breakouts(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.DataFrame:
    """Compute breakout and pullback indicators.
    
    Args:
        close: Close price series
        high: High price series
        low: Low price series
        
    Returns:
        DataFrame with breakout columns
    """
    out = pd.DataFrame(index=close.index)
    
    # Donchian channels
    high_20 = high.rolling(20).max()
    low_20 = low.rolling(20).min()
    high_55 = high.rolling(55).max()
    low_55 = low.rolling(55).min()
    
    out['donchian_pct_20'] = (close - low_20) / (high_20 - low_20 + 1e-12)
    out['donchian_pct_55'] = (close - low_55) / (high_55 - low_55 + 1e-12)
    
    # Breakout flags (1 if at or near high, 0 otherwise)
    out['breakout_20d'] = (close >= high_20.shift(1) * 0.999).astype(int)
    out['breakout_55d'] = (close >= high_55.shift(1) * 0.999).astype(int)
    
    # Z-score from moving average
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    out['zscore_dm20'] = (close - sma_20) / (std_20 + 1e-12)
    
    # RSI(2) for short-term mean reversion
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(2).mean()
    loss = -delta.clip(upper=0).rolling(2).mean()
    rs2 = gain / (loss + 1e-12)
    out['rsi2'] = 100 - (100 / (1 + rs2))
    
    return out


def compute_volume_features(close: pd.Series, volume: pd.Series, high: pd.Series, low: pd.Series) -> pd.DataFrame:
    """Compute advanced volume and flow indicators.
    
    Args:
        close: Close price series
        volume: Volume series
        high: High price series
        low: Low price series
        
    Returns:
        DataFrame with volume feature columns
    """
    out = pd.DataFrame(index=close.index)
    
    # Relative volume
    avg_vol_20 = volume.rolling(20).mean()
    out['rvol_20'] = volume / (avg_vol_20 + 1e-12)
    
    # Volume z-score
    vol_mean_60 = volume.rolling(60).mean()
    vol_std_60 = volume.rolling(60).std()
    out['rvol_z_60'] = (volume - vol_mean_60) / (vol_std_60 + 1e-12)
    
    # Chaikin Money Flow (20)
    mfm = ((close - low) - (high - close)) / (high - low + 1e-12)
    mfv = mfm * volume
    out['cmf_20'] = mfv.rolling(20).sum() / (volume.rolling(20).sum() + 1e-12)
    
    # Volume Price Trend
    out['vpt'] = (volume * close.pct_change()).cumsum()
    
    return out


def compute_volatility_features(open_: pd.Series, high: pd.Series, low: pd.Series, 
                                close: pd.Series) -> pd.DataFrame:
    """Compute range-based volatility estimators.
    
    Args:
        open_: Open price series
        high: High price series
        low: Low price series
        close: Close price series
        
    Returns:
        DataFrame with volatility columns
    """
    out = pd.DataFrame(index=close.index)
    
    # Parkinson volatility (20-day) - clip to reasonable bounds
    hl_ratio = np.log(high / (low + 1e-12))
    out['parkinson_20'] = np.sqrt((hl_ratio ** 2).rolling(20).mean() / (4 * np.log(2))).clip(0, 2.0)
    
    # Garman-Klass volatility (20-day) - clip to reasonable bounds
    hl_term = 0.5 * (np.log(high / (low + 1e-12)) ** 2)
    oc_term = (2 * np.log(2) - 1) * (np.log(close / (open_ + 1e-12)) ** 2)
    out['garman_klass_20'] = np.sqrt((hl_term - oc_term).rolling(20).mean()).clip(0, 2.0)
    
    # Rogers-Satchell volatility (20-day) - clip to reasonable bounds
    hc = np.log(high / (close + 1e-12))
    ho = np.log(high / (open_ + 1e-12))
    lc = np.log(low / (close + 1e-12))
    lo = np.log(low / (open_ + 1e-12))
    rs_term = hc * ho + lc * lo
    out['rogers_satchell_20'] = np.sqrt(rs_term.rolling(20).mean()).clip(0, 2.0)
    
    # ATR normalized
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    out['atr_norm'] = atr_14 / (close + 1e-12)
    
    # Downside volatility (20-day) - clip to reasonable bounds
    returns = close.pct_change()
    downside_returns = returns.where(returns < 0, 0)
    out['downside_vol_20'] = downside_returns.rolling(20).std().clip(0, 1.0)
    
    return out


def compute_gap_features(open_: pd.Series, close: pd.Series, atr_14: pd.Series) -> pd.DataFrame:
    """Compute overnight and intraday return features.
    
    Args:
        open_: Open price series
        close: Close price series
        atr_14: 14-day ATR series
        
    Returns:
        DataFrame with gap feature columns
    """
    out = pd.DataFrame(index=close.index)
    
    prev_close = close.shift(1)
    
    # Overnight return (open today / close yesterday - 1)
    out['overnight_ret'] = (open_ / (prev_close + 1e-12)) - 1
    
    # Intraday return (close today / open today - 1)
    out['intraday_ret'] = (close / (open_ + 1e-12)) - 1
    
    # Gap size
    gap = (open_ / (prev_close + 1e-12)) - 1
    out['gap_size'] = gap
    
    # Gap size relative to ATR
    out['gap_size_atr'] = gap / ((atr_14 / (close + 1e-12)) + 1e-12)
    
    return out


def compute_idiosyncratic_volatility(returns: pd.Series, market_returns: pd.Series, 
                                     window: int = 63) -> pd.Series:
    """Compute idiosyncratic volatility relative to market (SPY).
    
    Args:
        returns: Stock returns series
        market_returns: Market (SPY) returns series aligned to stock returns
        window: Rolling window for regression
        
    Returns:
        Series with idiosyncratic volatility
    """
    # Align the two series
    aligned = pd.DataFrame({'stock': returns, 'market': market_returns}).dropna()
    
    if len(aligned) < window:
        return pd.Series(np.nan, index=returns.index)
    
    result = pd.Series(index=returns.index, dtype=float)
    
    for i in range(window - 1, len(aligned)):
        window_data = aligned.iloc[i - window + 1:i + 1]
        stock_ret = window_data['stock'].values
        market_ret = window_data['market'].values
        
        # Compute beta via covariance / variance
        cov = np.cov(stock_ret, market_ret)[0, 1]
        var_market = np.var(market_ret)
        
        if var_market > 1e-12:
            beta = cov / var_market
            residuals = stock_ret - beta * market_ret
            idio_vol = np.std(residuals)
            # Clip to reasonable bounds (daily idio vol shouldn't exceed 100%)
            idio_vol = np.clip(idio_vol, 0, 1.0)
        else:
            idio_vol = np.std(stock_ret)
            idio_vol = np.clip(idio_vol, 0, 1.0)
        
        # FIX: use label-based assignment with .loc (aligned.index[i] is a Timestamp)
        result.loc[aligned.index[i]] = idio_vol
    
    return result.reindex(returns.index)

def compute_sector_relative_features(returns: pd.Series, sector_returns: pd.Series,
                                     sector_universe_returns: pd.DataFrame = None) -> pd.DataFrame:
    """Compute sector-relative return features.
    
    Args:
        returns: Stock returns series (e.g., 63-day)
        sector_returns: Sector ETF returns aligned to stock
        sector_universe_returns: Optional DataFrame of all stocks in sector for z-score
        
    Returns:
        DataFrame with sector-relative columns
    """
    out = pd.DataFrame(index=returns.index)
    
    # Sector relative return
    out['sector_rel_ret_63d'] = returns - sector_returns
    
    # If we have universe data, compute z-score within sector
    if sector_universe_returns is not None and not sector_universe_returns.empty:
        # Compute z-score: (stock_ret - sector_mean) / sector_std
        sector_mean = sector_universe_returns.mean(axis=1)
        sector_std = sector_universe_returns.std(axis=1)
        out['sector_z_63d'] = (returns - sector_mean) / (sector_std + 1e-12)
    else:
        out['sector_z_63d'] = np.nan
    
    return out


def compute_all_features(df: pd.DataFrame, market_df: Optional[pd.DataFrame] = None,
                        sector_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Compute all features (baseline + advanced) for a ticker.
    
    Args:
        df: DataFrame with OHLCV columns (including Adj Close for returns)
        market_df: Optional SPY DataFrame for market-relative features
        sector_df: Optional sector ETF DataFrame for sector-relative features
        
    Returns:
        DataFrame with all feature columns
    """
    # Start with baseline features
    features = compute_baseline_features(df)
    
    # Use adjusted close for returns (handles splits/dividends)
    close = _get_close_for_returns(df)
    high = df['High'] if 'High' in df.columns else close
    low = df['Low'] if 'Low' in df.columns else close
    open_ = df['Open'] if 'Open' in df.columns else close
    volume = df['Volume'] if 'Volume' in df.columns else pd.Series(0, index=df.index)
    
    # Multi-horizon returns
    ret_features = compute_multi_horizon_returns(close)
    features = pd.concat([features, ret_features], axis=1)
    
    # Trend quality
    trend_features = compute_trend_quality(close, window=63)
    features = pd.concat([features, trend_features], axis=1)
    
    # ADX
    adx_features = compute_adx(high, low, close, window=14)
    features = pd.concat([features, adx_features], axis=1)
    
    # Breakouts
    breakout_features = compute_breakouts(close, high, low)
    features = pd.concat([features, breakout_features], axis=1)
    
    # Volume features
    volume_features = compute_volume_features(close, volume, high, low)
    features = pd.concat([features, volume_features], axis=1)
    
    # Volatility features
    vol_features = compute_volatility_features(open_, high, low, close)
    features = pd.concat([features, vol_features], axis=1)
    
    # Gap features (need ATR from baseline)
    if 'atr_14' in features.columns:
        gap_features = compute_gap_features(open_, close, features['atr_14'])
        features = pd.concat([features, gap_features], axis=1)
    
    # Market-relative features (if SPY data provided)
    if market_df is not None:
        market_close = _get_close_for_returns(market_df)
        stock_returns = close.pct_change()
        market_returns = market_close.pct_change()
        # Align indices
        market_returns = market_returns.reindex(stock_returns.index, method='ffill')
        
        # Idiosyncratic volatility
        idio_vol = compute_idiosyncratic_volatility(stock_returns, market_returns, window=63)
        features['idio_vol_63'] = idio_vol
    
    # Sector-relative features (if sector ETF data provided)
    if sector_df is not None:
        sector_close = _get_close_for_returns(sector_df)
        # Compute 63-day returns for both
        stock_ret_63d = close.pct_change(63)
        sector_ret_63d = sector_close.pct_change(63)
        sector_ret_63d = sector_ret_63d.reindex(stock_ret_63d.index, method='ffill')

        sector_features = compute_sector_relative_features(stock_ret_63d, sector_ret_63d)
        features = pd.concat([features, sector_features], axis=1)
    
    return features
