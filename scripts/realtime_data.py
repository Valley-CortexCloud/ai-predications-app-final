#!/usr/bin/env python3
"""
Real-time market data fetching module for stock analysis.

Fetches and computes:
- Current prices and OHLCV data from yfinance
- Technical indicators (RSI, MACD, ATR, SMA, volume metrics)
- 52-week high/low statistics
- Recent performance metrics (1d, 5d, 21d, 63d returns)

All data is computed locally from live market data before sending to LLM.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import time


def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """Compute RSI indicator.
    
    Args:
        prices: Series of closing prices
        period: RSI period (default 14)
        
    Returns:
        Current RSI value (0-100)
    """
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1] if not rsi.empty else np.nan


def compute_macd(prices: pd.Series) -> Tuple[float, float]:
    """Compute MACD and signal line.
    
    Args:
        prices: Series of closing prices
        
    Returns:
        Tuple of (MACD value, Signal line value)
    """
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    
    return macd.iloc[-1] if not macd.empty else np.nan, signal.iloc[-1] if not signal.empty else np.nan


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Compute Average True Range.
    
    Args:
        df: DataFrame with High, Low, Close columns
        period: ATR period (default 14)
        
    Returns:
        Current ATR value
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr.iloc[-1] if not atr.empty else np.nan


def compute_sma(prices: pd.Series, period: int) -> float:
    """Compute Simple Moving Average.
    
    Args:
        prices: Series of closing prices
        period: SMA period
        
    Returns:
        Current SMA value
    """
    sma = prices.rolling(window=period).mean()
    return sma.iloc[-1] if not sma.empty else np.nan


def fetch_realtime_data(symbol: str, max_retries: int = 2) -> Optional[Dict]:
    """Fetch real-time market data and compute technical indicators.
    
    Args:
        symbol: Stock ticker symbol
        max_retries: Number of retry attempts for failed fetches
        
    Returns:
        Dictionary containing:
        - price: Current close price
        - date: Data timestamp
        - age_days: Age of data in days
        - rsi: RSI(14)
        - macd: MACD value
        - macd_signal: MACD signal line
        - atr: ATR(14)
        - sma50: 50-day SMA
        - sma200: 200-day SMA
        - volume: Current volume
        - avg_volume_20d: 20-day average volume
        - volume_ratio: Current volume / 20-day avg
        - high_52w: 52-week high
        - low_52w: 52-week low
        - pct_from_52w_high: % distance from 52w high
        - ret_1d: 1-day return
        - ret_5d: 5-day return
        - ret_21d: 21-day return (1 month)
        - ret_63d: 63-day return (3 months)
        - data_quality: Quality assessment (GOOD/FAIR/POOR)
        
        Returns None if data fetch fails after retries
    """
    for attempt in range(max_retries + 1):
        try:
            # Fetch data with sufficient history for 200-day SMA + 52-week high/low
            ticker = yf.Ticker(symbol)
            
            # Get historical data (1 year to cover all indicators)
            hist = ticker.history(period="1y", interval="1d")
            
            if hist.empty or len(hist) < 50:
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                return None
            
            # Get latest data point
            latest_date = hist.index[-1]
            latest_price = hist['Close'].iloc[-1]
            latest_volume = hist['Volume'].iloc[-1]
            
            # Calculate age of data
            if isinstance(latest_date, pd.Timestamp):
                latest_date_dt = latest_date.to_pydatetime()
            else:
                latest_date_dt = pd.to_datetime(latest_date).to_pydatetime()
            
            age_days = (datetime.now() - latest_date_dt.replace(tzinfo=None)).days
            
            # Compute technical indicators
            rsi = compute_rsi(hist['Close'], period=14)
            macd, macd_signal = compute_macd(hist['Close'])
            atr = compute_atr(hist, period=14)
            sma50 = compute_sma(hist['Close'], 50)
            sma200 = compute_sma(hist['Close'], 200)
            
            # Volume metrics
            avg_volume_20d = hist['Volume'].tail(20).mean()
            volume_ratio = latest_volume / avg_volume_20d if avg_volume_20d > 0 else np.nan
            
            # 52-week high/low
            high_52w = hist['High'].tail(252).max()
            low_52w = hist['Low'].tail(252).min()
            pct_from_52w_high = ((latest_price - high_52w) / high_52w * 100) if high_52w > 0 else np.nan
            
            # Returns at different horizons
            ret_1d = hist['Close'].pct_change(1).iloc[-1] if len(hist) >= 2 else np.nan
            ret_5d = hist['Close'].pct_change(5).iloc[-1] if len(hist) >= 6 else np.nan
            ret_21d = hist['Close'].pct_change(21).iloc[-1] if len(hist) >= 22 else np.nan
            ret_63d = hist['Close'].pct_change(63).iloc[-1] if len(hist) >= 64 else np.nan
            
            # Data quality assessment
            data_quality = "GOOD"
            if age_days > 1:
                data_quality = "FAIR"
            if age_days > 5 or len(hist) < 100:
                data_quality = "POOR"
            
            return {
                'symbol': symbol,
                'price': float(latest_price),
                'date': latest_date_dt.strftime('%Y-%m-%d'),
                'age_days': age_days,
                'rsi': float(rsi) if not np.isnan(rsi) else None,
                'macd': float(macd) if not np.isnan(macd) else None,
                'macd_signal': float(macd_signal) if not np.isnan(macd_signal) else None,
                'atr': float(atr) if not np.isnan(atr) else None,
                'sma50': float(sma50) if not np.isnan(sma50) else None,
                'sma200': float(sma200) if not np.isnan(sma200) else None,
                'volume': int(latest_volume) if not np.isnan(latest_volume) else None,
                'avg_volume_20d': float(avg_volume_20d) if not np.isnan(avg_volume_20d) else None,
                'volume_ratio': float(volume_ratio) if not np.isnan(volume_ratio) else None,
                'high_52w': float(high_52w) if not np.isnan(high_52w) else None,
                'low_52w': float(low_52w) if not np.isnan(low_52w) else None,
                'pct_from_52w_high': float(pct_from_52w_high) if not np.isnan(pct_from_52w_high) else None,
                'ret_1d': float(ret_1d) if not np.isnan(ret_1d) else None,
                'ret_5d': float(ret_5d) if not np.isnan(ret_5d) else None,
                'ret_21d': float(ret_21d) if not np.isnan(ret_21d) else None,
                'ret_63d': float(ret_63d) if not np.isnan(ret_63d) else None,
                'data_quality': data_quality
            }
            
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)
                continue
            print(f"  ❌ {symbol}: Failed to fetch data - {str(e)[:80]}")
            return None
    
    return None


def format_technical_data(data: Dict) -> str:
    """Format technical data for inclusion in LLM prompt.
    
    Args:
        data: Dictionary from fetch_realtime_data()
        
    Returns:
        Formatted string with technical data
    """
    if not data:
        return "Real-time data unavailable"
    
    # Helper to format values
    def fmt_val(val, decimals=2, prefix="", suffix=""):
        if val is None:
            return "N/A"
        return f"{prefix}{val:.{decimals}f}{suffix}"
    
    def fmt_pct(val):
        if val is None:
            return "N/A"
        return f"{val*100:+.2f}%"
    
    lines = [
        f"=== REAL-TIME DATA AS OF {datetime.now().strftime('%Y-%m-%d %H:%M')} ===",
        f"SYMBOL: {data['symbol']}",
        f"PRICE: ${data['price']:.2f} (as of {data['date']}, {data['age_days']} day(s) old)",
        "",
        "TECHNICALS (COMPUTED FROM LIVE DATA):",
        f"- RSI(14): {fmt_val(data['rsi'], 1)}",
        f"- MACD: {fmt_val(data['macd'], 2)} (Signal: {fmt_val(data['macd_signal'], 2)})",
        f"- ATR(14): ${fmt_val(data['atr'], 2)}",
        f"- 50-SMA: ${fmt_val(data['sma50'], 2)} | 200-SMA: ${fmt_val(data['sma200'], 2)}",
        f"- Volume: {fmt_val(data['volume'], 0):>12} ({fmt_val(data['volume_ratio'], 2)}x 20d avg)",
        f"- 52w High: ${fmt_val(data['high_52w'], 2)} | Distance: {fmt_val(data['pct_from_52w_high'], 1)}%",
        f"- 52w Low: ${fmt_val(data['low_52w'], 2)}",
        "",
        "RECENT PERFORMANCE:",
        f"- 1-day: {fmt_pct(data['ret_1d'])}",
        f"- 5-day: {fmt_pct(data['ret_5d'])}",
        f"- 21-day: {fmt_pct(data['ret_21d'])}",
        f"- 63-day: {fmt_pct(data['ret_63d'])}",
        "",
        f"DATA QUALITY: {data['data_quality']}",
        "=== END REAL-TIME DATA ===",
        ""
    ]
    
    return "\n".join(lines)


def print_data_validation(data: Dict) -> None:
    """Print validation summary for fetched data.
    
    Args:
        data: Dictionary from fetch_realtime_data()
    """
    if not data:
        print(f"  ✗ Data fetch failed")
        return
    
    symbol = data['symbol']
    price = data['price']
    date = data['date']
    age = data['age_days']
    quality = data['data_quality']
    
    # Status indicators
    price_status = "✓" if price and price > 0 else "✗"
    age_status = "✓" if age <= 1 else "⚠" if age <= 5 else "✗"
    tech_status = "✓" if data['rsi'] is not None and data['macd'] is not None else "⚠"
    
    # Safe formatting for potentially None values
    rsi_str = f"{data['rsi']:.1f}" if data['rsi'] is not None else 'N/A'
    macd_str = f"{data['macd']:.2f}" if data['macd'] is not None else 'N/A'
    atr_str = f"${data['atr']:.2f}" if data['atr'] is not None else 'N/A'
    
    print(f"[{symbol}] Fetching data...")
    print(f"  {price_status} Price data: ${price:.2f} (as of {date}, {age} day(s) old)")
    print(f"  {tech_status} Technicals computed: RSI={rsi_str}, MACD={macd_str}, ATR={atr_str}")
    print(f"  {age_status} Data quality: {quality}")


if __name__ == "__main__":
    # Test with a sample ticker
    import sys
    
    test_symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    print(f"Testing realtime_data module with {test_symbol}...")
    print("=" * 60)
    
    data = fetch_realtime_data(test_symbol)
    
    if data:
        print_data_validation(data)
        print()
        print(format_technical_data(data))
    else:
        print(f"Failed to fetch data for {test_symbol}")
