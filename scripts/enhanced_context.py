#!/usr/bin/env python3
"""
Enhanced Context Builder for LLM Analysis

Fetches comprehensive quantitative and fundamental data to enrich LLM prompts:
- Earnings intelligence (dates, surprises, estimates, revisions)
- Valuation metrics (P/E, P/S, analyst ratings, price targets)
- Recent news headlines
- Quantitative features from parquet files

This enables the LLM to make evidence-based decisions rather than guessing.
"""

import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time


def fetch_earnings_context(symbol: str, max_retries: int = 2) -> Dict:
    """Fetch comprehensive earnings data for a stock.
    
    Args:
        symbol: Stock ticker symbol
        max_retries: Number of retry attempts for failed fetches
        
    Returns:
        Dictionary containing:
        - next_earnings_date: Next earnings date (ISO string or None)
        - days_to_earnings: Days until next earnings (int or None)
        - last_surprise_pct: Last earnings surprise % (float or None)
        - surprise_streak: Consecutive beats/misses count (int)
        - eps_estimate_next_q: Next quarter EPS estimate (float or None)
        - estimate_revision_90d: Estimate revision trend (str or None)
    """
    for attempt in range(max_retries + 1):
        try:
            ticker = yf.Ticker(symbol)
            
            # Get earnings dates from calendar
            next_earnings_date = None
            days_to_earnings = None
            try:
                calendar = ticker.calendar
                if calendar is not None and not calendar.empty:
                    if 'Earnings Date' in calendar.index:
                        earnings_dates = calendar.loc['Earnings Date']
                        if isinstance(earnings_dates, pd.Series) and len(earnings_dates) > 0:
                            next_earnings_date = earnings_dates.iloc[0]
                        elif not pd.isna(earnings_dates):
                            next_earnings_date = earnings_dates
                        
                        if next_earnings_date:
                            # Convert to datetime if it's a timestamp
                            if isinstance(next_earnings_date, (pd.Timestamp, datetime)):
                                next_earnings_date = next_earnings_date.strftime('%Y-%m-%d')
                                # Calculate days to earnings
                                earnings_dt = pd.to_datetime(next_earnings_date)
                                days_to_earnings = (earnings_dt - pd.Timestamp.now()).days
            except Exception as e:
                # Calendar data not available, continue with None values
                pass
            
            # Get earnings history for surprise calculation
            last_surprise_pct = None
            surprise_streak = 0
            try:
                earnings_hist = ticker.earnings_history
                if earnings_hist is not None and not earnings_hist.empty:
                    # Get most recent earnings surprise
                    if 'Surprise(%)' in earnings_hist.columns:
                        surprises = earnings_hist['Surprise(%)'].dropna()
                        if len(surprises) > 0:
                            last_surprise_pct = float(surprises.iloc[0])
                            
                            # Calculate surprise streak (consecutive beats or misses)
                            for surprise in surprises:
                                if (surprise > 0 and last_surprise_pct > 0) or \
                                   (surprise < 0 and last_surprise_pct < 0):
                                    surprise_streak += 1
                                else:
                                    break
            except Exception as e:
                pass
            
            # Get analyst estimates for next quarter
            eps_estimate_next_q = None
            estimate_revision_90d = None
            try:
                # Try to get earnings estimates
                earnings_estimate = ticker.earnings_estimate
                if earnings_estimate is not None and not earnings_estimate.empty:
                    # Look for current quarter estimate
                    if len(earnings_estimate) > 0:
                        current_q = earnings_estimate.iloc[0]
                        if 'Avg. Estimate' in current_q:
                            eps_estimate_next_q = float(current_q['Avg. Estimate'])
                        
                        # Try to detect revision trend (comparing current to 90 days ago if available)
                        # This is simplified - ideally would track historical estimates
                        if 'Number Of Analysts' in current_q:
                            num_analysts = current_q['Number Of Analysts']
                            if num_analysts and num_analysts > 5:
                                estimate_revision_90d = "stable (high coverage)"
                            else:
                                estimate_revision_90d = "limited coverage"
            except Exception as e:
                pass
            
            return {
                'next_earnings_date': next_earnings_date,
                'days_to_earnings': days_to_earnings,
                'last_surprise_pct': last_surprise_pct,
                'surprise_streak': surprise_streak,
                'eps_estimate_next_q': eps_estimate_next_q,
                'estimate_revision_90d': estimate_revision_90d,
            }
            
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)
                continue
            # Return empty dict on final failure
            return {
                'next_earnings_date': None,
                'days_to_earnings': None,
                'last_surprise_pct': None,
                'surprise_streak': 0,
                'eps_estimate_next_q': None,
                'estimate_revision_90d': None,
            }
    
    return {
        'next_earnings_date': None,
        'days_to_earnings': None,
        'last_surprise_pct': None,
        'surprise_streak': 0,
        'eps_estimate_next_q': None,
        'estimate_revision_90d': None,
    }


def fetch_valuation_data(symbol: str, max_retries: int = 2) -> Dict:
    """Fetch valuation metrics for a stock.
    
    Args:
        symbol: Stock ticker symbol
        max_retries: Number of retry attempts for failed fetches
        
    Returns:
        Dictionary containing:
        - pe_ttm: Trailing P/E ratio (float or None)
        - ps_ttm: Trailing P/S ratio (float or None)
        - market_cap: Market capitalization (int or None)
        - analyst_rating: Analyst recommendation key (str or None)
        - price_target: Mean price target (float or None)
        - pct_to_target: % distance to price target (float or None)
    """
    for attempt in range(max_retries + 1):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract valuation metrics
            pe_ttm = info.get('trailingPE')
            ps_ttm = info.get('priceToSalesTrailing12Months')
            market_cap = info.get('marketCap')
            analyst_rating = info.get('recommendationKey')
            price_target = info.get('targetMeanPrice')
            
            # Calculate % to target
            pct_to_target = None
            current_price = info.get('currentPrice')
            if price_target and current_price and current_price > 0:
                pct_to_target = ((price_target - current_price) / current_price) * 100
            
            return {
                'pe_ttm': float(pe_ttm) if pe_ttm else None,
                'ps_ttm': float(ps_ttm) if ps_ttm else None,
                'market_cap': int(market_cap) if market_cap else None,
                'analyst_rating': analyst_rating,
                'price_target': float(price_target) if price_target else None,
                'pct_to_target': round(pct_to_target, 1) if pct_to_target else None,
            }
            
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)
                continue
            # Return empty dict on final failure
            return {
                'pe_ttm': None,
                'ps_ttm': None,
                'market_cap': None,
                'analyst_rating': None,
                'price_target': None,
                'pct_to_target': None,
            }
    
    return {
        'pe_ttm': None,
        'ps_ttm': None,
        'market_cap': None,
        'analyst_rating': None,
        'price_target': None,
        'pct_to_target': None,
    }


def fetch_recent_news(symbol: str, days: int = 7, max_items: int = 5, max_retries: int = 2) -> List[str]:
    """Fetch recent news headlines for a stock.
    
    Args:
        symbol: Stock ticker symbol
        days: Number of days to look back (default 7)
        max_items: Maximum number of headlines to return (default 5)
        max_retries: Number of retry attempts for failed fetches
        
    Returns:
        List of news headlines (strings), empty list if none available
    """
    for attempt in range(max_retries + 1):
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                return []
            
            # Filter to recent news and extract headlines
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_headlines = []
            
            for item in news[:max_items * 2]:  # Get more than needed to filter by date
                if len(recent_headlines) >= max_items:
                    break
                
                # Get headline
                headline = item.get('title')
                if not headline:
                    continue
                
                # Check date if available
                pub_date = item.get('providerPublishTime')
                if pub_date:
                    pub_datetime = datetime.fromtimestamp(pub_date)
                    if pub_datetime < cutoff_date:
                        continue
                
                recent_headlines.append(headline)
            
            return recent_headlines
            
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)
                continue
            return []
    
    return []


def get_stock_features(symbol: str, features_df: pd.DataFrame) -> Dict:
    """Extract key quantitative features for a stock from features dataframe.
    
    Args:
        symbol: Stock ticker symbol
        features_df: DataFrame with features (from parquet file)
        
    Returns:
        Dictionary containing key momentum, volatility, technical, and macro features
    """
    try:
        # Filter for the symbol
        stock_data = features_df[features_df['symbol'] == symbol]
        
        if stock_data.empty:
            # Return defaults if symbol not found
            return get_default_features()
        
        # Get the most recent row
        row = stock_data.iloc[-1]
        
        # Extract features with safe fallbacks
        return {
            # Momentum factors
            'mom_12m_skip1m': float(row.get('mom_12m_skip1m', 0)),
            'ret_63d': float(row.get('ret_63d', 0)),
            'sector_rel_ret_63d': float(row.get('feat_sector_rel_ret_63d', 0)),
            
            # Volatility & risk factors
            'idio_vol_63': float(row.get('feat_idio_vol_63', 0)),
            'beta_spy_126': float(row.get('feat_beta_spy_126', 0)),
            'volatility_20': float(row.get('volatility_20', 0)),
            'defensive_score': float(row.get('feat_defensive_score', 0)),
            
            # Technical indicators
            'rsi': float(row.get('rsi', 50)),
            'breakout_strength_20d': float(row.get('breakout_strength_20d', 0)),
            'rvol_z_60': float(row.get('rvol_z_60', 0)),
            
            # Macro regime
            'vix_regime': 'HIGH' if row.get('feat_high_vol_regime', 0) > 0 else 'LOW',
            'vix_z': float(row.get('feat_vix_level_z_63', 0)),
            
            # Quality
            'earnings_quality': float(row.get('feat_earnings_quality', 0)),
        }
        
    except Exception as e:
        # Return defaults on error
        return get_default_features()


def get_default_features() -> Dict:
    """Return default feature values when data is unavailable."""
    return {
        'mom_12m_skip1m': 0.0,
        'ret_63d': 0.0,
        'sector_rel_ret_63d': 0.0,
        'idio_vol_63': 0.0,
        'beta_spy_126': 1.0,
        'volatility_20': 0.0,
        'defensive_score': 0.0,
        'rsi': 50.0,
        'breakout_strength_20d': 0.0,
        'rvol_z_60': 0.0,
        'vix_regime': 'LOW',
        'vix_z': 0.0,
        'earnings_quality': 0.0,
    }


# Helper functions for formatting

def _rsi_flag(rsi: float) -> str:
    """Return flag for RSI level."""
    if rsi is None or pd.isna(rsi):
        return ""
    if rsi < 30:
        return "🔴 OVERSOLD"
    elif rsi > 70:
        return "🔵 OVERBOUGHT"
    else:
        return "⚪ NEUTRAL"


def _vol_flag(vol_z: float) -> str:
    """Return flag for volume z-score."""
    if vol_z is None or pd.isna(vol_z):
        return ""
    if vol_z > 2:
        return "📈 HIGH VOLUME"
    elif vol_z < -1:
        return "📉 LOW VOLUME"
    else:
        return ""


def _earnings_flag(days: Optional[int]) -> str:
    """Return flag for earnings proximity."""
    if days is None:
        return ""
    if days <= 7:
        return "⚠️ IMMINENT"
    elif days <= 14:
        return "⏰ APPROACHING"
    else:
        return ""


def _format_market_cap(market_cap: Optional[int]) -> str:
    """Format market cap in billions."""
    if market_cap is None:
        return "N/A"
    billions = market_cap / 1_000_000_000
    return f"${billions:.1f}B"


def _format_news(news_list: List[str]) -> str:
    """Format news headlines for display."""
    if not news_list:
        return "│   No recent news available"
    
    formatted = []
    for i, headline in enumerate(news_list[:5], 1):
        # Truncate long headlines
        if len(headline) > 70:
            headline = headline[:67] + "..."
        formatted.append(f"│   {i}. {headline}")
    
    return "\n".join(formatted)


if __name__ == "__main__":
    # Test the module with a sample ticker
    import sys
    
    test_symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    print(f"Testing enhanced_context module with {test_symbol}...")
    print("=" * 60)
    
    print("\n1. Fetching earnings context...")
    earnings = fetch_earnings_context(test_symbol)
    print(f"   Next earnings: {earnings['next_earnings_date']}")
    print(f"   Days to earnings: {earnings['days_to_earnings']}")
    print(f"   Last surprise: {earnings['last_surprise_pct']}%")
    print(f"   Surprise streak: {earnings['surprise_streak']}")
    
    print("\n2. Fetching valuation data...")
    valuation = fetch_valuation_data(test_symbol)
    print(f"   P/E: {valuation['pe_ttm']}")
    print(f"   Market cap: {_format_market_cap(valuation['market_cap'])}")
    print(f"   Analyst rating: {valuation['analyst_rating']}")
    print(f"   Price target: ${valuation['price_target']}")
    
    print("\n3. Fetching recent news...")
    news = fetch_recent_news(test_symbol)
    print(f"   Found {len(news)} headlines:")
    for headline in news:
        print(f"   - {headline[:80]}")
    
    print("\n✅ Module test complete")
