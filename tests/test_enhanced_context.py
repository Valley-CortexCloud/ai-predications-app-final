#!/usr/bin/env python3
"""
Unit tests for enhanced_context module.

Tests cover:
- Earnings context fetching
- Valuation data fetching
- News headline fetching
- Feature extraction from parquet files
- Data formatting helper functions
- Error handling and graceful degradation
"""

import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add scripts directory to path
sys.path.insert(0, 'scripts')

from enhanced_context import (
    fetch_earnings_context,
    fetch_valuation_data,
    fetch_recent_news,
    get_stock_features,
    get_default_features,
    _rsi_flag,
    _vol_flag,
    _earnings_flag,
    _format_market_cap,
    _format_news
)


# ============================================================================
# Tests for Helper Functions
# ============================================================================

def test_rsi_flag():
    """Test RSI flag generation."""
    assert "OVERSOLD" in _rsi_flag(25)
    assert "OVERBOUGHT" in _rsi_flag(75)
    assert "NEUTRAL" in _rsi_flag(50)
    assert _rsi_flag(None) == ""
    assert _rsi_flag(np.nan) == ""


def test_vol_flag():
    """Test volume z-score flag generation."""
    assert "HIGH VOLUME" in _vol_flag(2.5)
    assert "LOW VOLUME" in _vol_flag(-1.5)
    assert _vol_flag(0) == ""
    assert _vol_flag(None) == ""
    assert _vol_flag(np.nan) == ""


def test_earnings_flag():
    """Test earnings proximity flag generation."""
    assert "IMMINENT" in _earnings_flag(5)
    assert "APPROACHING" in _earnings_flag(10)
    assert _earnings_flag(30) == ""
    assert _earnings_flag(None) == ""


def test_format_market_cap():
    """Test market cap formatting."""
    assert _format_market_cap(1_000_000_000) == "$1.0B"
    assert _format_market_cap(50_000_000_000) == "$50.0B"
    assert _format_market_cap(None) == "N/A"


def test_format_news():
    """Test news headline formatting."""
    # Empty list
    assert "No recent news" in _format_news([])
    
    # Single headline
    news = ["Apple announces new product"]
    formatted = _format_news(news)
    assert "1." in formatted
    assert "Apple announces new product" in formatted
    
    # Long headline truncation
    long_headline = "A" * 100
    formatted = _format_news([long_headline])
    assert "..." in formatted
    assert len(formatted) < 200


# ============================================================================
# Tests for Feature Extraction
# ============================================================================

def test_get_stock_features():
    """Test stock feature extraction from dataframe."""
    # Create mock features dataframe
    features_df = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL'],
        'mom_12m_skip1m': [0.25, 0.15, 0.30],
        'ret_63d': [0.10, 0.05, 0.12],
        'feat_sector_rel_ret_63d': [0.02, -0.01, 0.03],
        'feat_idio_vol_63': [0.15, 0.12, 0.18],
        'feat_beta_spy_126': [1.2, 0.9, 1.1],
        'volatility_20': [0.02, 0.015, 0.025],
        'feat_defensive_score': [0.3, 0.6, 0.2],
        'rsi': [55, 48, 62],
        'breakout_strength_20d': [0.5, 0.2, 0.7],
        'rvol_z_60': [1.5, -0.5, 2.2],
        'feat_high_vol_regime': [0, 0, 1],
        'feat_vix_level_z_63': [0.5, -0.2, 1.5],
        'feat_earnings_quality': [0.7, 0.8, 0.6],
    })
    
    # Test successful extraction
    features = get_stock_features('AAPL', features_df)
    assert features['mom_12m_skip1m'] == 0.25
    assert features['ret_63d'] == 0.10
    assert features['rsi'] == 55
    assert features['vix_regime'] == 'LOW'
    
    # Test GOOGL has HIGH vix regime
    features = get_stock_features('GOOGL', features_df)
    assert features['vix_regime'] == 'HIGH'
    assert features['vix_z'] == 1.5
    
    # Test missing symbol returns defaults
    features = get_stock_features('UNKNOWN', features_df)
    assert features['mom_12m_skip1m'] == 0.0
    assert features['beta_spy_126'] == 1.0
    assert features['rsi'] == 50.0


def test_get_default_features():
    """Test default feature generation."""
    defaults = get_default_features()
    
    assert defaults['mom_12m_skip1m'] == 0.0
    assert defaults['beta_spy_126'] == 1.0
    assert defaults['rsi'] == 50.0
    assert defaults['vix_regime'] == 'LOW'
    assert isinstance(defaults, dict)
    assert len(defaults) > 10  # Should have multiple features


# ============================================================================
# Tests for Earnings Context
# ============================================================================

@patch('enhanced_context.yf.Ticker')
def test_fetch_earnings_context_success(mock_ticker):
    """Test successful earnings context fetching."""
    # Mock ticker object
    ticker_obj = Mock()
    
    # Mock calendar data
    calendar_data = pd.DataFrame({
        'Earnings Date': [datetime(2024, 2, 15)]
    })
    calendar_data = calendar_data.T
    ticker_obj.calendar = calendar_data
    
    # Mock earnings history
    earnings_hist = pd.DataFrame({
        'Surprise(%)': [5.0, 3.0, -2.0]
    })
    ticker_obj.earnings_history = earnings_hist
    
    # Mock earnings estimate
    earnings_estimate = pd.DataFrame({
        'Avg. Estimate': [2.50],
        'Number Of Analysts': [15]
    })
    ticker_obj.earnings_estimate = earnings_estimate
    
    mock_ticker.return_value = ticker_obj
    
    # Test function
    result = fetch_earnings_context('AAPL')
    
    assert result['next_earnings_date'] is not None
    assert result['last_surprise_pct'] == 5.0
    assert result['surprise_streak'] >= 1
    assert result['eps_estimate_next_q'] == 2.50


@patch('enhanced_context.yf.Ticker')
def test_fetch_earnings_context_no_data(mock_ticker):
    """Test earnings context with no data available."""
    ticker_obj = Mock()
    ticker_obj.calendar = None
    ticker_obj.earnings_history = None
    ticker_obj.earnings_estimate = None
    mock_ticker.return_value = ticker_obj
    
    result = fetch_earnings_context('AAPL')
    
    assert result['next_earnings_date'] is None
    assert result['days_to_earnings'] is None
    assert result['last_surprise_pct'] is None
    assert result['surprise_streak'] == 0


# ============================================================================
# Tests for Valuation Data
# ============================================================================

@patch('enhanced_context.yf.Ticker')
def test_fetch_valuation_data_success(mock_ticker):
    """Test successful valuation data fetching."""
    ticker_obj = Mock()
    ticker_obj.info = {
        'trailingPE': 25.5,
        'priceToSalesTrailing12Months': 6.2,
        'marketCap': 2_500_000_000_000,
        'recommendationKey': 'buy',
        'targetMeanPrice': 200.0,
        'currentPrice': 180.0
    }
    mock_ticker.return_value = ticker_obj
    
    result = fetch_valuation_data('AAPL')
    
    assert result['pe_ttm'] == 25.5
    assert result['ps_ttm'] == 6.2
    assert result['market_cap'] == 2_500_000_000_000
    assert result['analyst_rating'] == 'buy'
    assert result['price_target'] == 200.0
    assert result['pct_to_target'] > 0  # Should be positive since target > current


@patch('enhanced_context.yf.Ticker')
def test_fetch_valuation_data_missing_fields(mock_ticker):
    """Test valuation data with missing fields."""
    ticker_obj = Mock()
    ticker_obj.info = {
        'trailingPE': 25.5,
        # Other fields missing
    }
    mock_ticker.return_value = ticker_obj
    
    result = fetch_valuation_data('AAPL')
    
    assert result['pe_ttm'] == 25.5
    assert result['ps_ttm'] is None
    assert result['market_cap'] is None
    assert result['pct_to_target'] is None


# ============================================================================
# Tests for News Fetching
# ============================================================================

@patch('enhanced_context.yf.Ticker')
def test_fetch_recent_news_success(mock_ticker):
    """Test successful news fetching."""
    ticker_obj = Mock()
    
    # Mock news data
    now = datetime.now().timestamp()
    ticker_obj.news = [
        {'title': 'News 1', 'providerPublishTime': now - 3600},  # 1 hour ago
        {'title': 'News 2', 'providerPublishTime': now - 7200},  # 2 hours ago
        {'title': 'News 3', 'providerPublishTime': now - 86400 * 10},  # 10 days ago (should be filtered)
    ]
    mock_ticker.return_value = ticker_obj
    
    result = fetch_recent_news('AAPL', days=7)
    
    assert len(result) == 2  # Should only include first 2
    assert 'News 1' in result
    assert 'News 2' in result
    assert 'News 3' not in result


@patch('enhanced_context.yf.Ticker')
def test_fetch_recent_news_empty(mock_ticker):
    """Test news fetching with no news."""
    ticker_obj = Mock()
    ticker_obj.news = []
    mock_ticker.return_value = ticker_obj
    
    result = fetch_recent_news('AAPL')
    
    assert result == []


@patch('enhanced_context.yf.Ticker')
def test_fetch_recent_news_max_items(mock_ticker):
    """Test news fetching respects max_items limit."""
    ticker_obj = Mock()
    
    now = datetime.now().timestamp()
    ticker_obj.news = [
        {'title': f'News {i}', 'providerPublishTime': now - i * 3600}
        for i in range(20)
    ]
    mock_ticker.return_value = ticker_obj
    
    result = fetch_recent_news('AAPL', days=7, max_items=3)
    
    assert len(result) <= 3


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_context_building():
    """Test building a complete context dict (integration test)."""
    # Create mock features dataframe
    features_df = pd.DataFrame({
        'symbol': ['AAPL'],
        'mom_12m_skip1m': [0.25],
        'ret_63d': [0.10],
        'feat_sector_rel_ret_63d': [0.02],
        'feat_idio_vol_63': [0.15],
        'feat_beta_spy_126': [1.2],
        'volatility_20': [0.02],
        'feat_defensive_score': [0.3],
        'rsi': [55],
        'breakout_strength_20d': [0.5],
        'rvol_z_60': [1.5],
        'feat_high_vol_regime': [0],
        'feat_vix_level_z_63': [0.5],
        'feat_earnings_quality': [0.7],
    })
    
    # Get features
    context = get_stock_features('AAPL', features_df)
    
    # Verify context has all required keys
    required_keys = [
        'mom_12m_skip1m', 'ret_63d', 'sector_rel_ret_63d',
        'idio_vol_63', 'beta_spy_126', 'volatility_20',
        'defensive_score', 'rsi', 'breakout_strength_20d',
        'rvol_z_60', 'vix_regime', 'vix_z', 'earnings_quality'
    ]
    
    for key in required_keys:
        assert key in context, f"Missing key: {key}"
    
    # Verify values are correct types
    assert isinstance(context['mom_12m_skip1m'], float)
    assert isinstance(context['vix_regime'], str)
    assert context['vix_regime'] in ['LOW', 'HIGH']


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
