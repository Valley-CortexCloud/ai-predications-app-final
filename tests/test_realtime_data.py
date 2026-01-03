#!/usr/bin/env python3
"""
Unit tests for real-time data fetching and sentiment analysis modules.

Tests cover:
- Technical indicator calculations
- Sentiment aggregation logic
- Data validation and quality checks
- Prompt formatting
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

from realtime_data import (
    compute_rsi,
    compute_macd,
    compute_atr,
    compute_sma,
    fetch_realtime_data,
    format_technical_data,
    print_data_validation
)

from sentiment_analyzer import (
    fetch_stocktwits_sentiment,
    calculate_sentiment_score,
    format_sentiment_data,
    fetch_and_analyze_sentiment
)


# ============================================================================
# Tests for Technical Indicators
# ============================================================================

def test_compute_rsi():
    """Test RSI calculation."""
    # Create synthetic price data with upward trend
    prices = pd.Series([100, 102, 105, 103, 107, 110, 108, 112, 115, 113, 
                       117, 120, 118, 122, 125])
    
    rsi = compute_rsi(prices, period=14)
    
    # RSI should be between 0 and 100
    assert 0 <= rsi <= 100
    # With upward trend, RSI should be > 50
    assert rsi > 50


def test_compute_rsi_downtrend():
    """Test RSI with downward trend."""
    # Create synthetic price data with downward trend
    prices = pd.Series([125, 122, 118, 120, 117, 113, 115, 112, 108, 110,
                       107, 103, 105, 102, 100])
    
    rsi = compute_rsi(prices, period=14)
    
    # RSI should be between 0 and 100
    assert 0 <= rsi <= 100
    # With downward trend, RSI should be < 50
    assert rsi < 50


def test_compute_macd():
    """Test MACD calculation."""
    # Create synthetic price data
    prices = pd.Series(range(100, 150))
    
    macd, signal = compute_macd(prices)
    
    # Both values should be numeric
    assert not np.isnan(macd)
    assert not np.isnan(signal)
    # With upward trend, MACD should be positive
    assert macd > 0


def test_compute_atr():
    """Test ATR calculation."""
    # Create synthetic OHLC data
    data = {
        'High': [102, 105, 104, 107, 106],
        'Low': [98, 101, 100, 103, 102],
        'Close': [100, 103, 102, 105, 104]
    }
    df = pd.DataFrame(data)
    
    atr = compute_atr(df, period=3)
    
    # ATR should be positive
    assert atr > 0
    # ATR should be reasonable (less than average range)
    assert atr < 10


def test_compute_sma():
    """Test SMA calculation."""
    prices = pd.Series([100, 102, 104, 106, 108, 110, 112, 114, 116, 118])
    
    sma = compute_sma(prices, 5)
    
    # SMA should be numeric
    assert not np.isnan(sma)
    # SMA should be around the recent average
    expected = prices.tail(5).mean()
    assert abs(sma - expected) < 0.01


# ============================================================================
# Tests for Data Fetching
# ============================================================================

def test_fetch_realtime_data_structure():
    """Test that fetch_realtime_data returns correct structure."""
    with patch('realtime_data.yf.Ticker') as mock_ticker:
        # Mock yfinance response
        mock_hist = pd.DataFrame({
            'Close': [100] * 300,
            'High': [102] * 300,
            'Low': [98] * 300,
            'Volume': [1000000] * 300
        }, index=pd.date_range(start='2025-01-01', periods=300, freq='D'))
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_hist
        mock_ticker.return_value = mock_ticker_instance
        
        data = fetch_realtime_data('TEST')
        
        # Check all required fields are present
        assert data is not None
        assert 'symbol' in data
        assert 'price' in data
        assert 'date' in data
        assert 'rsi' in data
        assert 'macd' in data
        assert 'atr' in data
        assert 'sma50' in data
        assert 'sma200' in data
        assert 'data_quality' in data


def test_fetch_realtime_data_empty_history():
    """Test handling of empty history from yfinance."""
    with patch('realtime_data.yf.Ticker') as mock_ticker:
        # Mock empty response
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance
        
        data = fetch_realtime_data('INVALID')
        
        # Should return None for invalid ticker
        assert data is None


def test_format_technical_data():
    """Test formatting of technical data for LLM prompt."""
    test_data = {
        'symbol': 'TEST',
        'price': 100.50,
        'date': '2026-01-02',
        'age_days': 1,
        'rsi': 65.5,
        'macd': 2.5,
        'macd_signal': 2.0,
        'atr': 3.25,
        'sma50': 98.0,
        'sma200': 95.0,
        'volume': 5000000,
        'avg_volume_20d': 4000000,
        'volume_ratio': 1.25,
        'high_52w': 110.0,
        'low_52w': 85.0,
        'pct_from_52w_high': -8.64,
        'ret_1d': 0.015,
        'ret_5d': 0.025,
        'ret_21d': 0.05,
        'ret_63d': 0.12,
        'data_quality': 'GOOD'
    }
    
    formatted = format_technical_data(test_data)
    
    # Check that key elements are present
    assert 'TEST' in formatted
    assert '$100.50' in formatted
    assert 'RSI(14):' in formatted
    assert 'MACD:' in formatted
    assert 'GOOD' in formatted


def test_format_technical_data_with_none_values():
    """Test formatting handles None values gracefully."""
    test_data = {
        'symbol': 'TEST',
        'price': 100.50,
        'date': '2026-01-02',
        'age_days': 1,
        'rsi': None,
        'macd': None,
        'macd_signal': None,
        'atr': None,
        'sma50': None,
        'sma200': None,
        'volume': None,
        'avg_volume_20d': None,
        'volume_ratio': None,
        'high_52w': None,
        'low_52w': None,
        'pct_from_52w_high': None,
        'ret_1d': None,
        'ret_5d': None,
        'ret_21d': None,
        'ret_63d': None,
        'data_quality': 'POOR'
    }
    
    formatted = format_technical_data(test_data)
    
    # Should not crash and should contain N/A for missing values
    assert 'N/A' in formatted
    assert 'POOR' in formatted


# ============================================================================
# Tests for Sentiment Analysis
# ============================================================================

def test_calculate_sentiment_score_no_data():
    """Test sentiment calculation with no data."""
    sentiment = calculate_sentiment_score(None)
    
    assert sentiment['score'] == 0.0
    assert sentiment['label'] == 'NEUTRAL'
    assert sentiment['confidence'] == 'LOW'


def test_calculate_sentiment_score_bullish():
    """Test sentiment calculation with bullish data."""
    stocktwits_data = {
        'bullish_count': 15,
        'bearish_count': 5,
        'neutral_count': 5,
        'total_count': 25,
        'sentiment_ratio': 0.6,  # 15/25
        'source': 'stocktwits'
    }
    
    sentiment = calculate_sentiment_score(stocktwits_data)
    
    # Should be positive but neutral label (ratio 0.6 -> score 0.2, which is <=0.2)
    # Need higher ratio for BULLISH label (>0.2)
    assert sentiment['score'] > 0
    assert sentiment['confidence'] == 'HIGH'  # >= 20 messages


def test_calculate_sentiment_score_strongly_bullish():
    """Test sentiment calculation with strongly bullish data."""
    stocktwits_data = {
        'bullish_count': 18,
        'bearish_count': 2,
        'neutral_count': 5,
        'total_count': 25,
        'sentiment_ratio': 0.72,  # 18/25
        'source': 'stocktwits'
    }
    
    sentiment = calculate_sentiment_score(stocktwits_data)
    
    # Should be bullish (ratio 0.72 -> score 0.44, which is >0.2)
    assert sentiment['score'] > 0.2
    assert sentiment['label'] == 'BULLISH'
    assert sentiment['confidence'] == 'HIGH'  # >= 20 messages


def test_calculate_sentiment_score_bearish():
    """Test sentiment calculation with bearish data."""
    stocktwits_data = {
        'bullish_count': 3,
        'bearish_count': 12,
        'neutral_count': 5,
        'total_count': 20,
        'sentiment_ratio': 0.15,  # 3/20
        'source': 'stocktwits'
    }
    
    sentiment = calculate_sentiment_score(stocktwits_data)
    
    # Should be bearish (ratio 0.15 -> score -0.7)
    assert sentiment['score'] < 0
    assert sentiment['label'] == 'BEARISH'


def test_calculate_sentiment_score_neutral():
    """Test sentiment calculation with neutral data."""
    stocktwits_data = {
        'bullish_count': 5,
        'bearish_count': 5,
        'neutral_count': 2,
        'total_count': 12,
        'sentiment_ratio': 0.5,  # 5/10 with sentiment
        'source': 'stocktwits'
    }
    
    sentiment = calculate_sentiment_score(stocktwits_data)
    
    # Should be neutral (ratio 0.5 -> score 0.0)
    assert abs(sentiment['score']) < 0.2
    assert sentiment['label'] == 'NEUTRAL'
    assert sentiment['confidence'] == 'MEDIUM'  # 10-19 messages


def test_format_sentiment_data():
    """Test formatting of sentiment data."""
    sentiment = {
        'score': 0.35,
        'label': 'BULLISH',
        'confidence': 'HIGH',
        'reason': 'StockTwits: 25 messages (15 bullish, 5 bearish, 5 neutral)'
    }
    
    formatted = format_sentiment_data(sentiment)
    
    # Check key elements are present
    assert 'BULLISH' in formatted
    assert 'HIGH' in formatted
    assert 'StockTwits' in formatted


def test_fetch_stocktwits_sentiment_with_mock():
    """Test StockTwits API call with mocked response."""
    with patch('sentiment_analyzer.requests.get') as mock_get:
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'messages': [
                {'entities': {'sentiment': {'basic': 'Bullish'}}},
                {'entities': {'sentiment': {'basic': 'Bullish'}}},
                {'entities': {'sentiment': {'basic': 'Bearish'}}},
                {'entities': {'sentiment': None}},
            ]
        }
        mock_get.return_value = mock_response
        
        result = fetch_stocktwits_sentiment('TEST')
        
        assert result is not None
        assert result['bullish_count'] == 2
        assert result['bearish_count'] == 1
        assert result['neutral_count'] == 1
        assert result['total_count'] == 4


def test_fetch_stocktwits_sentiment_404():
    """Test handling of 404 (symbol not found)."""
    with patch('sentiment_analyzer.requests.get') as mock_get:
        # Mock 404 response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = fetch_stocktwits_sentiment('NOTFOUND')
        
        assert result is None


# ============================================================================
# Integration Tests
# ============================================================================

def test_fetch_and_analyze_sentiment_integration():
    """Test full sentiment analysis pipeline."""
    with patch('sentiment_analyzer.fetch_stocktwits_sentiment') as mock_fetch:
        # Mock StockTwits data
        mock_fetch.return_value = {
            'bullish_count': 10,
            'bearish_count': 5,
            'neutral_count': 5,
            'total_count': 20,
            'sentiment_ratio': 0.5,
            'source': 'stocktwits'
        }
        
        sentiment = fetch_and_analyze_sentiment('TEST')
        
        assert sentiment is not None
        assert 'score' in sentiment
        assert 'label' in sentiment
        assert 'confidence' in sentiment


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
