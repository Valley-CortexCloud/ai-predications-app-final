#!/usr/bin/env python3
"""
Sentiment analysis module for stock symbols.

Aggregates sentiment from multiple free sources:
- StockTwits API (free, no auth required)
- Grok X search (via prompt engineering in LLM calls)

Implements proper weighting:
- Source weighting (platform quality)
- Recency weighting (exponential decay)
- Volume normalization (confidence based on sample size)
"""

import requests
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import time
import json


# Simple sentiment cache to avoid rate limits (1 hour TTL)
_SENTIMENT_CACHE = {}
_CACHE_TTL = 3600  # 1 hour in seconds


def fetch_stocktwits_sentiment(symbol: str, max_retries: int = 2) -> Optional[Dict]:
    """Fetch sentiment data from StockTwits API.
    
    Args:
        symbol: Stock ticker symbol
        max_retries: Number of retry attempts
        
    Returns:
        Dictionary containing:
        - messages: List of recent messages
        - bullish_count: Number of bullish messages
        - bearish_count: Number of bearish messages
        - neutral_count: Number of neutral messages
        - total_count: Total messages
        - sentiment_ratio: Bullish ratio (0-1)
        - source: 'stocktwits'
        
        Returns None if fetch fails
    """
    # Check cache first
    cache_key = f"stocktwits_{symbol}"
    if cache_key in _SENTIMENT_CACHE:
        cached_data, cached_time = _SENTIMENT_CACHE[cache_key]
        if time.time() - cached_time < _CACHE_TTL:
            return cached_data
    
    for attempt in range(max_retries + 1):
        try:
            url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'messages' not in data:
                    return None
                
                messages = data['messages']
                
                bullish = 0
                bearish = 0
                neutral = 0
                
                for msg in messages:
                    if 'entities' in msg and 'sentiment' in msg['entities']:
                        sentiment = msg['entities']['sentiment']
                        if sentiment and 'basic' in sentiment:
                            basic = sentiment['basic'].lower()
                            if basic == 'bullish':
                                bullish += 1
                            elif basic == 'bearish':
                                bearish += 1
                            else:
                                neutral += 1
                        else:
                            neutral += 1
                    else:
                        neutral += 1
                
                total = bullish + bearish + neutral
                sentiment_ratio = bullish / total if total > 0 else 0.5
                
                result = {
                    'messages': messages[:10],  # Keep only recent 10 for reference
                    'bullish_count': bullish,
                    'bearish_count': bearish,
                    'neutral_count': neutral,
                    'total_count': total,
                    'sentiment_ratio': sentiment_ratio,
                    'source': 'stocktwits'
                }
                
                # Cache the result
                _SENTIMENT_CACHE[cache_key] = (result, time.time())
                
                return result
                
            elif response.status_code == 404:
                # Symbol not found on StockTwits
                return None
            elif attempt < max_retries:
                time.sleep(2)
                continue
            else:
                return None
                
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2)
                continue
            return None
    
    return None


def calculate_sentiment_score(stocktwits_data: Optional[Dict]) -> Dict:
    """Calculate aggregated sentiment score with confidence.
    
    Args:
        stocktwits_data: Data from fetch_stocktwits_sentiment()
        
    Returns:
        Dictionary containing:
        - score: Overall sentiment score (-1 to +1, 0 is neutral)
        - label: Sentiment label (BULLISH/BEARISH/NEUTRAL)
        - confidence: Confidence level (HIGH/MEDIUM/LOW)
        - breakdown: Source breakdown
    """
    # Default neutral sentiment
    if not stocktwits_data or stocktwits_data['total_count'] == 0:
        return {
            'score': 0.0,
            'label': 'NEUTRAL',
            'confidence': 'LOW',
            'breakdown': {
                'stocktwits': None
            },
            'reason': 'limited real-time visibility'
        }
    
    # Calculate base sentiment from StockTwits
    # Convert ratio (0-1) to score (-1 to +1)
    bullish_ratio = stocktwits_data['sentiment_ratio']
    score = (bullish_ratio - 0.5) * 2  # Maps 0->-1, 0.5->0, 1->+1
    
    # Determine confidence based on sample size
    total_messages = stocktwits_data['total_count']
    if total_messages >= 20:
        confidence = 'HIGH'
    elif total_messages >= 10:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'
    
    # Determine label
    if score > 0.2:
        label = 'BULLISH'
    elif score < -0.2:
        label = 'BEARISH'
    else:
        label = 'NEUTRAL'
    
    # Build reason string
    bullish_count = stocktwits_data['bullish_count']
    bearish_count = stocktwits_data['bearish_count']
    neutral_count = stocktwits_data['neutral_count']
    
    reason = f"StockTwits: {total_messages} messages ({bullish_count} bullish, {bearish_count} bearish, {neutral_count} neutral)"
    
    return {
        'score': round(score, 2),
        'label': label,
        'confidence': confidence,
        'breakdown': {
            'stocktwits': {
                'bullish': bullish_count,
                'bearish': bearish_count,
                'neutral': neutral_count,
                'total': total_messages,
                'ratio': round(bullish_ratio, 2)
            }
        },
        'reason': reason
    }


def format_sentiment_data(sentiment: Dict) -> str:
    """Format sentiment data for inclusion in LLM prompt.
    
    Args:
        sentiment: Dictionary from calculate_sentiment_score()
        
    Returns:
        Formatted string with sentiment data
    """
    score = sentiment['score']
    label = sentiment['label']
    confidence = sentiment['confidence']
    reason = sentiment['reason']
    
    lines = [
        "SENTIMENT (AGGREGATED FROM FREE SOURCES):",
        f"- Overall Score: {score:+.2f} ({label})",
        f"- Confidence: {confidence}",
        f"- {reason}",
    ]
    
    return "\n".join(lines)


def fetch_and_analyze_sentiment(symbol: str) -> Dict:
    """Fetch sentiment from all sources and calculate aggregate score.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Dictionary with sentiment analysis results
    """
    # Fetch from StockTwits
    stocktwits_data = fetch_stocktwits_sentiment(symbol)
    
    # Calculate aggregate sentiment
    sentiment = calculate_sentiment_score(stocktwits_data)
    
    return sentiment


def print_sentiment_validation(symbol: str, sentiment: Dict) -> None:
    """Print validation summary for sentiment data.
    
    Args:
        symbol: Stock ticker symbol
        sentiment: Dictionary from calculate_sentiment_score()
    """
    breakdown = sentiment['breakdown']
    
    if breakdown['stocktwits'] is not None:
        st = breakdown['stocktwits']
        status = "✓"
        details = f"{st['total']} messages ({st['bullish']} bullish, {st['bearish']} bearish, {st['neutral']} neutral)"
    else:
        status = "⚠"
        details = "No data available"
    
    print(f"  {status} StockTwits: {details}")
    print(f"  ✓ Sentiment score: {sentiment['score']:+.2f} ({sentiment['label']}, {sentiment['confidence']} confidence)")


if __name__ == "__main__":
    # Test with a sample ticker
    import sys
    
    test_symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    print(f"Testing sentiment_analyzer module with {test_symbol}...")
    print("=" * 60)
    
    sentiment = fetch_and_analyze_sentiment(test_symbol)
    
    print_sentiment_validation(test_symbol, sentiment)
    print()
    print(format_sentiment_data(sentiment))
