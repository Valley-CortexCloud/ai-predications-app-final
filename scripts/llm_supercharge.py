import pandas as pd
from pathlib import Path
from openai import OpenAI
import os
import datetime
import json
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import real-time data modules
try:
    from realtime_data import fetch_realtime_data, format_technical_data, print_data_validation
    from sentiment_analyzer import fetch_and_analyze_sentiment, format_sentiment_data, print_sentiment_validation
    from enhanced_context import (
        fetch_earnings_context, fetch_valuation_data, fetch_recent_news, get_stock_features,
        _rsi_flag, _vol_flag, _earnings_flag, _format_market_cap, _format_news
    )
    REALTIME_ENABLED = True
except ImportError:
    REALTIME_ENABLED = False
    print("⚠️  Real-time data modules not available - running in legacy mode")

# ============================================================================
# API Setup
# ============================================================================
xai_key = os.getenv("XAI_API_KEY")
if not xai_key: 
    raise ValueError("❌ XAI_API_KEY missing!")

client = OpenAI(api_key=xai_key, base_url="https://api.x.ai/v1")

# ============================================================================
# Load Top 20
# ============================================================================
csv_files = sorted(glob.glob("datasets/top20_*.csv"), reverse=True)
if not csv_files:
    raise FileNotFoundError("No top20 CSV found")

csv_path = csv_files[0]
print(f"📊 Loading from: {csv_path}")
df = pd.read_csv(csv_path)
stocks = df.head(20).reset_index(drop=True)
print(f"🚀 Supercharging {len(stocks)} stocks (ELITE mode, parallel)...\n")

# ============================================================================
# Load Features for Rich Context
# ============================================================================
features_files = sorted(glob.glob("datasets/features_*.parquet"), reverse=True)
if not features_files:
    # Try data_cache location as fallback
    features_files = sorted(glob.glob("data_cache/**/features_*.parquet", recursive=True), reverse=True)

if features_files:
    try:
        features_df = pd.read_parquet(features_files[0])
        print(f"📊 Loaded features from: {features_files[0]}")
        print(f"   Features shape: {features_df.shape}")
        FEATURES_AVAILABLE = True
    except Exception as e:
        features_df = None
        FEATURES_AVAILABLE = False
        print(f"⚠️ Failed to load features file: {e}")
else:
    features_df = None
    FEATURES_AVAILABLE = False
    print("⚠️ No features file found - using limited context")

# ============================================================================
# ELITE SYSTEM PROMPT (RESEARCH-OPTIMIZED FOR QUANT FUND PM)
# ============================================================================
ELITE_SYSTEM_PROMPT = """You are an elite quantitative portfolio manager at a $50B systematic hedge fund. Your edge comes from combining:
1. Rigorous quantitative signals (provided in the data section)
2. Real-time market intelligence
3. Asymmetric risk/reward identification

INVESTMENT MANDATE:
- Horizon: 63 trading days (3 months)
- Strategy: Momentum + Quality + Low Volatility factor tilt
- Universe: S&P 500 constituents
- Objective: Identify stocks with highest risk-adjusted alpha potential

YOUR TASK:
For the stock below, analyze ALL provided data to:
1. VALIDATE: Does the quantitative data support the model's ranking?
2. IDENTIFY: Are there any dislocations between model signals and real-time data?
3. ASSESS: Is the risk/reward asymmetric (upside > downside)?
4. DECIDE: Final conviction based on EVIDENCE from the provided data

CRITICAL RULES (STRICT COMPLIANCE):
- You MUST cite specific numbers from the provided data in your reasoning
- Default to HOLD unless clear asymmetric opportunity exists
- Earnings within 7 days = REDUCE conviction (binary event risk)
- Volume z-score > 2 required for breakout conviction upgrade
- RSI < 30 with positive momentum = potential mean reversion opportunity
- Never fabricate data points - use ONLY what's provided
- If data is missing, state "data unavailable" rather than guessing

CONVICTION FRAMEWORK:
- Strong Buy: Clear catalyst + technical setup + momentum alignment + capped downside + volume confirmation
- Buy: Positive factors outweigh risks, favorable entry point, reasonable risk/reward
- Hold: Mixed signals, no clear edge, fair value
- Avoid: Deteriorating fundamentals, broken technicals, or excessive binary risk

HALLUCINATION PREVENTION:
- NO access to options flow, insider transactions, or block trades
- Do NOT mention "unusual options activity", "dark pool", "gamma squeeze"
- Sentiment claims MUST reference the provided StockTwits data or state "limited visibility"
- News claims MUST reference the provided headlines or state "no recent news"

Output strict JSON with evidence-backed analysis."""

system_prompt = ELITE_SYSTEM_PROMPT

# ============================================================================
# Rich User Prompt Builder
# ============================================================================
def build_user_prompt(symbol: str, rank: int, context: Dict) -> str:
    """Build the user prompt with rich structured data"""
    
    # Helper to safely format values
    def fmt(val, decimals=2, mult=1):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "N/A"
        try:
            return f"{float(val) * mult:.{decimals}f}"
        except:
            return "N/A"
    
    def fmt_pct(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "N/A"
        try:
            return f"{float(val) * 100:+.1f}%"
        except:
            return "N/A"
    
    return f"""
═══════════════════════════════════════════════════════════════════════════════
STOCK ANALYSIS REQUEST: {symbol} | QUANT MODEL RANK: #{rank}/20
═══════════════════════════════════════════════════════════════════════════════

📊 QUANTITATIVE MODEL SIGNALS (why this stock ranked #{rank}):
┌─────────────────────────────────────────────────────────────────────────────┐
│ MOMENTUM FACTORS                                                            │
│   • 12-month return (skip 1m): {fmt_pct(context.get('mom_12m_skip1m'))}    │
│   • 63-day return: {fmt_pct(context.get('ret_63d'))}                       │
│   • Sector relative (63d): {fmt_pct(context.get('sector_rel_ret_63d'))}    │
│                                                                             │
│ VOLATILITY & RISK FACTORS                                                   │
│   • Idiosyncratic volatility (63d): {fmt(context.get('idio_vol_63'), 4)}   │
│   • Beta to SPY (126d): {fmt(context.get('beta_spy_126'), 2)}              │
│   • 20-day volatility: {fmt_pct(context.get('volatility_20'))}             │
│   • Defensive score: {fmt(context.get('defensive_score'), 2)}              │
│                                                                             │
│ TECHNICAL INDICATORS                                                        │
│   • RSI(14): {fmt(context.get('rsi'), 1)} {_rsi_flag(context.get('rsi'))}  │
│   • Breakout strength (20d): {fmt(context.get('breakout_strength_20d'), 2)}│
│   • Volume z-score (60d): {fmt(context.get('rvol_z_60'), 2)} {_vol_flag(context.get('rvol_z_60'))}
│   • Price vs 50-SMA: {context.get('pct_vs_sma50', 'N/A')}                  │
│   • Price vs 200-SMA: {context.get('pct_vs_sma200', 'N/A')}                │
│   • 52-week range position: {context.get('pct_52w_range', 'N/A')}          │
└─────────────────────────────────────────────────────────────────────────────┘

📈 EARNINGS INTELLIGENCE:
┌─────────────────────────────────────────────────────────────────────────────┐
│   • Next earnings date: {context.get('next_earnings_date', 'Unknown')}     │
│   • Days until earnings: {context.get('days_to_earnings', 'N/A')} {_earnings_flag(context.get('days_to_earnings'))}
│   • Last earnings surprise: {fmt(context.get('last_surprise_pct'), 1) + '%' if context.get('last_surprise_pct') else 'N/A'}
│   • Surprise streak: {context.get('surprise_streak', 'N/A')}               │
│   • Next Q EPS estimate: {context.get('eps_estimate_next_q', 'N/A')}       │
│   • Estimate revisions (90d): {context.get('estimate_revision_90d', 'N/A')}│
│   • Earnings quality score: {fmt(context.get('earnings_quality'), 1)}      │
└─────────────────────────────────────────────────────────────────────────────┘

💰 VALUATION CONTEXT:
┌─────────────────────────────────────────────────────────────────────────────┐
│   • P/E (TTM): {fmt(context.get('pe_ttm'), 1) if context.get('pe_ttm') else 'N/A'}
│   • P/S (TTM): {fmt(context.get('ps_ttm'), 2) if context.get('ps_ttm') else 'N/A'}
│   • Market Cap: {_format_market_cap(context.get('market_cap'))}            │
│   • Analyst Rating: {context.get('analyst_rating', 'N/A')}                 │
│   • Price Target: {'$' + fmt(context.get('price_target'), 2) if context.get('price_target') else 'N/A'}
│   • Distance to Target: {fmt(context.get('pct_to_target'), 1) + '%' if context.get('pct_to_target') else 'N/A'}
└─────────────────────────────────────────────────────────────────────────────┘

📰 RECENT NEWS (Last 7 Days):
┌─────────────────────────────────────────────────────────────────────────────┐
{_format_news(context.get('recent_news', []))}
└─────────────────────────────────────────────────────────────────────────────┘

🌡️ SENTIMENT SNAPSHOT:
┌─────────────────────────────────────────────────────────────────────────────┐
│   • StockTwits: {context.get('sentiment_label', 'N/A')} (score: {context.get('sentiment_score', 0):+.2f})
│   • Messages analyzed: {context.get('sentiment_total', 0)} ({context.get('sentiment_bullish', 0)} bullish, {context.get('sentiment_bearish', 0)} bearish)
│   • Confidence: {context.get('sentiment_confidence', 'LOW')}               │
└─────────────────────────────────────────────────────────────────────────────┘

🏛️ MACRO REGIME:
┌─────────────────────────────────────────────────────────────────────────────┐
│   • VIX Regime: {context.get('vix_regime', 'UNKNOWN')} VOLATILITY (z-score: {fmt(context.get('vix_z'), 2)})
│   • Market Condition: {'Risk-Off' if context.get('vix_z', 0) > 1 else 'Risk-On' if context.get('vix_z', 0) < -0.5 else 'Neutral'}
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
ANALYSIS REQUEST:
Based on ALL the data above, provide your assessment. You MUST reference 
specific data points in your reasoning. Output as strict JSON.
═══════════════════════════════════════════════════════════════════════════════
"""

# ============================================================================
# Single Stock Analysis (ELITE PROMPT, WITH RETRY + REAL-TIME DATA)
# ============================================================================
def analyze_stock(idx, symbol, original_rank, max_retries=2):
    """Analyze one stock with full elite prompt, real-time data, and retry logic"""
    
    # Build rich context from all available sources
    context = {}
    
    # Get quantitative features from parquet file
    if REALTIME_ENABLED and FEATURES_AVAILABLE and features_df is not None:
        try:
            quant_features = get_stock_features(symbol, features_df)
            context.update(quant_features)
        except Exception as e:
            print(f"  ⚠️  {symbol}: Failed to load features - {str(e)[:60]}")
    
    # Get real-time technical data
    if REALTIME_ENABLED:
        try:
            realtime_data = fetch_realtime_data(symbol)
            if realtime_data:
                context.update({
                    'price': realtime_data['price'],
                    'pct_vs_sma50': f"{((realtime_data['price'] / realtime_data['sma50']) - 1) * 100:+.1f}%" if realtime_data.get('sma50') else 'N/A',
                    'pct_vs_sma200': f"{((realtime_data['price'] / realtime_data['sma200']) - 1) * 100:+.1f}%" if realtime_data.get('sma200') else 'N/A',
                    'pct_52w_range': f"{realtime_data.get('pct_from_52w_high', 0):.1f}%" if realtime_data.get('pct_from_52w_high') else 'N/A',
                })
                # Override RSI from realtime if available
                if realtime_data.get('rsi'):
                    context['rsi'] = realtime_data['rsi']
        except Exception as e:
            print(f"  ⚠️  {symbol}: Real-time data fetch error - {str(e)[:60]}")
    
    # Get sentiment data
    if REALTIME_ENABLED:
        try:
            sentiment = fetch_and_analyze_sentiment(symbol)
            if sentiment:
                context.update({
                    'sentiment_score': sentiment['score'],
                    'sentiment_label': sentiment['label'],
                    'sentiment_confidence': sentiment['confidence'],
                    'sentiment_bullish': sentiment['breakdown'].get('stocktwits', {}).get('bullish', 0) if sentiment['breakdown'].get('stocktwits') else 0,
                    'sentiment_bearish': sentiment['breakdown'].get('stocktwits', {}).get('bearish', 0) if sentiment['breakdown'].get('stocktwits') else 0,
                    'sentiment_total': sentiment['breakdown'].get('stocktwits', {}).get('total', 0) if sentiment['breakdown'].get('stocktwits') else 0,
                })
        except Exception as e:
            print(f"  ⚠️  {symbol}: Sentiment fetch error - {str(e)[:60]}")
    
    # Get earnings context
    if REALTIME_ENABLED:
        try:
            earnings_ctx = fetch_earnings_context(symbol)
            context.update(earnings_ctx)
        except Exception as e:
            print(f"  ⚠️  {symbol}: Earnings context error - {str(e)[:60]}")
    
    # Get valuation data
    if REALTIME_ENABLED:
        try:
            valuation = fetch_valuation_data(symbol)
            context.update(valuation)
        except Exception as e:
            print(f"  ⚠️  {symbol}: Valuation data error - {str(e)[:60]}")
    
    # Get recent news
    if REALTIME_ENABLED:
        try:
            context['recent_news'] = fetch_recent_news(symbol)
        except Exception as e:
            context['recent_news'] = []
            print(f"  ⚠️  {symbol}: News fetch error - {str(e)[:60]}")
    
    # Build user prompt with rich context
    user_prompt = build_user_prompt(symbol, original_rank, context)
    
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="grok-4",
                messages=[
                    {"role": "system", "content":  system_prompt},
                    {"role": "user", "content":  user_prompt}
                ],
                temperature=0.2,
                max_tokens=800,  # Full elite analysis needs space
                response_format={"type": "json_object"}
            )

            raw = response.choices[0].message.content.strip()
            
            # Remove markdown if present
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].strip()

            # Parse JSON
            data = json.loads(raw)
            
            # Ensure core fields exist
            data['rank'] = int(original_rank)
            data['symbol'] = symbol
            
            # Track costs
            data['_tokens_used'] = response.usage.total_tokens
            data['_api_cost'] = (response.usage.prompt_tokens * 3.0 / 1_000_000) + \
                                (response.usage.completion_tokens * 15.0 / 1_000_000)
            
            return data

        except json.JSONDecodeError as e:
            if attempt < max_retries:
                print(f"  ⚠️  {symbol}:  JSON parse error, retrying...({attempt+1}/{max_retries})")
                time.sleep(1)
            else:
                print(f"  ❌ {symbol}: JSON failed after {max_retries} retries")
                return create_error_row(symbol, original_rank, f"JSON error: {str(e)[:100]}")
        
        except Exception as e:
            error_str = str(e).lower()
            if attempt < max_retries and ("rate" in error_str or "limit" in error_str):
                wait_time = 5 * (attempt + 1)  # Exponential backoff
                print(f"  ⚠️  {symbol}: Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  ❌ {symbol}: {str(e)[:80]}")
                return create_error_row(symbol, original_rank, str(e)[:150])
    
    return create_error_row(symbol, original_rank, "Max retries exceeded")

def create_error_row(symbol, original_rank, error_msg):
    """Fallback row for failed API calls"""
    return {
        "rank": int(original_rank),
        "symbol": symbol,
        "predicted_excess": "Analysis failed - using original quant rank",
        "sentiment": "error",
        "fundamental_edge":  f"ERROR: {error_msg}",
        "technical_outlook": "N/A",
        "conviction":  "Hold",
        "supercharged_rank": int(original_rank),  # Fallback to original
        "data_confidence":  "low",
        "_tokens_used": 0,
        "_api_cost": 0.0
    }

# ============================================================================
# Parallel Execution (8 workers = safe rate limiting)
# ============================================================================
results = []
total_cost = 0.0
total_tokens = 0

print(f"{'='*60}")
print(f"PROCESSING 20 STOCKS (ELITE GROK-4 ANALYSIS)")
if REALTIME_ENABLED:
    print(f"REAL-TIME DATA: ENABLED ✓")
else:
    print(f"REAL-TIME DATA: DISABLED (legacy mode)")
print(f"{'='*60}\n")

# Pre-fetch and validate real-time data for all stocks (with progress indication)
if REALTIME_ENABLED:
    print(f"{'='*60}")
    print("📊 REAL-TIME DATA PREFETCH")
    print(f"{'='*60}")
    
    for idx, row in stocks.iterrows():
        symbol = row['symbol']
        print(f"[{symbol}] Fetching data...")
        
        # Fetch data
        realtime_data = fetch_realtime_data(symbol)
        sentiment_data = fetch_and_analyze_sentiment(symbol)
        
        # Print validation
        if realtime_data:
            print_data_validation(realtime_data)
        else:
            print(f"  ✗ Price data: unavailable")
        
        if sentiment_data:
            print_sentiment_validation(symbol, sentiment_data)
        else:
            print(f"  ✗ Sentiment: unavailable")
        
        print(f"{'='*60}")
    
    print()

with ThreadPoolExecutor(max_workers=8) as executor:
    # Submit all jobs
    futures = {
        executor.submit(analyze_stock, idx, row['symbol'], idx + 1): (idx, row['symbol'])
        for idx, row in stocks.iterrows()
    }
    
    # Collect with progress tracking
    completed = 0
    for future in as_completed(futures):
        result = future.result()
        results.append(result)
        
        completed += 1
        total_cost += result.get('_api_cost', 0.0)
        total_tokens += result.get('_tokens_used', 0)
        
        # Status indicator
        if result.get('conviction') == 'error':
            status = "✗"
        elif result['rank'] != result['supercharged_rank']:
            status = "↕" if result['rank'] > result['supercharged_rank'] else "↓"
        else:
            status = "="
        
        print(f"  [{completed:2d}/20] {status} {result['symbol']:6s} | #{result['rank']:2d} → #{result['supercharged_rank']:2d} | {result['conviction']:12s} | ${result.get('_api_cost', 0):.4f}")

# ============================================================================
# DataFrame + Analytics
# ============================================================================
result_df = pd.DataFrame(results)
result_df = result_df.sort_values("supercharged_rank").reset_index(drop=True)

# Analytics
result_df['rank_change'] = result_df['rank'] - result_df['supercharged_rank']
result_df['grok_upgrade'] = result_df['rank_change'] > 0
result_df['grok_downgrade'] = result_df['rank_change'] < 0

# ============================================================================
# Summary
# ============================================================================
print(f"\n{'='*60}")
print("📊 ELITE GROK-4 SUPERCHARGE SUMMARY")
print(f"{'='*60}")
print(f"Stocks analyzed:     {len(result_df)}")
print(f"Successful:         {(result_df['conviction'] != 'error').sum()}")
print(f"Failed:             {(result_df['conviction'] == 'error').sum()}")
print(f"\nRank Changes:")
print(f"  Grok upgrades:    {result_df['grok_upgrade'].sum()} (moved UP in ranking)")
print(f"  Grok downgrades:  {result_df['grok_downgrade'].sum()} (moved DOWN in ranking)")
print(f"  Unchanged:        {(result_df['rank_change'] == 0).sum()}")
print(f"\n💰 Total API cost:   ${total_cost:.4f} ({total_tokens: ,} tokens)")
print(f"   Avg per stock:   ${total_cost/len(stocks):.4f}")
print(f"{'='*60}\n")

# Confidence warnings
if 'data_confidence' in result_df.columns:
    low_conf = result_df[result_df['data_confidence'] == 'low']
    if len(low_conf) > 0:
        print(f"⚠️  Low real-time confidence ({len(low_conf)} stocks): {', '.join(low_conf['symbol'].tolist())}")
        print(f"   → Grok had limited X sentiment visibility on these\n")

# Biggest moves
big_upgrades = result_df[result_df['rank_change'] >= 5].sort_values('supercharged_rank')
big_downgrades = result_df[result_df['rank_change'] <= -5].sort_values('supercharged_rank', ascending=False)

if len(big_upgrades) > 0:
    print(f"🚀 MAJOR UPGRADES (≥5 ranks):")
    for _, row in big_upgrades.iterrows():
        print(f"   {row['symbol']:6s}: #{row['rank']:2d} → #{row['supercharged_rank']:2d} ({row['rank_change']:2d}) | {row['conviction']}")
        print(f"      Reason: {row['predicted_excess'][: 80]}...")
    print()

if len(big_downgrades) > 0:
    print(f"⚠️  MAJOR DOWNGRADES (≥5 ranks):")
    for _, row in big_downgrades.iterrows():
        print(f"   {row['symbol']:6s}: #{row['rank']:2d} → #{row['supercharged_rank']:2d} ({row['rank_change']:2d}) | {row['conviction']}")
        print(f"      Reason: {row.get('fundamental_edge', 'N/A')[:80]}...")
    print()

# ============================================================================
# Save Results
# ============================================================================
today = datetime.date.today().strftime("%Y-%m-%d")
Path("datasets").mkdir(exist_ok=True)
output_path = f"datasets/supercharged_top20_{today}.csv"

# Remove internal tracking columns
result_df_clean = result_df.drop(columns=['_tokens_used', '_api_cost'], errors='ignore')
result_df_clean.to_csv(output_path, index=False)

# Metadata
metadata = {
    "run_date": today,
    "run_time": datetime.datetime.now().isoformat(),
    "num_stocks": len(stocks),
    "successful":  int((result_df['conviction'] != 'error').sum()),
    "failed": int((result_df['conviction'] == 'error').sum()),
    "total_api_cost_usd": round(total_cost, 4),
    "total_tokens":  total_tokens,
    "avg_cost_per_stock": round(total_cost / len(stocks), 4),
    "grok_upgrades": int(result_df['grok_upgrade'].sum()),
    "grok_downgrades": int(result_df['grok_downgrade'].sum()),
    "major_upgrades_5plus": int((result_df['rank_change'] >= 5).sum()),
    "major_downgrades_5plus": int((result_df['rank_change'] <= -5).sum())
}

metadata_path = f"datasets/supercharged_metadata_{today}.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Saved supercharged rankings → {output_path}")
print(f"✅ Saved run metadata → {metadata_path}")
print(f"\n🎯 TOTAL RUN COST: ${total_cost:.4f}")
print(f"   Annual cost (252 days): ${total_cost * 252:.2f}")
print(f"\n🔥 ELITE ANALYSIS COMPLETE - READY TO PRINT ALPHA\n")
