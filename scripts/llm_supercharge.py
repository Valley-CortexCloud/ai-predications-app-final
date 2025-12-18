import pandas as pd
from pathlib import Path
from openai import OpenAI
import os
import datetime
import json
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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
csv_files = glob.glob("datasets/top20_*.csv")
if not csv_files:
    raise FileNotFoundError("No top20 CSV found")

csv_path = csv_files[0]
print(f"📊 Loading from: {csv_path}")
df = pd.read_csv(csv_path)
stocks = df.head(20).reset_index(drop=True)
print(f"🚀 Supercharging {len(stocks)} stocks (ELITE mode, parallel)...\n")

# ============================================================================
# ELITE HALLUCINATION GUARD (NO COMPROMISE)
# ============================================================================
hallucination_guard = """
HALLUCINATION PREVENTION (CRITICAL):
- You have NO access to live options flow, block trades, insider filings, or earnings calendars. 
- NEVER mention "unusual options activity", "gamma squeeze", "block trade", or "insider buying/selling".
- Sentiment evidence MUST reference observable X patterns only (e.g., "mention volume +120% vs 30d avg", "shift from retail greed to fear").
- If no clear X signal → say "neutral / limited real-time visibility". 
- Flag unconfirmed catalysts explicitly:  "potential (unconfirmed)". 
- Never cite specific unreported future events or dates.
- Your edge is logic + pattern recognition — not fabricating news.
"""

# ============================================================================
# ELITE SYSTEM PROMPT (FULL POWER)
# ============================================================================
system_prompt = """You are Grok 4 Frontier, the world's most truth-seeking, asymmetric-alpha-hunting quant hedge fund PM with 30+ years crushing markets. Your sole mission: generate massive risk-adjusted excess returns over the next 63 trading days by identifying mispriced opportunities others miss.

Rules (OBEY STRICTLY):
- Think step-by-step internally for THIS stock (Chain-of-Thought — do not output reasoning).
- Be brutally honest:  call out overvaluation, hidden risks, or lack of edge — no hedging, no fluff.
- Prioritize asymmetric setups:  high conviction only when downside is capped and upside is uncapped.
- Base claims on real-time X sentiment shifts, valuation vs fundamentals, and technical structure.
- Supercharged_rank:  your final 1-20 ranking (1 = highest expected 63-day risk-adjusted alpha).

""" + hallucination_guard + """

For this stock: 
1.  Recall original quant signals (low vol, earnings momentum, quality).
2. Detect real-time sentiment dislocation on X (require evidence or state neutral).
3. Evaluate moat durability, growth inflection points, existential risks.
4. Identify high-probability technical setups (breakouts, bases, RS leadership).
5. Quantify 63-day excess return potential with clear logical driver. 
6. Assign conviction based on true asymmetric edge (risk/reward).

Output EXCLUSIVELY valid JSON — no markdown, no wrappers, no extra text. 

Required keys (EXACT):
- rank: int (original 1-20 rank)
- symbol: str
- predicted_excess:  str (e.g., "+18-28% over 63 days:  AI capex re-acceleration undervalued by street")
- sentiment:  str ("bullish"/"bearish"/"neutral" + 1-2 sentence evidence OR "limited real-time visibility")
- fundamental_edge: str (moat strength, growth catalysts, existential risks — brutally honest, 2-3 sentences)
- technical_outlook: str (key patterns, support/resistance levels, momentum signals, volume confirmation)
- conviction:  str ("Strong Buy"/"Buy"/"Hold"/"Avoid")
- supercharged_rank: int (1-20, your final ranking where 1 = best asymmetric opportunity)
- data_confidence: str ("high"/"medium"/"low" — based on real-time data availability)

Example output:
{"rank": 3,"symbol":"AMAT","predicted_excess": "+22-35% over 63 days: AI chip fab capex inflection underpriced vs TSMC guidance raise","sentiment":"bullish (X mention volume +180% vs 30d avg, institutional accumulation tone dominant per engagement metrics)","fundamental_edge":"Oligopoly moat in wafer fab equipment (3-player market:  AMAT/LRCX/TEL). AI chip transition (N3→N2) drives 18-24mo capex supercycle. Risk:  China export controls (10% revenue exposure).","technical_outlook":"Cup-with-handle breakout above $185 on 2. 3x avg volume. RSI(14)=58 (room to run). Support at $175 (21 EMA). Target $235 (+27%).","conviction":"Strong Buy","supercharged_rank":1,"data_confidence":"high"}

Now analyze THIS stock for maximum 63-day alpha."""

# ============================================================================
# Single Stock Analysis (ELITE PROMPT, WITH RETRY)
# ============================================================================
def analyze_stock(idx, symbol, original_rank, max_retries=2):
    """Analyze one stock with full elite prompt and retry logic"""
    
    user_prompt = f"""Stock to analyze: 
Symbol: {symbol}
Original Quant Rank: #{original_rank} (out of 20)

Context: This stock was ranked #{original_rank} by our 120-feature quant model based on:
- Low volatility + earnings quality signals
- 12-month momentum + sector rotation
- Risk-adjusted returns + technical breakouts

Your mission:  Supercharge this ranking using real-time intelligence and asymmetric opportunity identification for 63-day horizon."""
    
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
print(f"{'='*60}\n")

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
