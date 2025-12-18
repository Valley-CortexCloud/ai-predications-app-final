import pandas as pd
from pathlib import Path
from openai import OpenAI
import os
import datetime
import json
import glob

# ============================================================================
# API Key Validation
# ============================================================================
xai_key = os.getenv("XAI_API_KEY")
if not xai_key:
    raise ValueError("❌ XAI_API_KEY is MISSING from environment!")

client = OpenAI(
    api_key=xai_key,
    base_url="https://api.x.ai/v1"
)

# ============================================================================
# Load Top 20 Stocks
# ============================================================================
csv_files = glob.glob("datasets/top20_*.csv")
if not csv_files:
    raise FileNotFoundError("No top20_*.csv file found in datasets/")

csv_path = csv_files[0]
print(f"📊 Loading top20 data from: {csv_path}")
df = pd.read_csv(csv_path)
stocks = df['symbol'].tolist()[:20]

# ============================================================================
# Hallucination Guard
# ============================================================================
hallucination_guard = """
HALLUCINATION PREVENTION (CRITICAL):
- You have NO access to live options flow, block trades, insider filings, or earnings calendars.
- NEVER mention "unusual options activity", "gamma squeeze", "block trade", or "insider buying/selling".
- Sentiment evidence MUST reference observable X patterns only (e.g., "mention volume +120% vs 30d avg", "shift from retail greed to fear").
- If no clear X signal → say "neutral / limited real-time visibility".
- Flag unconfirmed catalysts explicitly: "potential (unconfirmed)".
- Never cite specific unreported future events or dates.
- Your edge is logic + pattern recognition — not fabricating news."""

# ============================================================================
# Elite Prompt
# ============================================================================
system_prompt = f"""You are Grok 4 Frontier, the world's most truth-seeking, asymmetric-alpha-hunting quant hedge fund PM with 30+ years crushing markets. Your sole mission: generate massive risk-adjusted excess returns over the next 63 trading days by identifying mispriced opportunities others miss.

Rules (OBEY STRICTLY):
- Think step-by-step internally for EACH stock (Chain-of-Thought — do not output).
- Be brutally honest: call out overvaluation, hidden risks, or lack of edge — no hedging, no fluff.
- Prioritize asymmetric setups: high conviction only when downside is capped and upside is uncapped.
- Base claims on real-time X sentiment shifts, valuation vs fundamentals, and technical structure.
- Supercharged_rank: 1 = highest expected 63-day risk-adjusted alpha.
""" + hallucination_guard + """

For each stock:
1. Recall original quant signals (low vol, earnings momentum, quality).
2. Detect real-time sentiment dislocation on X (require evidence or state neutral).
3. Evaluate moat durability, growth inflection, existential risks.
4. Identify high-probability technical setups (breakouts, bases, RS leadership).
5. Quantify 63-day excess return potential with logical driver.
6. Assign conviction based on true asymmetric edge.

Output EXCLUSIVELY a valid JSON array — no markdown, no wrappers, no extra text.

Strict keys:
- rank: int (original 1-20)
- symbol: str
- predicted_excess: str
- sentiment: str
- fundamental_edge: str
- technical_outlook: str
- conviction: str ("Strong Buy"/"Buy"/"Hold"/"Avoid")
- supercharged_rank: int (1 = best)
- data_confidence: str ("high"/"medium"/"low")

Few-shot example:
[{{"rank":1,"symbol":"NVDA","predicted_excess":"+22-35% over 63 days: AI capex re-acceleration undervalued","sentiment":"bullish (X mention volume +180% vs avg, institutional tone dominant)","fundamental_edge":"Unassailable GPU moat; risk: ongoing geopolitical bans","technical_outlook":"Cup-with-handle breakout above $120 on volume","conviction":"Strong Buy","supercharged_rank":1,"data_confidence":"high"}}]

Now supercharge these {len(stocks)} quant-ranked stocks for maximum 63-day alpha."""

# ============================================================================
# API Call
# ============================================================================
print(f"🚀 Calling Grok-4 to supercharge {len(stocks)} stocks...")
response = client.chat.completions.create(
    model="grok-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Stocks (original quant order):\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(stocks)])}
    ],
    temperature=0.2,
    max_tokens=4096,
    response_format={"type": "json_object"}
)

# ============================================================================
# Cost Tracking
# ============================================================================
output_tokens = response.usage.completion_tokens
rough_input_tokens = 3500 + (len(stocks) * 60)
total_cost = (rough_input_tokens * 3.0 / 1_000_000) + (output_tokens * 15.0 / 1_000_000)
print(f"💰 Estimated API cost: ${total_cost:.4f} (~{response.usage.total_tokens:,} tokens)")

# ============================================================================
# Parse Response Robustly
# ============================================================================
raw_content = response.choices[0].message.content.strip()

# Remove markdown fencing
if "```json" in raw_content:
    raw_content = raw_content.split("```json")[1].split("```")[0].strip()
elif "```" in raw_content:
    raw_content = raw_content. split("```")[1].strip()

try:
    parsed = json.loads(raw_content)

    # Extract stock data
    if isinstance(parsed, list):
        # Format 1: Direct array [{"rank": 1,... }, {"rank":2,...}]
        data = parsed
        print(f"✅ Received direct array ({len(data)} items)")
    
    elif isinstance(parsed, dict):
        # Format 2: Wrapped {"supercharged":  [{... }, {...}]}
        found = False
        for key in ["stocks", "results", "data", "rankings", "array", "output", "supercharged", "analysis"]:
            if key in parsed and isinstance(parsed[key], list):
                data = parsed[key]
                found = True
                print(f"✅ Extracted {len(data)} items from wrapper key:  '{key}'")
                break
        
        if not found: 
            # Format 3: Single stock dict (edge case)
            data = [parsed]
            print(f"⚠️  WARNING: Received single dict (expected list for {len(stocks)} stocks)")
    
    else:
        raise ValueError(f"Unexpected response type: {type(parsed)}")

    # Final safety: ensure data is a list
    if not isinstance(data, list):
        data = [data]

except json.JSONDecodeError as e:
    print(f"❌ JSON PARSING FAILED!")
    print(f"Error:  {e}")
    print(f"Raw response (first 500 chars):\n{raw_content[:500]}")
    raise ValueError(f"Grok returned invalid JSON")

# ============================================================================
# VALIDATION:  Confirm Stock Count
# ============================================================================
expected_count = len(stocks)
actual_count = len(data)

if actual_count != expected_count:
    print(f"\n🚨 STOCK COUNT MISMATCH!")
    print(f"   Expected: {expected_count}")
    print(f"   Got: {actual_count}")
    
    # Debug: show structure
    print(f"\n   Parsed type: {type(parsed)}")
    if isinstance(parsed, dict):
        print(f"   Dict keys: {list(parsed.keys())}")
        for k, v in parsed.items():
            if isinstance(v, (list, dict)):
                print(f"     {k}: {type(v).__name__} (len={len(v)})")
    
    print(f"\n   → MANUAL REVIEW REQUIRED!")
    # Don't crash - let user inspect output
else:
    print(f"✅ Successfully extracted {actual_count} stocks")

# Create DataFrame
result_df = pd.DataFrame(data)
# ============================================================================
# VALIDATION:  Confirm Stock Count
# ============================================================================
expected_count = len(stocks)
actual_count = len(result_df)

if actual_count != expected_count:
    print(f"\n⚠️  WARNING: Expected {expected_count} stocks, got {actual_count}")
    print(f"   This may indicate Grok skipped stocks or returned duplicates.")
    print(f"   Parsed structure type: {type(parsed)}")
    
    if isinstance(parsed, dict):
        print(f"   Top-level keys: {list(parsed.keys())}")
    
    # Don't crash, but flag for manual review
    print(f"   → MANUAL REVIEW REQUIRED!\n")
else:
    print(f"✅ Successfully extracted {actual_count} stocks\n")
# ============================================================================
# Post-Run Analytics & Save
# ============================================================================
result_df['rank_change'] = result_df['rank'] - result_df['supercharged_rank']
result_df['grok_upgrade'] = result_df['rank_change'] > 0
result_df['grok_downgrade'] = result_df['rank_change'] < 0

print(f"\n{'='*60}")
print("📊 GROK-4 SUPERCHARGE RESULTS")
print(f"{'='*60}")
print(f"Stocks analyzed: {len(result_df)}")
print(f"Grok upgrades: {result_df['grok_upgrade'].sum()}")
print(f"Grok downgrades: {result_df['grok_downgrade'].sum()}")
print(f"Unchanged: {(result_df['rank_change'] == 0).sum()}")

# Low confidence
if 'data_confidence' in result_df.columns:
    low_conf = result_df[result_df['data_confidence'] == 'low']
    if len(low_conf) > 0:
        print(f"\n⚠️ Low visibility ({len(low_conf)}): {', '.join(low_conf['symbol'].tolist())}")

# Hallucination check
def detect_hallucination_flags(df):
    flags = []
    for _, row in df.iterrows():
        text = " ".join(str(row.get(k, "")) for k in ['sentiment', 'predicted_excess', 'fundamental_edge', 'technical_outlook'])
        if any(bad in text.lower() for bad in ["options activity", "gamma", "block trade", "insider buy", "insider sell"]):
            flags.append(f"{row['symbol']}: Unverifiable flow mention")
    return flags

hallucination_flags = detect_hallucination_flags(result_df)
if hallucination_flags:
    print(f"\n🚨 HALLUCINATION FLAGS ({len(hallucination_flags)}):")
    for f in hallucination_flags:
        print(f" • {f}")

print(f"{'='*60}\n")

# Save
today = datetime.date.today().strftime("%Y-%m-%d")
Path("datasets").mkdir(exist_ok=True)
output_path = f"datasets/supercharged_top20_{today}.csv"
result_df.to_csv(output_path, index=False)

metadata = {
    "run_date": today,
    "num_stocks": len(stocks),
    "api_cost_usd": round(total_cost, 4),
    "total_tokens": response.usage.total_tokens,
    "grok_upgrades": int(result_df['grok_upgrade'].sum()),
    "grok_downgrades": int(result_df['grok_downgrade'].sum()),
    "hallucination_flags": len(hallucination_flags)
}
with open(f"datasets/supercharged_metadata_{today}.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Saved → {output_path}")
print(f"🎯 Total cost: ${total_cost:.4f}")
