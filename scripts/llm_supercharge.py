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
# Hallucination Guard (Strong but Realistic for Grok)
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
# Elite Prompt — Asymmetric Alpha for 63-Day Horizon
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
- predicted_excess: str (e.g., "+18-28% over 63 days: catalyst X undervalued")
- sentiment: str ("bullish"/"bearish"/"neutral" + brief evidence or "limited real-time visibility")
- fundamental_edge: str (moat/growth/risks — brutally honest)
- technical_outlook: str (key patterns, levels, momentum)
- conviction: str ("Strong Buy"/"Buy"/"Hold"/"Avoid")
- supercharged_rank: int (1 = best opportunity)
- data_confidence: str ("high"/"medium"/"low")

Few-shot example:
[{{"rank":1,"symbol":"NVDA","predicted_excess":"+22-35% over 63 days: AI capex re-acceleration undervalued","sentiment":"bullish (X mention volume +180% vs avg, institutional tone dominant)","fundamental_edge":"Unassailable GPU moat; risk: ongoing geopolitical bans","technical_outlook":"Cup-with-handle breakout above $120 on volume","conviction":"Strong Buy","supercharged_rank":1,"data_confidence":"high"}}]

Now supercharge these {len(stocks)} quant-ranked stocks for maximum 63-day alpha."""

# ============================================================================
# API Call — Full Grok 4
# ============================================================================
print(f"🚀 Calling Grok-4 to supercharge {len(stocks)} stocks (Monday pre-market run)...")
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
# Cost Tracking (Correct Grok 4 Pricing)
# ============================================================================
output_tokens = response.usage.completion_tokens
rough_input_tokens = 3500 + (len(stocks) * 60)  # Conservative estimate
total_cost = (rough_input_tokens * 3.0 / 1_000_000) + (output_tokens * 15.0 / 1_000_000)
print(f"💰 Estimated API cost: ${total_cost:.4f} (~{response.usage.total_tokens:,} tokens)")

# ============================================================================
# Parse Response Robustly
# ============================================================================
raw_content = response.choices[0].message.content.strip()

if "```json" in raw_content:
    raw_content = raw_content.split("```json")[1].split("```")[0].strip()
elif "```" in raw_content:
    raw_content = raw_content.split("```")[1].strip()

try:
    parsed = json.loads(raw_content)
except json.JSONDecodeError as e:
    raise ValueError(f"JSON parsing failed: {e}\nFirst 500 chars: {raw_content[:500]}")

if isinstance(parsed, dict):
    for key in ["stocks", "results", "data", "rankings", "array", "output"]:
        if key in parsed and isinstance(parsed[key], list):
            data = parsed[key]
            break
    else:
        data = [parsed]
elif isinstance(parsed, list):
    data = parsed
else:
    raise ValueError(f"Unexpected structure: {type(parsed)}")

result_df = pd.DataFrame(data)

# ============================================================================
# Post-Run Analytics
# ============================================================================
result_df['rank_change'] = result_df['rank'] - result_df['supercharged_rank']
result_df['grok_upgrade'] = result_df['rank_change'] > 0
result_df['grok_downgrade'] = result_df['rank_change'] < 0

print(f"\n{'='*60}")
print("📊 GROK-4 SUPERCHARGE RESULTS (Monday Pre-Market)")
print(f"{'='*60}")
print(f"Stocks analyzed: {len(result_df)}")
print(f"Grok upgrades: {result_df['grok_upgrade'].sum()}")
print(f"Grok downgrades: {result_df['grok_downgrade'].sum()}")
print(f"Unchanged: {(result_df['rank_change'] == 0).sum()}")

# Low confidence stocks
if 'data_confidence' in result_df.columns:
    low_conf = result_df[result_df['data_confidence'] == 'low']
    if len(low_conf) > 0:
        print(f"\n⚠️ Low real-time visibility ({len(low_conf)} stocks): {', '.join(low_conf['symbol'].tolist())}")

# Hallucination flags
def detect_hallucination_flags(df):
    flags = []
    for _, row in df.iterrows():
        s = row['symbol']
        text = " ".join(str(row.get(k, "")) for k in ['sentiment', 'predicted_excess', 'fundamental_edge', 'technical_outlook'])
        if any(bad in text.lower() for bad in ["options activity", "gamma", "block trade", "insider buy", "insider sell"]):
            flags.append(f"{s}: Mentioned unverifiable flow data")
        if "+100%" in str(row.get('predicted_excess', '')) and "short squeeze" not in text.lower():
            flags.append(f"{s}: Extreme return without squeeze justification")
    return flags

hallucination_flags = detect_hallucination_flags(result_df)
if hallucination_flags:
    print(f"\n🚨 POTENTIAL HALLUCINATION FLAGS ({len(hallucination_flags)}):")
    for f in hallucination_flags:
        print(f"  • {f}")

print(f"{'='*60}\n")

# ============================================================================
# Save Results + Metadata
# ============================================================================
today = datetime.date.today().strftime("%Y-%m-%d")
Path("datasets").mkdir(exist_ok=True)

output_path = f"datasets/supercharged_top20_{today}.csv"
result_df.to_csv(output_path, index=False)

metadata = {
    "run_date": today,
    "run_time_et": datetime.datetime.now().isoformat(),
    "num_stocks": len(stocks),
    "api_cost_usd": round(total_cost, 4),
    "total_tokens": response.usage.total_tokens,
    "grok_upgrades": int(result_df['grok_upgrade'].sum()),
    "grok_downgrades": int(result_df['grok_downgrade'].sum()),
    "hallucination_flags": len(hallucination_flags)
}
metadata_path = f"datasets/supercharged_metadata_{today}.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Saved rankings → {output_path}")
print(f"✅ Saved metadata → {metadata_path}")
print(f"🎯 Total cost: ${total_cost:.4f}")
