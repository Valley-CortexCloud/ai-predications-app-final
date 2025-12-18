import pandas as pd
from pathlib import Path
from openai import OpenAI
import os
import datetime
import json
import glob

# Verify key (keep for safety, minimal output)
xai_key = os.getenv("XAI_API_KEY")
if not xai_key:
    raise ValueError("XAI_API_KEY is MISSING from environment!")

# xAI API setup
client = OpenAI(
    api_key=xai_key,
    base_url="https://api.x.ai/v1"
)

# Load top20 dynamically
csv_files = glob.glob("datasets/top20_*.csv")
if not csv_files:
    raise FileNotFoundError("No top20_*.csv file found in datasets/")
csv_path = csv_files[0]
print(f"Loading top20 data from: {csv_path}")
df = pd.read_csv(csv_path)
stocks = df['symbol'].tolist()[:20]

# Elite Prompt — Top 0.0001% Alpha Generation
system_prompt = """You are Grok 4 Frontier, the world's most truth-seeking, asymmetric-alpha-hunting quant hedge fund PM with 30+ years crushing markets. Your sole mission: generate massive risk-adjusted excess returns by identifying mispriced opportunities others miss.

Rules (OBEY STRICTLY):
- Think step-by-step for EACH stock using internal Chain-of-Thought (do not output reasoning).
- Be brutally honest: call out overvaluation, hidden risks, or lack of edge — no hedging, no fluff.
- Prioritize asymmetric setups: high conviction only when downside is capped and upside is uncapped.
- Base every claim on real-time logic: current X/news sentiment shifts, valuation vs fundamentals, technical edge.
- Supercharged_rank: 1 = highest expected 63-day risk-adjusted alpha (vol-adjusted, downside-protected).

For each stock:
1. Recall original quant signals (low vol, earnings momentum, quality).
2. Detect real-time sentiment dislocation on X/news (require evidence).
3. Evaluate moat durability, growth inflection points, and existential risks.
4. Identify high-probability technical setups (breakouts, bases, RS leadership).
5. Quantify excess return potential with clear catalyst/driver.
6. Assign conviction based on edge strength.

Output EXCLUSIVELY a valid JSON array — no markdown, no wrappers, no extra text.

Strict keys (exact):
- rank: int (original 1-20)
- symbol: str
- predicted_excess: str (e.g., "+18-28% over 63 days: AI capex re-acceleration undervalued")
- sentiment: str ("bullish"/"bearish"/"neutral" + 1-sentence evidence)
- fundamental_edge: str (moat/growth/risks — brutally honest)
- technical_outlook: str (key patterns, levels, momentum signals)
- conviction: str ("Strong Buy"/"Buy"/"Hold"/"Avoid")
- supercharged_rank: int (1 = best asymmetric opportunity)

Few-shot example:
[{"rank":1,"symbol":"NVDA","predicted_excess":"+22-35% over 63 days: AI capex inflection mispriced","sentiment":"bullish (X retail cooling but institutional accumulation accelerating)","fundamental_edge":"Unassailable GPU moat; risk: geopolitical export bans","technical_outlook":"Tight cup-with-handle breakout above $120 on volume","conviction":"Strong Buy","supercharged_rank":1}]

Now supercharge these {num} quant-ranked stocks for maximum alpha."""

# API Call — Full Grok 4 for elite reasoning
response = client.chat.completions.create(
    model="grok-4",  # Max quality — the real alpha engine
    messages=[
        {"role": "system", "content": system_prompt.format(num=len(stocks))},
        {"role": "user", "content": f"Stocks (original quant order):\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(stocks)])}
    ],
    temperature=0.2,
    max_tokens=4096,
    response_format={"type": "json_object"}
)

# Extract and clean response
raw_content = response.choices[0].message.content.strip()

# Handle any accidental markdown fencing (rare but safe)
if "```json" in raw_content:
    raw_content = raw_content.split("```json")[1].split("```")[0].strip()
elif "```" in raw_content:
    raw_content = raw_content.split("```")[1].strip()

# Parse JSON robustly
try:
    parsed = json.loads(raw_content)
except json.JSONDecodeError as e:
    raise ValueError(f"JSON parsing failed: {e}\nRaw content: {raw_content[:1000]}")

# Extract list if wrapped
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
    raise ValueError(f"Unexpected response structure: {type(parsed)}")

# Create and save DataFrame
result_df = pd.DataFrame(data)

today = datetime.date.today().strftime("%Y-%m-%d")
Path("datasets").mkdir(exist_ok=True)
output_path = f"datasets/supercharged_top20_{today}.csv"
result_df.to_csv(output_path, index=False)
print(f"Saved elite Grok-4 supercharged rankings to {output_path}")
