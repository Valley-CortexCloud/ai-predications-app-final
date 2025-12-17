import pandas as pd
from pathlib import Path
from openai import OpenAI  # Works for xAI API
import os
import datetime
import json
import glob

# xAI API setup (set XAI_API_KEY in secrets/env)
client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1"  # xAI endpoint
)

# Load top20 dynamically
csv_files = glob.glob("datasets/top20_*.csv")
if not csv_files:
    raise FileNotFoundError("No top20_*.csv file found in datasets/")
csv_path = csv_files[0]
print(f"Loading top20 data from: {csv_path}")
df = pd.read_csv(csv_path)
stocks = df['symbol'].tolist()[:20]

# Prompt (optimized for Grok 4's strengths: deeper reasoning, less fluff)
system_prompt = """You are Grok 4, a maximally truth-seeking hedge fund analyst with unparalleled reasoning on markets.
Re-evaluate these {num} quantitatively ranked stocks (low vol, earnings momentum, quality factors).
For each:
- Current rank + predicted 63-day excess return rationale.
- Real-time sentiment from X/news (bullish/bearish/neutral, with evidence).
- Fundamental edge: moat strength, growth drivers, key risks (be brutally honest).
- Technical outlook: patterns, support/resistance, momentum signals.
- Conviction level: Strong Buy / Buy / Hold / Avoid (no hedging).
- Supercharged rank: 1 = best opportunity.
Prioritize asymmetric upside/low downside. Output ONLY a strict JSON array of objects with keys: rank, symbol, predicted_excess, sentiment, fundamental_edge, technical_outlook, conviction, supercharged_rank."""

human_prompt = "Stocks (ranked order): {stocks_list}"

# Call Grok 4 Fast (best balance for your task)
response = client.chat.completions.create(
    model="grok-4-fast",  # Or "grok-4" for max quality
    messages=[
        {"role": "system", "content": system_prompt.format(num=len(stocks))},
        {"role": "user", "content": human_prompt.format(stocks_list=", ".join([f"{i+1}. {s}" for i, s in enumerate(stocks)]))}
    ],
    temperature=0.2,  # Low for consistent rankings
    response_format={"type": "json_object"}  # Enforces JSON (Grok supports this natively)
)

# Parse (Grok is reliable on JSON)
data = json.loads(response.choices[0].message.content)

# Save as before
result_df = pd.DataFrame(data)
today = datetime.date.today().strftime("%Y-%m-%d")
Path("datasets").mkdir(exist_ok=True)
output_path = f"datasets/supercharged_top20_{today}.csv"
result_df.to_csv(output_path, index=False)
print(f"Saved supercharged rankings to {output_path}")
