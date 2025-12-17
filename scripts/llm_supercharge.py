import pandas as pd
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate  # ← Fixed
from langchain_core.output_parsers import StrOutputParser  # ← Recommended addition
import os
import datetime
import json  # ← Add this for safe JSON parsing

# Load top20
df = pd.read_csv("datasets/top20_latest.csv")  # your filename
stocks = df['symbol'].tolist()[:20]

# LLM setup (Groq example — fastest)
llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.3)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a world-class hedge fund analyst with 20+ years experience.
Task: Re-evaluate these {num} stocks ranked by a strong quantitative model (low vol, earnings momentum, quality).
For each:
- Current rank and predicted 63-day excess.
- Sentiment from recent X/news (bullish/bearish/neutral).
- Fundamental edge (moat, growth drivers, risks).
- Technical outlook.
- Overall conviction: Strong Buy / Buy / Hold / Avoid.
- Final supercharged rank (1 best).
Output strict JSON array of objects with keys: rank, symbol, predicted_excess, sentiment, fundamental_edge, technical_outlook, conviction, supercharged_rank."""),
    ("human", "Stocks: {stocks_list}")
])

# Add parser for cleaner output
chain = prompt | llm | StrOutputParser()

response = chain.invoke({
    "num": len(stocks),
    "stocks_list": ", ".join([f"{i+1}. {s}" for i, s in enumerate(stocks)])
})

# Parse the JSON response
try:
    data = json.loads(response.strip())
except json.JSONDecodeError as e:
    print("JSON parsing failed:", e)
    print("Raw response:", response)
    raise

# Convert to DataFrame (adjust columns if needed)
result_df = pd.DataFrame(data)

# Add today's date
today = datetime.date.today().strftime("%Y-%m-%d")
Path("datasets").mkdir(exist_ok=True)

# Save the supercharged file
output_path = f"datasets/supercharged_top20_{today}.csv"
result_df.to_csv(output_path, index=False)
print(f"Saved supercharged rankings to {output_path}")
