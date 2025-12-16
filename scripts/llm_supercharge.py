import pandas as pd
from pathlib import Path
from langchain_groq import ChatGroq   # or langchain_openai
from langchain.prompts import ChatPromptTemplate
import os
import datetime

# Load top20
df = pd.read_csv("datasets/top20_latest.csv")  # your filename
stocks = df['symbol'].tolist()[:20]

# LLM setup (Groq example — fastest)
llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.3)  # or mixtral-8x7b

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

Output strict JSON array."""),
    ("human", "Stocks: {stocks_list}")
])

chain = prompt | llm

response = chain.invoke({
    "num": len(stocks),
    "stocks_list": ", ".join([f"{i+1}. {s}" for i, s in enumerate(stocks)])
})

# Parse JSON response, save new CSV/report
# Add columns: sentiment, conviction, notes, new_rank
# Email artifact

today = datetime.date.today().strftime("%Y-%m-%d")
Path("datasets").mkdir(exist_ok=True)
# Save parsed df as supercharged_top20_{today}.csv
