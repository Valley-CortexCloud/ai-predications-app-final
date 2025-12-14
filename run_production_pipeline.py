#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT = Path(__file__).parent
TICKER_CACHE_DIR = ROOT / "data_cache" / "10y_ticker_features"
ETF_CACHE_DIR = ROOT / "data_cache" / "_etf_cache"
MODEL_DIR = ROOT / "model"
DATASETS_DIR = ROOT / "datasets"
PREDICTIONS_PATH = DATASETS_DIR / "predictions_today.csv"

def run(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR:", result.stderr)  # Only real errors
        sys.exit(1)
    if result.stderr:  # Warnings ok
        print("WARNINGS:", result.stderr)
    print(result.stdout)

def main():
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"=== Daily Top 20 Pipeline - {today} ===")

    # 1. Fetch only latest data for universe + benchmarks (incremental)
    run(f"python3 scripts/fetch_history_bulletproof.py --universe sp500 --period 1d --out-dir {TICKER_CACHE_DIR} --max-workers 8")
    run(f"python3 scripts/fetch_history_bulletproof.py --universe nasdaq --period 1d --out-dir {TICKER_CACHE_DIR} --max-workers 8")
    run(f'python3 scripts/fetch_history_bulletproof.py --tickers "SPY,^VIX,TLT,^BTC-USD,XLK,XLF,XLV,XLE,XLI,XLP,XLY,XLB,XLRE,XLU,XLC" --period 1d --out-dir {ETF_CACHE_DIR}')
    
    # 2. Augment + enhance ONLY new/changed tickers (fast)
    run(f"python3 scripts/augment_caches_fast.py --processes 6 --cache-dir {TICKER_CACHE_DIR} --overwrite")
    run(f"python3 scripts/enhance_features_final.py --processes 4 --cache-dir {TICKER_CACHE_DIR} --overwrite")

    # 3. Earnings calendar (weekly is fine — run only on Mondays)
    if datetime.now().weekday() == 0:  # Monday
        run(f"python3 scripts/build_earnings_calendar.py --start 2014-11-01 --out data/earnings.csv --verbose")

    # 4. Build today's feature dataset (latest date only)
    run(f"python3 scripts/build_labels_final.py --cache-dir {CACHE_DIR} --output {DATASETS_DIR}/today_features.parquet --earnings-file data/earnings.csv")

    # Load today's dataset to check VIX
    today_features_path = DATASETS_DIR / "today_features.parquet"
    today_df = pd.read_parquet(today_features_path)
    
    # feat_vix_level_z_63 is the same for all rows on a given date
    current_vix_z = today_df['feat_vix_level_z_63'].iloc[0]  # safe, all identical
    
    print(f"\nCurrent feat_vix_level_z_63: {current_vix_z:.2f}")
    
    if current_vix_z > 1.5:
        print("⚠️  HIGH VIX REGIME DETECTED (>1.5) — SKIPPING TRADES TODAY (model trained low-vol only)")
        # Optional: save empty top20 or "cash" signal
        pd.DataFrame({"note": ["High VIX - No trades"]}).to_csv(DATASETS_DIR / f"top20_{today}.csv")
    else:
        print("✅ Low VIX regime — proceeding with predictions")
        # Proceed to apply_ranker and top20
        run(f"python3 scripts/apply_ranker.py --dataset {today_features_path} --model-dir {MODEL_DIR} --out-dir {DATASETS_DIR}")
        
    # Then your existing top20 post-process...
    # 5. Apply frozen model → predictions
    run(f"python3 scripts/apply_ranker.py --dataset {DATASETS_DIR}/today_features.parquet --model-dir {MODEL_DIR} --out-dir {DATASETS_DIR}")

    # 6. Post-process: Top 20 + risk overlay
    pred_df = pd.read_csv(PREDICTIONS_PATH)
    latest_date = pred_df['date'].max()
    today_df = pred_df[pred_df['date'] == latest_date].sort_values('pred', ascending=False)

    # Quick-kill filter (apply here too)
    valid = today_df[
        (today_df.get('adv20_dollar', 10e9) >= 10_000_000) &
        (today_df.get('price', 100) >= 15) &
        (today_df.get('price', 100) <= 3000)
    ]

    top20 = valid.head(20)[['symbol', 'pred']].reset_index(drop=True)
    top20.index += 1
    print("\n" + "="*50)
    print("TODAY'S TOP 20 LONG PORTFOLIO (63-day horizon)")
    print("="*50)
    print(top20.to_string())
    print("="*50)

    # Save final list
    top20.to_csv(DATASETS_DIR / f"top20_{today}.csv")
    print(f"Saved to {DATASETS_DIR / f'top20_{today}.csv'}")

if __name__ == "__main__":
    main()
