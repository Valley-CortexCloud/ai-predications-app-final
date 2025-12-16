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
PREDICTIONS_PATH = DATASETS_DIR / "predictions.csv"

def run(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("STDOUT:", result.stdout)   # ADD THIS
    print("STDERR:", result.stderr)   # ADD THIS
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(1)

def main():
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"=== Daily Top 20 Pipeline - {today} ===")

    # 1. Fetch only latest data for universe + benchmarks (incremental)
    run(f"python3 scripts/fetch_history_bulletproof.py --universe sp500 --period 2y --out-dir {TICKER_CACHE_DIR} --max-workers 1")
    run(f"python3 scripts/fetch_history_bulletproof.py --universe nasdaq --period 2y --out-dir {TICKER_CACHE_DIR} --max-workers 1")
    run(f'python3 scripts/fetch_history_bulletproof.py --tickers "SPY,^VIX,TLT,BTC-USD,XLK,XLF,XLV,XLE,XLI,XLP,XLY,XLB,XLRE,XLU,XLC" --period 2y --adjusted --out-dir {ETF_CACHE_DIR}')
    
    # 2. Augment + enhance ONLY new/changed tickers (fast)
    run(f"python3 scripts/augment_caches_fast.py --processes 2 --cache-dir {TICKER_CACHE_DIR} --overwrite")
    run(f"python3 scripts/enhance_features_final.py --processes 2 --cache-dir {TICKER_CACHE_DIR} --overwrite")

    # 3. Earnings calendar (weekly is fine — run only on Mondays)
    # 3. Earnings calendar (Mondays OR if missing)
    #earnings_file = Path("data/earnings.csv")
    #if datetime.now().weekday() == 0 or not earnings_file.exists():
    #    run(f"python3 scripts/build_earnings_calendar.py --start 2020-01-01 --out data/earnings.csv --verbose")
    # 4. Build today's feature dataset (latest date only)
    run(f"python3 scripts/build_labels_final.py --cache-dir {TICKER_CACHE_DIR} --output {DATASETS_DIR}/today_features.parquet --earnings-file data/earnings.csv")

    # Load today's dataset to check VIX
    today_features_path = DATASETS_DIR / "today_features.parquet"
    today_df = pd.read_parquet(today_features_path)
    
    # ===== VIX DIAGNOSTICS =====
    print("\n" + "=" * 60)
    print("VIX DIAGNOSTICS")
    print("=" * 60)

    if 'feat_vix_level_z_63' in today_df.columns:
        vix_z = today_df['feat_vix_level_z_63']
        print(f"feat_vix_level_z_63 unique values: {vix_z.unique()[:10]}")
        print(f"feat_vix_level_z_63 stats:")
        print(vix_z.describe())
        print(f"  - All zeros: {(vix_z == 0).all()}")
        print(f"  - All NaN: {vix_z.isna().all()}")
    else:
        print("⚠️  feat_vix_level_z_63 MISSING from features!")

    # Check other VIX-related features
    vix_features = [c for c in today_df.columns if 'vix' in c.lower()]
    print(f"\nAll VIX features found: {vix_features}")
    for feat in vix_features:
        print(f"  {feat}: min={today_df[feat].min():.4f}, max={today_df[feat].max():.4f}, mean={today_df[feat].mean():.4f}")

    # Check the raw VIX file
    vix_file = ETF_CACHE_DIR / "^VIX_2y_adj.parquet"
    if vix_file.exists():
        vix_df = pd.read_parquet(vix_file)
        print(f"\n✓ VIX file exists: {vix_file}")
        print(f"  Shape: {vix_df.shape}")
        print(f"  Columns: {list(vix_df.columns)}")
        print(f"  Date range: {vix_df.index.min()} to {vix_df.index.max()}")
        print(f"  Last 5 Close values:\n{vix_df['Close'].tail()}")
        print(f"  Close stats: min={vix_df['Close'].min():.2f}, max={vix_df['Close'].max():.2f}, mean={vix_df['Close'].mean():.2f}")
    else:
        print(f"\n⚠️  VIX file NOT FOUND at {vix_file}")

    print("=" * 60)
    
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
    
    # ===== PREDICTION DIAGNOSTICS =====
    print("\n" + "=" * 60)
    print("PREDICTION DIAGNOSTICS")
    print("=" * 60)

    print(f"Predictions DataFrame shape: {pred_df.shape}")
    print(f"Columns: {list(pred_df.columns)}")

    if 'pred' in pred_df.columns:
        print(f"\nPrediction statistics:")
        print(pred_df['pred'].describe())
        print(f"\nPredictions > 0: {(pred_df['pred'] > 0).sum()} ({(pred_df['pred'] > 0).sum() / len(pred_df) * 100:.1f}%)")
        print(f"Predictions < 0: {(pred_df['pred'] < 0).sum()} ({(pred_df['pred'] < 0).sum() / len(pred_df) * 100:.1f}%)")
        print(f"Predictions == 0: {(pred_df['pred'] == 0).sum()}")
        
        print(f"\nTop 10 positive predictions:")
        top_positive = pred_df.nlargest(10, 'pred')[['symbol', 'pred', 'date']]
        print(top_positive.to_string(index=False))
        
        print(f"\nTop 10 negative predictions (worst):")
        top_negative = pred_df.nsmallest(10, 'pred')[['symbol', 'pred', 'date']]
        print(top_negative.to_string(index=False))
    else:
        print("⚠️  'pred' column MISSING from predictions!")

    if 'date' in pred_df.columns:
        print(f"\nDate range in predictions: {pred_df['date'].min()} to {pred_df['date'].max()}")
        print(f"Unique dates: {pred_df['date'].nunique()}")
        latest_date = pred_df['date'].max()
        print(f"Latest date: {latest_date}")
        latest_count = (pred_df['date'] == latest_date).sum()
        print(f"Predictions for latest date: {latest_count}")

    print("=" * 60)
    
    latest_date = pred_df['date'].max()
    today_df = pred_df[pred_df['date'] == latest_date].sort_values('pred', ascending=False)

    # Quick-kill filter (apply here too)
    valid = today_df.copy()
    
    # Only apply filters if columns exist
    if 'adv20_dollar' in valid.columns:
        valid = valid[valid['adv20_dollar'] >= 10_000_000]
    
    if 'price' in valid.columns:
        valid = valid[(valid['price'] >= 15) & (valid['price'] <= 3000)]

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
