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
    run(f'python3 scripts/fetch_history_bulletproof.py --tickers "SPY,^VIX,TLT,BTC-USD,XLK,XLF,XLV,XLE,XLI,XLP,XLY,XLB,XLRE,XLU,XLC" --max-workers 1 --period 2y --out-dir {ETF_CACHE_DIR}')
    
    # ===== DEBUG SPY PARQUET AFTER FETCH =====
    print(f"\n{'='*60}")
    print("DEBUG:  SPY PARQUET INSPECTION AFTER FETCH")
    print(f"{'='*60}")
    
    spy_files = list(ETF_CACHE_DIR.glob("SPY_*.parquet"))
    print(f"SPY files found in {ETF_CACHE_DIR}:  {[f.name for f in spy_files]}")
    
    if spy_files:
        spy_file = spy_files[0]
        print(f"\nInspecting:  {spy_file.name}")
        
        # Load SPY directly
        spy_df = pd.read_parquet(spy_file)
        
        print(f"  Shape: {spy_df.shape}")
        print(f"  Index type: {type(spy_df.index)}")
        print(f"  Index name: {spy_df.index. name}")
        print(f"  Column type: {type(spy_df.columns)}")
        print(f"  Columns: {list(spy_df.columns)}")
        print(f"  Has 'Adj Close': {'Adj Close' in spy_df.columns}")
        print(f"  Has 'Close': {'Close' in spy_df.columns}")
        
        print(f"\n  First 3 rows:")
        print(spy_df.head(3))
        
        print(f"\n  Last 3 rows:")
        print(spy_df.tail(3))
        
        # Check if columns are MultiIndex
        if isinstance(spy_df.columns, pd.MultiIndex):
            print(f"\n  ⚠️  WARNING:  Columns are MultiIndex!")
            print(f"  MultiIndex levels: {spy_df.columns. nlevels}")
            print(f"  Level 0 values:  {spy_df.columns.get_level_values(0).tolist()}")
            if spy_df.columns.nlevels > 1:
                print(f"  Level 1 values: {spy_df.columns. get_level_values(1).tolist()}")
        
        # Check data types
        print(f"\n  Data types:")
        print(spy_df.dtypes)
        
    else:
        print(f"  ⚠️  No SPY parquet files found!")
    
    print(f"{'='*60}\n")
    # ===== END SPY DEBUG =====
    
    # 2. Augment + enhance ONLY new/changed tickers (fast)
    run(f"python3 scripts/augment_caches_fast.py --processes 2 --cache-dir {TICKER_CACHE_DIR} --overwrite")
      # ===== DEBUG SPY PARQUET AFTER ENHANCE =====
    print(f"\n{'='*60}")
    print("DEBUG: SPY PARQUET INSPECTION AFTER ENHANCE")
    print(f"{'='*60}")
    
    if spy_files:
        spy_file = spy_files[0]
        print(f"\nRe-inspecting: {spy_file.name}")
        
        # Reload SPY
        spy_df_after = pd.read_parquet(spy_file)
        
        print(f"  Shape: {spy_df_after.shape}")
        print(f"  Columns: {list(spy_df_after.columns)}")
        print(f"  Has 'Adj Close': {'Adj Close' in spy_df_after.columns}")
        print(f"  Has 'Close':  {'Close' in spy_df_after.columns}")
        
        # Check if columns changed
        if spy_files: 
            original_cols = set(spy_df.columns)
            after_cols = set(spy_df_after.columns)
            
            if original_cols != after_cols: 
                print(f"\n  ⚠️  WARNING:  Columns CHANGED between fetch and enhance!")
                added = after_cols - original_cols
                removed = original_cols - after_cols
                if added:
                    print(f"  Added: {added}")
                if removed:
                    print(f"  Removed: {removed}")
            else:
                print(f"\n  ✓ Columns unchanged")
    
    print(f"{'='*60}\n")
    # ===== END SPY DEBUG AFTER =====
    run(f"python3 scripts/enhance_features_final.py --processes 2 --cache-dir {TICKER_CACHE_DIR} --overwrite")
      # ===== DEBUG SPY PARQUET AFTER ENHANCE =====
    print(f"\n{'='*60}")
    print("DEBUG: SPY PARQUET INSPECTION AFTER ENHANCE")
    print(f"{'='*60}")
    
    if spy_files:
        spy_file = spy_files[0]
        print(f"\nRe-inspecting: {spy_file.name}")
        
        # Reload SPY
        spy_df_after = pd.read_parquet(spy_file)
        
        print(f"  Shape: {spy_df_after.shape}")
        print(f"  Columns: {list(spy_df_after.columns)}")
        print(f"  Has 'Adj Close': {'Adj Close' in spy_df_after.columns}")
        print(f"  Has 'Close':  {'Close' in spy_df_after.columns}")
        
        # Check if columns changed
        if spy_files: 
            original_cols = set(spy_df.columns)
            after_cols = set(spy_df_after.columns)
            
            if original_cols != after_cols: 
                print(f"\n  ⚠️  WARNING:  Columns CHANGED between fetch and enhance!")
                added = after_cols - original_cols
                removed = original_cols - after_cols
                if added:
                    print(f"  Added: {added}")
                if removed:
                    print(f"  Removed: {removed}")
            else:
                print(f"\n  ✓ Columns unchanged")
    
    print(f"{'='*60}\n")
    # ===== END SPY DEBUG AFTER =====
    # 3. Earnings calendar (weekly is fine — run only on Mondays)
    # 3. Earnings calendar (Mondays OR if missing)
    #earnings_file = Path("data/earnings.csv")
    #if datetime.now().weekday() == 0 or not earnings_file.exists():
    #    run(f"python3 scripts/build_earnings_calendar.py --start 2020-01-01 --out data/earnings.csv --verbose")
    # 4. Build today's feature dataset (latest date only)
    run(f"python3 scripts/build_labels_final.py --cache-dir {TICKER_CACHE_DIR} --output {DATASETS_DIR}/today_features.parquet --earnings-file data/earnings.csv --production-only")

    # Load and validate dataset
    today_features_path = DATASETS_DIR / "today_features.parquet"
    full_df = pd.read_parquet(today_features_path)

    print(f"\n{'='*60}")
    print("DATASET VALIDATION")
    print(f"{'='*60}")
    print(f"Total rows loaded: {len(full_df)}")
    print(f"Date range: {full_df['date'].min()} to {full_df['date'].max()}")
    print(f"Unique dates: {full_df['date'].nunique()}")
    print(f"Symbols: {full_df['symbol'].nunique()}")

    # Filter to LATEST DATE ONLY for inference
    latest_date = full_df['date'].max()
    today_df = full_df[full_df['date'] == latest_date].copy()

    print(f"\n📅 LATEST DATE: {latest_date}")
    print(f"Rows for this date: {len(today_df)}")
    print(f"Symbols for this date: {today_df['symbol'].nunique()}")

    # Validate this is actually "today" or very recent
    actual_today = datetime.now().date()
    days_behind = (pd.to_datetime(actual_today) - pd.to_datetime(latest_date)).days
    print(f"Days behind current date: {days_behind}")

    if days_behind > 5:
        print(f"⚠️  WARNING: Latest data is {days_behind} days old! Expected fresh data.")
        print(f"   This means predictions are for {days_behind} days ago, not today.")
        print(f"   The model will predict {latest_date} + 63 days = {pd.to_datetime(latest_date) + pd.Timedelta(days=63)}")
    elif days_behind < -1:
        print(f"⚠️  WARNING: Latest date is {abs(days_behind)} days in the FUTURE! Data issue detected.")
    else:
        print(f"✓ Data is fresh (within {days_behind} days of current date)")

    # Extract VIX from LATEST date (all rows have same value for cross-sectional features)
    if 'feat_vix_level_z_63' in today_df.columns:
        current_vix_z = today_df['feat_vix_level_z_63'].iloc[0]
        print(f"\nCurrent feat_vix_level_z_63 (from {latest_date}): {current_vix_z:.4f}")
        
        # Show VIX distribution to verify consistency
        vix_unique = today_df['feat_vix_level_z_63'].nunique()
        if vix_unique > 1:
            print(f"⚠️  WARNING: VIX feature has {vix_unique} different values on same date! Should be 1.")
            print(f"   Values: {today_df['feat_vix_level_z_63'].unique()[:10]}")
    else:
        print("⚠️  ERROR: feat_vix_level_z_63 not found in features!")
        current_vix_z = 0.0

    print(f"{'='*60}\n")
    
    # Save TODAY-ONLY dataset for model prediction
    today_only_path = DATASETS_DIR / "today_only.parquet"
    today_df.to_parquet(today_only_path)
    print(f"✓ Saved {len(today_df)} rows (latest date only) to {today_only_path}\n")
    
    if current_vix_z > 1.5:
        print("⚠️  HIGH VIX REGIME DETECTED (>1.5) — SKIPPING TRADES TODAY (model trained low-vol only)")
        # Optional: save empty top20 or "cash" signal
        pd.DataFrame({"note": ["High VIX - No trades"]}).to_csv(DATASETS_DIR / f"top20_{today}.csv")
    else:
        print("✅ Low VIX regime — proceeding with predictions")
        # Proceed to apply_ranker with today-only data
        run(f"python3 scripts/apply_ranker.py --dataset {today_only_path} --model-dir {MODEL_DIR} --out-dir {DATASETS_DIR}")

    # 5. Load predictions (should only be for latest date now)
    pred_df = pd.read_csv(PREDICTIONS_PATH)
    # ===== ADD THIS SECTION =====
    print(f"\n{'='*60}")
    print("FULL PREDICTION DISTRIBUTION")
    print(f"{'='*60}")
    print(f"Total predictions: {len(pred_df)}")
    print(f"\nPrediction statistics:")
    print(f"  Min:   {pred_df['pred']. min():.6f}")
    print(f"  25%:   {pred_df['pred']. quantile(0.25):.6f}")
    print(f"  50%:  {pred_df['pred'].median():.6f}")
    print(f"  75%:  {pred_df['pred'].quantile(0.75):.6f}")
    print(f"  Max:  {pred_df['pred'].max():.6f}")
    print(f"  Mean: {pred_df['pred']. mean():.6f}")
    print(f"  Std:  {pred_df['pred'].std():.6f}")
    
    positive_count = (pred_df['pred'] > 0).sum()
    negative_count = (pred_df['pred'] < 0).sum()
    zero_count = (pred_df['pred'] == 0).sum()
    
    print(f"\nPrediction signs:")
    print(f"  Positive: {positive_count}/{len(pred_df)} ({positive_count/len(pred_df)*100:.1f}%)")
    print(f"  Negative: {negative_count}/{len(pred_df)} ({negative_count/len(pred_df)*100:.1f}%)")
    print(f"  Zero:  {zero_count}/{len(pred_df)}")
    
    print(f"\nTop 10 predictions:")
    print(pred_df.nlargest(10, 'pred')[['symbol', 'pred']].to_string(index=False))
    
    print(f"\nBottom 10 predictions:")
    print(pred_df.nsmallest(10, 'pred')[['symbol', 'pred']].to_string(index=False))
    print(f"{'='*60}\n")
    # ===== END ADDITION =====
    
    print(f"\n{'='*60}")
    print("PREDICTION VALIDATION")
    print(f"{'='*60}")
    print(f"Predictions loaded: {len(pred_df)}")

    if 'date' in pred_df.columns:
        print(f"Date range: {pred_df['date'].min()} to {pred_df['date'].max()}")
        print(f"Unique dates: {pred_df['date'].nunique()}")
        
        # Should only be ONE date
        if pred_df['date'].nunique() > 1:
            print(f"⚠️  WARNING: Multiple dates in predictions! Using latest...")
            latest_pred_date = pred_df['date'].max()
            pred_df = pred_df[pred_df['date'] == latest_pred_date].copy()
            print(f"Filtered to {latest_pred_date}: {len(pred_df)} rows")
        
        final_date = pred_df['date'].iloc[0]
        print(f"Final prediction date: {final_date}")
    else:
        print("⚠️  WARNING: No 'date' column in predictions")
        final_date = "unknown"

    print(f"{'='*60}\n")

    # Now filter by prediction quality
    today_pred = pred_df.copy()  # Already filtered to latest date
    
    # 6. Post-process: Top 20 + risk overlay
    # Sort by prediction
    today_pred = today_pred.sort_values('pred', ascending=False)

    # Quick-kill filter (apply here too)
    valid = today_pred.copy()

    if 'adv20_dollar' in valid.columns:
        valid = valid[valid['adv20_dollar'] >= 10_000_000]
    
    if 'price' in valid.columns:
        valid = valid[(valid['price'] >= 15) & (valid['price'] <= 3000)]

    top20 = valid.head(20)[['symbol', 'pred']].reset_index(drop=True)
    top20.index += 1
    
    print(f"\n{'='*60}")
    print("FINAL PIPELINE VALIDATION")
    print(f"{'='*60}")

    # Validate VIX check happened on correct date
    print(f"VIX checked on date: {latest_date}")
    print(f"VIX z-score: {current_vix_z:.4f}")
    print(f"VIX regime decision: {'HIGH (skip trades)' if current_vix_z > 1.5 else 'LOW (trade active)'}")

    # Validate predictions
    print(f"\nPredictions generated: {len(top20)}")
    print(f"Prediction date: {final_date}")

    # Check if dates match
    if str(latest_date) != str(final_date):
        print(f"⚠️  WARNING: Feature date ({latest_date}) != Prediction date ({final_date})")

    # Check prediction quality
    if len(top20) > 0:
        best_pred = top20['pred'].iloc[0]
        worst_pred = top20['pred'].iloc[-1]
        print(f"\nTop 20 prediction range: {worst_pred:.6f} to {best_pred:.6f}")
        
        if best_pred < 0:
            print(f"⚠️  WARNING: Best prediction is negative ({best_pred:.6f})")
            print(f"   Model expects ALL stocks to have negative excess returns")
            print(f"   This may indicate:")
            print(f"     - Market regime shift (bearish outlook)")
            print(f"     - Feature drift from training distribution")
            print(f"     - Data quality issue")
        
        positive_count = (top20['pred'] > 0).sum()
        print(f"Positive predictions in Top 20: {positive_count}/20")

    print(f"{'='*60}\n")
    
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
