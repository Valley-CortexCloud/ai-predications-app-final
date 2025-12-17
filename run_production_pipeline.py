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
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(1)

def inspect_spy_parquet(label, spy_files, previous_df=None):
    """Helper function to inspect SPY parquet and compare to previous state"""
    print(f"\n{'='*60}")
    print(f"DEBUG: SPY PARQUET INSPECTION {label}")
    print(f"{'='*60}")
    
    if not spy_files:
        print(f"  ⚠️  No SPY parquet files found!")
        return None
    
    spy_file = spy_files[0]
    print(f"\nInspecting: {spy_file.name}")
    
    # Load SPY
    spy_df = pd.read_parquet(spy_file)
    
    print(f"  Shape: {spy_df.shape}")
    print(f"  Index type: {type(spy_df.index).__name__}")
    print(f"  Index name: {spy_df.index.name}")
    print(f"  Column type: {type(spy_df.columns).__name__}")
    print(f"  Columns: {list(spy_df.columns)}")
    print(f"  Has 'Adj Close': {'Adj Close' in spy_df.columns}")
    print(f"  Has 'Close': {'Close' in spy_df.columns}")
    
    # Check if columns are MultiIndex
    if isinstance(spy_df.columns, pd.MultiIndex):
        print(f"\n  ⚠️  WARNING: Columns are MultiIndex!")
        print(f"  MultiIndex levels: {spy_df.columns.nlevels}")
        print(f"  Level 0 values: {spy_df.columns.get_level_values(0).tolist()}")
        if spy_df.columns.nlevels > 1:
            print(f"  Level 1 values: {spy_df.columns.get_level_values(1).tolist()}")
    
    print(f"\n  First 3 rows:")
    print(spy_df.head(3))
    
    print(f"\n  Data types:")
    print(spy_df.dtypes)
    
    # Compare to previous if provided
    if previous_df is not None:
        print(f"\n  📊 COMPARISON TO PREVIOUS STATE:")
        prev_cols = set(previous_df.columns)
        curr_cols = set(spy_df.columns)
        
        if prev_cols == curr_cols:
            print(f"  ✓ Columns unchanged ({len(curr_cols)} columns)")
        else:
            print(f"  ⚠️  COLUMNS CHANGED!")
            added = curr_cols - prev_cols
            removed = prev_cols - curr_cols
            if added:
                print(f"    Added: {added}")
            if removed:
                print(f"    Removed: {removed}")
        
        # Check shape
        if previous_df.shape == spy_df.shape:
            print(f"  ✓ Shape unchanged:  {spy_df.shape}")
        else:
            print(f"  ⚠️  Shape changed: {previous_df.shape} → {spy_df.shape}")
    
    print(f"{'='*60}\n")
    return spy_df

def main():
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"=== Daily Top 20 Pipeline - {today} ===")

    # 1.Fetch only latest data for universe + benchmarks (incremental)
    run(f"python3 scripts/fetch_history_bulletproof.py --universe sp500 --period 2y --adjusted --out-dir {TICKER_CACHE_DIR} --max-workers 1")
    run(f"python3 scripts/fetch_history_bulletproof.py --universe nasdaq --period 2y --adjusted --out-dir {TICKER_CACHE_DIR} --max-workers 1")
    run(f'python3 scripts/fetch_history_bulletproof.py --tickers "SPY,^VIX,TLT,BTC-USD,XLK,XLF,XLV,XLE,XLI,XLP,XLY,XLB,XLRE,XLU,XLC" --max-workers 1 --period 2y --out-dir {ETF_CACHE_DIR}')
    
    # Get SPY file list once
    spy_files = list(ETF_CACHE_DIR.glob("SPY_*.parquet"))
    
    # DEBUG 1: After fetch
    spy_after_fetch = inspect_spy_parquet("AFTER FETCH", spy_files)

    print(f"\n{'='*60}")
    print("DEBUG: ETF FILES BEFORE AUGMENT")
    print(f"{'='*60}")
    etf_files = list(ETF_CACHE_DIR.glob("*.parquet"))
    print(f"Files in {ETF_CACHE_DIR}:")
    for f in sorted(etf_files):
        print(f"  {f. name}")
    print(f"{'='*60}\n")
    # 2.Augment + enhance ONLY new/changed tickers (fast)
    run(f"python3 scripts/augment_caches_fast.py --processes 2 --cache-dir {TICKER_CACHE_DIR} --overwrite")
    print(f"\n{'='*60}")
    print("DEBUG: ETF FILES AFTER AUGMENT")
    print(f"{'='*60}")
    etf_files_after = list(ETF_CACHE_DIR.glob("*.parquet"))
    print(f"Files in {ETF_CACHE_DIR}:")
    for f in sorted(etf_files_after):
        print(f"  {f.name}")
        
    new_files = set(etf_files_after) - set(etf_files)
    if new_files:
        print(f"\n⚠️  NEW FILES CREATED:")
        for f in new_files:
            print(f"  {f.name}")
    print(f"{'='*60}\n")
    # DEBUG 2: After augment
    spy_after_augment = inspect_spy_parquet("AFTER AUGMENT", spy_files, previous_df=spy_after_fetch)
    
    run(f"python3 scripts/enhance_features_final.py --processes 2 --cache-dir {TICKER_CACHE_DIR} --overwrite")
    
    # DEBUG 3: After enhance
    spy_after_enhance = inspect_spy_parquet("AFTER ENHANCE", spy_files, previous_df=spy_after_augment)
    
    # 3.Build today's feature dataset (latest date only)
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

    print(f"\n📅 LATEST DATE:  {latest_date}")
    print(f"Rows for this date: {len(today_df)}")
    print(f"Symbols for this date: {today_df['symbol'].nunique()}")

    # Validate this is actually "today" or very recent
    actual_today = datetime.now().date()
    days_behind = (pd.to_datetime(actual_today) - pd.to_datetime(latest_date)).days
    print(f"Days behind current date: {days_behind}")

    if days_behind > 5:
        print(f"⚠️  WARNING: Latest data is {days_behind} days old! Expected fresh data.")
    elif days_behind < -1:
        print(f"⚠️  WARNING: Latest date is {abs(days_behind)} days in the FUTURE!")
    else:
        print(f"✓ Data is fresh (within {days_behind} days of current date)")

    # Extract VIX
    if 'feat_vix_level_z_63' in today_df.columns:
        current_vix_z = today_df['feat_vix_level_z_63'].iloc[0]
        print(f"\nCurrent feat_vix_level_z_63: {current_vix_z:.4f}")
        
        vix_unique = today_df['feat_vix_level_z_63'].nunique()
        if vix_unique > 1:
            print(f"⚠️  WARNING: VIX has {vix_unique} values on same date!")
    else:
        current_vix_z = 0.0

    print(f"{'='*60}\n")
    
    # Save TODAY-ONLY dataset
    today_only_path = DATASETS_DIR / "today_only.parquet"
    today_df.to_parquet(today_only_path)
    print(f"✓ Saved {len(today_df)} rows to {today_only_path}\n")
    
    if current_vix_z > 1.5:
        print("⚠️  HIGH VIX - SKIPPING TRADES")
        pd.DataFrame({"note": ["High VIX"]}).to_csv(DATASETS_DIR / f"top20_{today}.csv")
        return
    
    print("✅ Low VIX — proceeding with predictions")
    run(f"python3 scripts/apply_ranker.py --dataset {today_only_path} --model-dir {MODEL_DIR} --out-dir {DATASETS_DIR}")

    # Load predictions
    pred_df = pd.read_csv(PREDICTIONS_PATH)
    
    # Prediction distribution
    print(f"\n{'='*60}")
    print("FULL PREDICTION DISTRIBUTION")
    print(f"{'='*60}")
    print(f"Total:  {len(pred_df)}")
    print(f"\nStats:")
    print(f"  Min:   {pred_df['pred'].min():.6f}")
    print(f"  25%:   {pred_df['pred'].quantile(0.25):.6f}")
    print(f"  50%:  {pred_df['pred'].median():.6f}")
    print(f"  75%:  {pred_df['pred'].quantile(0.75):.6f}")
    print(f"  Max:  {pred_df['pred'].max():.6f}")
    print(f"  Mean: {pred_df['pred'].mean():.6f}")
    
    pos = (pred_df['pred'] > 0).sum()
    neg = (pred_df['pred'] < 0).sum()
    print(f"\nSigns:  Positive={pos}, Negative={neg}")
    
    print(f"\nTop 10:")
    print(pred_df.nlargest(10, 'pred')[['symbol', 'pred']].to_string(index=False))
    print(f"{'='*60}\n")
    
    # Validation
    if 'date' in pred_df.columns:
        final_date = pred_df['date'].iloc[0]
        print(f"Prediction date: {final_date}")
        if str(latest_date) != str(final_date):
            print(f"⚠️  Date mismatch: {latest_date} vs {final_date}")
    
    # Filter and create top 20
    today_pred = pred_df.sort_values('pred', ascending=False)
    valid = today_pred.copy()
    
    if 'adv20_dollar' in valid.columns:
        valid = valid[valid['adv20_dollar'] >= 10_000_000]
    if 'price' in valid.columns:
        valid = valid[(valid['price'] >= 15) & (valid['price'] <= 3000)]
    
    top20 = valid.head(20)[['symbol', 'pred']].reset_index(drop=True)
    top20.index += 1
    
    # Final validation
    print(f"\n{'='*60}")
    print("FINAL VALIDATION")
    print(f"{'='*60}")
    print(f"VIX z-score: {current_vix_z:.4f}")
    print(f"Top 20 range: {top20['pred'].iloc[-1]:.6f} to {top20['pred'].iloc[0]:.6f}")
    
    if top20['pred'].iloc[0] < 0:
        print(f"\nℹ️  All predictions negative (normal for ranking models)")
        print(f"   Higher (less negative) = better performance")
    
    print(f"{'='*60}\n")
    
    print("\n" + "="*50)
    print("TODAY'S TOP 20 LONG PORTFOLIO")
    print("="*50)
    print(top20.to_string())
    print("="*50)
    
    top20.to_csv(DATASETS_DIR / f"top20_{today}.csv")
    print(f"\nSaved to {DATASETS_DIR / f'top20_{today}.csv'}")

if __name__ == "__main__":
    main()
