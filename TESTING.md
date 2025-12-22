# Testing Guide for Phase 1 Implementation

## Overview

This document provides testing procedures for the institutional-grade append-only data architecture.

## 1. Test Incremental Mode on Sample Tickers

### 1.1 Initial Fetch (Baseline)

```bash
# Create test directory
mkdir -p test_output

# Fetch initial data for 5 tickers
python scripts/fetch_history_bulletproof.py \
  --tickers "AAPL,MSFT,GOOGL,AMZN,TSLA" \
  --period 2y \
  --adjusted \
  --out-dir test_output \
  --max-workers 2

# Check output
ls -lh test_output/
```

**Expected:**
- 5 files created: `AAPL_2y_adj.parquet`, `MSFT_2y_adj.parquet`, etc.
- Each file should be ~50-100 KB
- Total: ~300-500 KB

**Validation:**
```python
import pandas as pd

# Check AAPL data
df = pd.read_parquet("test_output/AAPL_2y_adj.parquet")
print(f"Rows: {len(df)}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Duplicates: {df.index.duplicated().sum()}")
```

### 1.2 Incremental Update (No New Data)

```bash
# Run incremental mode immediately (should skip, already up-to-date)
python scripts/fetch_history_bulletproof.py \
  --tickers "AAPL,MSFT,GOOGL,AMZN,TSLA" \
  --period 2y \
  --adjusted \
  --out-dir test_output \
  --incremental
```

**Expected Output:**
```
[1/5] · AAPL (skip: up-to-date)
[2/5] · MSFT (skip: up-to-date)
[3/5] · GOOGL (skip: up-to-date)
[4/5] · AMZN (skip: up-to-date)
[5/5] · TSLA (skip: up-to-date)

OK: 0 | Skipped: 5 | Empty: 0 | Errors: 0
```

### 1.3 Simulate Incremental Update (Manual)

```python
# Simulate old data by truncating AAPL to 30 days ago
import pandas as pd
from datetime import timedelta

df = pd.read_parquet("test_output/AAPL_2y_adj.parquet")
cutoff = df.index.max() - timedelta(days=30)
df_old = df[df.index <= cutoff]
df_old.to_parquet("test_output/AAPL_2y_adj.parquet")

print(f"Truncated AAPL to {len(df_old)} rows (last date: {df_old.index.max().date()})")
```

Now run incremental:
```bash
python scripts/fetch_history_bulletproof.py \
  --tickers "AAPL" \
  --period 2y \
  --adjusted \
  --out-dir test_output \
  --incremental
```

**Expected Output:**
```
[1/1] ✓ AAPL (appended_XX_rows)
OK: 1 | Skipped: 0 | Empty: 0 | Errors: 0
Incremental: XX total rows appended across 1 tickers
```

**Validation:**
```python
# Check appended data
df_after = pd.read_parquet("test_output/AAPL_2y_adj.parquet")
print(f"Rows after: {len(df_after)}")
print(f"Last date: {df_after.index.max().date()}")
print(f"Duplicates: {df_after.index.duplicated().sum()}")  # Should be 0
```

## 2. Test Feature Augmentation (Incremental)

### 2.1 Initial Augmentation

```bash
# Augment features for test tickers
python scripts/augment_caches_fast.py \
  --tickers "AAPL,MSFT" \
  --cache-dir test_output \
  --processes 1
```

**Expected:**
- Creates `AAPL_2y_adj_features.parquet`, `MSFT_2y_adj_features.parquet`
- ~500+ rows each
- ~50-100 feature columns

**Validation:**
```python
df = pd.read_parquet("test_output/AAPL_2y_adj_features.parquet")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)[:10]}...")  # First 10 columns
print(f"Has OHLCV: {all(c in df.columns for c in ['Open', 'High', 'Low', 'Close', 'Volume'])}")
```

### 2.2 Incremental Augmentation (No New Data)

```bash
python scripts/augment_caches_fast.py \
  --tickers "AAPL,MSFT" \
  --cache-dir test_output \
  --incremental \
  --processes 1
```

**Expected Output:**
```
[1/2] ⊘ AAPL: up-to-date (last: 2024-XX-XX)
[2/2] ⊘ MSFT: up-to-date (last: 2024-XX-XX)

Success: 0 | Skipped: 2 | Errors: 0
```

### 2.3 Incremental Augmentation (With New Data)

```python
# Add new row to raw data
import pandas as pd
from datetime import timedelta

df = pd.read_parquet("test_output/AAPL_2y_adj.parquet")
last_date = df.index.max()
new_date = last_date + timedelta(days=1)

# Create synthetic new row
new_row = df.iloc[-1].copy()
new_row.name = new_date
df_new = pd.concat([df, new_row.to_frame().T])

df_new.to_parquet("test_output/AAPL_2y_adj.parquet")
print(f"Added synthetic row for {new_date.date()}")
```

Now run incremental augmentation:
```bash
python scripts/augment_caches_fast.py \
  --tickers "AAPL" \
  --cache-dir test_output \
  --incremental \
  --processes 1
```

**Expected Output:**
```
[1/1] ✓ AAPL: +1 rows (total: XXX, XX features)

Success: 1 | Skipped: 0 | Errors: 0
Incremental: 1 total rows added
```

## 3. Test Feature Enhancement (Incremental)

Similar pattern to augmentation:

```bash
# Initial enhancement
python scripts/enhance_features_final.py \
  --tickers "AAPL,MSFT" \
  --cache-dir test_output \
  --processes 1

# Incremental (no new data)
python scripts/enhance_features_final.py \
  --tickers "AAPL,MSFT" \
  --cache-dir test_output \
  --incremental \
  --processes 1
```

## 4. Test Snapshot Creation

### 4.1 Create Test Snapshot

```bash
# Use real data from data_cache (if available)
python scripts/create_snapshot.py \
  --features-dir data_cache/10y_ticker_features \
  --output-dir test_snapshots
```

**Expected Output:**
```
Creating Feature Snapshot
Found XXX enhanced feature files
✓ Loaded XXX tickers

📅 Snapshot Date: 2024-XX-XX
   Symbols: XXX
   Features: XXX

✓ Saved feature matrix: test_snapshots/2024-XX-XX/feature_matrix.parquet
✓ Saved universe: test_snapshots/2024-XX-XX/universe.csv (XXX symbols)
✓ Saved metadata: test_snapshots/2024-XX-XX/metadata.json

Snapshot created successfully!
```

### 4.2 Verify Snapshot Structure

```bash
ls -lh test_snapshots/2024-XX-XX/
```

**Expected:**
```
feature_matrix.parquet  (~10-20 MB compressed)
universe.csv            (~10 KB)
metadata.json           (~1 KB)
```

### 4.3 Inspect Metadata

```bash
cat test_snapshots/2024-XX-XX/metadata.json | python -m json.tool
```

**Expected:**
```json
{
  "snapshot_date": "2024-XX-XX",
  "created_at": "2024-XX-XXTXX:XX:XX",
  "git_commit": "abc123...",
  "symbol_count": XXX,
  "feature_count": XXX,
  "all_nan_columns": [],
  "file_checksums": {
    "feature_matrix.parquet": "...",
    "universe.csv": "..."
  }
}
```

## 5. Test Snapshot Validation

```bash
python scripts/validate_snapshot.py \
  --snapshot-dir test_snapshots/2024-XX-XX
```

**Expected Output:**
```
Validating Snapshot: 2024-XX-XX
✓ Snapshot directory exists
✓ feature_matrix.parquet exists
✓ universe.csv exists
✓ metadata.json exists

Freshness Check:
✓ Data is fresh (within 7 days)

Completeness Check:
✓ Sufficient symbols (XXX >= 400)
✓ Sufficient features (XXX >= 100)
✓ No all-NaN columns

Data Integrity Check:
✓ 'symbol' column present
✓ All critical features present
✓ Acceptable NaN percentage

✅ VALIDATION PASSED
```

## 6. Test Production Inference

### 6.1 Mock Production Run

```bash
# Requires: model/ directory with trained model
python scripts/production_inference.py \
  --snapshot-dir test_snapshots/2024-XX-XX \
  --output-dir test_predictions \
  --model-dir model
```

**Expected Output:**
```
Production Inference Pipeline
Snapshot: 2024-XX-XX

Step 1: Loading feature matrix from snapshot
✓ Loaded XXX rows, XXX columns

Step 2: Building labels (production mode)
✓ Step 2: Building labels (production mode) completed

Step 3: Applying ranker
✓ Step 3: Applying ranker completed

Step 4: Verifying predictions
✓ Predictions saved: test_predictions/predictions_2024-XX-XX.csv

Top 10 predictions:
  symbol       pred
    AAPL  -0.123456
    MSFT  -0.234567
     ...       ...

✅ Production Inference Complete
```

## 7. Integration Test (Full Pipeline)

```bash
# Clean test directory
rm -rf test_full_pipeline
mkdir -p test_full_pipeline

# 1. Fetch (initial)
python scripts/fetch_history_bulletproof.py \
  --tickers "AAPL,MSFT,GOOGL" \
  --period 1y \
  --adjusted \
  --out-dir test_full_pipeline

# 2. Augment (initial)
python scripts/augment_caches_fast.py \
  --tickers "AAPL,MSFT,GOOGL" \
  --cache-dir test_full_pipeline \
  --processes 1

# 3. Enhance (initial)
python scripts/enhance_features_final.py \
  --tickers "AAPL,MSFT,GOOGL" \
  --cache-dir test_full_pipeline \
  --processes 1

# 4. Simulate passage of time (truncate data)
python -c "
import pandas as pd
from pathlib import Path
from datetime import timedelta

cache_dir = Path('test_full_pipeline')
for ticker in ['AAPL', 'MSFT', 'GOOGL']:
    file = cache_dir / f'{ticker}_1y_adj.parquet'
    if file.exists():
        df = pd.read_parquet(file)
        cutoff = df.index.max() - timedelta(days=7)
        df_old = df[df.index <= cutoff]
        df_old.to_parquet(file)
        print(f'Truncated {ticker} to {df_old.index.max().date()}')
"

# 5. Incremental update (all 3 steps)
python scripts/fetch_history_bulletproof.py \
  --tickers "AAPL,MSFT,GOOGL" \
  --period 1y \
  --adjusted \
  --out-dir test_full_pipeline \
  --incremental

python scripts/augment_caches_fast.py \
  --tickers "AAPL,MSFT,GOOGL" \
  --cache-dir test_full_pipeline \
  --incremental \
  --processes 1

python scripts/enhance_features_final.py \
  --tickers "AAPL,MSFT,GOOGL" \
  --cache-dir test_full_pipeline \
  --incremental \
  --processes 1
```

**Expected:**
- All 3 scripts should append new data
- No errors
- No duplicate dates in output

## 8. Verify No Duplicates

```python
import pandas as pd
from pathlib import Path

cache_dir = Path('test_full_pipeline')

# Check each ticker
for ticker in ['AAPL', 'MSFT', 'GOOGL']:
    # Check raw data
    raw_file = cache_dir / f'{ticker}_1y_adj.parquet'
    if raw_file.exists():
        df = pd.read_parquet(raw_file)
        dups = df.index.duplicated().sum()
        print(f"{ticker} raw: {len(df)} rows, {dups} duplicates")
        assert dups == 0, f"Duplicates found in {ticker}!"
    
    # Check features
    feat_file = cache_dir / f'{ticker}_1y_adj_features.parquet'
    if feat_file.exists():
        df = pd.read_parquet(feat_file)
        
        # Normalize index if needed
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        
        dups = df.index.duplicated().sum()
        print(f"{ticker} features: {len(df)} rows, {dups} duplicates")
        assert dups == 0, f"Duplicates found in {ticker} features!"

print("\n✅ All checks passed - no duplicates found")
```

## 9. Maintenance Test

```bash
# Show disk usage
python scripts/maintenance.py --disk-usage

# Verify integrity
python scripts/maintenance.py --verify-integrity

# Clean snapshots (dry run)
python scripts/maintenance.py --clean-snapshots --keep 3 --dry-run

# Actually clean (if needed)
# python scripts/maintenance.py --clean-snapshots --keep 3
```

## 10. Cleanup

```bash
# Remove test directories
rm -rf test_output test_snapshots test_predictions test_full_pipeline
```

## Success Criteria Checklist

- [ ] Incremental fetch: appends only new data, no duplicates
- [ ] Incremental augment: processes only new rows with proper lookback
- [ ] Incremental enhance: processes only new rows with proper lookback
- [ ] Snapshot creation: single parquet with metadata and git hash
- [ ] Snapshot validation: checks freshness, completeness, integrity
- [ ] Production inference: loads snapshot and generates predictions
- [ ] No duplicate dates in any output file
- [ ] All scripts handle edge cases gracefully (already up-to-date, empty data, etc.)

## Notes

- Tests assume Python dependencies are installed (`pip install -r requirements.txt`)
- Some tests require actual market data (may fail on weekends/holidays)
- Integration test creates ~50-100 MB of temporary files
- Production inference test requires trained model in `model/` directory
