# Earnings Quality Fix - Testing Guide

## Summary of Changes

Fixed `feat_earnings_quality` being zero in production mode by replacing the earnings calendar features approach with `pd.merge_asof()`.

### Key Changes
1. **`load_earnings_events()`** - Now returns DataFrame (not dict) with eps_actual, eps_estimate, eps_surprise_pct
2. **Earnings merge** - Uses `merge_asof(direction='backward')` with 120-day tolerance
3. **Earnings quality** - Computed directly from eps_surprise_pct (-100 to +300)
4. **Logging** - Comprehensive earnings merge statistics and VIX validation

## Testing the Fix

### Prerequisites
```bash
# Install dependencies
pip install numpy pandas pyarrow

# Ensure you have an earnings CSV file
# Format: symbol,earnings_date,eps_actual,eps_estimate,eps_surprise_pct
```

### Unit Tests
```bash
# Run the earnings merge tests
python3 test_earnings_merge.py

# Expected output:
# ✅ All tests PASSED (3/3)
```

### Production Mode Test
```bash
# Test with production mode (requires ticker cache and earnings file)
python3 scripts/build_labels_final.py \
  --cache-dir data_cache/10y_ticker_features \
  --output datasets/today_features.parquet \
  --earnings-file data/earnings.csv \
  --production-only

# Expected output should include:
# 📊 Earnings Data Statistics:
#   Total earnings events: X
#   Earnings in last 90 days: Y events for Z symbols
#
# 📈 Earnings Merge Results:
#   Stocks with earnings data: ~350/501 (70%)
#
# 📊 Earnings Quality Statistics:
#   Min: -XX.XX, Max: XXX.XX, Mean: X.XX
#   Non-zero: ~70%
#
# 📊 VIX Features Validation:
#   feat_vix_level_z_63: Min, Max, Mean
#   High/Low volatility regime distribution
```

## Expected Behavior After Fix

### Before (Broken)
- ❌ `feat_earnings_quality` = 0 for all stocks in production
- ❌ No earnings data merged for single-date inference
- ❌ Coverage: 0% in production mode

### After (Fixed)
- ✅ `feat_earnings_quality` non-zero for ~70% of stocks
- ✅ Most recent earnings (within 120 days) merged for each stock
- ✅ Coverage: ~70% in both training and production modes
- ✅ Comprehensive debug logging showing merge statistics
- ✅ VIX features validated with output

## Verification Checklist

After running in production mode, verify:

1. **Earnings merge statistics logged**
   - [ ] "📊 Earnings Data Statistics" shows total events and symbols
   - [ ] "📈 Earnings Merge Results" shows ~70% coverage
   
2. **Earnings quality computed**
   - [ ] "📊 Earnings Quality Statistics" shows non-zero values
   - [ ] Min/Max/Mean are reasonable (-100 to +300 range)
   - [ ] Sample shows non-zero values for specific stocks
   
3. **VIX features validated**
   - [ ] "📊 VIX Features Validation" section present
   - [ ] VIX regime distribution shown (high/low volatility)
   - [ ] Sample VIX values displayed

4. **Output file created**
   - [ ] `datasets/today_features.parquet` exists
   - [ ] Contains `feat_earnings_quality` column
   - [ ] Contains `feat_prev_earn_surprise_pct` column
   - [ ] Values are non-zero for stocks with recent earnings

## Configuration

Earnings-related constants in `scripts/build_labels_final.py`:

```python
EARNINGS_LOOKBACK_DAYS = 90   # Days to look back for statistics
EARNINGS_TOLERANCE_DAYS = 120  # Max days for merge_asof (~1 quarter)
```

Adjust these if needed for your specific use case.

## Troubleshooting

### "No earnings data loaded"
- Ensure earnings CSV file exists and is readable
- Check file format: `symbol,earnings_date,eps_actual,eps_estimate,eps_surprise_pct`
- Verify symbols in earnings file match ticker cache symbols

### "Earnings quality still zero"
- Check earnings file date range covers recent quarters
- Verify tolerance (120 days) is appropriate
- Check logs for "Stocks with earnings data" percentage

### "File not found" errors
- Ensure `--cache-dir` points to correct ticker features directory
- Verify ticker cache files exist (e.g., `AAPL_2y_raw_features_enhanced.parquet`)
- Check SPY benchmark file exists in cache directory
