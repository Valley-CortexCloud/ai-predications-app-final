# Universe/Data Separation Refactor Summary

## 🎯 Objective Achieved
Successfully refactored the data pipeline to separate ticker universe management from price data fetching. The gold file `config/ticker_universe.csv` is now the single source of truth for all data operations.

## 📦 Key Changes

### New Workflows (3 files)

1. **`.github/workflows/universe-update.yml`** - Universe Update (Weekly - Saturday 10 PM UTC)
   - Scrapes S&P 500 from Wikipedia
   - Scrapes NASDAQ 100 from Wikipedia
   - Merges and deduplicates into `config/ticker_universe.csv`
   - Builds sector map `config/sector_map.csv`
   - Auto-commits changes with summary (no PR review needed)
   - **Runtime:** 5-10 minutes

2. **`.github/workflows/data-refresh.yml`** - Data Refresh (Weekly - Sunday 3 AM UTC)
   - **VALIDATES** `config/ticker_universe.csv` first (fail fast)
   - Fetches price data using `--ticker-file config/ticker_universe.csv`
   - Removes redundant universe scraping during data fetch
   - Fetches benchmark ETFs
   - Augments and enhances features
   - Creates snapshot
   - **Runtime:** 30-60 minutes

3. **`.github/workflows/earnings-refresh.yml`** - Earnings Update (Weekly - Sunday 2 AM UTC)
   - Updates earnings calendar from gold universe
   - Runs BEFORE data-refresh
   - **Runtime:** 5-10 minutes

### New Scripts (1 file)

4. **`scripts/validate_ticker_universe.py`** - Universe Validation
   - Checks file exists and is readable
   - Validates CSV structure (required columns: symbol, name, exchange, source)
   - Checks for duplicate symbols
   - Validates no empty symbols
   - Validates symbol format (uppercase, alphanumeric with - or .)
   - Spot-checks random sample with Yahoo Finance
   - **Exit codes:** 0 = valid, 1 = invalid (blocks data fetch)

### Modified Scripts (2 files)

5. **`scripts/fetch_history_bulletproof.py`** - Enhanced ticker input
   - Added `--ticker-file` alias for `--tickers-file`
   - **NEW:** CSV file support - reads 'symbol' column from CSV
   - Updated docstring emphasizing `--ticker-file` as primary method
   - `--universe` option kept as fallback (not recommended for production)
   - Validates CSV structure when reading files

6. **`scripts/fetch_ticker_universe.py`** - Enhanced output control
   - Added `--output` argument for flexible output path
   - Used by universe-update.yml workflow

### Removed Files (1 file)

7. **`.github/workflows/update-ticker-data.yml`** - DELETED
   - Replaced by separate universe-update.yml and data-refresh.yml
   - Old workflow did too many things in one job

### Updated Documentation (1 file)

8. **`README.md`** - Updated workflow documentation
   - New pipeline diagram showing 4 separate stages
   - Updated automated workflows section
   - Updated script usage examples
   - Emphasized `--ticker-file` as primary method

## 🎨 Architecture Benefits

### Before (Problems)
```
update-ticker-data.yml (Sunday 3 AM)
├─ Scrape Wikipedia for S&P 500
├─ Scrape Wikipedia for NASDAQ 100  
├─ Build sector map
├─ Fetch prices (--universe sp500)
├─ Fetch prices (--universe nasdaq) ← REDUNDANT
└─ Augment/enhance/snapshot

Issues:
❌ If Wikipedia scrape fails, entire pipeline corrupted
❌ Redundant fetching (sp500 + nasdaq)
❌ No validation before fetching
❌ Universe changes mixed with data changes
❌ --universe bypasses gold file
```

### After (Solution)
```
Saturday 10 PM → universe-update.yml
                 ├─ Scrape and merge
                 ├─ Build sector map
                 └─ Commit gold files
                 
Sunday 2 AM    → earnings-refresh.yml
                 └─ Update earnings from gold universe
                 
Sunday 3 AM    → data-refresh.yml
                 ├─ VALIDATE gold universe (fail fast)
                 ├─ Fetch prices (--ticker-file config/ticker_universe.csv)
                 └─ Augment/enhance/snapshot

Benefits:
✅ Gold file committed BEFORE data fetch
✅ Validation catches corrupt universe early
✅ Single source of truth (config/ticker_universe.csv)
✅ No redundant fetching
✅ Clear separation of concerns
✅ Easier debugging (isolated workflows)
```

## 📊 Workflow Schedule

```
SATURDAY 10 PM UTC    → universe-update.yml
                        └── Scrape Wikipedia, update gold files, commit

SUNDAY 2 AM UTC       → earnings-refresh.yml
                        └── Update earnings from gold universe

SUNDAY 3 AM UTC       → data-refresh.yml
                        ├── VALIDATE gold universe first
                        └── Fetch prices, augment, create snapshot

MONDAY 8:30 AM UTC    → weekly-predictions.yml
                        └── Run ML predictions on fresh snapshot

MONDAY 9:00 AM UTC    → portfolio-validation.yml
                        └── Grok exit detection, send email
```

## 🧪 Testing

### Validation Script Tests
```bash
# Test validation with good universe
python scripts/validate_ticker_universe.py --file config/ticker_universe.csv --skip-yahoo

# Test validation with bad universe
python scripts/validate_ticker_universe.py --file /tmp/bad_universe.csv

# Run automated test suite
python test_validation.py
```

### CSV Parsing Tests
```bash
# Test CSV input to fetch script
python scripts/fetch_history_bulletproof.py \
  --ticker-file config/ticker_universe.csv \
  --limit 5 --period 5d --out-dir /tmp/test
```

All tests passing ✅

## 🔒 Security & Reliability

1. **Fail Fast:** Validation runs BEFORE any expensive data fetching
2. **Single Source of Truth:** All scripts read from committed gold file
3. **Atomic Updates:** Universe committed separately from data
4. **Change Tracking:** Added/removed tickers logged in commit messages
5. **No Mid-Workflow Corruption:** Universe can't change during data fetch

## 📝 Migration Notes

### For Manual Runs
Old way:
```bash
python scripts/fetch_history_bulletproof.py --universe sp500 --period 2y
```

New way (recommended):
```bash
python scripts/fetch_history_bulletproof.py --ticker-file config/ticker_universe.csv --period 2y
```

Old way still works but not recommended for production.

### For Workflows
- Old `update-ticker-data.yml` workflow removed
- New workflows are independent and can run separately
- Universe updates happen BEFORE data refresh
- Validation prevents bad universe from reaching data fetch

## ✅ Success Criteria

- [x] Universe update separated from data fetch
- [x] Validation script catches bad data
- [x] Gold file is single source of truth
- [x] CSV parsing works correctly
- [x] Documentation updated
- [x] Tests passing
- [x] Old workflow removed
- [x] No breaking changes (--universe still works as fallback)

## 🚀 Next Steps

1. Monitor first run of new workflows
2. Verify validation catches issues in production
3. Consider adding more validation checks if needed
4. Track universe changes over time in git history
