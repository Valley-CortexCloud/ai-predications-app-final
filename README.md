# AI Stock Predictions - Production Grade Application

A production-grade stock prediction system with institutional-grade data architecture featuring append-only data lakes and weekly snapshots for fast, reproducible inference.

## 🏗️ Architecture

### Data Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    WEEKLY DATA PIPELINE                      │
│               (Sunday 3 AM UTC - Offline)                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌───────────────────────────────────────────┐
    │ 1. Fetch Ticker Data (--incremental)      │
    │    • S&P 500 + NASDAQ universe            │
    │    • Benchmark ETFs (SPY, VIX, etc.)      │
    │    • Append only new data since last run  │
    └───────────────────────────────────────────┘
                            │
                            ▼
    ┌───────────────────────────────────────────┐
    │ 2. Augment Features (--incremental)       │
    │    • Technical indicators                 │
    │    • Market-relative features             │
    │    • Rolling statistics                   │
    └───────────────────────────────────────────┘
                            │
                            ▼
    ┌───────────────────────────────────────────┐
    │ 3. Enhance Features (--incremental)       │
    │    • Cross-sectional ranks                │
    │    • Sector-relative features             │
    │    • VIX regime indicators                │
    └───────────────────────────────────────────┘
                            │
                            ▼
    ┌───────────────────────────────────────────┐
    │ 4. Create Snapshot                        │
    │    • Single parquet file (all tickers)    │
    │    • Metadata (git hash, checksums)       │
    │    • Saved to data/snapshots/YYYY-MM-DD/  │
    └───────────────────────────────────────────┘
                            │
                            ▼
    ┌───────────────────────────────────────────┐
    │ 5. Commit Data to Repo                    │
    │    • data_cache/ (raw + features)         │
    │    • data/snapshots/ (weekly snapshots)   │
    └───────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  PRODUCTION INFERENCE                        │
│              (Monday 4:30 AM ET - < 5 min)                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌───────────────────────────────────────────┐
    │ 1. Validate Snapshot Freshness            │
    │    • Age < 7 days                         │
    │    • Symbol count >= 400                  │
    │    • Feature count >= 100                 │
    └───────────────────────────────────────────┘
                            │
                            ▼
    ┌───────────────────────────────────────────┐
    │ 2. Load Latest Snapshot (Fast)            │
    │    • Pre-computed features                │
    │    • No API calls required                │
    └───────────────────────────────────────────┘
                            │
                            ▼
    ┌───────────────────────────────────────────┐
    │ 3. Apply Ranker Model                     │
    │    • Generate predictions                 │
    │    • Create Top 20 portfolio              │
    └───────────────────────────────────────────┘
                            │
                            ▼
    ┌───────────────────────────────────────────┐
    │ 4. LLM Supercharge (Optional)             │
    │    • Grok analysis of Top 20              │
    │    • Narrative generation                 │
    └───────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Git
- ~2 GB disk space for data

### Installation

```bash
# Clone repository
git clone https://github.com/Valley-CortexCloud/ai-predications-app-final.git
cd ai-predications-app-final

# Install dependencies
pip install -r requirements.txt
```

### Initial Setup (First Time Only)

```bash
# 1. Fetch ticker universe and sector mappings
python scripts/fetch_ticker_universe.py
python scripts/build_sector_map.py

# 2. Update earnings calendar
python scripts/update_earnings_incremental.py

# 3. Fetch initial data (may take 30-60 min for full universe)
python scripts/fetch_history_bulletproof.py --universe sp500 --period 2y --adjusted --out-dir data_cache/10y_ticker_features --max-workers 8

# 4. Augment features
python scripts/augment_caches_fast.py --processes 4 --cache-dir data_cache/10y_ticker_features

# 5. Enhance features
python scripts/enhance_features_final.py --processes 4 --cache-dir data_cache/10y_ticker_features --sector-map config/sector_map.csv

# 6. Create first snapshot
python scripts/create_snapshot.py --features-dir data_cache/10y_ticker_features --output-dir data/snapshots
```

### Weekly Data Updates (Incremental)

```bash
# Update data incrementally (only fetches new dates)
python scripts/fetch_history_bulletproof.py --universe sp500 --period 2y --adjusted --incremental --max-workers 8
python scripts/augment_caches_fast.py --incremental --processes 4
python scripts/enhance_features_final.py --incremental --processes 4

# Create new snapshot
python scripts/create_snapshot.py

# Validate snapshot
python scripts/validate_snapshot.py
```

### Production Inference (Fast)

```bash
# Run inference using latest snapshot (< 5 min)
python scripts/production_inference.py --snapshots-root data/snapshots --output-dir datasets --model-dir model
```

## 📋 Script Reference

### Core Data Pipeline Scripts

#### `scripts/fetch_history_bulletproof.py`
Fetch OHLCV data from yfinance with robust error handling.

**Key Features:**
- `--incremental`: Append only new data since last date
- Automatic deduplication
- Session-aligned dates (NYSE calendar)
- Adjusted vs raw close handling

**Usage:**
```bash
# Full fetch
python scripts/fetch_history_bulletproof.py --universe sp500 --period 2y --adjusted

# Incremental update (only fetch new dates)
python scripts/fetch_history_bulletproof.py --universe sp500 --period 2y --adjusted --incremental
```

#### `scripts/augment_caches_fast.py`
Compute technical features from raw OHLCV data.

**Key Features:**
- `--incremental`: Only compute features for new rows
- Lookback window: 252 days (1 year trading days)
- Parallel processing with multiprocessing
- Market-relative features (SPY correlation)

**Usage:**
```bash
# Full augmentation
python scripts/augment_caches_fast.py --processes 4

# Incremental (only process new dates)
python scripts/augment_caches_fast.py --incremental --processes 4
```

#### `scripts/enhance_features_final.py`
Add market-relative, cross-sectional, and interaction features.

**Key Features:**
- `--incremental`: Only enhance new rows
- Lookback window: 126 days
- VIX regime indicators
- Sector-relative features
- Momentum quality metrics

**Usage:**
```bash
# Full enhancement
python scripts/enhance_features_final.py --processes 4

# Incremental (only process new dates)
python scripts/enhance_features_final.py --incremental --processes 4
```

### Snapshot Management Scripts

#### `scripts/create_snapshot.py`
Create point-in-time snapshot of all ticker features.

**Output:**
- `data/snapshots/YYYY-MM-DD/feature_matrix.parquet` - All tickers, latest date
- `data/snapshots/YYYY-MM-DD/universe.csv` - List of symbols
- `data/snapshots/YYYY-MM-DD/metadata.json` - Git hash, checksums, stats

**Usage:**
```bash
python scripts/create_snapshot.py --features-dir data_cache/10y_ticker_features --output-dir data/snapshots
```

#### `scripts/validate_snapshot.py`
Validate snapshot is fresh and complete.

**Checks:**
- Snapshot age < 7 days
- Symbol count >= 400
- Feature count >= 100
- No all-NaN columns
- Data integrity

**Usage:**
```bash
# Validate latest snapshot
python scripts/validate_snapshot.py

# Validate specific snapshot
python scripts/validate_snapshot.py --snapshot-dir data/snapshots/2024-12-16
```

#### `scripts/production_inference.py`
Fast production inference using cached snapshots.

**Flow:**
1. Load latest snapshot (pre-computed features)
2. Build labels (production mode)
3. Apply ranker model
4. Generate predictions

**Target Runtime:** < 5 minutes

**Usage:**
```bash
python scripts/production_inference.py --snapshots-root data/snapshots --output-dir datasets --model-dir model
```

## 🤖 Automated Workflows

### Weekly Data Update (`.github/workflows/update-ticker-data.yml`)
**Schedule:** Sunday 3 AM UTC

**Steps:**
1. Update ticker universe and sector mappings
2. Fetch data incrementally (S&P 500, NASDAQ, benchmarks)
3. Augment features incrementally
4. Enhance features incrementally
5. Create snapshot
6. Validate snapshot
7. Commit all data to repository

**Runtime:** 30-60 minutes (offline, doesn't matter)

### Weekly Predictions (`.github/workflows/weekly-predictions.yml`)
**Schedule:** Monday 4:30 AM ET

**Steps:**
1. Validate snapshot freshness
2. Run production inference (using cached snapshot)
3. Generate Top 20 portfolio
4. LLM supercharge (Grok analysis)
5. Email results
6. Commit predictions only

**Runtime:** < 5 minutes (fast!)

### Portfolio Validation (`.github/workflows/portfolio-validation.yml`)
**Schedule:** Monday 5:00 AM ET

**Steps:**
1. Run portfolio validator (exit detection with Grok)
2. Generate proposed portfolio changes
3. Upload proposed changes as artifact
4. Email for human review
5. Commit proposed file

**Runtime:** ~2-3 minutes

## 📊 Data Storage

### Expected Sizes (after 3 months of weekly runs)

```
data_cache/
├── 10y_ticker_features/          # ~150 MB
│   ├── AAPL_2y_adj.parquet
│   ├── AAPL_2y_adj_features.parquet
│   └── AAPL_2y_adj_features_enhanced.parquet
└── _etf_cache/                   # ~10 MB
    ├── SPY_2y_raw.parquet
    └── ^VIX_2y_raw.parquet

data/
└── snapshots/                    # ~150 MB (12 weekly snapshots)
    ├── 2024-09-15/
    ├── 2024-09-22/
    └── 2024-12-22/
        ├── feature_matrix.parquet  # ~12 MB compressed
        ├── universe.csv            # ~10 KB
        └── metadata.json           # ~1 KB

Total: ~400 MB (well within GitHub limits)
```

### Git Repository Size Projection

- **Initial:** ~50 MB
- **After 3 months:** ~500-700 MB
- **After 1 year:** ~1.5-2 GB

**Note:** GitHub free tier limit is 5 GB, so this is sustainable.

## ✅ Validation & Testing

### Test Incremental Mode

```bash
# Test with small set of tickers
python scripts/fetch_history_bulletproof.py --tickers "AAPL,MSFT,GOOGL,AMZN,TSLA" --period 2y --adjusted --incremental
python scripts/augment_caches_fast.py --tickers "AAPL,MSFT,GOOGL,AMZN,TSLA" --incremental
python scripts/enhance_features_final.py --tickers "AAPL,MSFT,GOOGL,AMZN,TSLA" --incremental
```

### Validate No Duplicates

```python
import pandas as pd

# Check for duplicate dates
df = pd.read_parquet("data_cache/10y_ticker_features/AAPL_2y_adj.parquet")
assert df.index.duplicated().sum() == 0, "Duplicate dates found!"
print(f"✓ No duplicates in {len(df)} rows")
```

### Test Snapshot Creation

```bash
# Create test snapshot
python scripts/create_snapshot.py

# Validate it
python scripts/validate_snapshot.py

# Test production inference
python scripts/production_inference.py
```

## 🎯 Success Criteria

- [x] All 3 scripts accept `--incremental` flag
- [x] Snapshots include metadata with git hash
- [x] Production inference designed for < 5 min runtime
- [x] Weekly workflows configured
- [x] Git repo size projection < 2 GB after 3 months

## 📝 Best Practices

### Incremental Updates
- Run incremental mode weekly to keep data fresh
- Validate after each incremental update
- Monitor gap warnings (> 7 days)

### Snapshot Management
- Keep 12 weekly snapshots (rolling 3 months)
- Always validate before production use
- Check git commit hash for reproducibility

### Production Inference
- Always use validated snapshots
- Check data age (< 7 days)
- Monitor prediction distribution

## 💼 Portfolio Intelligence Engine (NEW)

Transform predictions into portfolio action with AI-powered exit detection and automated trade execution.

### Features
- **Exit Detection**: Grok AI monitors holdings for deterioration signals
- **Asymmetric Holding**: Let winners run, ruthlessly exit failures
- **Low Turnover**: 15-25% monthly max (Renaissance-inspired)
- **Human-in-the-Loop**: Manual confirmation before trades (V1)
- **Alpaca Integration**: Automated order generation and submission

### Quick Start
```bash
# Initialize portfolio
python scripts/portfolio_tracker.py --init

# Weekly validation (runs automatically via GitHub Actions)
python scripts/portfolio_validator.py --dry-run  # Test mode

# Generate orders from confirmed changes
python scripts/trade_executor.py --confirmed data/portfolio/confirmed_2024-01-15.csv --paper

# Sync portfolio state
python scripts/portfolio_tracker.py --sync
python scripts/portfolio_tracker.py --report
```

### Key Scripts
- `scripts/portfolio_validator.py` - Exit detection with Grok AI
- `scripts/trade_executor.py` - Order generation and Alpaca submission  
- `scripts/portfolio_tracker.py` - Portfolio state management

### Documentation
📚 **[Complete Portfolio Architecture Guide](docs/PORTFOLIO_ARCHITECTURE.md)**

Covers:
- Architecture principles and data flow
- Exit score thresholds and rationale
- Position sizing formulas
- Weekly workflow and examples
- Configuration and troubleshooting

## 🌐 Web Dashboard + Email Confirmation System (NEW)

Review and approve portfolio rotations via interactive web dashboard or email reply.

### Features
- **🎨 Interactive Dashboard**: Single-page web UI with TailwindCSS + Alpine.js
  - Toggle SELL ↔ HOLD for current holdings
  - Toggle BUY ↔ SKIP for new recommendations
  - Color-coded exit scores (red >70, green <60)
  - Real-time token expiry countdown
- **📧 Enhanced Email Notifications**: Review link + quick reply support
  - Summary of proposed trades in email
  - One-click dashboard access with secure token
  - Quick CONFIRM/DENY via email reply (optional setup)
- **🔒 Security**: Cryptographic tokens with 24-hour expiration, single-use
- **🚀 GitHub Actions**: Automated confirmation processing via workflows
- **📱 Mobile-Friendly**: Responsive design works on all devices
- **🆓 Free Tier**: GitHub Pages + Actions only, no external services required

### Quick Start
```bash
# 1. Enable GitHub Pages
# Go to: Settings → Pages → Source: /docs

# 2. Generate dashboard data and send email
python scripts/generate_dashboard_data.py \
  --proposed data/portfolio/proposed_2026-01-05.csv
  
python scripts/send_proposal_email.py \
  --proposed data/portfolio/proposed_2026-01-05.csv \
  --token <generated-token> \
  --date 2026-01-05

# 3. User reviews via dashboard at:
# https://valley-cortexcloud.github.io/ai-predications-app-final/dashboard/

# 4. Confirmation triggers GitHub Action to execute trades
```

### Key Files
- `docs/dashboard/index.html` - Interactive web dashboard
- `scripts/generate_dashboard_data.py` - Convert proposals to JSON with tokens
- `scripts/send_proposal_email.py` - Enhanced email with review link
- `.github/workflows/dashboard-confirmation.yml` - Process dashboard submissions
- `.github/workflows/email-confirmation.yml` - Process email replies
- `data/portfolio/tokens/` - Secure token storage

### Setup Guide
📚 **[Confirmation System Setup Guide](docs/CONFIRMATION_SETUP.md)**

Complete instructions for:
- GitHub Pages configuration
- Personal Access Token creation
- Email automation setup (Zapier/Google Apps Script)
- End-to-end testing
- Troubleshooting and security best practices

### Workflow Integration
The confirmation system is integrated into the weekly portfolio validation workflow:
1. **Monday 5 AM ET**: `portfolio-validation.yml` runs
2. Generates `proposed_*.csv` with exit scores and recommendations
3. Creates dashboard JSON data with secure token
4. Commits data to repo (triggers GitHub Pages deployment)
5. Sends email with dashboard review link
6. **User reviews** via dashboard or email reply
7. **Confirmation triggers** trade execution in Alpaca paper account
8. Updates `tracker.csv` and sends confirmation email

## 🐛 Troubleshooting

### "No data to append" in incremental mode
- Files are already up-to-date
- This is expected behavior, not an error

### Gap warnings in fetch script
- Stock was halted or delisted
- Market holiday period
- Usually safe to ignore if gap < 7 days

### Snapshot validation fails
- Check snapshot age (may need fresh data)
- Verify symbol count (universe changes)
- Review all-NaN columns (some features expected)

### "XAI_API_KEY not set"
- Required for portfolio_validator.py
- Set environment variable: `export XAI_API_KEY="your-key"`

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Email: jvalley19@gmail.com

## 📄 License

Production-grade stock predictions application.
