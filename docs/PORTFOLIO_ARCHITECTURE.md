# Portfolio Intelligence Engine Architecture

## Executive Summary

The Portfolio Intelligence Engine transforms our weekly prediction system into a complete portfolio rotation strategy by adding AI-powered exit detection, automated trade execution, and systematic portfolio management. This is the "alpha multiplier" component that enables asymmetric holding strategies inspired by Renaissance Technologies.

**Key Innovation**: Use Grok AI to monitor positions for "failure signals" across dimensions our quant model can't see—sentiment shifts, management scandals, competitive threats, and technical breakdowns.

## Core Principles

### 1. Signal Fusion (Quant + AI)
- **Entry**: Pure quant model (120+ features, 63-day horizon)
- **Exit**: AI-augmented monitoring (Grok detects deterioration)
- **Rebalance**: Weekly validation cycle (Monday 5 AM ET)

### 2. Asymmetric Holding Optimization
- Let winners compound for full 63-day horizon
- Ruthlessly exit on fundamental/sentiment/technical deterioration
- Minimum 15-day hold period (except catastrophic exits)
- Target: 70% of exits occur naturally at 63-day horizon

### 3. Low Turnover by Design
- Max 25% monthly portfolio turnover
- Only rotate if current holding shows "failure"
- Prefer HOLD over premature exit (winners run longer than you think)
- Turnover constraint enforced algorithmically

### 4. Human-in-the-Loop (V1)
- AI proposes changes → Human reviews → Execute
- Manual confirmation prevents AI over-trading
- V2 roadmap: Graduated autonomy with kill switches

### 5. Risk Controls
- Max 15 positions (concentration + diversification balance)
- Max 15% position size (20% for Strong Buy conviction)
- Position sizing with conviction weighting (Strong Buy: 1.4x, Buy: 1.1x)
- Paper trading default (require explicit --live flag)

## Quick Start Guide

### Initial Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Initialize portfolio tracker
python scripts/portfolio_tracker.py --init

# Set environment variables
export XAI_API_KEY="your-key-here"
export ALPACA_API_KEY="your-key-here"
export ALPACA_API_SECRET="your-secret-here"
```

### Weekly Workflow
1. GitHub Actions runs portfolio_validator.py automatically (Monday 5 AM ET)
2. Review proposed_YYYY-MM-DD.csv from email
3. Save approved version as confirmed_YYYY-MM-DD.csv
4. Run trade_executor.py to generate orders
5. Submit orders and update tracker

## File Reference

### Core Scripts

- **`scripts/portfolio_validator.py`** - Exit detection engine (queries Grok)
- **`scripts/trade_executor.py`** - Order generation and submission
- **`scripts/portfolio_tracker.py`** - Portfolio state management

### Data Files

- **`data/portfolio/tracker.csv`** - Current portfolio state
- **`data/portfolio/proposed_*.csv`** - AI-generated recommendations  
- **`data/portfolio/confirmed_*.csv`** - Human-approved changes
- **`data/portfolio/orders_*.json`** - Trade orders for Alpaca

## Configuration

### Exit Score Thresholds
- `<60`: HOLD (no material deterioration)
- `60-75`: WATCH (early warning signals)
- `75-85`: EXIT (material deterioration, sell if >15 days)
- `>85`: CATASTROPHIC EXIT (override minimum hold period)

### Portfolio Constraints
- Max positions: 15
- Max position size: 15%
- Max monthly turnover: 25%
- Min hold period: 15 days

### Position Sizing
```python
CONVICTION_MULTIPLIERS = {
    'Strong Buy': 1.4,  # 21% → capped at 15%
    'Buy': 1.1,         # 16.5% → capped at 15%
    'Hold': 0.8,        # 12%
    'Avoid': 0.0        # Skip
}
```

## Environment Variables

Required for production:
- `XAI_API_KEY` - Grok API key
- `ALPACA_API_KEY` - Alpaca trading key
- `ALPACA_API_SECRET` - Alpaca secret
- `EMAIL_USER` - Gmail for notifications
- `EMAIL_PASS` - Gmail app password

## Example Usage

### Dry Run (No Grok API Calls)
```bash
python scripts/portfolio_validator.py --dry-run
```

### Generate Orders (Manual Mode)
```bash
python scripts/trade_executor.py \
  --confirmed data/portfolio/confirmed_2024-01-15.csv \
  --paper
```

### Auto-Submit to Alpaca (V2)
```bash
python scripts/trade_executor.py \
  --confirmed data/portfolio/confirmed_2024-01-15.csv \
  --auto \
  --paper
```

### Portfolio Management
```bash
# Sync prices and calculate P&L
python scripts/portfolio_tracker.py --sync

# Generate performance report
python scripts/portfolio_tracker.py --report

# Update from fills
python scripts/portfolio_tracker.py --update-fills fills_2024-01-15.json
```

## Risk Disclaimers

⚠️ **This is experimental software for educational purposes**

- No guarantees on performance
- AI can hallucinate or miss signals
- Always review recommendations manually
- Test thoroughly in paper mode first
- Never risk more than you can afford to lose

## Support

- GitHub Issues: [Open an issue](https://github.com/Valley-CortexCloud/ai-predications-app-final/issues)
- Email: jvalley19@gmail.com

---

**Version**: 1.0.0  
**Last Updated**: January 2024
