The production grade stock predictions application.

## API Keys (Optional but Recommended)

For complete earnings calendar coverage (503/503 tickers vs ~347/503 with Yahoo-only), add these free API keys as GitHub Secrets:

1. **Finnhub** (recommended - best coverage): https://finnhub.io/register
2. **FMP** (backup): https://financialmodelingprep.com/developer
3. **Alpha Vantage** (backup): https://www.alphavantage.co/support/#api-key

### How to Add API Keys

1. Go to your repository on GitHub
2. Navigate to: **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add each secret with these exact names:
   - `FINNHUB_API_KEY`
   - `FMP_API_KEY`
   - `ALPHAVANTAGE_API_KEY`

### Expected Coverage

- **With API keys**: ~100% ticker coverage (503/503) using Finnhub as primary source
- **Without API keys**: ~85-90% coverage with improved Yahoo parsing (vs previous 69%)
- **Fallback chain**: Finnhub → FMP → Alpha Vantage → Yahoo

