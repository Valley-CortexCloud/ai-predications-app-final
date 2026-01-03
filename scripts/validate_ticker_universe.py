#!/usr/bin/env python3
"""
Validate ticker_universe.csv before price fetching.

Checks:
1. File exists and is readable
2. Required columns present (symbol, name, exchange, source)
3. No duplicate symbols
4. No empty symbols
5. Symbol format is valid (uppercase, no spaces)
6. Random sample of tickers are valid Yahoo Finance symbols

Exit codes:
  0 = Valid
  1 = Invalid (blocks price fetch)
"""
import argparse
import sys
import random
import pandas as pd
import yfinance as yf
from pathlib import Path

def validate_structure(df: pd.DataFrame) -> list:
    """Check CSV structure."""
    errors = []
    
    # Required columns
    required = ['symbol', 'name', 'exchange', 'source']
    # Case-insensitive check - use a copy to avoid modifying original
    cols_lower = [c.lower() for c in df.columns]
    
    missing = [c for c in required if c not in cols_lower]
    if missing:
        errors.append(f"Missing required columns: {missing}")
    
    return errors

def validate_symbols(df: pd.DataFrame) -> list:
    """Check symbol column quality."""
    errors = []
    warnings = []
    
    if 'symbol' not in df.columns:
        return ["No 'symbol' column found"]
    
    symbols = df['symbol'].astype(str)
    
    # Empty symbols
    empty = symbols.str.strip().eq('').sum()
    if empty > 0:
        errors.append(f"Found {empty} empty symbols")
    
    # Duplicates
    dupes = symbols[symbols.duplicated()].tolist()
    if dupes:
        errors.append(f"Duplicate symbols: {dupes[:10]}")
    
    # Invalid format (should be uppercase, alphanumeric with optional - or .)
    invalid = symbols[~symbols.str.match(r'^[A-Z0-9\-\.]+$', na=False)]
    if len(invalid) > 0:
        # Could be lowercase - warn but don't fail
        lowercase = symbols[symbols.str.match(r'^[a-z0-9\-\.]+$', na=False)]
        if len(lowercase) > 0:
            warnings.append(f"Found {len(lowercase)} lowercase symbols (will be uppercased)")
        
        truly_invalid = invalid[~invalid.str.match(r'^[a-zA-Z0-9\-\.]+$', na=False)]
        if len(truly_invalid) > 0:
            errors.append(f"Invalid symbol format: {truly_invalid.head(10).tolist()}")
    
    return errors

def validate_yahoo_sample(df: pd.DataFrame, sample_size: int = 10, seed: int = 42) -> list:
    """Spot-check random tickers are valid on Yahoo Finance."""
    errors = []
    warnings = []
    
    if 'symbol' not in df.columns:
        return ["No 'symbol' column found"]
    
    symbols = df['symbol'].astype(str).str.upper().str.strip().tolist()
    
    # Random sample
    random.seed(seed)
    sample = random.sample(symbols, min(sample_size, len(symbols)))
    
    print(f"Validating {len(sample)} random tickers: {sample}")
    
    failed = []
    for sym in sample:
        try:
            ticker = yf.Ticker(sym)
            # Try multiple methods to verify ticker exists
            valid = False
            
            # Method 1: Check fast_info (most reliable for valid tickers)
            try:
                last_price = ticker.fast_info.get('lastPrice')
                if last_price is None:
                    last_price = ticker.fast_info.get('previousClose')
                if last_price is not None and last_price > 0:
                    valid = True
            except:
                pass
            
            # Method 2: Check info dict (fallback)
            if not valid:
                try:
                    info = ticker.info
                    if info and (info.get('regularMarketPrice') or info.get('previousClose')):
                        valid = True
                except:
                    pass
            
            # Method 3: Try to get recent history (final fallback)
            if not valid:
                try:
                    hist = ticker.history(period='5d')
                    if hist is not None and len(hist) > 0:
                        valid = True
                except:
                    pass
            
            if not valid:
                failed.append(sym)
                
        except Exception as e:
            failed.append(sym)
    
    if failed:
        fail_rate = len(failed) / len(sample)
        if fail_rate > 0.5:  # More than 50% failed = error (more lenient)
            errors.append(f"Too many invalid tickers ({len(failed)}/{len(sample)}): {failed}")
        elif fail_rate > 0:
            # Just warn, don't fail
            print(f"  ⚠️  Warning: Some tickers may be invalid ({len(failed)}/{len(sample)}): {failed}")
    
    return errors

def main():
    parser = argparse.ArgumentParser(description="Validate ticker universe CSV")
    parser.add_argument('--file', default='config/ticker_universe.csv', help='Path to universe CSV')
    parser.add_argument('--sample-size', type=int, default=10, help='Number of tickers to validate with Yahoo')
    parser.add_argument('--skip-yahoo', action='store_true', help='Skip Yahoo validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    args = parser.parse_args()
    
    print(f"=" * 60)
    print(f"Validating: {args.file}")
    print(f"=" * 60)
    
    path = Path(args.file)
    
    # Check file exists
    if not path.exists():
        print(f"❌ FATAL: File not found: {path}")
        return 1
    
    # Load CSV once
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"❌ FATAL: Cannot read CSV: {e}")
        return 1
    
    print(f"Loaded {len(df)} rows")
    
    all_errors = []
    
    # Structure validation
    print("\n📋 Checking structure...")
    errors = validate_structure(df)
    all_errors.extend(errors)
    for e in errors:
        print(f"  ❌ {e}")
    if not errors:
        print(f"  ✅ Structure OK")
    
    # Symbol validation - create lowercase version for validation
    print("\n🔤 Checking symbols...")
    df_lower = df.copy()
    df_lower.columns = df_lower.columns.str.lower()
    errors = validate_symbols(df_lower)
    all_errors.extend(errors)
    for e in errors:
        print(f"  ❌ {e}")
    if not errors:
        print(f"  ✅ Symbols OK ({len(df_lower)} unique)")
    
    # Yahoo validation
    if not args.skip_yahoo:
        print(f"\n🌐 Validating {args.sample_size} random tickers with Yahoo...")
        errors = validate_yahoo_sample(df, args.sample_size, args.seed)
        all_errors.extend(errors)
        for e in errors:
            print(f"  ❌ {e}")
        if not errors:
            print(f"  ✅ Yahoo validation passed")
    
    # Final result
    print(f"\n" + "=" * 60)
    if all_errors:
        print(f"❌ VALIDATION FAILED ({len(all_errors)} errors)")
        for e in all_errors:
            print(f"   - {e}")
        return 1
    else:
        print(f"✅ VALIDATION PASSED")
        print(f"   Universe: {len(df)} tickers")
        return 0

if __name__ == '__main__':
    sys.exit(main())
