#!/usr/bin/env python3
"""
Test suite for ticker universe validation.

Tests:
1. Valid universe passes
2. Missing columns are caught
3. Duplicate symbols are caught
4. Empty symbols are caught
5. CSV parsing with symbol column works
"""
import tempfile
from pathlib import Path
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from validate_ticker_universe import validate_structure, validate_symbols
import pandas as pd


def test_valid_universe():
    """Test that a valid universe passes validation."""
    df = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL'],
        'name': ['Apple', 'Microsoft', 'Google'],
        'exchange': ['NASDAQ', 'NASDAQ', 'NASDAQ'],
        'source': ['SP500', 'SP500', 'SP500']
    })
    
    errors = validate_structure(df)
    assert len(errors) == 0, f"Expected no errors, got: {errors}"
    
    errors = validate_symbols(df)
    assert len(errors) == 0, f"Expected no errors, got: {errors}"


def test_missing_columns():
    """Test that missing required columns are caught."""
    df = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT'],
        'company': ['Apple', 'Microsoft']
    })
    
    errors = validate_structure(df)
    assert len(errors) > 0, "Expected errors for missing columns"
    assert 'Missing required columns' in errors[0]


def test_duplicate_symbols():
    """Test that duplicate symbols are caught."""
    df = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'AAPL'],  # Duplicate AAPL
        'name': ['Apple', 'Microsoft', 'Apple Dup'],
        'exchange': ['NASDAQ', 'NASDAQ', 'NASDAQ'],
        'source': ['SP500', 'SP500', 'SP500']
    })
    
    errors = validate_symbols(df)
    assert len(errors) > 0, "Expected errors for duplicate symbols"
    assert 'Duplicate symbols' in errors[0]


def test_empty_symbols():
    """Test that empty symbols are caught."""
    df = pd.DataFrame({
        'symbol': ['AAPL', '', 'MSFT'],  # Empty symbol
        'name': ['Apple', 'Empty', 'Microsoft'],
        'exchange': ['NASDAQ', 'NASDAQ', 'NASDAQ'],
        'source': ['SP500', 'SP500', 'SP500']
    })
    
    errors = validate_symbols(df)
    assert len(errors) > 0, "Expected errors for empty symbols"
    assert 'empty symbols' in errors[0]


def test_csv_parsing_integration():
    """Test that CSV files are parsed correctly with symbol column."""
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('symbol,name,exchange,source\n')
        f.write('AAPL,Apple,NASDAQ,SP500\n')
        f.write('MSFT,Microsoft,NASDAQ,SP500\n')
        f.write('GOOGL,Google,NASDAQ,SP500\n')
        temp_path = f.name
    
    try:
        # Test that we can read and parse it
        df = pd.read_csv(temp_path)
        df.columns = df.columns.str.lower()
        
        assert 'symbol' in df.columns, "CSV should have symbol column"
        assert len(df) == 3, "CSV should have 3 rows"
        
        errors = validate_structure(df)
        assert len(errors) == 0, "Valid CSV should pass structure validation"
        
        errors = validate_symbols(df)
        assert len(errors) == 0, "Valid CSV should pass symbol validation"
    finally:
        Path(temp_path).unlink()


def test_case_insensitive_columns():
    """Test that column names are case-insensitive."""
    df = pd.DataFrame({
        'Symbol': ['AAPL', 'MSFT'],  # Uppercase
        'Name': ['Apple', 'Microsoft'],
        'Exchange': ['NASDAQ', 'NASDAQ'],
        'Source': ['SP500', 'SP500']
    })
    
    errors = validate_structure(df)
    assert len(errors) == 0, "Column names should be case-insensitive"


if __name__ == '__main__':
    # Run tests
    print("Running validation tests...")
    
    test_valid_universe()
    print("✅ test_valid_universe passed")
    
    test_missing_columns()
    print("✅ test_missing_columns passed")
    
    test_duplicate_symbols()
    print("✅ test_duplicate_symbols passed")
    
    test_empty_symbols()
    print("✅ test_empty_symbols passed")
    
    test_csv_parsing_integration()
    print("✅ test_csv_parsing_integration passed")
    
    test_case_insensitive_columns()
    print("✅ test_case_insensitive_columns passed")
    
    print("\n✅ All validation tests passed!")
