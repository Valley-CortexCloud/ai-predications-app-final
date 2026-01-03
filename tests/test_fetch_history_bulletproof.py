#!/usr/bin/env python3
"""
Unit tests for fetch_history_bulletproof.py script.

Tests cover:
- CSV file parsing (with headers)
- Text file parsing (line-by-line)
- Symbol normalization
- Error handling
"""

import sys
import pytest
import pandas as pd
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from fetch_history_bulletproof import resolve_universe


class TestResolveUniverse:
    """Tests for the resolve_universe function."""
    
    def test_csv_with_symbol_column(self):
        """Test CSV parsing with proper 'symbol' column."""
        with NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("symbol,name,exchange,source\n")
            f.write("AAPL,Apple Inc.,NASDAQ,SP500\n")
            f.write("MSFT,Microsoft Corp.,NASDAQ,SP500\n")
            f.write("GOOGL,Alphabet Inc.,NASDAQ,SP500\n")
            f.flush()
            
            tickers = resolve_universe(None, None, f.name)
            
            # Clean up
            Path(f.name).unlink()
            
            assert len(tickers) == 3
            assert "AAPL" in tickers
            assert "MSFT" in tickers
            assert "GOOGL" in tickers
            # Should NOT include header values
            assert "SYMBOL" not in tickers
            assert "NAME" not in tickers
            assert "EXCHANGE" not in tickers
            assert "SOURCE" not in tickers
    
    def test_csv_with_uppercase_symbol_column(self):
        """Test CSV parsing with 'Symbol' (uppercase) column."""
        with NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Symbol,Name\n")
            f.write("TSLA,Tesla Inc.\n")
            f.write("NVDA,NVIDIA Corp.\n")
            f.flush()
            
            tickers = resolve_universe(None, None, f.name)
            
            # Clean up
            Path(f.name).unlink()
            
            assert len(tickers) == 2
            assert "TSLA" in tickers
            assert "NVDA" in tickers
    
    def test_csv_with_ticker_column(self):
        """Test CSV parsing with 'ticker' column instead of 'symbol'."""
        with NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("ticker,company\n")
            f.write("AMD,Advanced Micro Devices\n")
            f.write("INTC,Intel Corp.\n")
            f.flush()
            
            tickers = resolve_universe(None, None, f.name)
            
            # Clean up
            Path(f.name).unlink()
            
            assert len(tickers) == 2
            assert "AMD" in tickers
            assert "INTC" in tickers
    
    def test_csv_fallback_to_first_column(self):
        """Test CSV parsing falls back to first column if no symbol column."""
        with NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("stock_code,description\n")
            f.write("FB,Facebook\n")
            f.write("AMZN,Amazon\n")
            f.flush()
            
            tickers = resolve_universe(None, None, f.name)
            
            # Clean up
            Path(f.name).unlink()
            
            assert len(tickers) == 2
            assert "FB" in tickers
            assert "AMZN" in tickers
    
    def test_csv_with_commas_in_data(self):
        """Test CSV parsing handles commas in company names correctly."""
        with NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("symbol,name,exchange,source\n")
            f.write("A,\"Agilent Technologies, Inc.\",NYSE,SP500\n")
            f.write("ABBV,AbbVie Inc.,NYSE,SP500\n")
            f.flush()
            
            tickers = resolve_universe(None, None, f.name)
            
            # Clean up
            Path(f.name).unlink()
            
            assert len(tickers) == 2
            assert "A" in tickers
            assert "ABBV" in tickers
            # Should not split on commas within quoted fields
            assert "AGILENT" not in tickers
            assert "TECHNOLOGIES" not in tickers
    
    def test_csv_with_empty_values(self):
        """Test CSV parsing handles empty/null values."""
        with NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("symbol,name\n")
            f.write("AAPL,Apple\n")
            f.write(",Missing Symbol\n")
            f.write("MSFT,Microsoft\n")
            f.flush()
            
            tickers = resolve_universe(None, None, f.name)
            
            # Clean up
            Path(f.name).unlink()
            
            # Should skip rows with empty symbols
            assert len(tickers) == 2
            assert "AAPL" in tickers
            assert "MSFT" in tickers
    
    def test_txt_file_one_per_line(self):
        """Test plain text file parsing (one ticker per line)."""
        with NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("AAPL\n")
            f.write("MSFT\n")
            f.write("# This is a comment\n")
            f.write("GOOGL\n")
            f.write("\n")  # empty line
            f.write("  TSLA  \n")  # with spaces
            f.flush()
            
            tickers = resolve_universe(None, None, f.name)
            
            # Clean up
            Path(f.name).unlink()
            
            assert len(tickers) == 4
            assert "AAPL" in tickers
            assert "MSFT" in tickers
            assert "GOOGL" in tickers
            assert "TSLA" in tickers
    
    def test_txt_file_with_dots(self):
        """Test text file handles dots in ticker symbols (converts to dash)."""
        with NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("BRK.B\n")
            f.write("BF.B\n")
            f.flush()
            
            tickers = resolve_universe(None, None, f.name)
            
            # Clean up
            Path(f.name).unlink()
            
            assert len(tickers) == 2
            assert "BRK-B" in tickers
            assert "BF-B" in tickers
    
    def test_explicit_tickers_parameter(self):
        """Test explicit tickers parameter (comma-separated)."""
        tickers = resolve_universe(None, "AAPL,MSFT,GOOGL", None)
        
        assert len(tickers) == 3
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "GOOGL" in tickers
    
    def test_explicit_tickers_with_spaces(self):
        """Test explicit tickers parameter handles spaces."""
        tickers = resolve_universe(None, "AAPL, MSFT , GOOGL", None)
        
        assert len(tickers) == 3
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "GOOGL" in tickers
    
    def test_file_not_found(self):
        """Test graceful handling of non-existent file."""
        tickers = resolve_universe(None, None, "/nonexistent/file.csv")
        
        assert len(tickers) == 0
    
    def test_empty_parameters(self):
        """Test behavior when no tickers specified."""
        tickers = resolve_universe(None, None, None)
        
        assert len(tickers) == 0
    
    def test_csv_symbol_normalization(self):
        """Test CSV symbols are uppercased and normalized."""
        with NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("symbol,name\n")
            f.write("aapl,Apple\n")
            f.write("brk.b,Berkshire Hathaway\n")
            f.flush()
            
            tickers = resolve_universe(None, None, f.name)
            
            # Clean up
            Path(f.name).unlink()
            
            assert len(tickers) == 2
            assert "AAPL" in tickers
            assert "BRK-B" in tickers
    
    def test_actual_ticker_universe_csv(self):
        """Test with the actual ticker_universe.csv file if it exists."""
        ticker_file = Path(__file__).parent.parent / "config" / "ticker_universe.csv"
        
        if ticker_file.exists():
            tickers = resolve_universe(None, None, str(ticker_file))
            
            # Should have many tickers
            assert len(tickers) > 0
            
            # Should NOT include header values
            assert "SYMBOL" not in tickers
            assert "NAME" not in tickers
            assert "EXCHANGE" not in tickers
            assert "SOURCE" not in tickers
            
            # Should include actual tickers from the file
            # Read the file to verify
            df = pd.read_csv(ticker_file)
            expected_tickers = df['symbol'].astype(str).str.upper().tolist()
            
            assert len(tickers) == len(expected_tickers)
            for expected in expected_tickers[:5]:  # Check first 5
                assert expected in tickers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
