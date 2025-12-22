#!/usr/bin/env python3
"""
Test suite for enhanced validation system.
Tests ValidationTracker and snapshot validation functionality.
"""
import sys
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import validation components
from scripts.validate_snapshot import ValidationTracker, validate_snapshot


def test_validation_tracker():
    """Test ValidationTracker check tracking and summary"""
    print("\n" + "="*60)
    print("TEST: ValidationTracker")
    print("="*60)
    
    tracker = ValidationTracker()
    
    # Add passing checks
    tracker.check("Check 1", True, "")
    tracker.check("Check 2", True, "")
    
    # Add failing checks
    tracker.check("Check 3", False, "This check failed")
    
    print("\nTracker state:")
    print(f"  Passed: {tracker.passed}")
    print(f"  Failed: {tracker.failed}")
    print(f"  Total checks: {len(tracker.checks)}")
    
    # Validate state
    assert tracker.passed == 2, "Should have 2 passed checks"
    assert tracker.failed == 1, "Should have 1 failed check"
    assert len(tracker.checks) == 3, "Should have 3 total checks"
    
    # Test summary
    result = tracker.summary()
    assert result is False, "Summary should return False when checks failed"
    
    print("\n✅ ValidationTracker test PASSED")
    return True


def test_validation_tracker_all_pass():
    """Test ValidationTracker with all passing checks"""
    print("\n" + "="*60)
    print("TEST: ValidationTracker (all pass)")
    print("="*60)
    
    tracker = ValidationTracker()
    
    # All passing checks
    tracker.check("Check 1", True, "")
    tracker.check("Check 2", True, "")
    tracker.check("Check 3", True, "")
    
    # Test summary
    result = tracker.summary()
    assert result is True, "Summary should return True when all checks pass"
    
    print("\n✅ ValidationTracker all-pass test PASSED")
    return True


def create_test_snapshot(tmpdir: Path, valid: bool = True) -> Path:
    """Create a test snapshot directory with mock data"""
    snapshot_dir = tmpdir / "2024-12-22"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    # Create feature matrix
    n_symbols = 450 if valid else 350  # Test symbol count validation
    n_features = 120 if valid else 80  # Test feature count validation
    
    symbols = [f"TICKER{i:03d}" for i in range(n_symbols)]
    
    # Create mock feature data
    data = {
        'symbol': symbols,
        'Close': np.random.uniform(50, 500, n_symbols),
        'Volume': np.random.uniform(1e6, 1e8, n_symbols),
        'Adj Close': np.random.uniform(50, 500, n_symbols),
    }
    
    # Add random features
    for i in range(n_features - 3):
        data[f'feat_{i}'] = np.random.randn(n_symbols)
    
    # Add beta feature for testing
    if valid:
        data['feat_beta_spy_126'] = np.random.uniform(-2, 2, n_symbols)
    
    df = pd.DataFrame(data)
    df.to_parquet(snapshot_dir / "feature_matrix.parquet", index=False)
    
    # Create universe file
    universe_df = pd.DataFrame({'symbol': symbols})
    universe_df.to_csv(snapshot_dir / "universe.csv", index=False)
    
    # Create metadata
    data_date = datetime.now() - timedelta(days=1 if valid else 10)
    created_at = datetime.now() - timedelta(hours=1 if valid else 72)
    
    metadata = {
        'snapshot_date': data_date.strftime('%Y-%m-%d'),
        'data_date': data_date.strftime('%Y-%m-%d'),
        'created_at': created_at.isoformat(),
        'symbol_count': n_symbols,
        'feature_count': n_features,
        'git_commit': 'abc123def456' if valid else '',
        'all_nan_columns': []
    }
    
    with open(snapshot_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return snapshot_dir


def test_snapshot_validation_pass():
    """Test snapshot validation with valid data"""
    print("\n" + "="*60)
    print("TEST: Snapshot validation (should pass)")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        snapshot_dir = create_test_snapshot(tmpdir, valid=True)
        
        # Run validation
        result = validate_snapshot(snapshot_dir)
        
        assert result is True, "Valid snapshot should pass validation"
    
    print("\n✅ Snapshot validation pass test PASSED")
    return True


def test_snapshot_validation_fail():
    """Test snapshot validation with invalid data"""
    print("\n" + "="*60)
    print("TEST: Snapshot validation (should fail)")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        snapshot_dir = create_test_snapshot(tmpdir, valid=False)
        
        # Run validation
        result = validate_snapshot(snapshot_dir)
        
        assert result is False, "Invalid snapshot should fail validation"
    
    print("\n✅ Snapshot validation fail test PASSED")
    return True


def test_snapshot_validation_missing_files():
    """Test snapshot validation with missing required files"""
    print("\n" + "="*60)
    print("TEST: Snapshot validation (missing files)")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        snapshot_dir = tmpdir / "2024-12-22"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Only create one file (missing others)
        metadata = {
            'snapshot_date': '2024-12-22',
            'data_date': '2024-12-22',
            'created_at': datetime.now().isoformat(),
            'symbol_count': 500,
            'feature_count': 120,
            'git_commit': 'abc123',
            'all_nan_columns': []
        }
        with open(snapshot_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Run validation
        result = validate_snapshot(snapshot_dir)
        
        assert result is False, "Snapshot with missing files should fail"
    
    print("\n✅ Missing files test PASSED")
    return True


def test_snapshot_validation_stale_data():
    """Test snapshot validation with stale data"""
    print("\n" + "="*60)
    print("TEST: Snapshot validation (stale data)")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        snapshot_dir = tmpdir / "2024-12-22"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Create valid feature matrix
        n_symbols = 450
        symbols = [f"TICKER{i:03d}" for i in range(n_symbols)]
        data = {
            'symbol': symbols,
            'Close': np.random.uniform(50, 500, n_symbols),
            'Volume': np.random.uniform(1e6, 1e8, n_symbols),
            'Adj Close': np.random.uniform(50, 500, n_symbols),
        }
        for i in range(100):
            data[f'feat_{i}'] = np.random.randn(n_symbols)
        
        df = pd.DataFrame(data)
        df.to_parquet(snapshot_dir / "feature_matrix.parquet", index=False)
        
        # Create universe
        universe_df = pd.DataFrame({'symbol': symbols})
        universe_df.to_csv(snapshot_dir / "universe.csv", index=False)
        
        # Create metadata with old date (> 5 days)
        old_date = datetime.now() - timedelta(days=10)
        metadata = {
            'snapshot_date': old_date.strftime('%Y-%m-%d'),
            'data_date': old_date.strftime('%Y-%m-%d'),
            'created_at': old_date.isoformat(),
            'symbol_count': n_symbols,
            'feature_count': 103,
            'git_commit': 'abc123',
            'all_nan_columns': []
        }
        with open(snapshot_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Run validation
        result = validate_snapshot(snapshot_dir)
        
        assert result is False, "Stale snapshot should fail validation"
    
    print("\n✅ Stale data test PASSED")
    return True


def run_all_tests():
    """Run all validation tests"""
    print("\n" + "="*60)
    print("RUNNING VALIDATION SYSTEM TESTS")
    print("="*60)
    
    tests = [
        test_validation_tracker,
        test_validation_tracker_all_pass,
        test_snapshot_validation_pass,
        test_snapshot_validation_fail,
        test_snapshot_validation_missing_files,
        test_snapshot_validation_stale_data,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n❌ Test FAILED: {test.__name__}")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed > 0:
        print("\n❌ Some tests FAILED")
        return False
    else:
        print("\n✅ All tests PASSED")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
