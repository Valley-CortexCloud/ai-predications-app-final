#!/usr/bin/env python3
"""
Minimal validation demo without dependencies.
Shows the ValidationTracker in action.
"""

class ValidationTracker:
    """Track validation checks and generate summary"""
    def __init__(self):
        self.checks = []
        self.passed = 0
        self.failed = 0
    
    def check(self, name: str, condition: bool, error_msg: str = ""):
        """Register a validation check"""
        self.checks.append({
            'name': name,
            'passed': condition,
            'error': error_msg if not condition else None
        })
        if condition:
            self.passed += 1
            print(f"  ✓ {name}")
        else:
            self.failed += 1
            print(f"  ✗ {name}: {error_msg}")
    
    def summary(self) -> bool:
        """Print summary and return overall pass/fail"""
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        if self.failed == 0:
            print(f"✅ {self.passed}/{total} CHECKS PASSED - DATA VALIDATED!")
            print(f"{'='*60}")
            return True
        else:
            print(f"❌ {self.failed}/{total} CHECKS FAILED")
            print(f"{'='*60}")
            print(f"\nFailed checks:")
            for check in self.checks:
                if not check['passed']:
                    print(f"  • {check['name']}: {check['error']}")
            return False


def demo_success_case():
    """Demonstrate successful validation"""
    print(f"\n{'='*60}")
    print("DEMONSTRATION: Success Case")
    print(f"{'='*60}\n")
    
    tracker = ValidationTracker()
    
    print("📁 FILE EXISTENCE CHECKS")
    tracker.check("Feature matrix exists", True, "")
    tracker.check("Universe file exists", True, "")
    tracker.check("Metadata file exists", True, "")
    
    print("\n🕐 DATA FRESHNESS CHECKS")
    tracker.check("Snapshot age < 48 hours", True, "")
    tracker.check("Data age < 5 days", True, "")
    
    print("\n📊 DATA COMPLETENESS CHECKS")
    tracker.check("Universe size >= 400 symbols", True, "")
    tracker.check("Feature count >= 100", True, "")
    tracker.check("Feature matrix matches universe", True, "")
    
    print("\n🔬 FEATURE QUALITY CHECKS")
    tracker.check("No features with >50% NaN", True, "")
    tracker.check("No infinite values", True, "")
    tracker.check("No constant features", True, "")
    
    print("\n⚡ CRITICAL FEATURE CHECKS")
    tracker.check("OHLCV features present", True, "")
    tracker.check("Beta values in range [-5, 5]", True, "")
    tracker.check("Volatility values in range [0, 10]", True, "")
    
    print("\n🔐 METADATA INTEGRITY CHECKS")
    tracker.check("Git commit hash present", True, "")
    tracker.check("Feature count matches metadata", True, "")
    
    passed = tracker.summary()
    
    if passed:
        print(f"\n🚀 SNAPSHOT READY FOR PRODUCTION INFERENCE")
        print(f"   Symbols: 503")
        print(f"   Features: 142")
        print(f"   Data date: 2024-12-21")
        print(f"   Age: 1 days")
    
    return passed


def demo_failure_case():
    """Demonstrate validation failure"""
    print(f"\n\n{'='*60}")
    print("DEMONSTRATION: Failure Case")
    print(f"{'='*60}\n")
    
    tracker = ValidationTracker()
    
    print("📁 FILE EXISTENCE CHECKS")
    tracker.check("Feature matrix exists", True, "")
    tracker.check("Universe file exists", True, "")
    tracker.check("Metadata file exists", True, "")
    
    print("\n🕐 DATA FRESHNESS CHECKS")
    tracker.check("Snapshot age < 48 hours", False, "Snapshot is 168.3 hours old (max: 48)")
    tracker.check("Data age < 5 days", False, "Data is 7 days old (max: 5)")
    
    print("\n📊 DATA COMPLETENESS CHECKS")
    tracker.check("Universe size >= 400 symbols", True, "")
    tracker.check("Feature count >= 100", False, "Only 87 features (expected >= 100)")
    tracker.check("Feature matrix matches universe", True, "")
    
    print("\n🔬 FEATURE QUALITY CHECKS")
    tracker.check("No features with >50% NaN", True, "")
    tracker.check("No infinite values", False, "3 features contain Inf: ['feat_vol_expansion', 'feat_upside_quality']")
    tracker.check("No constant features", True, "")
    
    passed = tracker.summary()
    
    if not passed:
        print(f"\n⛔ SNAPSHOT VALIDATION FAILED - DO NOT USE FOR PRODUCTION")
    
    return passed


if __name__ == "__main__":
    print("="*60)
    print("ENHANCED VALIDATION SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Demo success case
    success = demo_success_case()
    assert success is True, "Success case should pass"
    
    # Demo failure case
    failure = demo_failure_case()
    assert failure is False, "Failure case should fail"
    
    print("\n\n" + "="*60)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Features:")
    print("  • Clear X/Y checks passed format")
    print("  • Organized validation sections with emojis")
    print("  • Specific error messages for failed checks")
    print("  • Summary with list of all failures")
    print("  • Boolean return for exit codes")
    print("="*60)
