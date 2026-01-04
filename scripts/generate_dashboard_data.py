#!/usr/bin/env python3
"""
Dashboard Data Generator - Convert proposed CSV to JSON for web dashboard

Generates secure tokens and prepares data for dashboard consumption.

Usage:
    python scripts/generate_dashboard_data.py --proposed data/portfolio/proposed_2026-01-05.csv
"""

import argparse
import json
import secrets
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

TOKEN_EXPIRY_HOURS = 24

# Get dashboard URL from environment variable
# Set this in your environment or in .github/workflows/portfolio-validation.yml
DASHBOARD_BASE_URL = os.environ.get('DASHBOARD_URL')
if not DASHBOARD_BASE_URL or DASHBOARD_BASE_URL == 'YOUR_CLOUDFLARE_WORKER_URL':
    print("⚠️  WARNING: DASHBOARD_URL not set or using placeholder")
    print("   Dashboard links in output will not work properly")
    print("   Set DASHBOARD_URL environment variable to your Cloudflare Worker URL")
    print("   See cloudflare-worker/SETUP.md for deployment instructions")
    DASHBOARD_BASE_URL = "YOUR_CLOUDFLARE_WORKER_URL"


# ============================================================================
# Token Management
# ============================================================================

def generate_token() -> str:
    """Generate a cryptographically secure random token"""
    return secrets.token_urlsafe(32)


def save_token(date: str, token: str, output_dir: Path) -> None:
    """Save token to file for validation"""
    token_dir = output_dir / 'tokens'
    token_dir.mkdir(parents=True, exist_ok=True)
    
    token_file = token_dir / f"{date}.json"
    
    token_data = {
        'token': token,
        'date': date,
        'created': datetime.now().isoformat(),
        'expires': (datetime.now() + timedelta(hours=TOKEN_EXPIRY_HOURS)).isoformat(),
        'used': False
    }
    
    with open(token_file, 'w') as f:
        json.dump(token_data, f, indent=2)
    
    print(f"✅ Token saved: {token_file}")


# ============================================================================
# Data Conversion
# ============================================================================

def convert_proposed_to_json(
    proposed_df: pd.DataFrame,
    tracker_df: pd.DataFrame,
    date: str,
    token: str
) -> Dict:
    """
    Convert proposed CSV to JSON format for dashboard
    
    Args:
        proposed_df: Proposed actions DataFrame
        tracker_df: Current tracker DataFrame
        date: Date string (YYYY-MM-DD)
        token: Security token
        
    Returns:
        Dictionary ready for JSON serialization
    """
    
    # Separate holdings (positions we currently own) from recommendations (new buys)
    holdings = []
    recommendations = []
    
    for _, row in proposed_df.iterrows():
        symbol = row.get('symbol', '')
        action = row.get('proposed_action', '').upper()
        
        # Build base item
        item = {
            'symbol': symbol,
            'proposed_action': action,
            'reason': row.get('reason', 'No reason provided')
        }
        
        # Check if this is a current holding or a new recommendation
        is_holding = not tracker_df.empty and symbol in tracker_df['symbol'].values
        
        if is_holding:
            # This is a current holding - add exit score and P&L info
            tracker_row = tracker_df[tracker_df['symbol'] == symbol].iloc[0]
            holdings.append({
                **item,
                'exit_score': int(row.get('exit_score', 0)) if 'exit_score' in row else 0,
                'days_held': int(tracker_row.get('days_held', 0)),
                'pnl_pct': float(tracker_row.get('pnl_pct', 0.0)),
                'entry_price': float(tracker_row.get('entry_price', 0.0)),
                'current_price': float(tracker_row.get('current_price', 0.0)),
            })
        else:
            # This is a new recommendation
            recommendations.append({
                **item,
                'conviction': row.get('conviction', 'Buy'),
                'supercharged_rank': int(row.get('supercharged_rank', 0)) if 'supercharged_rank' in row else 0,
            })
    
    # Calculate turnover estimate (simplified)
    total_positions = len(tracker_df) if not tracker_df.empty else 1
    sells = sum(1 for h in holdings if h['proposed_action'] == 'SELL')
    turnover_pct = (sells / total_positions * 100) if total_positions > 0 else 0
    
    return {
        'date': date,
        'token': token,
        'generated': datetime.now().isoformat(),
        'expires': (datetime.now() + timedelta(hours=TOKEN_EXPIRY_HOURS)).isoformat(),
        'turnover_pct': round(turnover_pct, 1),
        'holdings': holdings,
        'recommendations': recommendations,
        'summary': {
            'total_sells': sum(1 for h in holdings if h['proposed_action'] == 'SELL'),
            'total_buys': len(recommendations),
            'total_holds': sum(1 for h in holdings if h['proposed_action'] == 'HOLD'),
        }
    }


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate dashboard data from proposed portfolio changes'
    )
    parser.add_argument(
        '--proposed',
        type=Path,
        required=True,
        help='Path to proposed_*.csv file'
    )
    parser.add_argument(
        '--tracker',
        type=Path,
        default=Path('data/portfolio/tracker.csv'),
        help='Path to tracker.csv (default: data/portfolio/tracker.csv)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/portfolio'),
        help='Output directory for tokens (default: data/portfolio)'
    )
    parser.add_argument(
        '--dashboard-dir',
        type=Path,
        default=Path('docs/dashboard/data'),
        help='Dashboard data directory (default: docs/dashboard/data)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.proposed.exists():
        print(f"❌ Proposed file not found: {args.proposed}")
        sys.exit(1)
    
    # Extract date from filename (proposed_YYYY-MM-DD.csv)
    try:
        date_str = args.proposed.stem.replace('proposed_', '')
        datetime.strptime(date_str, '%Y-%m-%d')  # Validate format
    except ValueError:
        print(f"❌ Invalid filename format. Expected: proposed_YYYY-MM-DD.csv")
        sys.exit(1)
    
    print(f"📊 Generating dashboard data for {date_str}")
    
    # Load data
    print(f"   Loading proposed changes: {args.proposed}")
    proposed_df = pd.read_csv(args.proposed)
    
    print(f"   Loading tracker: {args.tracker}")
    tracker_df = pd.DataFrame()
    if args.tracker.exists():
        tracker_df = pd.read_csv(args.tracker)
    else:
        print("   ⚠️  Tracker not found, treating all as new recommendations")
    
    # Generate token
    token = generate_token()
    print(f"🔐 Generated token: {token[:16]}...")
    
    # Save token for validation
    save_token(date_str, token, args.output_dir)
    
    # Convert to JSON
    dashboard_data = convert_proposed_to_json(
        proposed_df,
        tracker_df,
        date_str,
        token
    )
    
    # Save dashboard JSON
    args.dashboard_dir.mkdir(parents=True, exist_ok=True)
    dashboard_file = args.dashboard_dir / f"{date_str}.json"
    
    with open(dashboard_file, 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    print(f"✅ Dashboard data saved: {dashboard_file}")
    
    # Print summary
    print(f"\n📋 Summary:")
    print(f"   Sells: {dashboard_data['summary']['total_sells']}")
    print(f"   Buys: {dashboard_data['summary']['total_buys']}")
    print(f"   Holds: {dashboard_data['summary']['total_holds']}")
    print(f"   Turnover: {dashboard_data['turnover_pct']:.1f}%")
    
    # Print review URL
    review_url = f"{DASHBOARD_BASE_URL}/?token={token}&date={date_str}"
    print(f"\n🔗 Review URL:")
    print(f"   {review_url}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
