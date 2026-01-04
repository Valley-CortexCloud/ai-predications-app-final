#!/usr/bin/env python3
"""
Email Proposal Sender - Enhanced email with review link and quick reply

Sends portfolio rotation proposals via email with dashboard review link.

Usage:
    python scripts/send_proposal_email.py \
        --proposed data/portfolio/proposed_2026-01-05.csv \
        --token abc123... \
        --date 2026-01-05
"""

import argparse
import json
import os
import smtplib
import sys
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List

import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

# Get dashboard URL from environment variable
# Set this in your environment or in .github/workflows/portfolio-validation.yml
DASHBOARD_BASE_URL = os.environ.get('DASHBOARD_URL')
if not DASHBOARD_BASE_URL or DASHBOARD_BASE_URL == 'https://portfolio-dashboard.alpha-work.workers.dev':
    print("⚠️  WARNING: DASHBOARD_URL not set or using placeholder")
    print("   Dashboard links in emails will not work properly")
    print("   Set DASHBOARD_URL environment variable to your Cloudflare Worker URL")
    print("   See cloudflare-worker/SETUP.md for deployment instructions")
    DASHBOARD_BASE_URL = "https://portfolio-dashboard.alpha-work.workers.dev"

TOKEN_EXPIRY_HOURS = 24


# ============================================================================
# Email Template
# ============================================================================

def build_email_body(
    proposed_df: pd.DataFrame,
    date: str,
    token: str,
    review_url: str
) -> str:
    """
    Build email body with summary and action links
    
    Args:
        proposed_df: Proposed actions DataFrame
        date: Date string (YYYY-MM-DD)
        token: Security token
        review_url: Full dashboard review URL
        
    Returns:
        Email body as plain text
    """
    
    # Count actions
    sells = proposed_df[proposed_df['proposed_action'].str.upper() == 'SELL']
    buys = proposed_df[proposed_df['proposed_action'].str.upper() == 'BUY']
    holds = proposed_df[proposed_df['proposed_action'].str.upper() == 'HOLD']
    
    # Build sells section
    sells_section = ""
    if not sells.empty:
        sells_section = "🔴 SELLS ({}):\n".format(len(sells))
        for _, row in sells.iterrows():
            symbol = row['symbol']
            exit_score = row.get('exit_score', 0)
            reason = row.get('reason', 'No reason provided')
            # Truncate reason to 50 chars
            reason_short = reason[:50] + "..." if len(reason) > 50 else reason
            sells_section += f"   • {symbol} (Score: {exit_score}) - {reason_short}\n"
    
    # Build buys section
    buys_section = ""
    if not buys.empty:
        buys_section = "🟢 BUYS ({}):\n".format(len(buys))
        for _, row in buys.iterrows():
            symbol = row['symbol']
            conviction = row.get('conviction', 'Buy')
            reason = row.get('reason', 'No reason provided')
            # Truncate reason to 50 chars
            reason_short = reason[:50] + "..." if len(reason) > 50 else reason
            buys_section += f"   • {symbol} - {conviction} - {reason_short}\n"
    
    # Build holds section (only if there are some)
    holds_section = ""
    if not holds.empty and len(holds) <= 5:  # Only show if 5 or fewer
        holds_section = f"\n🔵 HOLDS ({len(holds)}):\n"
        for _, row in holds.iterrows():
            holds_section += f"   • {row['symbol']}\n"
    elif not holds.empty:
        holds_section = f"\n🔵 HOLDS: {len(holds)} positions (view dashboard for details)\n"
    
    # Calculate expiry
    expires_dt = datetime.now() + timedelta(hours=TOKEN_EXPIRY_HOURS)
    expires_str = expires_dt.strftime('%Y-%m-%d %H:%M ET')
    
    # Build full email
    email_body = f"""🧠 Portfolio Intelligence Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 Rotation Date: {date}
⏰ Expires: {expires_str} (24 hours from now)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 PROPOSED CHANGES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{sells_section}
{buys_section}{holds_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 ACTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▶️ FULL REVIEW & MODIFY:
{review_url}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📧 QUICK EMAIL REPLY (FUTURE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Reply with ONE word:
  CONFIRM  →  Execute all proposed trades
  DENY     →  Cancel, no trades

(Email reply automation requires setup - see CONFIRMATION_SETUP.md)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔐 Token: {token[:16]}...
📊 Summary: {len(sells)} sells, {len(buys)} buys, {len(holds)} holds

---
Portfolio Intelligence Engine v1.0
Renaissance-inspired exit detection with Grok AI
"""
    
    return email_body


# ============================================================================
# Email Sending
# ============================================================================

def send_email(
    to_email: str,
    subject: str,
    body: str,
    from_email: str = None,
    smtp_user: str = None,
    smtp_pass: str = None
) -> bool:
    """
    Send email via Gmail SMTP
    
    Args:
        to_email: Recipient email
        subject: Email subject
        body: Email body (plain text)
        from_email: Sender email (defaults to smtp_user)
        smtp_user: Gmail username (from env EMAIL_USER)
        smtp_pass: Gmail app password (from env EMAIL_PASS)
        
    Returns:
        True if sent successfully
    """
    
    # Get credentials from environment
    smtp_user = smtp_user or os.environ.get('EMAIL_USER')
    smtp_pass = smtp_pass or os.environ.get('EMAIL_PASS')
    
    if not smtp_user or not smtp_pass:
        print("⚠️  Email credentials not found in environment")
        print("   Set EMAIL_USER and EMAIL_PASS to enable email sending")
        return False
    
    from_email = from_email or smtp_user
    
    # Create message
    msg = MIMEMultipart()
    msg['From'] = f'Portfolio Intelligence <{from_email}>'
    msg['To'] = to_email
    msg['Subject'] = subject
    
    msg.attach(MIMEText(body, 'plain'))
    
    # Send via Gmail SMTP
    try:
        print(f"📧 Sending email to {to_email}...")
        
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)
        server.quit()
        
        print(f"✅ Email sent successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Send portfolio rotation proposal email'
    )
    parser.add_argument(
        '--proposed',
        type=Path,
        required=True,
        help='Path to proposed_*.csv file'
    )
    parser.add_argument(
        '--token',
        type=str,
        required=True,
        help='Security token for dashboard access'
    )
    parser.add_argument(
        '--date',
        type=str,
        required=True,
        help='Date string (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--to',
        type=str,
        default='jvalley19@gmail.com',
        help='Recipient email address'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print email body without sending'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.proposed.exists():
        print(f"❌ Proposed file not found: {args.proposed}")
        sys.exit(1)
    
    # Load proposed changes
    print(f"📊 Loading proposed changes: {args.proposed}")
    proposed_df = pd.read_csv(args.proposed)
    
    # Build review URL
    review_url = f"{DASHBOARD_BASE_URL}/?token={args.token}&date={args.date}"
    
    # Build email body
    email_body = build_email_body(
        proposed_df,
        args.date,
        args.token,
        review_url
    )
    
    # Subject line
    subject = f"📊 Portfolio Review Required - {args.date}"
    
    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN - Email body:")
        print("="*60)
        print(f"\nTo: {args.to}")
        print(f"Subject: {subject}")
        print(f"\n{email_body}")
        print("="*60)
        return 0
    
    # Send email
    success = send_email(
        to_email=args.to,
        subject=subject,
        body=email_body
    )
    
    if success:
        print(f"✅ Portfolio rotation email sent to {args.to}")
        return 0
    else:
        print(f"❌ Failed to send email")
        return 1


if __name__ == '__main__':
    sys.exit(main())
