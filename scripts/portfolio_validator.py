#!/usr/bin/env python3
"""
Portfolio Validator - Core Exit Detection Engine

Queries Grok for EXIT_SCORE on each holding to detect positions that should be sold.
Uses hallucination-guarded prompts for Renaissance-inspired asymmetric holding strategy.

Usage:
    python scripts/portfolio_validator.py [--dry-run] [--tracker data/portfolio/tracker.csv]
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from openai import OpenAI
import os

# ============================================================================
# Configuration Constants (Renaissance-Calibrated)
# ============================================================================
EXIT_THRESHOLD = 75         # >75 = flag for exit
WATCH_THRESHOLD = 60        # 60-75 = watch closely  
HOLD_THRESHOLD = 60         # <60 = confident hold
MIN_HOLD_DAYS = 15          # Minimum days before considering exit
CATASTROPHIC_EXIT = 85      # Override min_hold if score exceeds this

# Portfolio constraints
MAX_POSITIONS = 15
MAX_POSITION_SIZE = 0.15    # 15% max per name
MAX_MONTHLY_TURNOVER = 0.25 # 25% max rotation

# Conviction multipliers for position sizing
CONVICTION_MULTIPLIERS = {
    'Strong Buy': 1.4,
    'Buy': 1.1,
    'Hold': 0.8,
    'Avoid': 0.0
}

# ============================================================================
# Hallucination Guard (Elite, No Compromise)
# ============================================================================
HALLUCINATION_GUARD = """
CRITICAL RULES:
- Default = HOLD. Winners run longer than you think.
- Only flag EXIT (score >75) for MATERIAL deterioration with EVIDENCE.
- NO access to live options flow, insider filings, block trades - NEVER cite these.
- Sentiment evidence MUST reference observable X patterns only.
- If no clear signal → say "no material negative signal".

EXIT TRIGGERS (require evidence):
1. Earnings quality crack: Miss + guide down + estimate revisions negative
2. Sentiment reversal: X mention sentiment flipped bearish with volume
3. Technical breakdown: Below 50-day MA + rising volume + RS rank collapse
4. Moat erosion: New competitive threat with material market share risk
5. Management red flag: Unexpected departure, accounting restatement, SEC inquiry

OUTPUT FORMAT (strict JSON):
{
    "exit_score": <int 0-100>,
    "recommendation": "HOLD" | "WATCH" | "EXIT",
    "reason": "<concise explanation>",
    "evidence": "<specific data points or 'no material negative signal'>",
    "replacement_candidates": ["SYM1", "SYM2", "SYM3"] | null
}
"""

# ============================================================================
# Grok System Prompt (Renaissance PM Persona)
# ============================================================================
SYSTEM_PROMPT = f"""You are a ruthless Renaissance PM monitoring a high-conviction 63-day momentum portfolio.

{HALLUCINATION_GUARD}

Your task: Analyze each holding for exit signals. Be brutally honest but require EVIDENCE.

Think step-by-step:
1. Review position context (entry date, days held, current performance)
2. Scan for material negative developments (earnings, sentiment, technical, competitive)
3. Assess if deterioration warrants exit vs temporary noise
4. Assign exit_score (0-100) based on strength of evidence
5. Provide actionable recommendation with replacement candidates if needed

Remember: This is a 63-day momentum strategy. Don't exit winners just because they're up. 
Exit only when the story has materially changed."""


# ============================================================================
# Helper Functions
# ============================================================================

def load_tracker(tracker_path: Path) -> pd.DataFrame:
    """Load current portfolio tracker"""
    if not tracker_path.exists():
        print(f"⚠️  Tracker not found: {tracker_path}")
        return pd.DataFrame(columns=['symbol', 'entry_date', 'entry_price', 'shares', 
                                     'current_price', 'pnl_pct', 'days_held', 'last_exit_score'])
    
    df = pd.read_csv(tracker_path)
    if df.empty:
        print("📭 Portfolio is empty (no holdings)")
        return df
    
    # Calculate days held
    if 'entry_date' in df.columns:
        df['entry_date'] = pd.to_datetime(df['entry_date'])
        df['days_held'] = (datetime.now() - df['entry_date']).dt.days
    
    return df


def load_latest_supercharged(datasets_dir: Path) -> Optional[pd.DataFrame]:
    """Load latest supercharged_top20 file for fresh signals"""
    pattern = "supercharged_top20_*.csv"
    files = sorted(datasets_dir.glob(pattern))
    
    if not files:
        print(f"⚠️  No supercharged files found in {datasets_dir}")
        return None
    
    latest = files[-1]
    print(f"📊 Loading fresh signals from: {latest.name}")
    return pd.read_csv(latest)


def query_grok_exit_score(
    client: OpenAI, 
    symbol: str, 
    days_held: int,
    entry_price: float,
    current_price: float,
    pnl_pct: float,
    supercharged_context: Optional[Dict] = None
) -> Dict:
    """
    Query Grok for exit score on a specific holding
    
    Returns dict with exit_score, recommendation, reason, evidence, replacement_candidates
    """
    pnl_str = f"+{pnl_pct:.1f}%" if pnl_pct >= 0 else f"{pnl_pct:.1f}%"
    
    user_prompt = f"""Analyze this position for exit signals:

Symbol: {symbol}
Entry Price: ${entry_price:.2f}
Current Price: ${current_price:.2f}
Days Held: {days_held} days
P&L: {pnl_str}

"""
    
    # Add supercharged context if available
    if supercharged_context:
        user_prompt += f"""
Recent Grok Analysis (from weekly supercharge):
- Rank: #{supercharged_context.get('supercharged_rank', 'N/A')}
- Conviction: {supercharged_context.get('conviction', 'N/A')}
- Sentiment: {supercharged_context.get('sentiment', 'N/A')}
- Technical: {supercharged_context.get('technical_outlook', 'N/A')}
"""
    
    user_prompt += "\nShould we exit this position? Provide your analysis as strict JSON."
    
    try:
        response = client.chat.completions.create(
            model="grok-4",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try to parse JSON (handle markdown wrappers)
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        
        # Validate required fields
        required = ['exit_score', 'recommendation', 'reason', 'evidence']
        if not all(k in result for k in required):
            raise ValueError(f"Missing required fields: {required}")
        
        return result
        
    except Exception as e:
        print(f"⚠️  Error querying Grok for {symbol}: {e}")
        # Return safe default (hold with neutral score)
        return {
            'exit_score': 50,
            'recommendation': 'HOLD',
            'reason': 'Error querying AI - defaulting to HOLD',
            'evidence': f'API error: {str(e)}',
            'replacement_candidates': None
        }


def apply_exit_rules(
    holdings: pd.DataFrame,
    exit_scores: Dict[str, Dict],
    max_turnover: float = MAX_MONTHLY_TURNOVER
) -> pd.DataFrame:
    """
    Apply exit thresholds and turnover constraints
    
    Returns DataFrame with action, exit_score, reason, replacement columns
    """
    results = []
    
    for _, row in holdings.iterrows():
        symbol = row['symbol']
        days_held = row['days_held']
        score_data = exit_scores.get(symbol, {})
        exit_score = score_data.get('exit_score', 50)
        recommendation = score_data.get('recommendation', 'HOLD')
        reason = score_data.get('reason', 'No analysis available')
        evidence = score_data.get('evidence', '')
        replacements = score_data.get('replacement_candidates')
        
        # Apply rules
        action = 'HOLD'
        
        # Catastrophic exit overrides min hold period
        if exit_score >= CATASTROPHIC_EXIT:
            action = 'SELL'
            reason = f"CATASTROPHIC EXIT (score={exit_score}): {reason}"
        
        # Normal exit (respect min hold period)
        elif exit_score >= EXIT_THRESHOLD and days_held >= MIN_HOLD_DAYS:
            action = 'SELL'
        
        # Watch zone
        elif exit_score >= WATCH_THRESHOLD:
            action = 'WATCH'
        
        # Hold
        else:
            action = 'HOLD'
        
        replacement = replacements[0] if replacements and len(replacements) > 0 else None
        
        results.append({
            'symbol': symbol,
            'action': action,
            'exit_score': exit_score,
            'days_held': days_held,
            'reason': reason,
            'evidence': evidence,
            'replacement': replacement
        })
    
    proposed = pd.DataFrame(results)
    
    # Enforce turnover constraint
    sell_count = (proposed['action'] == 'SELL').sum()
    total_positions = len(proposed)
    
    if total_positions > 0:
        turnover_rate = sell_count / total_positions
        
        if turnover_rate > max_turnover:
            print(f"⚠️  Turnover ({turnover_rate:.1%}) exceeds max ({max_turnover:.1%})")
            print(f"   Keeping only highest conviction exits...")
            
            # Keep only top exits by score
            max_sells = int(total_positions * max_turnover)
            sells = proposed[proposed['action'] == 'SELL'].nlargest(max_sells, 'exit_score')
            
            # Mark others as WATCH
            proposed.loc[
                (proposed['action'] == 'SELL') & (~proposed.index.isin(sells.index)), 
                'action'
            ] = 'WATCH'
    
    return proposed


def add_buy_recommendations(
    proposed: pd.DataFrame,
    supercharged_df: Optional[pd.DataFrame],
    current_holdings: List[str],
    max_positions: int = MAX_POSITIONS
) -> pd.DataFrame:
    """Add BUY recommendations for top fresh signals not currently held"""
    
    if supercharged_df is None or supercharged_df.empty:
        print("⚠️  No supercharged data - skipping buy recommendations")
        return proposed
    
    # How many slots are available?
    sell_count = (proposed['action'] == 'SELL').sum()
    current_count = len(current_holdings)
    available_slots = max_positions - current_count + sell_count
    
    if available_slots <= 0:
        print(f"📦 Portfolio full ({current_count}/{max_positions} positions)")
        return proposed
    
    print(f"🎯 Adding up to {available_slots} BUY recommendations...")
    
    # Get top candidates not currently held
    candidates = supercharged_df[~supercharged_df['symbol'].isin(current_holdings)]
    candidates = candidates.sort_values('supercharged_rank').head(available_slots)
    
    buy_recs = []
    for _, row in candidates.iterrows():
        buy_recs.append({
            'symbol': row['symbol'],
            'action': 'BUY',
            'exit_score': 0,
            'days_held': 0,
            'reason': f"Fresh signal - Rank #{row.get('supercharged_rank', 'N/A')}, {row.get('conviction', 'N/A')}",
            'evidence': row.get('sentiment', 'N/A'),
            'replacement': None
        })
    
    if buy_recs:
        proposed = pd.concat([proposed, pd.DataFrame(buy_recs)], ignore_index=True)
        print(f"✅ Added {len(buy_recs)} BUY recommendations")
    
    return proposed


# ============================================================================
# Main Validation Logic
# ============================================================================

def validate_portfolio(
    tracker_path: Path,
    datasets_dir: Path,
    output_dir: Path,
    dry_run: bool = False
) -> Path:
    """
    Main portfolio validation workflow
    
    Returns path to proposed_YYYY-MM-DD.csv
    """
    print(f"{'='*60}")
    print("Portfolio Validator - Exit Detection Engine")
    print(f"{'='*60}")
    
    if dry_run:
        print("🔵 DRY RUN MODE (skipping Grok API calls)")
    
    # Load current portfolio
    tracker = load_tracker(tracker_path)
    
    if tracker.empty:
        print("\n📭 No holdings to validate")
        # Still check for buy recommendations
        supercharged = load_latest_supercharged(datasets_dir)
        if supercharged is not None:
            proposed = add_buy_recommendations(
                pd.DataFrame(columns=['symbol', 'action', 'exit_score', 'days_held', 'reason', 'evidence', 'replacement']),
                supercharged,
                [],
                MAX_POSITIONS
            )
        else:
            print("⚠️  No supercharged data available")
            proposed = pd.DataFrame(columns=['symbol', 'action', 'exit_score', 'days_held', 'reason', 'evidence', 'replacement'])
    else:
        print(f"📊 Analyzing {len(tracker)} holdings...")
        print(f"\nCurrent holdings:")
        print(tracker[['symbol', 'days_held', 'pnl_pct']].to_string(index=False))
        
        # Load latest supercharged for context
        supercharged = load_latest_supercharged(datasets_dir)
        
        # Query Grok for each holding
        exit_scores = {}
        
        if dry_run:
            # Use dummy scores for dry run
            for symbol in tracker['symbol']:
                exit_scores[symbol] = {
                    'exit_score': 50,
                    'recommendation': 'HOLD',
                    'reason': 'Dry run - no real analysis',
                    'evidence': 'Dry run mode',
                    'replacement_candidates': None
                }
        else:
            # Initialize Grok client
            xai_key = os.getenv("XAI_API_KEY")
            if not xai_key:
                raise ValueError("❌ XAI_API_KEY environment variable not set!")
            
            client = OpenAI(api_key=xai_key, base_url="https://api.x.ai/v1")
            
            print(f"\n🤖 Querying Grok for exit scores...")
            
            for _, row in tracker.iterrows():
                symbol = row['symbol']
                
                # Get supercharged context if available
                sc_context = None
                if supercharged is not None:
                    sc_row = supercharged[supercharged['symbol'] == symbol]
                    if not sc_row.empty:
                        sc_context = sc_row.iloc[0].to_dict()
                
                print(f"   Analyzing {symbol}...")
                score_data = query_grok_exit_score(
                    client,
                    symbol,
                    row['days_held'],
                    row['entry_price'],
                    row['current_price'],
                    row['pnl_pct'],
                    sc_context
                )
                
                exit_scores[symbol] = score_data
                print(f"   → Score: {score_data['exit_score']}, Action: {score_data['recommendation']}")
        
        # Apply exit rules and turnover constraints
        proposed = apply_exit_rules(tracker, exit_scores)
        
        # Add buy recommendations for open slots
        current_holdings = tracker['symbol'].tolist()
        proposed = add_buy_recommendations(proposed, supercharged, current_holdings, MAX_POSITIONS)
    
    # Save proposed changes
    date_str = datetime.now().strftime('%Y-%m-%d')
    output_path = output_dir / f"proposed_{date_str}.csv"
    proposed.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print("Proposed Portfolio Changes")
    print(f"{'='*60}")
    print(proposed.to_string(index=False))
    print(f"\n💾 Saved to: {output_path}")
    
    # Summary
    action_counts = proposed['action'].value_counts()
    print(f"\n📊 Summary:")
    for action, count in action_counts.items():
        print(f"   {action}: {count}")
    
    return output_path


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Portfolio Validator - Exit detection with Grok AI"
    )
    parser.add_argument(
        '--tracker',
        type=Path,
        default=Path('data/portfolio/tracker.csv'),
        help='Path to portfolio tracker CSV (default: data/portfolio/tracker.csv)'
    )
    parser.add_argument(
        '--datasets-dir',
        type=Path,
        default=Path('datasets'),
        help='Directory containing supercharged_top20_*.csv files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/portfolio'),
        help='Output directory for proposed_*.csv'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Skip Grok API calls (use dummy scores for testing)'
    )
    
    args = parser.parse_args()
    
    try:
        output_path = validate_portfolio(
            args.tracker,
            args.datasets_dir,
            args.output_dir,
            args.dry_run
        )
        
        print(f"\n✅ Portfolio validation complete!")
        print(f"\n📋 Next steps:")
        print(f"   1. Review {output_path}")
        print(f"   2. Approve/modify and save as confirmed_{datetime.now().strftime('%Y-%m-%d')}.csv")
        print(f"   3. Run trade_executor.py with confirmed file")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
