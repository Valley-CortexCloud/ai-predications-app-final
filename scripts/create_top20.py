#!/usr/bin/env python3
"""
Create Top 20 portfolio from predictions.

Usage:
    python scripts/create_top20.py [--predictions predictions_YYYY-MM-DD.csv] [--output datasets/]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

DATASETS_DIR = Path(__file__).parent.parent / "datasets"


def create_top20(predictions_file: Path, output_dir: Path, min_liquidity: float = 10_000_000,
                min_price: float = 15.0, max_price: float = 3000.0) -> Path:
    """
    Create Top 20 portfolio from predictions.
    
    Args:
        predictions_file: Path to predictions CSV
        output_dir: Output directory
        min_liquidity: Minimum ADV20 dollar volume
        min_price: Minimum price
        max_price: Maximum price
        
    Returns:
        Path to Top 20 CSV
    """
    print(f"{'='*60}")
    print("Creating Top 20 Portfolio")
    print(f"{'='*60}")
    print(f"Input: {predictions_file}")
    
    # Load predictions
    pred_df = pd.read_csv(predictions_file)
    print(f"Loaded {len(pred_df)} predictions")
    
    # Filter and sort
    valid = pred_df.copy()
    
    # Apply filters
    initial_count = len(valid)
    
    if 'adv20_dollar' in valid.columns:
        valid = valid[valid['adv20_dollar'] >= min_liquidity]
        print(f"After liquidity filter (>= ${min_liquidity:,.0f}): {len(valid)} symbols")
    
    if 'price' in valid.columns:
        valid = valid[(valid['price'] >= min_price) & (valid['price'] <= max_price)]
        print(f"After price filter (${min_price}-${max_price}): {len(valid)} symbols")
    
    # Sort by prediction
    valid = valid.sort_values('pred', ascending=False)
    
    # Top 20
    top20 = valid.head(20)[['symbol', 'pred']].reset_index(drop=True)
    top20.index += 1  # 1-based indexing
    
    # Determine output filename
    if predictions_file.stem.startswith('predictions_'):
        date = predictions_file.stem.split('_')[1]
    else:
        from datetime import datetime
        date = datetime.now().strftime('%Y-%m-%d')
    
    output_path = output_dir / f"top20_{date}.csv"
    top20.to_csv(output_path)
    
    print(f"\n{'='*60}")
    print("Top 20 Portfolio")
    print(f"{'='*60}")
    print(top20.to_string())
    print(f"{'='*60}")
    print(f"\nSaved to: {output_path}")
    print(f"Prediction range: {top20['pred'].iloc[-1]:.6f} to {top20['pred'].iloc[0]:.6f}")
    
    if top20['pred'].iloc[0] < 0:
        print(f"\nℹ️  All predictions negative (normal for ranking models)")
        print(f"   Higher (less negative) = better performance")
    
    return output_path


def find_latest_predictions(datasets_dir: Path) -> Path:
    """Find the most recent predictions file"""
    pred_files = sorted(datasets_dir.glob('predictions_*.csv'))
    
    if not pred_files:
        # Try default name
        default_pred = datasets_dir / 'predictions.csv'
        if default_pred.exists():
            return default_pred
        raise FileNotFoundError(f"No predictions found in {datasets_dir}")
    
    return pred_files[-1]


def parse_args():
    parser = argparse.ArgumentParser(description='Create Top 20 portfolio from predictions')
    parser.add_argument('--predictions', type=str, default=None,
                       help='Path to predictions CSV (default: latest in datasets/)')
    parser.add_argument('--output-dir', type=str, default=str(DATASETS_DIR),
                       help='Output directory')
    parser.add_argument('--min-liquidity', type=float, default=10_000_000,
                       help='Minimum ADV20 dollar volume (default: 10M)')
    parser.add_argument('--min-price', type=float, default=15.0,
                       help='Minimum price (default: $15)')
    parser.add_argument('--max-price', type=float, default=3000.0,
                       help='Maximum price (default: $3000)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine predictions file
    if args.predictions:
        predictions_file = Path(args.predictions)
    else:
        try:
            predictions_file = find_latest_predictions(output_dir)
            print(f"Using latest predictions: {predictions_file.name}\n")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return 1
    
    if not predictions_file.exists():
        print(f"❌ Predictions file not found: {predictions_file}")
        return 1
    
    # Create Top 20
    try:
        top20_path = create_top20(
            predictions_file,
            output_dir,
            min_liquidity=args.min_liquidity,
            min_price=args.min_price,
            max_price=args.max_price
        )
        return 0
    except Exception as e:
        print(f"\n❌ Failed to create Top 20: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
