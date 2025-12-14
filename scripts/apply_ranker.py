#!/usr/bin/env python3
"""
Apply a trained LightGBM LambdaRank model (from train_alpha_ranker.py artifacts)
to any compatible dataset and write predictions CSV.

- Reads metrics.json to get feature list and preprocessing flags (winsor/zscore).
- Performs the same per-date winsorization and z-scoring, in-place, on the OOU dataset.
- Outputs: <out_dir>/predictions.csv with columns [symbol, date, pred, excess_63d (if present)].
"""
import argparse
import json
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import lightgbm as lgb

def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        s = df["date"]
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "date", "Date": "date"})
        s = df["date"]
    elif "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
        s = df["date"]
    else:
        df = df.reset_index().rename(columns={"index": "date"})
        s = df["date"]
    df["date"] = pd.to_datetime(s, errors="coerce").dt.normalize()
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)

def winsorize_inplace(df: pd.DataFrame, feat_cols: List[str], q: float) -> None:
    if q <= 0 or not feat_cols:
        return
    for _, g in df.groupby("date"):
        lo = g[feat_cols].quantile(q)
        hi = g[feat_cols].quantile(1 - q)
        clipped = g[feat_cols].clip(lower=lo, upper=hi, axis=1)
        df.loc[g.index, feat_cols] = clipped.astype(np.float32).values

def zscore_inplace(df: pd.DataFrame, feat_cols: List[str]) -> None:
    if not feat_cols:
        return
    for _, g in df.groupby("date"):
        mu = g[feat_cols].mean()
        sd = g[feat_cols].std(ddof=0).replace(0, np.nan)
        z = (g[feat_cols] - mu) / sd
        z = z.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
        df.loc[g.index, feat_cols] = z.values

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Parquet or CSV with features and columns [date, symbol, feat_*]")
    ap.add_argument("--model-dir", required=True, help="Directory with model.txt and metrics.json from training")
    ap.add_argument("--out-dir", required=True, help="Directory to write predictions.csv")
    ap.add_argument("--start-date", type=str, default=None, help="Optional YYYY-MM-DD start date filter")
    ap.add_argument("--end-date", type=str, default=None, help="Optional YYYY-MM-DD end date filter")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    p = Path(args.dataset)
    if p.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    if "symbol" not in df.columns:
        raise SystemExit("Dataset must contain 'symbol' column.")
    df = ensure_date_column(df)

    if args.start_date:
        df = df[df["date"] >= pd.Timestamp(args.start_date)]
    if args.end_date:
        df = df[df["date"] <= pd.Timestamp(args.end_date)]
    if df.empty:
        raise SystemExit("No rows after date filtering.")

    # Load model + metrics
    mdir = Path(args.model_dir)
    model = lgb.Booster(model_file=str(mdir / "model.txt"))
    with open(mdir / "metrics.json", "r") as f:
        info = json.load(f)

    feat_cols = info.get("features", [])
    if not feat_cols:
        # fallback: all feat_* numeric columns
        feat_cols = [c for c in df.columns if c.startswith("feat_") and pd.api.types.is_numeric_dtype(df[c])]
    missing = [c for c in feat_cols if c not in df.columns]
    for c in missing:
        df[c] = 0.0
    # Cast in order and float32
    df[feat_cols] = df[feat_cols].astype(np.float32)

    # Per-date preprocessing to match training
    if info.get("winsor", False):
        winsorize_inplace(df, feat_cols, float(info.get("winsor_q", 0.01)))
    if info.get("xsec_z", False):
        zscore_inplace(df, feat_cols)

    # Predict
    preds = model.predict(df[feat_cols], num_iteration=model.best_iteration if getattr(model, "best_iteration", 0) else None)
    out = df[["symbol", "date"]].copy()
    out["pred"] = preds
    if "excess_63d" in df.columns:
        out["excess_63d"] = pd.to_numeric(df["excess_63d"], errors="coerce")

    out_path = out_dir / "predictions.csv"
    out.sort_values(["date", "pred"], ascending=[True, False]).to_csv(out_path, index=False)
    print("Wrote predictions:", out_path)

if __name__ == "__main__":
    main()
