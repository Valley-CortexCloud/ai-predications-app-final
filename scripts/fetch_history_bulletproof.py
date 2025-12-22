#!/usr/bin/env python3
"""
Fetch reliable daily OHLCV for many tickers, aligned to NYSE sessions (no off-by-one drift).

Key behavior:
- Universe selection: sp500, nasdaq, explicit --tickers list, or --tickers-file
- Limit or random sample
- Parallel downloads with retries and resume
- Adjusted or unadjusted Close (default: RAW; only sets Close=Adj Close when --adjusted is provided)
- Output files: data_cache/{TICKER}_{period}_{tag}.parquet (e.g., AAPL_5y_adj.parquet or AAPL_5y_raw.parquet)
- Robust handling of yfinance MultiIndex columns (any layout)
- Strict date normalization (tz-naive, session-aligned, no weekend rows)
- Sanity prints per ticker to detect accidental adjustment
- Non-blocking sanity cross-check against Yahoo for last few bars (warns but does not fail run)

Examples:
  # S&P 500, adjusted, parallel, resume:
  python3 scripts/fetch_universe_history.py --universe sp500 --period 5y --adjusted --max-workers 8

  # NASDAQ, unadjusted, first 30:
  python3 scripts/fetch_universe_history.py --universe nasdaq --period 5y --limit 30

  # Explicit tickers (20), random sample of 10:
  python3 scripts/fetch_universe_history.py --tickers AAPL,MSFT,GOOGL,META,NVDA,AMZN,TSLA,AVGO,PEP,KO,JPM,V,MA,HD,COST,LIN,MRK,LLY,UNH,ORCL --sample 10 --seed 42

  # From file, overwrite existing:
  python3 scripts/fetch_universe_history.py --tickers-file tickers.txt --overwrite
"""
import argparse
import concurrent.futures
import os
import random
from io import StringIO
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Try to use repo's centralized cache path if available
CACHE_DIR_DEFAULT = Path("data_cache")
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data_paths import CACHE_DIR as REPO_CACHE_DIR  # optional
    CACHE_DIR_DEFAULT = REPO_CACHE_DIR
except Exception:
    pass

# Optional calendars (prefer exchange_calendars, fallback to pandas_market_calendars, then weekdays)
_CALENDAR = None
try:
    import exchange_calendars as ec
    _CALENDAR = ("exchange_calendars", ec.get_calendar("XNYS"))
except Exception:
    try:
        import pandas_market_calendars as mcal
        _CALENDAR = ("pandas_market_calendars", mcal.get_calendar("XNYS"))
    except Exception:
        _CALENDAR = None


# ----------------------- Universe helpers -----------------------
def fetch_sp500_tickers() -> List[str]:
    """Scrape S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        # FIX: Wrap the HTML string
        df = pd.read_html(StringIO(resp.text))[0]
        syms = df["Symbol"].astype(str).tolist()
        return [s.replace(".", "-").strip().upper() for s in syms]
    except Exception as e:
        print(f"Failed to fetch S&P 500 tickers: {e}")
        return []


def fetch_nasdaq_tickers() -> List[str]:
    """Fetch NASDAQ-listed tickers from nasdaqtrader; fallback to S&P500 on failure."""
    try:
        url = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        symbols = []
        for line in resp.text.splitlines():
            parts = line.split("|")
            if parts and parts[0] not in ("Symbol", "File Creation Time"):
                sym = parts[0].strip().upper()
                if sym:
                    symbols.append(sym)
        return [s.replace(".", "-") for s in symbols]
    except Exception as e:
        print(f"Warning: failed to fetch NASDAQ master list: {e}; falling back to S&P 500")
        return fetch_sp500_tickers()


def resolve_universe(universe: Optional[str], tickers: Optional[str], tickers_file: Optional[str]) -> List[str]:
    if tickers:
        return [t.strip().upper().replace(".", "-") for t in tickers.split(",") if t.strip()]
    if tickers_file:
        p = Path(tickers_file)
        if not p.exists():
            print(f"Tickers file not found: {p}")
            return []
        out = []
        for line in p.read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s.upper().replace(".", "-"))
        return out
    u = (universe or "").lower()
    if u == "sp500":
        return fetch_sp500_tickers()
    if u == "nasdaq":
        return fetch_nasdaq_tickers()
    print("No tickers specified; use --universe {sp500|nasdaq} or --tickers/--tickers-file")
    return []


# ----------------------- Calendar & date utils -----------------------
def _sessions_in_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Return NYSE session dates (tz-naive date-only Timestamps) for [start, end]."""
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()
    if _CALENDAR is None:
        days = pd.date_range(s, e, freq="B")  # business days (weekdays)
        return pd.to_datetime(days.date)
    lib, cal = _CALENDAR
    if lib == "exchange_calendars":
        sessions = cal.sessions_in_range(s, e)  # tz-aware
        return pd.to_datetime(pd.Index(sessions.date))
    else:
        sched = cal.schedule(start_date=s.date(), end_date=e.date())
        return pd.to_datetime(sched.index.date)


def _safe_set_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """Make index tz-naive date-only, deduped and sorted (no day shifts)."""
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, errors="coerce")
    if idx.tz is not None:
        try:
            idx = idx.tz_convert("UTC").tz_localize(None)
        except Exception:
            idx = idx.tz_localize(None)
    idx = idx.normalize()
    df = df.copy()
    df.index = idx
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _auto_correct_minus_one_day(df: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """
    If index dates are consistently one day early (off-calendar), shift such rows +1 day when safe.
    Only shift d -> d+1 if d ∉ sessions, (d+1) ∈ sessions, and (d+1) not already used.
    """
    if len(df) == 0:
        return df
    idx = df.index
    is_session = pd.Index(idx).isin(sessions)
    if is_session.all():
        return df
    df = df.copy()
    taken = set(idx)
    moves = []
    for d, ok in zip(idx, is_session):
        if ok:
            continue
        t = d + pd.Timedelta(days=1)
        if t in sessions and t not in taken:
            moves.append((d, t))
    if not moves:
        return df
    for d, t in moves:
        row = df.loc[[d]].copy()
        df = df.drop(index=d)
        row.index = pd.DatetimeIndex([t])
        df = pd.concat([df, row], axis=0)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


# ----------------------- yfinance flattener -----------------------
def _flatten_yf_columns(df: pd.DataFrame, ticker: Optional[str] = None) -> pd.DataFrame:
    """
    Flatten yfinance DataFrame columns to a single level with OHLCV columns.
    Handles both (Ticker -> OHLCV) and (OHLCV -> Ticker) MultiIndex layouts, plus odd variants.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    # 1) Try selecting by whichever level contains the ticker symbol
    if ticker is not None:
        for lvl in range(df.columns.nlevels):
            try:
                vals = df.columns.get_level_values(lvl)
                if ticker in vals:
                    sub = df.xs(ticker, level=lvl, axis=1, drop_level=True)
                    if isinstance(sub, pd.Series):
                        sub = sub.to_frame()
                    if {"Open", "High", "Low", "Close"}.issubset(set(sub.columns)):
                        return sub
            except Exception:
                pass

    # 2) Reconstruct a single-level OHLCV by picking columns for each "price" key
    price_keys = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    picks = {}
    for p in price_keys:
        try:
            matches = [col for col in df.columns if (isinstance(col, tuple) and p in col)]
            if ticker is not None:
                # Prefer (p, ticker) or (ticker, p) tuples that include our ticker
                matches = [col for col in matches if ticker in col] or matches
            if matches:
                picks[p] = df[matches[0]]
        except Exception:
            pass
    if picks:
        out = pd.concat(picks, axis=1)
        if isinstance(out.columns, pd.MultiIndex):
            try:
                out.columns = out.columns.get_level_values(0)
            except Exception:
                out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
        return out

    # 3) Try dropping levels to see if "Open/High/Low/Close" appear
    for lvl in range(df.columns.nlevels):
        try:
            tmp = df.droplevel(lvl, axis=1)
            if {"Open", "High", "Low", "Close"}.issubset(set(tmp.columns)):
                return tmp
        except Exception:
            pass

    # 4) Last resort: flatten tuples by taking their first element
    try:
        flat = df.copy()
        flat.columns = [c[0] if isinstance(c, tuple) else c for c in flat.columns]
        return flat
    except Exception:
        return df


# ----------------------- Fetch core -----------------------
def fetch_yf_aligned(
    ticker: str,
    start: Optional[str],
    end: Optional[str],
    period: str,
    adjusted: bool,
) -> pd.DataFrame:
    """Fetch 1d OHLCV via yfinance, flatten columns, normalize dates, align to sessions."""
    kwargs = dict(interval="1d", auto_adjust=False, actions=False, progress=False)  # no group_by
    if start or end:
        yfdf = yf.download(ticker, start=start, end=end, **kwargs)
        if ticker == "SPY":  # Debug only for SPY
          print(f"SPY RAW COLUMNS BEFORE FLATTEN: {list(yfdf.columns)}")
          print(f"SPY COLUMN TYPE: {type(yfdf. columns)}")
    else:
        yfdf = yf.download(ticker, period=period, **kwargs)
        if ticker == "SPY":  # Debug only for SPY
          print(f"SPY RAW COLUMNS BEFORE FLATTEN: {list(yfdf.columns)}")
          print(f"SPY COLUMN TYPE: {type(yfdf. columns)}")
        
    if yfdf is None or len(yfdf) == 0:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    yfdf = _flatten_yf_columns(yfdf, ticker=ticker)

    # Ensure expected columns
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c not in yfdf.columns:
            yfdf[c] = np.nan

    # Normalize index to date-only (no tz shifts)
    yfdf = _safe_set_date_index(yfdf)

    # Align to NYSE sessions and auto-correct consistent -1d
    sessions = _sessions_in_range(yfdf.index.min(), yfdf.index.max())
    yfdf = _auto_correct_minus_one_day(yfdf, sessions)
    yfdf = yfdf.loc[yfdf.index.intersection(sessions)].sort_index()

    # Adjusted option (only when requested)
    if adjusted and "Adj Close" in yfdf.columns and yfdf["Adj Close"].notna().any():
        yfdf["Close"] = yfdf["Adj Close"]

    # Final sanity: no weekends
    if (yfdf.index.weekday >= 5).any():
        bad = yfdf.index[yfdf.index.weekday >= 5][:5]
        raise RuntimeError(f"Weekend dates present even after alignment: {list(bad)}")

    return yfdf[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]


# ----------------------- Worker -----------------------
def fetch_and_save(
    ticker: str,
    out_dir: Path,
    period: str,
    start: Optional[str],
    end: Optional[str],
    adjusted: bool,
    overwrite: bool,
    incremental: bool,
    max_retries: int,
    backoff_base: float,
) -> Tuple[str, str]:
    """
    Returns (ticker, status) where status in {"ok","skip","empty","error:<msg>","ok:appended_N_rows"}.
    """
    tag = "adj" if adjusted else "raw"
    out_path = out_dir / f"{ticker}_{period}_{tag}.parquet"
    
    # Incremental mode: append only new data
    if incremental and out_path.exists():
        try:
            existing_df = pd.read_parquet(out_path)
            existing_df = _safe_set_date_index(existing_df)
            
            if len(existing_df) == 0:
                # Empty file, treat as new fetch
                pass
            else:
                last_date = existing_df.index.max()
                
                # Fetch only new data (last_date + 1 day to today)
                start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
                
                if start_date >= end_date:
                    return (ticker, "skip:up-to-date")
                
                # Fetch new data with retries
                df_new = None
                wait = backoff_base
                last_err = None
                for attempt in range(max_retries):
                    try:
                        df_new = fetch_yf_aligned(ticker, start=start_date, end=end_date, 
                                                 period=None, adjusted=adjusted)
                        break
                    except Exception as e:
                        last_err = str(e)
                        time.sleep(wait)
                        wait = min(wait * 2.0, 30.0)
                
                if df_new is None or df_new.empty:
                    return (ticker, "skip:no-new-data")
                
                # Append and deduplicate
                df_combined = pd.concat([existing_df, df_new]).sort_index()
                df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                
                # Validation: check for gaps > 5 trading days
                date_diffs = df_combined.index.to_series().diff()
                max_gap = date_diffs.max().days if len(date_diffs) > 1 else 0
                if max_gap > 7:  # Allow some flexibility for holidays
                    print(f"Warning: {ticker} has gap of {max_gap} days")
                
                # Save combined data
                out_dir.mkdir(parents=True, exist_ok=True)
                df_combined.to_parquet(out_path)
                return (ticker, f"ok:appended_{len(df_new)}_rows")
                
        except Exception as e:
            print(f"Warning: {ticker} incremental failed: {e}, falling back to full fetch")
            # Fall through to normal fetch
    
    # Normal mode or fallback
    if out_path.exists() and not overwrite and not incremental:
        return (ticker, "skip")

    wait = backoff_base
    last_err = None
    for attempt in range(max_retries):
        try:
            df = fetch_yf_aligned(ticker, start=start, end=end, period=period, adjusted=adjusted)
            if df is None or df.empty:
                return (ticker, "empty")

            # Print accidental-adjustment sanity for visibility
            if "Adj Close" in df.columns:
                diff = (df["Close"].astype(float) - df["Adj Close"].astype(float)).abs()
                frac_equal = float((diff < 1e-8).mean())
                mode = "ADJ" if adjusted else "RAW"
                print(f"{ticker} [{mode}] Close==AdjClose fraction={frac_equal:.1%}")

            # Non-blocking sanity cross-check: last few bars vs fresh Yahoo (warn only)
            try:
                end_dt = df.index.max()
                start_dt = end_dt - pd.Timedelta(days=7)
                chk = yf.download(
                    ticker,
                    start=start_dt.date().isoformat(),
                    end=(end_dt + pd.Timedelta(days=1)).date().isoformat(),
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                    progress=False,
                )
                chk = _flatten_yf_columns(chk, ticker=ticker)
                if not isinstance(chk.index, pd.DatetimeIndex):
                    chk.index = pd.to_datetime(chk.index, errors="coerce")
                if chk.index.tz is not None:
                    try:
                        chk.index = chk.index.tz_convert("UTC").tz_localize(None)
                    except Exception:
                        chk.index = chk.index.tz_localize(None)
                chk.index = chk.index.normalize()
                chk = chk[~chk.index.duplicated(keep="last")].sort_index()
                req = ["Open", "High", "Low", "Close"]
                if not set(req).issubset(chk.columns):
                    # Final flatten fallback
                    if isinstance(chk.columns, pd.MultiIndex):
                        try:
                            chk.columns = chk.columns.get_level_values(0)
                        except Exception:
                            chk.columns = [c[0] if isinstance(c, tuple) else c for c in chk.columns]
                if set(req).issubset(chk.columns):
                    idx = df.index.intersection(chk.index)[-3:]
                    if len(idx) >= 1:
                        max_abs = (df.loc[idx, req].astype(float) - chk.loc[idx, req].astype(float)).abs().max().max()
                        if not np.isfinite(max_abs) or max_abs > 2.0:
                            print(f"Warning: {ticker} sanity mismatch on recent bars (max_abs={max_abs:.2f})")
                else:
                    print(f"Warning: {ticker} sanity skipped (unrecognized columns: {list(chk.columns)[:6]})")
            except Exception as se:
                print(f"Warning: {ticker} sanity check failed: {se}")

            # Save
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_path)
            return (ticker, "ok")
        except Exception as e:
            last_err = str(e)
            time.sleep(wait)
            wait = min(wait * 2.0, 30.0)

    return (ticker, f"error:{last_err or 'unknown'}")


# ----------------------- CLI -----------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fetch reliable daily OHLCV for many tickers (session-aligned).")
    # Universe selection
    p.add_argument("--universe", choices=["sp500", "nasdaq"], default=None, help="Universe to fetch")
    p.add_argument("--tickers", default=None, help="Comma-separated list of tickers (overrides --universe)")
    p.add_argument("--tickers-file", default=None, help="File with one ticker per line")
    # Size controls
    p.add_argument("--limit", type=int, default=0, help="Use only first N tickers (0 = all)")
    p.add_argument("--sample", type=int, default=0, help="Random sample size from the resolved universe (0 = disabled)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    # Date/period
    p.add_argument("--start", default=None, help="YYYY-MM-DD (optional)")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (optional)")
    p.add_argument("--period", default="5y", help="yfinance period if --start/--end not given, e.g., 5y, 10y")
    p.add_argument("--adjusted", action="store_true", help="Use adjusted close as Close")
    # IO/exec
    p.add_argument("--out-dir", default=str(CACHE_DIR_DEFAULT), help="Output directory (default: data_cache)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    p.add_argument("--incremental", action="store_true", help="Append new data to existing files (only fetch since last date)")
    p.add_argument("--max-workers", type=int, default=8, help="Parallel workers (threads)")
    p.add_argument("--max-retries", type=int, default=5, help="Max retries per ticker")
    p.add_argument("--backoff-base", type=float, default=1.0, help="Initial backoff seconds")
    return p


def main():
    args = build_arg_parser().parse_args()

    tickers = resolve_universe(args.universe, args.tickers, args.tickers_file)
    if not tickers:
        print("No tickers to process.")
        return 1

    # Limit / sample
    if args.limit and args.limit > 0:
        tickers = tickers[: args.limit]
    if args.sample and args.sample > 0:
        random.seed(args.seed)
        if args.sample < len(tickers):
            tickers = random.sample(tickers, args.sample)

    # Summary
    print(f"Tickers to fetch: {len(tickers)} (adjusted={args.adjusted})")
    print(f"Output dir: {args.out_dir}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parallel fetch
    statuses = []
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futs = []
        for t in tickers:
            fut = ex.submit(
                fetch_and_save,
                t,
                out_dir,
                args.period,
                args.start,
                args.end,
                args.adjusted,
                args.overwrite,
                args.incremental,
                args.max_retries,
                args.backoff_base,
            )
            futs.append(fut)
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            ticker, status = fut.result()
            statuses.append((ticker, status))
            if status == "ok" or status.startswith("ok:"):
                print(f"[{i}/{len(futs)}] ✓ {ticker}" + (f" ({status.split(':')[1]})" if ":" in status else ""))
            elif status == "skip" or status.startswith("skip:"):
                reason = status.split(":", 1)[1] if ":" in status else ""
                print(f"[{i}/{len(futs)}] · {ticker} (skip{': ' + reason if reason else ''})")
            elif status == "empty":
                print(f"[{i}/{len(futs)}] ◦ {ticker} (empty)")
            else:
                print(f"[{i}/{len(futs)}] × {ticker} ({status})")

    elapsed = time.time() - start_time
    ok = sum(1 for _, s in statuses if s == "ok" or s.startswith("ok:"))
    skip = sum(1 for _, s in statuses if s == "skip" or s.startswith("skip:"))
    empty = sum(1 for _, s in statuses if s == "empty")
    err = [f"{t}:{s}" for t, s in statuses if s.startswith("error:")]
    
    # Count appended rows for incremental mode
    appended = [s for _, s in statuses if s.startswith("ok:appended_")]
    total_appended = sum(int(s.split("_")[1]) for s in appended) if appended else 0

    print("\n" + "=" * 60)
    print(f"Done in {elapsed:.1f}s")
    print(f"OK: {ok} | Skipped: {skip} | Empty: {empty} | Errors: {len(err)}")
    if args.incremental and total_appended > 0:
        print(f"Incremental: {total_appended} total rows appended across {len(appended)} tickers")
    if err:
        print("Errors (first 10):")
        for e in err[:10]:
            print(f"  - {e}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
