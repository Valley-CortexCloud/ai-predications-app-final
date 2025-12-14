#!/usr/bin/env python3
"""
Build earnings calendar CSV with EPS actual/estimate/surprise and release timing.

Output CSV columns:
  - symbol
  - earnings_date (YYYY-MM-DD)
  - trading_date  (YYYY-MM-DD)   [session when info is tradable; pre=possibly same day, else next session]
  - source (finnhub|fmp|alphavantage|yahoo|nasdaq)
  - confidence (actual|estimated)  [actual if eps_actual present or date <= today]
  - release_time (pre|post|during|unknown)
  - eps_actual (float)
  - eps_estimate (float)
  - eps_surprise_pct (float)      [(actual - estimate)/|estimate|, provider value if available]

New flags:
  --tickers AAPL or --tickers AAPL,MU           (test specific tickers)
  --provider-order finnhub,fmp,alphavantage,yahoo  (choose provider priority)
  --disable-yahoo                               (skip Yahoo dates-only fallback)
  --require-eps                                 (drop rows missing EPS actual or estimate)

Provider notes:
  - Finnhub (recommended): FINNHUB_API_KEY
  - FMP: FMP_API_KEY; legacy endpoints deprecated; use earning_calendar or earnings-surprises
  - Alpha Vantage (free ~5/min, ~25/day): ALPHAVANTAGE_API_KEY
  - Yahoo: dates-only fallback; often blocked
  - Nasdaq: optional sweep (--use-nasdaq-fallback) to fill dates/some EPS

Examples:
  export FINNHUB_API_KEY=...
  export FMP_API_KEY=...
  export ALPHAVANTAGE_API_KEY=...

  # Test a few tickers with Finnhub+AV, no Yahoo pollution, keep only rows with EPS
  python3 scripts/build_earnings_calendar.py --tickers AAPL,MU --start 2020-01-01 \
    --out data/earnings_test.csv --provider-order finnhub,alphavantage --disable-yahoo --require-eps --verbose
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple, Dict, Any
import pandas as pd
import numpy as np
import datetime as dt
import sys
import time

# Project root to import CACHE_DIR
sys.path.insert(0, str(Path(__file__).parent.parent))
from pathlib import Path

# ADD at top
ROOT = Path(__file__).parent.parent
TICKER_CACHE_DIR = ROOT / "data_cache" / "10y_ticker_features"
ETF_CACHE_DIR = ROOT / "data_cache" / "_etf_cache"
# -------- Utilities --------

# Add near the other utils (only once; skip if you already added them)
def finnhub_symbol_norm(sym: str) -> str:
    """
    Finnhub uses dot for share classes (e.g., BRK.B). Many lists use hyphen (BRK-B).
    Convert Yahoo-style class tickers to Finnhub style; otherwise return uppercased symbol.
    """
    s = str(sym).upper().strip()
    if "-" in s:
        parts = s.split("-")
        if len(parts) == 2 and parts[0] and parts[1] and parts[1].isalnum():
            s = parts[0] + "." + parts[1]
    return s

def _http_get_with_backoff(url: str, params: Dict[str, Any], headers: Dict[str, str] | None = None, max_retries: int = 3, verbose: bool = False):
    import requests, time as _time
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=20, headers=headers or {"User-Agent": "Mozilla/5.0"})
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                wait = float(ra) if ra else (3.0 + attempt * 2.0)
                if verbose:
                    print(f"[HTTP429] Backing off {wait:.1f}s for {url}")
                _time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                _time.sleep(1.5 + attempt * 1.5)
            else:
                raise last_err

def yahoo_symbol_norm(sym: str) -> str:
    s = str(sym).upper()
    if "." in s and s not in ("GOOG", "GOOGL"):  # e.g., BRK.B -> BRK-B
        s = s.replace(".", "-")
    return s

def parse_tickers_arg(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    toks = [t.strip().upper() for t in str(raw).split(",")]
    return [t for t in toks if t]

def parse_provider_order(raw: Optional[str]) -> List[str]:
    default = ["finnhub", "fmp", "alphavantage", "yahoo"]
    if not raw:
        return default
    xs = [x.strip().lower() for x in str(raw).split(",") if x.strip()]
    valid = {"finnhub","fmp","alphavantage","yahoo"}
    return [x for x in xs if x in valid] or default

def list_symbols_from_cache() -> List[str]:
    root = Path(CACHE_DIR)
    # Use rglob for recursive search, but limit to _10y_ files only
    files = sorted(root.rglob("*_10y_*.parquet"))
    syms = []
    for fp in files:
        if not fp.is_file() or fp.name.endswith("_features.parquet"):
            continue
        sym = fp.stem.split("_")[0].upper()
        if sym == "SPY":
            continue
        syms.append(sym)
    return sorted(set(syms))

def load_symbols(symbols_file: Optional[str]) -> List[str]:
    if not symbols_file:
        return list_symbols_from_cache()
    df = pd.read_csv(symbols_file)
    if "symbol" not in df.columns:
        raise SystemExit(f"{symbols_file} must have a 'symbol' column")
    syms = df["symbol"].astype(str).str.upper().str.strip().tolist()
    # Dedup preserving order
    seen: Set[str] = set()
    out: List[str] = []
    for s in syms:
        if s and s not in seen and s != "SPY":
            seen.add(s)
            out.append(s)
    return out

def _normalize_dates(dts: Iterable[pd.Timestamp]) -> List[pd.Timestamp]:
    xs: List[pd.Timestamp] = []
    for d in dts:
        ts = pd.to_datetime(d, errors="coerce")
        if pd.isna(ts):
            continue
        try:
            if getattr(ts, "tzinfo", None) is not None:
                ts = ts.tz_localize(None)
        except Exception:
            ts = pd.to_datetime(ts.date())
        xs.append(pd.to_datetime(ts).normalize())
    xs = sorted(set(xs))
    return xs

def _compute_surprise_pct(eps_actual: Optional[float], eps_estimate: Optional[float]) -> Optional[float]:
    if eps_actual is None or np.isnan(eps_actual):
        return None
    if eps_estimate is None or np.isnan(eps_estimate):
        return None
    if eps_estimate == 0:
        diff = eps_actual - eps_estimate
        return float(diff) if np.isfinite(diff) else None
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (eps_actual - eps_estimate) / abs(eps_estimate)
    return float(v) if np.isfinite(v) else None

# -------- Providers --------

def try_imports_yf():
    try:
        import yfinance as yf_mod
        return yf_mod
    except Exception:
        return None

def fetch_yahoo_events(symbol: str, start: Optional[dt.date], end: Optional[dt.date], verbose: bool=False) -> List[Dict[str, Any]]:
    """
    Updated yfinance fetch with EPS parsing. Requires latest yfinance (pip install --upgrade yfinance).
    Handles historical and future (via calendar fallback).
    """
    evs: List[Dict[str, Any]] = []
    yf_mod = try_imports_yf()
    if yf_mod is None:
        if verbose:
            print("[Yahoo] yfinance not available.")
        return evs
    limit_sweep = (100, 50, 20, 10)  # Respect max limit
    try:
        t = yf_mod.Ticker(yahoo_symbol_norm(symbol))
        df = None
        for lim in limit_sweep:
            for attempt in range(1, 6):
                try:
                    df = t.get_earnings_dates(limit=lim)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        break
                except Exception as e:
                    err_str = str(e).lower()
                    if '429' in err_str or 'rate limit' in err_str:
                        sleep_sec = 2 ** attempt + 1
                        if verbose:
                            print(f"[Yahoo] 429 for {symbol} (lim={lim}, att={attempt}); sleep {sleep_sec}s")
                        time.sleep(sleep_sec)
                    elif '401' in err_str:
                        if verbose:
                            print(f"[Yahoo] 401 for {symbol}; headers/proxy may be needed.")
                        break
                    else:
                        if verbose:
                            print(f"[Yahoo] Error for {symbol} (lim={lim}): {e}")
                        break
            if isinstance(df, pd.DataFrame) and not df.empty:
                break
            # Scrape fallback if standard fails (in latest yfinance)
            try:
                df = t.get_earnings_dates_using_scrape(limit=lim)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    if verbose:
                        print(f"[Yahoo] Used scrape fallback for {symbol} (lim={lim})")
                    break
            except AttributeError:
                if verbose:
                    print("[Yahoo] Scrape fallback unavailable; update yfinance to 0.2.66+")
            except Exception as e:
                if verbose:
                    print(f"[Yahoo] Scrape fallback error for {symbol}: {e}")

        if not isinstance(df, pd.DataFrame) or df.empty:
            if verbose:
                print(f"[Yahoo] No earnings data for {symbol}")
        else:
            # Parse dates (index or column)
            date_series = None
            if isinstance(df.index, pd.DatetimeIndex) and df.index.notna().any():
                date_series = pd.to_datetime(df.index, errors="coerce")
            else:
                for cand in ("Earnings Date", "EarningsDate", "date", "Date"):
                    if cand in df.columns:
                        date_series = pd.to_datetime(df[cand], errors="coerce")
                        break
            if date_series is not None:
                # EPS columns
                eps_est_col = next((c for c in df.columns if "estimate" in c.lower() and "eps" in c.lower()), None)
                eps_act_col = next((c for c in df.columns if ("reported" in c.lower() or "actual" in c.lower()) and "eps" in c.lower()), None)
                surpr_col = next((c for c in df.columns if "surprise" in c.lower()), None)

                for i in range(len(df)):
                    d_raw = date_series[i] if isinstance(date_series, pd.DatetimeIndex) else date_series.iloc[i]
                    d = pd.to_datetime(d_raw, errors="coerce")
                    if pd.isna(d):
                        continue
                    d = d.normalize()
                    if start and d.date() < start:
                        continue
                    if end and d.date() > end:
                        continue
                    row = df.iloc[i]
                    eps_est = float(row[eps_est_col]) if eps_est_col and pd.notna(row[eps_est_col]) else np.nan
                    eps_act = float(row[eps_act_col]) if eps_act_col and pd.notna(row[eps_act_col]) else np.nan
                    surpr = float(row[surpr_col]) if surpr_col and pd.notna(row[surpr_col]) else _compute_surprise_pct(eps_act, eps_est)
                    evs.append({
                        "symbol": symbol,
                        "earnings_date": d,
                        "source": "yahoo",
                        "eps_actual": eps_act,
                        "eps_estimate": eps_est,
                        "eps_surprise_pct": surpr,
                        "release_time": "unknown",
                    })

        # Fallback for future/upcoming
        if not evs or all(pd.to_datetime(e["earnings_date"]).date() < dt.date.today() for e in evs):
            try:
                cal = t.calendar
                if isinstance(cal, pd.DataFrame) and not cal.empty:
                    dcols = [c for c in cal.columns if "earn" in c.lower()]
                    vals = []
                    for c in dcols:
                        vals.extend(pd.to_datetime(cal[c].astype(str), errors="coerce"))
                    for d in _normalize_dates(vals):
                        if start and d.date() < start:
                            continue
                        if end and d.date() > end:
                            continue
                        evs.append({
                            "symbol": symbol,
                            "earnings_date": d,
                            "source": "yahoo",
                            "eps_actual": np.nan,
                            "eps_estimate": np.nan,
                            "eps_surprise_pct": np.nan,
                            "release_time": "unknown",
                        })
            except Exception as e:
                if verbose:
                    print(f"[Yahoo] Calendar fallback error for {symbol}: {e}")

    except Exception as e:
        if verbose:
            print(f"[Yahoo] General error for {symbol}: {e}")
    return evs

def fetch_finnhub_events(symbol: str, start: Optional[dt.date], end: Optional[dt.date], token: Optional[str], verbose: bool=False) -> List[Dict[str, Any]]:
    """
    Pull announcement dates/timing from calendar endpoint and enrich EPS from company earnings history.
    - Correctly parse /stock/earnings keys: actual, estimate, surprisePercent
    - Request limit=200 to go back to 2020+
    - Map fiscal period to announcement date when available
    """
    evs: List[Dict[str, Any]] = []
    if not token:
        return evs

    fsym = finnhub_symbol_norm(symbol)
    from_iso = (start or dt.date(2010, 1, 1)).isoformat()
    to_iso = (end or dt.date.today()).isoformat()

    # 1) Release dates + timing (bmo/amc/dmh) — build mapping period -> announcement date
    period_to_ann: Dict[pd.Timestamp, pd.Timestamp] = {}
    cal_url = "https://finnhub.io/api/v1/calendar/earnings"
    try:
        r = _http_get_with_backoff(
            cal_url,
            {"symbol": fsym, "from": from_iso, "to": to_iso, "token": token},
            max_retries=3,
            verbose=verbose,
        )
        data = r.json() or {}
        rows = data.get("earningsCalendar") or data.get("data") or []
        for row in rows:
            ann = pd.to_datetime(row.get("date") or row.get("period"), errors="coerce")
            if pd.isna(ann):
                continue
            ann = pd.to_datetime(ann).normalize()
            if start and ann.date() < start:
                continue
            if end and ann.date() > end:
                continue

            # Build mapping from period -> announcement date when Finnhub provides it
            per = pd.to_datetime(row.get("period"), errors="coerce") if row.get("period") else pd.NaT
            if not pd.isna(per):
                period_to_ann[pd.to_datetime(per).normalize()] = ann

            # Calendar sometimes includes EPS
            act = row.get("epsActual")
            est = row.get("epsEstimate")
            try:
                act = float(act) if act is not None and str(act) != "" else np.nan
                est = float(est) if est is not None and str(est) != "" else np.nan
            except Exception:
                act = np.nan
                est = np.nan

            hour = str(row.get("hour") or "").lower()  # bmo=pre, amc=post, dmh=during
            if "bmo" in hour:
                rt = "pre"
            elif "amc" in hour:
                rt = "post"
            elif "dmh" in hour:
                rt = "during"
            else:
                rt = "unknown"

            evs.append({
                "symbol": symbol,
                "earnings_date": ann,
                "source": "finnhub",
                "eps_actual": act,
                "eps_estimate": est,
                "eps_surprise_pct": _compute_surprise_pct(act, est),
                "release_time": rt,
            })
    except Exception as e:
        if verbose:
            print(f"[Finnhub] {symbol} calendar error: {e}")

    # 2) EPS history via /stock/earnings (deeper history, correct keys)
    earn_url = "https://finnhub.io/api/v1/stock/earnings"
    try:
        r2 = _http_get_with_backoff(
            earn_url,
            {"symbol": fsym, "limit": 200, "token": token},
            max_retries=3,
            verbose=verbose,
        )
        js = r2.json() or []
        if isinstance(js, dict):
            js = js.get("earnings") or []
        existing_dates = set(pd.to_datetime(x["earnings_date"]).normalize() for x in evs if "earnings_date" in x)
        for row in js:
            # Finnhub keys: period (YYYY-MM-DD), actual, estimate, surprisePercent
            per = pd.to_datetime(row.get("period"), errors="coerce")
            if pd.isna(per):
                continue
            per = pd.to_datetime(per).normalize()
            if start and per.date() < start:
                continue
            if end and per.date() > end:
                continue

            # Map to announcement date if we have it; else fallback to period (less ideal for trading-time alignment)
            ann = period_to_ann.get(per, per)
            if ann in existing_dates:
                continue

            def f2(x):
                try:
                    return float(x)
                except Exception:
                    return np.nan

            act = f2(row.get("actual"))
            est = f2(row.get("estimate"))
            surpr = f2(row.get("surprisePercent"))
            evs.append({
                "symbol": symbol,
                "earnings_date": ann,
                "source": "finnhub",
                "eps_actual": act,
                "eps_estimate": est,
                "eps_surprise_pct": surpr if np.isfinite(surpr) else _compute_surprise_pct(act, est),
                "release_time": "unknown",
            })
    except Exception as e:
        if verbose:
            print(f"[Finnhub] {symbol} earnings history error: {e}")

    return evs
def fetch_fmp_events(symbol: str, start: Optional[dt.date], end: Optional[dt.date], token: Optional[str], verbose: bool=False) -> List[Dict[str, Any]]:
    """
    FMP current endpoints:
      - Primary: v3/earning_calendar?symbol=...&from=...&to=...
      - Fallback: v3/earnings-surprises?symbol=...&limit=...
    """
    evs: List[Dict[str, Any]] = []
    if not token:
        return evs
    import requests

    s = (start or dt.date(2010, 1, 1)).isoformat()
    e = (end or dt.date.today()).isoformat()

    def f2(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    # 1) New earning_calendar endpoint
    cal_url = "https://financialmodelingprep.com/api/v3/earning_calendar"
    try:
        r = requests.get(
            cal_url,
            params={"symbol": symbol, "from": s, "to": e, "apikey": token},
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if r.status_code == 403:
            if verbose:
                print(f"[FMP] {symbol} forbidden (plan or key).")
            raise requests.HTTPError("403 Forbidden")
        r.raise_for_status()
        js = r.json() or []
        if isinstance(js, dict) and js.get("Error Message"):
            if verbose:
                print(f"[FMP] {symbol} earning_calendar error: {js.get('Error Message')}")
            js = []
        rows = js if isinstance(js, list) else []
        for row in rows:
            d = pd.to_datetime(row.get("date"), errors="coerce")
            if pd.isna(d):
                continue
            d = pd.to_datetime(d).normalize()
            if start and d.date() < start:
                continue
            if end and d.date() > end:
                continue
            act = f2(row.get("eps"))
            est = f2(row.get("epsEstimated") or row.get("epsEstimate"))
            tstr = str(row.get("time") or "").lower()
            if "bmo" in tstr or "before" in tstr:
                rt = "pre"
            elif "amc" in tstr or "after" in tstr:
                rt = "post"
            elif "dmh" in tstr or "during" in tstr:
                rt = "during"
            else:
                rt = "unknown"
            evs.append({
                "symbol": symbol,
                "earnings_date": d,
                "source": "fmp",
                "eps_actual": act,
                "eps_estimate": est,
                "eps_surprise_pct": _compute_surprise_pct(act, est),
                "release_time": rt,
            })
        if evs:
            return evs
    except Exception as ex:
        if verbose:
            print(f"[FMP] {symbol} earning_calendar request failed: {ex}")

    # 2) Fallback: earnings-surprises (historical EPS actual/estimate)
    es_url = "https://financialmodelingprep.com/api/v3/earnings-surprises"
    try:
        r2 = requests.get(
            es_url,
            params={"symbol": symbol, "limit": 200, "apikey": token},
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if r2.status_code == 403:
            if verbose:
                print(f"[FMP] {symbol} earnings-surprises forbidden (plan or key).")
            raise requests.HTTPError("403 Forbidden")
        r2.raise_for_status()
        js2 = r2.json() or []
        if isinstance(js2, dict) and js2.get("Error Message"):
            if verbose:
                print(f"[FMP] {symbol} earnings-surprises error: {js2.get('Error Message')}")
            js2 = []
        rows2 = js2 if isinstance(js2, list) else []
        for row in rows2:
            d = pd.to_datetime(row.get("date"), errors="coerce")
            if pd.isna(d):
                continue
            d = pd.to_datetime(d).normalize()
            if start and d.date() < start:
                continue
            if end and d.date() > end:
                continue
            lc = {k.lower(): k for k in row.keys()}
            act_key = next((lc[k] for k in lc if "actual" in k and "eps" in k), None)
            est_key = next((lc[k] for k in lc if ("estimate" in k or "estimated" in k) and "eps" in k), None)
            act = f2(row.get(act_key)) if act_key else np.nan
            est = f2(row.get(est_key)) if est_key else np.nan
            evs.append({
                "symbol": symbol,
                "earnings_date": d,
                "source": "fmp",
                "eps_actual": act,
                "eps_estimate": est,
                "eps_surprise_pct": _compute_surprise_pct(act, est),
                "release_time": "unknown",
            })
    except Exception as ex:
        if verbose:
            print(f"[FMP] {symbol} earnings-surprises request failed: {ex}")

    return evs

def _av_detect_rate_limit(js: Dict[str, Any]) -> Tuple[bool, str]:
    text = (js.get("Note") or js.get("Information") or js.get("Error Message") or "") if isinstance(js, dict) else ""
    t = str(text).lower()
    if not t:
        return (False, "")
    if "5 requests per minute" in t and "25 requests per day" in t:
        return (True, "minute_or_daily")
    if "requests per minute" in t:
        return (True, "minute")
    if "requests per day" in t or "upgrade" in t or "premium" in t:
        return (True, "daily")
    return (True, "unknown")

def fetch_alpha_vantage_events(
    symbol: str,
    token: Optional[str],
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], bool, bool]:
    """
    Alpha Vantage EARNINGS endpoint: reported/estimated EPS and surprise%.
    Returns (events, rate_limited_flag, daily_limit_flag).
    """
    evs: List[Dict[str, Any]] = []
    if not token:
        return evs, False, False
    import requests
    try:
        url = "https://www.alphavantage.co/query"
        params = {"function": "EARNINGS", "symbol": symbol, "apikey": token}
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        js = r.json() or {}
        limited, kind = _av_detect_rate_limit(js)
        if limited:
            if verbose:
                msg = js.get("Note") or js.get("Information") or js.get("Error Message")
                print(f"[AlphaV] {symbol}: limited ({kind}) -> {msg}")
            # treat 'daily' and 'minute_or_daily' as potentially daily cap
            return [], True, (kind in ("daily","minute_or_daily"))
        q = js.get("quarterlyEarnings") or js.get("quarterly_earnings") or []
        for row in q:
            d = row.get("reportedDate") or row.get("reportDate") or row.get("fiscalDateEnding")
            if not d:
                continue
            dtv = pd.to_datetime(d, errors="coerce")
            if pd.isna(dtv):
                continue
            dtv = dtv.normalize()
            if start and dtv.date() < start:
                continue
            if end and dtv.date() > end:
                continue
            def f2(x):
                try:
                    return float(x)
                except Exception:
                    return np.nan
            act = f2(row.get("reportedEPS") or row.get("epsActual"))
            est = f2(row.get("estimatedEPS") or row.get("epsEstimate"))
            surpr = f2(row.get("surprisePercentage") or row.get("surprise"))
            evs.append({
                "symbol": symbol,
                "earnings_date": dtv,
                "source": "alphavantage",
                "eps_actual": act,
                "eps_estimate": est,
                "eps_surprise_pct": surpr if np.isfinite(surpr) else _compute_surprise_pct(act, est),
                "release_time": "unknown",
            })
    except Exception as e:
        if verbose:
            print(f"[AlphaV] {symbol} error: {e}")
        return evs, False, False
    return evs, False, False

def fetch_nasdaq_daily(date: dt.date) -> List[Dict[str, Any]]:
    # Simple JSON endpoint; requires headers
    import requests
    url = f"https://api.nasdaq.com/api/calendar/earnings?date={date.isoformat()}"
    try:
        r = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json, text/plain, */*",
                "Origin": "https://www.nasdaq.com",
                "Referer": "https://www.nasdaq.com/",
            },
        )
        r.raise_for_status()
        data = r.json()
        rows = (data or {}).get("data", {}).get("rows", []) or []
        out: List[Dict[str, Any]] = []
        for row in rows:
            sym = str(row.get("symbol") or row.get("symbol_r")).upper().strip() if row else None
            d = pd.to_datetime(row.get("date") or row.get("reportDate"), errors="coerce")
            if not sym or pd.isna(d):
                continue
            d = pd.to_datetime(d).normalize()
            act = row.get("eps") or row.get("actualEps")
            est = row.get("epsconsensus") or row.get("consensusEPS") or row.get("epsForecast")
            def f2(x):
                try:
                    return float(x)
                except Exception:
                    return np.nan
            act = f2(act)
            est = f2(est)
            tstr = str(row.get("time") or row.get("when")).lower()
            if "bmo" in tstr or "before" in tstr:
                rt = "pre"
            elif "amc" in tstr or "after" in tstr:
                rt = "post"
            elif "dmh" in tstr or "during" in tstr or "open" in tstr or "close" in tstr:
                rt = "during"
            else:
                rt = "unknown"
            out.append({
                "symbol": sym,
                "earnings_date": d,
                "source": "nasdaq",
                "eps_actual": act,
                "eps_estimate": est,
                "eps_surprise_pct": _compute_surprise_pct(act, est),
                "release_time": rt,
            })
        return out
    except Exception:
        return []

# -------- Trading date normalization --------

def is_business_day_bdays(ts: pd.Timestamp) -> bool:
    return len(pd.bdate_range(ts, ts)) == 1

def make_trading_date_func(mode: str):
    """
    Returns f(ts: pd.Timestamp, release_time: str) -> pd.Timestamp
    - pre: same session if valid day; else next
    - post/during/unknown: next session
    """
    mode = (mode or "bdays").lower()
    if mode == "none":
        def _identity(ts: pd.Timestamp, release_time: str) -> pd.Timestamp:
            return pd.Timestamp(ts).normalize()
        return _identity

    if mode == "nyse":
        try:
            import pandas_market_calendars as pmc  # type: ignore
            cal = pmc.get_calendar("XNYS")
            valid_days_cache: Dict[pd.Timestamp, bool] = {}
            def _is_valid(ts: pd.Timestamp) -> bool:
                tsn = pd.Timestamp(ts).normalize()
                if tsn in valid_days_cache:
                    return valid_days_cache[tsn]
                v = cal.valid_days(start_date=tsn, end_date=tsn)
                ok = len(v) > 0 and pd.Timestamp(v[0]).normalize() == tsn
                valid_days_cache[tsn] = ok
                return ok
            def _nyse_next(ts: pd.Timestamp, release_time: str) -> pd.Timestamp:
                base = pd.Timestamp(ts).normalize()
                if release_time == "pre" and _is_valid(base):
                    return base
                valid = cal.valid_days(start_date=base, end_date=base + pd.Timedelta(days=365))
                next_days = valid[valid > base]
                if len(next_days) == 0:
                    return (base + pd.tseries.offsets.BDay(1)).normalize()
                return pd.Timestamp(next_days[0]).normalize()
            return _nyse_next
        except Exception:
            mode = "bdays"

    def _bdays_next(ts: pd.Timestamp, release_time: str) -> pd.Timestamp:
        base = pd.Timestamp(ts).normalize()
        if release_time == "pre" and is_business_day_bdays(base):
            return base
        return (base + pd.tseries.offsets.BDay(1)).normalize()
    return _bdays_next

def fix_trading_dates(df: pd.DataFrame, normalize_to: str = "bdays") -> pd.DataFrame:
    if "earnings_date" not in df.columns:
        return df.copy()
    out = df.copy()
    out["earnings_date"] = pd.to_datetime(out["earnings_date"], errors="coerce")
    try:
        out["earnings_date"] = out["earnings_date"].dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    out["earnings_date"] = out["earnings_date"].dt.normalize()
    out = out.dropna(subset=["earnings_date"]).copy()

    to_trading = make_trading_date_func(normalize_to)
    if "release_time" not in out.columns:
        out["release_time"] = "unknown"
    out["release_time"] = out["release_time"].fillna("unknown").astype(str).str.lower()

    out["trading_date"] = [
        to_trading(ts, rt if rt in ("pre", "post", "during") else "unknown")
        for ts, rt in zip(out["earnings_date"], out["release_time"])
    ]
    return out

# -------- Dedupe logic --------

def collapse_near_duplicates(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """
    Collapse near-duplicate earnings events per symbol within window_days.
    Ranking preference:
      - confidence: actual > estimated
      - has_eps: both eps_actual and eps_estimate present
      - source quality: finnhub > fmp > alphavantage > yahoo > nasdaq
      - later earnings_date wins ties
    """
    source_rank = {"finnhub": 4, "fmp": 3, "alphavantage": 2, "yahoo": 1, "nasdaq": 0}

    MIN_DATE_SCORE = -9_223_372_036_854_775_808  # very small int64 for missing dates

    def _date_score(ts_val) -> int:
        ts = pd.to_datetime(ts_val, errors="coerce")
        if pd.isna(ts):
            return MIN_DATE_SCORE
        try:
            return int(pd.Timestamp(ts).value)  # ns since epoch as int64
        except Exception:
            return MIN_DATE_SCORE

    def _score_row(r: pd.Series) -> Tuple[int, int, int, int]:
        conf = str(r.get("confidence") or "").lower()
        conf_score = 1 if conf == "actual" else 0
        has_eps = int(pd.notna(r.get("eps_actual")) and pd.notna(r.get("eps_estimate")))
        src = str(r.get("source") or "").lower()
        src_score = source_rank.get(src, -1)
        dscore = _date_score(r.get("earnings_date"))
        return (conf_score, has_eps, src_score, dscore)

    def _dedupe_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("earnings_date").reset_index(drop=True)
        if len(g) <= 1 or window_days <= 0:
            return g.copy()
        clusters: List[pd.DataFrame] = []
        cur_start = 0
        for i in range(1, len(g)):
            if (g.loc[i, "earnings_date"] - g.loc[i - 1, "earnings_date"]).days <= window_days:
                continue
            clusters.append(g.iloc[cur_start:i])
            cur_start = i
        clusters.append(g.iloc[cur_start:])
        picks: List[pd.Series] = []
        for cl in clusters:
            if len(cl) == 1:
                picks.append(cl.iloc[0]); continue
            scores = cl.apply(_score_row, axis=1)
            best_pos = max(range(len(cl)), key=lambda idx: scores.iloc[idx])
            picks.append(cl.iloc[best_pos])
        return pd.DataFrame(picks)

    parts: List[pd.DataFrame] = []
    for _, gg in df.groupby("symbol", sort=False):
        parts.append(_dedupe_group(gg))
    if parts:
        return pd.concat(parts, ignore_index=True)
    return df.copy()

# -------- Main builder --------

def build_earnings_calendar(
    symbols: List[str],
    start: dt.date,
    end: dt.date,
    use_nasdaq_fallback: bool,
    dedupe_window_days: int,
    normalize_to: str,
    provider_order: List[str],
    disable_yahoo: bool,
    require_eps: bool,
    av_min_interval: float = 12.5,
    av_retry_cooldown: float = 65.0,
    av_max_retries: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    yf_mod = try_imports_yf()

    finnhub_key = os.getenv("FINNHUB_API_KEY", "").strip() or None
    fmp_key = os.getenv("FMP_API_KEY", "").strip() or None
    av_key = os.getenv("ALPHAVANTAGE_API_KEY", "").strip() or None

    rows: List[Dict[str, Any]] = []
    last_av_call = 0.0
    enforced_av_interval = max(av_min_interval, 12.5)  # AV free tier safety
    av_daily_exhausted = False

    def try_provider(sym: str, prov: str) -> List[Dict[str, Any]]:
        nonlocal last_av_call, av_daily_exhausted
        if prov == "finnhub":
            return fetch_finnhub_events(sym, start, end, finnhub_key, verbose=verbose) if finnhub_key else []
        if prov == "fmp":
            return fetch_fmp_events(sym, start, end, fmp_key, verbose=verbose) if fmp_key else []
        if prov == "alphavantage":
            if not av_key or av_daily_exhausted:
                return []
            now = time.time()
            sleep_need = enforced_av_interval - (now - last_av_call)
            if sleep_need > 0:
                time.sleep(sleep_need)
            evs, limited, daily_flag = fetch_alpha_vantage_events(sym, av_key, start, end, verbose=verbose)
            last_av_call = time.time()
            retries = 0
            # If limited: one cooldown retry
            while not evs and limited and not daily_flag and retries < av_max_retries:
                if verbose:
                    print(f"[AlphaV] {sym}: minute cap. Cooldown {av_retry_cooldown:.0f}s, retry {retries+1}/{av_max_retries} ...")
                time.sleep(av_retry_cooldown)
                evs, limited, daily_flag = fetch_alpha_vantage_events(sym, av_key, start, end, verbose=verbose)
                last_av_call = time.time()
                retries += 1
            if daily_flag:
                av_daily_exhausted = True
                if verbose:
                    print("[AlphaV] Daily cap reached; skipping AV for remaining symbols.")
            return evs
        if prov == "yahoo":
            if disable_yahoo:
                return []
            return fetch_yahoo_events(sym, start, end, verbose=verbose)
        return []

    for i, sym in enumerate(symbols, 1):
        ev: List[Dict[str, Any]] = []
        for prov in provider_order:
            ev = try_provider(sym, prov)
            if ev:
                break

        if ev:
            rows.extend(ev)
        else:
            print(f"{sym}: No earnings events found (no provider data)")

        if verbose or i % 25 == 0 or i == len(symbols):
            print(f"Per-symbol pass {i}/{len(symbols)} ... ({len(ev)} events for {sym})")

    # Optional Nasdaq sweep to fill remaining symbols
    if use_nasdaq_fallback:
        have_symbols = set(r["symbol"] for r in rows)
        missing_symbols = sorted(set(symbols) - have_symbols)
        if missing_symbols:
            print(f"Nasdaq fallback for {len(missing_symbols)} missing symbols; sweeping weekdays from {start} to {end}.")
            cur = start
            all_rows: List[Dict[str, Any]] = []
            while cur <= end:
                if cur.weekday() < 5:
                    try:
                        dr = fetch_nasdaq_daily(cur)
                        if dr:
                            all_rows.extend(dr)
                    except Exception:
                        pass
                cur = cur + dt.timedelta(days=1)
            if all_rows:
                df_day = pd.DataFrame(all_rows)
                df_day = df_day[df_day["symbol"].isin(missing_symbols)]
                rows.extend(df_day.to_dict("records"))

    if not rows:
        return pd.DataFrame(columns=[
            "symbol","earnings_date","trading_date","source","confidence",
            "release_time","eps_actual","eps_estimate","eps_surprise_pct"
        ])

    df = pd.DataFrame(rows)
    # Normalize dates/types
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce")
    try:
        df["earnings_date"] = df["earnings_date"].dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    df["earnings_date"] = df["earnings_date"].dt.normalize()
    df = df.dropna(subset=["earnings_date"]).copy()

    # Filter by window
    if start:
        df = df[df["earnings_date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["earnings_date"] <= pd.Timestamp(end)]

    # Ensure columns
    for c in ("eps_actual","eps_estimate","eps_surprise_pct"):
        if c not in df.columns:
            df[c] = np.nan
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "release_time" not in df.columns:
        df["release_time"] = "unknown"
    df["source"] = df.get("source", "unknown").astype(str)

    # Confidence
    today = pd.Timestamp(dt.date.today())
    df["confidence"] = np.where(df["eps_actual"].notna() | (df["earnings_date"] <= today), "actual", "estimated")

    # Early dedupe by exact key
    df = df.drop_duplicates(subset=["symbol","earnings_date","source"]).copy()

    # Collapse near duplicates within window
    if dedupe_window_days > 0:
        df = collapse_near_duplicates(df, window_days=dedupe_window_days)

    # Optional: keep only rows that have both EPS (testing)
    if require_eps:
        df = df[df[["eps_actual","eps_estimate"]].notna().all(axis=1)].copy()

    # Compute trading_date with timing-aware logic
    df = fix_trading_dates(df, normalize_to=normalize_to)

    # Final formatting
    df = df.sort_values(["symbol","earnings_date"]).copy()
    df["earnings_date"] = df["earnings_date"].dt.strftime("%Y-%m-%d")
    df["trading_date"] = df["trading_date"].dt.strftime("%Y-%m-%d")

    cols = ["symbol","earnings_date","trading_date","source","confidence","release_time","eps_actual","eps_estimate","eps_surprise_pct"]
    for c in cols:
        if c not in df.columns:
            df[c] = "" if c in ("source","confidence","release_time") else np.nan
    return df[cols]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/earnings.csv")
    ap.add_argument("--symbols-file", type=str, default=None, help="CSV with 'symbol' column (e.g., configs/sector_map.csv)")
    ap.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers for quick testing (e.g., AAPL or AAPL,MU). Overrides --symbols-file if provided.")
    ap.add_argument("--start", type=str, default="2020-01-01")
    ap.add_argument("--end", type=str, default=None, help="Default: today")
    ap.add_argument("--use-nasdaq-fallback", action="store_true", help="Scrape Nasdaq daily if per-symbol providers return little/no data.")
    ap.add_argument("--dedupe-window-days", type=int, default=2, help="Collapse events within N days per symbol; keep best row (default: 2). 0 to disable.")
    ap.add_argument("--normalize-to", type=str, choices=["none", "bdays", "nyse"], default="bdays", help="Trading date normalization (default: bdays).")
    ap.add_argument("--provider-order", type=str, default="finnhub,fmp,alphavantage,yahoo", help="Comma-separated provider priority, e.g., finnhub,alphavantage or alphavantage,yahoo.")
    ap.add_argument("--disable-yahoo", action="store_true", help="Skip Yahoo dates-only fallback.")
    ap.add_argument("--require-eps", action="store_true", help="Drop rows without both eps_actual and eps_estimate (useful for testing).")
    ap.add_argument("--alpha-min-interval", type=float, default=12.5, help="Min seconds between Alpha Vantage calls (free tier ~5/min). Enforced minimum is 12.5s.")
    ap.add_argument("--alpha-retry-cooldown", type=float, default=65.0, help="Cooldown seconds before retrying Alpha Vantage once if rate-limited.")
    ap.add_argument("--alpha-max-retries", type=int, default=1, help="Max retries on Alpha Vantage after minute limit (default 1).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Symbol selection precedence: --tickers > --symbols-file > cache
    if args.tickers:
        symbols = parse_tickers_arg(args.tickers)
        if not symbols:
            raise SystemExit("No valid tickers parsed from --tickers argument.")
    else:
        symbols = load_symbols(args.symbols_file)

    start_dt = pd.to_datetime(args.start).date()
    end_dt = (pd.to_datetime(args.end).date() if args.end else dt.date.today())

    df = build_earnings_calendar(
        symbols=symbols,
        start=start_dt,
        end=end_dt,
        use_nasdaq_fallback=args.use_nasdaq_fallback,
        dedupe_window_days=args.dedupe_window_days,
        normalize_to=args.normalize_to,
        provider_order=parse_provider_order(args.provider_order),
        disable_yahoo=args.disable_yahoo,
        require_eps=args.require_eps,
        av_min_interval=args.alpha_min_interval,
        av_retry_cooldown=args.alpha_retry_cooldown,
        av_max_retries=args.alpha_max_retries,
        verbose=args.verbose,
    )
    if df.empty:
        print("No earnings events found; not writing a file.")
        return
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows across {df['symbol'].nunique()} symbols to {out_path}")

if __name__ == "__main__":
    main()
