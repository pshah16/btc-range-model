"""Streamlit app for live hourly BTC next-close prediction.

Run:
    streamlit run btc_hourly_app.py

The app:
  * pulls the latest hourly BTC + macro + Fear&Greed data
  * builds the same features used at training time
  * applies the saved model and emits a next-hour forecast with 95% CI
  * plots the last N hours of actuals plus the forecast
  * auto-refreshes every REFRESH_SECONDS (default 600 = 10 min)
"""
import os, sys, time, warnings, joblib, requests, json
warnings.filterwarnings("ignore")
from datetime import datetime, timezone, timedelta, date as _date
from pathlib import Path

# Make the repo root importable so `from paths import …` works regardless
# of the cwd from which Streamlit is launched.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from paths import (
    HOURLY_MODEL, DAILY_MODEL_CT, CONE_7D_MODEL, DAY_TYPE_MODEL,
    BINANCE_HOURLY_CSV,
    BOOKMARKS_FILE as _BOOKMARKS_PATH, RUNTIME_DIR,
)

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

# ════════════════════════════════════════════════════════════════════════
# CONFIG
ASSETS_PATH      = str(HOURLY_MODEL)
REFRESH_SECONDS  = 60           # auto-refresh interval (1 min — rolling forecast)
LOOKBACK_HOURS   = 24           # how many past hours to show
CACHE_TTL        = 300          # data cache lifetime (seconds)
BAND_PCT         = 0.005        # ±0.5% forecast band (around prediction)
# ════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="BTC Hourly Forecaster", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")
st.title("📈 Bitcoin — Live hourly next-close forecast")
st.caption(
    "Live feed: BTC + ETH + macro (Yahoo) + Fear & Greed (alternative.me). "
    "Model: ridge regression on log-returns. "
    "**Honest framing:** ±3 % accuracy on hourly close is trivial (hit-rate "
    "~100 %); the model's real signal is in **direction accuracy ~54 %** "
    "and **tight CI**."
)

# ──────────────────────────── load model ──────────────────────────────
@st.cache_resource
def load_assets():
    if not os.path.exists(ASSETS_PATH):
        st.error(f"Model artefacts not found at {ASSETS_PATH}.\n"
                 "Run `python train_hourly_model.py` first.")
        st.stop()
    return joblib.load(ASSETS_PATH)

A = load_assets()
model     = A["model"]
sigma     = A["sigma"]
feat_cols = A["feat_cols"]
best_name = A.get("best_name","ridge")


@st.cache_resource
def _training_cutoffs():
    """Return {model_label: train_end_date_or_None} for the 4 artefacts.

    Used by the historical-replay banner to warn the user when their
    picked date falls inside any model's training window — predictions
    for those dates are in-sample fit, not honest out-of-sample forecasts.
    """
    out = {}
    # Hourly: stored as ISO datetime on newer artefacts; fall back to test_start.
    he = A.get("train_end") or A.get("test_start")
    out["hourly close"] = pd.Timestamp(he).normalize() if he else None
    # Daily H/L
    if os.path.exists(str(DAILY_MODEL_CT)):
        try:
            meta = joblib.load(DAILY_MODEL_CT).get("calibration_meta", {})
            out["daily H/L"] = pd.Timestamp(meta.get("train_end")) if meta.get("train_end") else None
        except Exception:
            out["daily H/L"] = None
    # 7-day cone
    if os.path.exists(str(CONE_7D_MODEL)):
        try:
            meta = joblib.load(CONE_7D_MODEL).get("calibration_meta", {})
            out["7-day cone"] = pd.Timestamp(meta.get("train_end")) if meta.get("train_end") else None
        except Exception:
            out["7-day cone"] = None
    # 3-class
    if os.path.exists(str(DAY_TYPE_MODEL)):
        try:
            meta = joblib.load(DAY_TYPE_MODEL).get("calibration_meta", {})
            out["3-class day type"] = pd.Timestamp(meta.get("train_end")) if meta.get("train_end") else None
        except Exception:
            out["3-class day type"] = None
    return out


def render_replay_in_sample_warning(target_date):
    """If `target_date` falls inside any model's training window, show a
    yellow warning explaining the predictions on this date are in-sample
    fit (memorisation), not honest out-of-sample forecasts."""
    if target_date is None:
        return
    td = pd.Timestamp(target_date).normalize()
    affected = []
    for name, end in _training_cutoffs().items():
        if end is not None and td <= end:
            affected.append(f"**{name}** (train ≤ {end.date()})")
    if affected:
        st.warning(
            "⚠️  **In-sample replay.**  You picked **"
            f"{td.date()}** — this date falls inside the training window of: "
            + ", ".join(affected) + ". The predictions and "
            "look-back metrics shown for in-sample dates are MEMORISATION, "
            "not honest forecasts; they will look unrealistically accurate. "
            "Pick a date AFTER each model's `train_end` to see genuine "
            "out-of-sample behaviour."
        )

with st.sidebar:
    st.markdown(
        "**Auto-refresh:** every "
        f"{REFRESH_SECONDS // 60} min. Click **Refresh now** to force.")
    if st.button("Refresh now", use_container_width=True):
        # Clear BOTH caches so retrained joblibs are picked up too — without
        # this, `cache_resource`-decorated loaders (`load_assets`,
        # `_load_cone_7d`, `_load_day_type`) hold onto the previous artefact
        # for the lifetime of the session.
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.markdown(
        "_Hourly BTC bars update every hour; macro and F&G update less often._"
    )
    # Model freshness — train_end of each artefact, so users can tell at a
    # glance which version of each model is live in the UI.
    st.markdown("---")
    st.caption("**Model freshness** (`train_end`)")
    for label, end in _training_cutoffs().items():
        end_str = f"`{end.date()}`" if end is not None else "_unknown_"
        st.caption(f"&bull; {label}: {end_str}", unsafe_allow_html=True)

# ───────────────────────── fetch helpers ──────────────────────────────
def _flat(df, name):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open","High","Low","Close","Volume"]].copy()
    df.columns = [f"{name}_{c.lower()}" for c in df.columns]
    idx = pd.to_datetime(df.index)
    if idx.tz is not None: idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx
    return df[~df.index.duplicated(keep="last")].sort_index()

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_data():
    SYMS = {"btc":"BTC-USD","eth":"ETH-USD","spx":"^GSPC","ndx":"^IXIC",
            "vix":"^VIX","gold":"GC=F","dxy":"DX-Y.NYB","tnx":"^TNX"}
    parts = {}
    for name, sym in SYMS.items():
        parts[name] = _flat(yf.download(sym, period="2y", interval="60m",
                                        progress=False, auto_adjust=False), name)
    btc = parts["btc"]
    grid = pd.date_range(btc.index.min().floor("h"), btc.index.max().floor("h"),
                         freq="h")
    df = pd.DataFrame(index=grid)
    for name, d in parts.items():
        agg = {f"{name}_open":"first", f"{name}_high":"max", f"{name}_low":"min",
               f"{name}_close":"last", f"{name}_volume":"last"}
        d = d.resample("h").agg(agg)
        d[f"{name}_volume"] = d[f"{name}_volume"].replace(0, np.nan)
        d = d.reindex(grid).ffill(limit=168 if name not in ("btc","eth") else 4)
        df = df.join(d)
    df = df.dropna(subset=["btc_close"])

    # Fear & Greed daily, forward-filled to hourly.
    #
    # IMPORTANT (causality): alternative.me's current-day F&G record is
    # re-computed throughout the UTC day. Using the value stamped at
    # 2026-05-21 00:00 UTC at hour 14:00 UTC would be look-ahead, because
    # the value at fetch time reflects information from past 14:00 UTC.
    # We lag by 1 day so hours of date D use D-1's finalised value (this
    # mirrors src/train_hourly_model.py).
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=0", timeout=20).json()
        fng = pd.DataFrame(r["data"])
        fng["dt"]    = pd.to_datetime(fng["timestamp"].astype(int), unit="s").dt.normalize()
        fng["value"] = fng["value"].astype(int)
        fng = fng[["dt","value"]].sort_values("dt").drop_duplicates("dt").set_index("dt")
        fng = fng.shift(1, freq="D")   # ← 1-day causal lag (anti-leak)
        fng_h = fng.reindex(df.index.normalize()).ffill()
        df["fng"] = fng_h["value"].values
    except Exception:
        df["fng"] = 50  # neutral fallback
    df["fng_d24"] = pd.Series(df["fng"].values, index=df.index).diff(24).values
    df["fng_d7"]  = pd.Series(df["fng"].values, index=df.index).diff(24*7).values
    return df

@st.cache_data(ttl=30, show_spinner=False)
def fetch_live_spot():
    """Binance public ticker — true real-time BTC/USDT price (no API key)."""
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/price",
                         params={"symbol":"BTCUSDT"}, timeout=10)
        return float(r.json()["price"]), datetime.now(timezone.utc)
    except Exception:
        return None, None

# ─────────────────────── Date bookmarks ────────────────────────────────
RUNTIME_DIR.mkdir(exist_ok=True)
BOOKMARKS_FILE = str(_BOOKMARKS_PATH)


def load_bookmarks():
    """Return dict of {category: [{"date": "YYYY-MM-DD", "label": ""}, ...]}."""
    if not os.path.exists(BOOKMARKS_FILE):
        return {}
    try:
        with open(BOOKMARKS_FILE) as f:
            data = json.load(f)
        # Normalize legacy formats (list of strings) → list of dicts
        for cat, entries in list(data.items()):
            data[cat] = [(e if isinstance(e, dict) else {"date": e, "label": ""})
                         for e in entries]
        return data
    except Exception:
        return {}


def save_bookmarks(data):
    with open(BOOKMARKS_FILE, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def add_bookmark(category, d, label=""):
    data = load_bookmarks()
    cat = data.setdefault(category, [])
    iso = d.isoformat() if isinstance(d, _date) else str(d)
    # Don't duplicate (same date in same category)
    if not any(e["date"] == iso for e in cat):
        cat.append({"date": iso, "label": label or ""})
        cat.sort(key=lambda e: e["date"])
        save_bookmarks(data)
    return data


def delete_bookmark(category, iso_date):
    data = load_bookmarks()
    if category in data:
        data[category] = [e for e in data[category] if e["date"] != iso_date]
        if not data[category]:
            del data[category]
        save_bookmarks(data)
    return data


# ──────── DAILY H/L forecast (7am-CT day boundary = 12:00 UTC) ────────
# Bar D covers [D 12:00 UTC, D+1 12:00 UTC). Indexed by start date D.
ANCHOR_HOUR_UTC = 12  # 7am CDT (summer) / 6am CST (winter)


def _rebucket_12utc(hourly):
    """Group hourly OHLCV into 24h bars starting at ANCHOR_HOUR_UTC.

    Returns bars indexed by start date D. Drops incomplete bars
    (anything other than 24 hours)."""
    if hourly.empty:
        # Return an empty frame with the correct columns and a DatetimeIndex
        # so callers that check .empty or iterate over .index don't crash.
        g = pd.DataFrame(columns=["open","high","low","close","volume"])
        g.index = pd.DatetimeIndex([], name="bar_start")
        return g
    h = hourly.copy()
    h["bucket"] = (h.index - pd.Timedelta(hours=ANCHOR_HOUR_UTC)).normalize()
    g = h.groupby("bucket").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), n_hours=("close", "size"),
    )
    g = g[g["n_hours"] == 24].drop(columns="n_hours")
    g.index.name = "bar_start"
    return g


@st.cache_data(ttl=600, show_spinner="Fetching BTC hourly from Binance …")
def _fetch_binance_hourly(days_back=None):
    """Return BTC hourly OHLCV with full history.

    Strategy: load `binance_hourly_btc.csv` (saved during training; covers
    2017-08 → save-time), then top-up from the Binance public API for any
    hours after the CSV's last row. This guarantees the daily-forecast
    pipeline can build features for ANY date the historical-tab picker
    allows, while staying fast on the common case (top-up = a few API calls).

    `days_back` is kept for backward compatibility — when set, only the last
    `days_back` days are returned (used by the live `_fetch_daily_raw` if
    the CSV is missing)."""
    CSV_PATH = str(BINANCE_HOURLY_CSV)
    parts = []

    if os.path.exists(CSV_PATH):
        csv_df = pd.read_csv(CSV_PATH, index_col="timestamp_utc", parse_dates=True)
        csv_df.index = csv_df.index.tz_localize(None)
        parts.append(csv_df)
        # Top up: API from the hour AFTER csv's last row → now
        start_ms = int((csv_df.index.max() + pd.Timedelta(hours=1)).timestamp() * 1000)
    else:
        # Fallback: pull ~days_back days from API (covers cold-start cases)
        d = days_back if days_back is not None else 400
        start_ms = int(datetime.now(timezone.utc).timestamp() * 1000) - d * 86400_000

    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    cursor = start_ms
    rows = []
    while cursor < end_ms:
        params = dict(symbol="BTCUSDT", interval="1h",
                      startTime=cursor, limit=1000)
        try:
            r = requests.get("https://api.binance.com/api/v3/klines",
                             params=params, timeout=30)
            batch = r.json()
        except Exception:
            break
        # Guard: Binance returns an error dict (e.g. rate-limit) instead of a
        # list of klines — a non-empty dict would pass `if not batch` but then
        # fail on `batch[-1][0]`.  Also coerce the open-time to int to handle
        # API versions that return timestamps as strings.
        if not batch or not isinstance(batch, list) or not isinstance(batch[-1], (list, tuple)):
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + 3600_000
        time.sleep(0.1)
    if rows:
        cols = ["open_time","open","high","low","close","volume",
                "close_time","qv","n","tb","tq","ig"]
        new_df = pd.DataFrame(rows, columns=cols)
        new_df["ts"] = pd.to_datetime(new_df["open_time"], unit="ms",
                                      utc=True).dt.tz_convert(None)
        for c in ["open","high","low","close","volume"]:
            new_df[c] = new_df[c].astype(float)
        new_df = new_df.set_index("ts")[["open","high","low","close","volume"]]
        parts.append(new_df)

    if not parts:
        # Return with a DatetimeIndex (not the default RangeIndex) so that
        # _rebucket_12utc can safely do `h.index - pd.Timedelta(...)`.
        empty = pd.DataFrame(columns=["open","high","low","close","volume"])
        empty.index = pd.DatetimeIndex([], name="ts")
        return empty
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if days_back is not None and len(df):
        cutoff = df.index.max() - pd.Timedelta(days=days_back)
        df = df.loc[df.index >= cutoff]
    return df


@st.cache_data(ttl=3600*6, show_spinner="Fetching daily macro + on-chain …")
def _fetch_daily_raw():
    """Build the daily-bar DataFrame anchored at 12:00 UTC.

      - BTC OHLCV: rebucketed from Binance hourly into 12:00→12:00 UTC bars.
      - Macro (Yahoo daily, indexed by calendar date): SPX/NDX/VIX/Gold/DXY/TNX/ETH.
        Joined to bar D using calendar date D — macro closes for D are
        published by ~21:00 UTC on day D, well before bar D ends (D+1 12:00).
      - On-chain (blockchain.info, daily UTC): same calendar-date join.
    Cached 6 h."""
    # 1. BTC 12:00-UTC daily bars (full history → any picked date is fully featured)
    btc_hourly = _fetch_binance_hourly()
    if btc_hourly.empty:
        st.error(
            "⚠️ Could not fetch BTC hourly data from Binance. "
            "The Binance API may be temporarily unavailable or rate-limiting this server. "
            "The daily H/L forecast requires this data — please wait a minute and refresh."
        )
        st.stop()
    btc_daily = _rebucket_12utc(btc_hourly).add_prefix("btc_")

    # 2. Macro daily from Yahoo (calendar-date indexed)
    START = (btc_daily.index.min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    END = (datetime.now(timezone.utc).date()
           + pd.Timedelta(days=1).to_pytimedelta()).strftime("%Y-%m-%d")
    SYMS = {"eth":"ETH-USD","spx":"^GSPC","ndx":"^IXIC",
            "vix":"^VIX","gold":"GC=F","dxy":"DX-Y.NYB","tnx":"^TNX"}
    parts = []
    for name, sym in SYMS.items():
        d = yf.download(sym, start=START, end=END, progress=False,
                        auto_adjust=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = [c[0] for c in d.columns]
        d = d[["Open","High","Low","Close","Volume"]].copy()
        d.columns = [f"{name}_{c.lower()}" for c in d.columns]
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        parts.append(d)
    mkt = parts[0]
    for p in parts[1:]: mkt = mkt.join(p, how="outer")

    # 3. On-chain (blockchain.info)
    ONCHAIN = ["hash-rate","difficulty","n-transactions","miners-revenue",
               "n-unique-addresses","transaction-fees-usd","mempool-size",
               "estimated-transaction-volume-usd","market-cap","avg-block-size",
               "cost-per-transaction"]
    oc_parts = []
    for s in ONCHAIN:
        try:
            j = requests.get(f"https://api.blockchain.info/charts/{s}?timespan=all&format=json&sampled=false",
                             timeout=30).json()
            idx = pd.to_datetime([x["x"] for x in j["values"]], unit="s").normalize()
            ser = pd.Series([x["y"] for x in j["values"]], index=idx,
                            name=f"oc_{s.replace('-','_')}")
            ser = ser[~ser.index.duplicated(keep="last")]
            oc_parts.append(ser)
        except Exception:
            pass
    oc = pd.concat(oc_parts, axis=1) if oc_parts else pd.DataFrame()

    # 4. Join — bar D's aux data comes from calendar date D
    df = btc_daily.join(mkt, how="left").join(oc, how="left").sort_index()
    df = df.loc[df["btc_close"].notna()].ffill(limit=5)
    return df


@st.cache_data(ttl=86400, show_spinner="Computing daily H/L forecast …")
def compute_daily_forecast(target_date_iso):
    """Apply the 12:00-UTC (7am-CT) daily model as it was trained.

    `target_date_iso` = ISO date of the bar BEING PREDICTED. That bar covers
    [target 12:00 UTC, target+1 12:00 UTC) = [target 7am CT, target+1 7am CT).
    Only data completed before the target bar starts is used — i.e. bars with
    start date ≤ target − 1 day (the latest such bar closes at target 7am CT,
    which is the target bar's open). The cache key is the target date, so the
    prediction recomputes once per 12:00-UTC rollover automatically."""
    path = str(DAILY_MODEL_CT)
    if not os.path.exists(path):
        return None
    AD = joblib.load(path)
    mh, ml = AD["hi_model"], AD["lo_model"]
    sh, sl = AD["sigma_hi"], AD["sigma_lo"]
    fc = AD["feat_cols"]

    df = _fetch_daily_raw().copy()

    # Truncate to bars that close at or before the target bar's open
    # (target 12:00 UTC = target 7am CT). Those are bars with start ≤ target−1.
    target_date = pd.Timestamp(target_date_iso)
    asof_cutoff = target_date - pd.Timedelta(days=1)
    df = df.loc[df.index <= asof_cutoff]
    if df.empty:
        return None
    df = df.sort_index().ffill(limit=5)

    # 3. Daily features (same as the daily training notebook)
    f = pd.DataFrame(index=df.index)
    c = df["btc_close"]; h = df["btc_high"]; l_ = df["btc_low"]; v = df["btc_volume"]
    ret = np.log(c).diff()
    for k in [1,3,5,7,14,30]: f[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [5,10,20,30]:    f[f"vol_{k}"] = ret.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h-l_),(h-prev_c).abs(),(l_-prev_c).abs()],axis=1).max(axis=1)
    for k in [7,14,30]: f[f"atr_{k}"] = tr.rolling(k).mean()/c
    f["range_today"] = (h-l_)/c
    f["range_ma7"]   = ((h-l_)/c).rolling(7).mean()
    f["range_ma30"]  = ((h-l_)/c).rolling(30).mean()
    f["range_std30"] = ((h-l_)/c).rolling(30).std()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain/loss.replace(0,np.nan)
    f["rsi_14"] = 100 - 100/(1+rs)
    e12 = c.ewm(span=12,adjust=False).mean(); e26 = c.ewm(span=26,adjust=False).mean()
    macd = e12 - e26
    f["macd"]      = macd/c
    f["macd_sig"]  = macd.ewm(span=9,adjust=False).mean()/c
    f["macd_hist"] = (macd-macd.ewm(span=9,adjust=False).mean())/c
    ma20=c.rolling(20).mean(); sd20=c.rolling(20).std()
    f["bb_width"]   = (4*sd20)/ma20
    f["dist_hi_30"] = c/c.rolling(30).max()-1
    f["dist_lo_30"] = c/c.rolling(30).min()-1
    f["dist_hi_90"] = c/c.rolling(90).max()-1
    f["vol_chg_1"]    = np.log(v).diff()
    f["vol_z_20"]     = (np.log(v)-np.log(v).rolling(20).mean())/np.log(v).rolling(20).std()
    f["vol_ma_ratio"] = v/v.rolling(20).mean()
    dow = df.index.dayofweek
    for i in range(6): f[f"dow_{i}"] = (dow==i).astype(float)
    def mret(name, ks=(1,5,20)):
        s = df[f"{name}_close"]
        for k in ks: f[f"{name}_ret_{k}"] = np.log(s).diff(k)
        f[f"{name}_vol_20"] = np.log(s).diff().rolling(20).std()
    for nm in ["spx","ndx","vix","gold","dxy","tnx","eth"]: mret(nm)
    f["btc_spx_corr_30"]  = ret.rolling(30).corr(np.log(df["spx_close"]).diff())
    f["btc_ndx_corr_30"]  = ret.rolling(30).corr(np.log(df["ndx_close"]).diff())
    f["btc_gold_corr_30"] = ret.rolling(30).corr(np.log(df["gold_close"]).diff())
    f["btc_dxy_corr_30"]  = ret.rolling(30).corr(np.log(df["dxy_close"]).diff())
    for col in [x for x in df.columns if x.startswith("oc_")]:
        # Local `s_log` instead of `sl` — `sl` is bound earlier to AD["sigma_lo"]
        # and needed downstream for the LOW CI bands.
        s = df[col].astype(float); s_log = np.log(s.replace(0,np.nan))
        f[f"{col}_d1"]  = s_log.diff(1)
        f[f"{col}_d7"]  = s_log.diff(7)
        f[f"{col}_z30"] = (s_log - s_log.rolling(30).mean())/s_log.rolling(30).std()
    nh, nl = h.shift(-1), l_.shift(-1)
    y_hi = (nh-c)/c; y_lo = (c-nl)/c
    # Smoothed lag features (must match `src/pipeline_ct.py` exactly).
    f["y_hi_ema3"] = y_hi.shift(1).ewm(span=3, adjust=False).mean()
    f["y_lo_ema3"] = y_lo.shift(1).ewm(span=3, adjust=False).mean()
    f["y_hi_ema7"] = y_hi.shift(1).ewm(span=7, adjust=False).mean()
    f["y_lo_ema7"] = y_lo.shift(1).ewm(span=7, adjust=False).mean()
    # Anti-mean-reversion features
    prev_3_hi = h.shift(1).rolling(3).max()
    prev_3_lo = l_.shift(1).rolling(3).min()
    f["above_3d_high"] = (c > prev_3_hi).astype(float)
    f["below_3d_low"]  = (c < prev_3_lo).astype(float)
    f["bo_strength_up"] = (c / prev_3_hi - 1).clip(lower=0)
    f["bo_strength_dn"] = (1 - c / prev_3_lo).clip(lower=0)
    _y_hi_lag = y_hi.shift(1)
    _y_lo_lag = y_lo.shift(1)
    f["y_hi_surprise"] = _y_hi_lag - _y_hi_lag.ewm(span=7, adjust=False).mean()
    f["y_lo_surprise"] = _y_lo_lag - _y_lo_lag.ewm(span=7, adjust=False).mean()
    # LOW-specific / downside regime features
    neg_ret = ret.clip(upper=0)
    f["dn_vol_5"]  = neg_ret.rolling(5).std()
    f["dn_vol_20"] = neg_ret.rolling(20).std()
    sma50 = c.rolling(50).mean()
    f["below_sma50"] = (c < sma50).astype(float)
    f["below_sma50_5d"] = f["below_sma50"].rolling(5).min().fillna(0)

    f = f.replace([np.inf,-np.inf], np.nan)
    F = f[fc].dropna()
    if F.empty:
        return None
    # df was already truncated to bars with start ≤ target − 1 day, so the
    # last row is the as-of bar — the one whose close coincides with the
    # target bar's open (target 12:00 UTC = target 7am CT).
    asof = F.index[-1]
    close_asof = float(c.loc[asof])
    # The target bar covers [target 12:00 UTC, target+1 12:00 UTC).
    target_window_start = target_date + pd.Timedelta(hours=ANCHOR_HOUR_UTC)
    target_window_end   = target_date + pd.Timedelta(days=1, hours=ANCHOR_HOUR_UTC)

    # Robust scalar coercion (assets can come back as numpy 0-d arrays,
    # Series, or pure floats depending on how joblib reloaded them).
    def _scalar(x):
        if hasattr(x, "item"):
            try: return float(x.item())
            except Exception: pass
        if hasattr(x, "iloc"):
            return float(x.iloc[0])
        return float(np.asarray(x).ravel()[0])

    # Predict — either single model or ensemble (mean of constituents),
    # optionally blended with climatological mean offset.
    row = F.loc[[asof]]
    if AD.get("ensemble") and AD.get("constituents"):
        pred_hi_list = [_scalar(c["m_hi"].predict(row)[0]) for c in AD["constituents"]]
        pred_lo_list = [_scalar(c["m_lo"].predict(row)[0]) for c in AD["constituents"]]
        yhi = float(np.mean(pred_hi_list))
        ylo = float(np.mean(pred_lo_list))
        if AD.get("blended") and float(AD.get("alpha", 1.0)) < 1.0:
            a = float(AD["alpha"])
            yhi = a * yhi + (1 - a) * float(AD.get("mu_hi", yhi))
            ylo = a * ylo + (1 - a) * float(AD.get("mu_lo", ylo))
    else:
        yhi = _scalar(mh.predict(row)[0])
        ylo = _scalar(ml.predict(row)[0])

    # Direction head (optional, present in newer artefacts). Reparameterise
    # into (half-range m, asymmetry d), replace d with a blend of ensemble's
    # d and a classifier-driven d, then reconstruct yhi/ylo.
    # β is adaptive: trend_str = min(|ret_5|/trend_sat, 1); β_eff =
    # β_base × (1 − reduction × trend_str). On trending days β shrinks
    # (favour direction head); in chop β stays near β_base.
    dh = AD.get("direction_head")
    p_bull = None
    beta_eff = None
    if dh is not None and dh.get("classifier") is not None:
        try:
            clf = dh["classifier"]
            beta_base    = float(dh.get("beta", 1.0))
            reduction    = float(dh.get("beta_trend_reduction", 0.0))
            trend_sat    = float(dh.get("trend_saturation", 0.05))
            trend_feat   = dh.get("trend_feature", "ret_5")
            d_bull_mean  = float(dh.get("d_bull_mean", 0.0))
            d_bear_mean  = float(dh.get("d_bear_mean", 0.0))
            p_bull = float(clf.predict_proba(row)[0, 1])
            try:
                ret_val = float(row.iloc[0][trend_feat])
                trend_str = min(abs(ret_val) / trend_sat, 1.0)
            except Exception:
                trend_str = 0.0
            beta_eff = max(0.0, min(1.0, beta_base * (1.0 - reduction * trend_str)))
            m_pred = (yhi + ylo) / 2.0
            d_pred = (yhi - ylo) / 2.0
            d_dir  = p_bull * d_bull_mean + (1.0 - p_bull) * d_bear_mean
            d_blend = beta_eff * d_pred + (1.0 - beta_eff) * d_dir
            yhi = m_pred + d_blend
            ylo = m_pred - d_blend
        except Exception:
            # If the classifier fails, fall back to ensemble-only prediction
            p_bull = None
            beta_eff = None

    sh  = _scalar(sh); sl = _scalar(sl)
    clip0 = lambda x: float(max(float(x), 0.0))
    pred_high = close_asof * (1 + clip0(yhi))
    pred_low  = close_asof * (1 - clip0(ylo))
    band_hi_up = close_asof * (1 + yhi + 1.96*sh)
    band_hi_dn = close_asof * (1 + clip0(yhi - 1.96*sh))
    band_lo_up = close_asof * (1 - clip0(ylo - 1.96*sl))
    band_lo_dn = close_asof * (1 - clip0(ylo + 1.96*sl))

    return dict(
        as_of_date=asof, close_asof=close_asof,
        target_date=target_date,
        target_window_start=target_window_start,
        target_window_end=target_window_end,
        pred_high=pred_high, high_ci_lo=band_hi_dn, high_ci_hi=band_hi_up,
        pred_low =pred_low,  low_ci_lo =band_lo_dn, low_ci_hi =band_lo_up,
        p_bull=p_bull, beta_eff=beta_eff,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def compute_daily_series(end_target_date_iso, days_back=7):
    """Build a series of (pred_high, pred_low, actual_high, actual_low) for the
    last `days_back`+1 target days ending at `end_target_date`. Each prediction
    is generated by `compute_daily_forecast(target_date)`, which internally
    uses data through target − 1 day (i.e. through target 7am CT).

    The cache TTL is 1 hour so the series refreshes when underlying data does."""
    end_target = pd.Timestamp(end_target_date_iso)
    daily_df = _fetch_daily_raw()
    rows = []
    for i in range(days_back, -1, -1):
        target_date = end_target - pd.Timedelta(days=i)
        pred = compute_daily_forecast(target_date.strftime("%Y-%m-%d"))
        if pred is None:
            continue
        ts = pd.Timestamp(target_date)
        actual_h = (float(daily_df.loc[ts, "btc_high"])
                    if ts in daily_df.index and pd.notna(daily_df.loc[ts, "btc_high"])
                    else np.nan)
        actual_l = (float(daily_df.loc[ts, "btc_low"])
                    if ts in daily_df.index and pd.notna(daily_df.loc[ts, "btc_low"])
                    else np.nan)
        rows.append(dict(
            target_date=ts,
            as_of_date=pred["as_of_date"],
            close_asof=float(pred["close_asof"]),
            pred_high=float(pred["pred_high"]),
            pred_low =float(pred["pred_low"]),
            actual_high=actual_h,
            actual_low =actual_l,
        ))
    return pd.DataFrame(rows)


@st.cache_resource
def _load_cone_7d():
    """Load the 7-day close-price regime-cone artefact (or None if absent)."""
    p = str(CONE_7D_MODEL)
    if not os.path.exists(p):
        return None
    return joblib.load(p)


@st.cache_data(ttl=3600, show_spinner=False)
def compute_7d_close_cone_forecast(asof_date_iso):
    """Forecast BTC close 7 days after `asof_date_iso` using the regime cone.

    The cone is parameter-free at inference: classify the as-of bar's
    ``range_ma30`` into one of three training-set terciles, take the
    regime's median forward 7-day log-return, multiply against the
    as-of close, and apply a fixed ±9.7 % band (the headline empirical
    band-width from notebooks/btc_7d_close_research.ipynb).

    Returns ``None`` if the cone artefact is missing or the daily data
    can't reach back 49 days. Otherwise returns a dict with:

      history     – DataFrame of 7 weekly close observations spaced 7
                    days apart ending at the as-of bar
      pred_date   – target Timestamp = as_of + 7 days
      pred_close  – predicted USD close at pred_date
      lower / upper – ±9.7 % band on pred_close
      regime, regime_label – tercile index and short label
      band_pct    – the fixed band width (0.097)
      asof_close  – the as-of-bar close used as the anchor
    """
    cone = _load_cone_7d()
    if cone is None:
        return None
    daily = _fetch_daily_raw().copy()
    if daily.empty:
        return None

    asof = pd.Timestamp(asof_date_iso)
    # Snap to the latest available bar at or before asof
    bars_avail = daily.loc[daily.index <= asof]
    if bars_avail.empty:
        return None
    asof_t = bars_avail.index[-1]

    # range_ma30 needs 30 prior daily bars; weekly history needs 49 days back
    c = bars_avail["btc_close"]; h = bars_avail["btc_high"]; l_ = bars_avail["btc_low"]
    range_today = (h - l_) / c
    range_ma30  = range_today.rolling(30).mean()
    rm30 = range_ma30.loc[asof_t]
    if not np.isfinite(rm30):
        return None

    edges = np.asarray(cone["regime_edges"], dtype=float)
    regime = int(np.searchsorted(edges, float(rm30), side="right"))
    regime = max(0, min(len(edges), regime))
    regime_label = ["low vol","mid vol","high vol"][regime] if regime < 3 else f"r{regime}"

    stats = cone["regime_stats"][regime]
    # Median forward 7-day log-return for this regime
    med_logret = float(stats[0.50] if 0.50 in stats else stats["0.5"])
    asof_close = float(c.loc[asof_t])
    pred_close = asof_close * float(np.exp(med_logret))
    band_pct = float(cone.get("band_pct", 0.097))
    lower = pred_close * (1 - band_pct)
    upper = pred_close * (1 + band_pct)
    pred_date = asof_t + pd.Timedelta(days=7)

    # Helper: snap a target date to the latest bar at or before it.
    def _snap(d):
        if d in c.index and pd.notna(c.loc[d]):
            return d
        prior = c.loc[c.index <= d]
        return prior.index[-1] if not prior.empty else None

    # 7 weekly target dates ending at asof_t (oldest → newest).
    hist_dates, hist_closes = [], []
    for k in range(6, -1, -1):
        snapped = _snap(asof_t - pd.Timedelta(days=7 * k))
        if snapped is not None:
            hist_dates.append(snapped); hist_closes.append(float(c.loc[snapped]))
    history = pd.DataFrame({"close": hist_closes}, index=pd.DatetimeIndex(hist_dates))

    # Historical predictions: for each target date d in `history.index`,
    # find the prediction-anchor date 7 days before, classify its regime
    # via range_ma30 at that anchor, and apply the regime median return.
    # The "actual" at d is just history.close[d].
    rows = []
    for d in history.index:
        a = _snap(d - pd.Timedelta(days=7))
        if a is None or not np.isfinite(range_ma30.loc[a]):
            continue
        a_close = float(c.loc[a])
        rm30_a  = float(range_ma30.loc[a])
        r       = int(np.searchsorted(edges, rm30_a, side="right"))
        r       = max(0, min(len(edges), r))
        m       = float(cone["regime_stats"][r][0.50])
        p_close = a_close * float(np.exp(m))
        rows.append(dict(
            anchor_date   = a,
            target_date   = d,
            anchor_close  = a_close,
            pred_close    = p_close,
            lower         = p_close * (1 - band_pct),
            upper         = p_close * (1 + band_pct),
            actual_close  = float(c.loc[d]),
            regime        = r,
        ))
    hist_preds = pd.DataFrame(rows)

    # In historical-replay mode the +7d target may already be in the past;
    # surface the realized close if data is available for that bar so the
    # caller can plot the actual alongside the forecast star.
    actual_pred_close, actual_pred_date = None, None
    last_avail = c.index[-1]
    if pred_date <= last_avail:
        snapped_pred = _snap(pred_date)
        if snapped_pred is not None and pd.notna(c.loc[snapped_pred]):
            actual_pred_close = float(c.loc[snapped_pred])
            actual_pred_date  = snapped_pred

    return dict(
        history          = history,
        hist_preds       = hist_preds,
        pred_date        = pred_date,
        pred_close       = pred_close,
        lower            = lower,
        upper            = upper,
        regime           = regime,
        regime_label     = regime_label,
        band_pct         = band_pct,
        asof_close       = asof_close,
        asof_date        = asof_t,
        regime_median_logret = med_logret,
        actual_pred_close = actual_pred_close,
        actual_pred_date  = actual_pred_date,
    )


@st.cache_resource
def _load_day_type():
    """Load the 3-class day-type GBM artefact (or None if missing)."""
    p = str(DAY_TYPE_MODEL)
    if not os.path.exists(p):
        return None
    return joblib.load(p)


@st.cache_data(ttl=86400, show_spinner="Classifying day-type …")
def compute_day_type_forecast(target_date_iso):
    """Classify the next 12:00-UTC bar as BigUpper / BigLower / Quiet.

    Uses the H/L model's predictions + the cone regime + a handful of
    raw daily features. Returns:
      predicted_class    – str
      probability        – top-class probability (0-1)
      proba_by_class     – {class: probability}
      target_date        – ISO date of the bar being classified
      as_of_date         – the latest completed bar used
      band_realized      – the test-set selective-accuracy table from the
                           artefact (for the caption)
    Returns None if the artefact is missing or the model can't load.
    """
    art = _load_day_type()
    if art is None:
        return None
    gbm   = art["model"]
    FEATS = art["feature_columns"]
    cone  = _load_cone_7d()

    # Recompute the same features the training script used at the cutoff
    df = _fetch_daily_raw().copy()
    target_date = pd.Timestamp(target_date_iso)
    asof_cutoff = target_date - pd.Timedelta(days=1)
    df = df.loc[df.index <= asof_cutoff].sort_index().ffill(limit=5)
    if df.empty:
        return None
    c = df["btc_close"]; h = df["btc_high"]; l_ = df["btc_low"]; v = df["btc_volume"]
    f = pd.DataFrame(index=df.index)
    ret = np.log(c).diff()
    for k in [3, 7, 14]:           f[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [10, 20, 30]:         f[f"vol_{k}"] = ret.rolling(k).std()
    prev_c = c.shift(1)
    tr_ = pd.concat([(h - l_), (h - prev_c).abs(), (l_ - prev_c).abs()], axis=1).max(axis=1)
    for k in [7, 14, 30]:          f[f"atr_{k}"] = tr_.rolling(k).mean() / c
    f["range_today"] = (h - l_) / c
    f["range_ma7"]   = ((h - l_) / c).rolling(7).mean()
    f["range_ma30"]  = ((h - l_) / c).rolling(30).mean()
    f["range_std30"] = ((h - l_) / c).rolling(30).std()
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    f["macd"]      = macd / c
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    ma20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
    f["bb_width"]   = (4 * sd20) / ma20
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    f["rsi_14"] = 100 - 100 / (1 + rs)
    dow = df.index.dayofweek
    for i in range(6):
        f[f"dow_{i}"] = (dow == i).astype(float)

    # Cone regime one-hots
    edges = np.asarray(art.get("regime_edges") or cone["regime_edges"])
    reg   = int(np.searchsorted(edges, float(f["range_ma30"].iloc[-1]), side="right").clip(0, 2))
    for r in (0, 1, 2):
        f[f"regime_{r}"] = float(r == reg)

    # H/L model predictions and direction-head probability (use only the
    # as-of row to keep this cheap)
    fc = A["feat_cols"]
    daily_forecast = compute_daily_forecast(target_date_iso)
    if daily_forecast is None:
        return None
    close_asof = daily_forecast["close_asof"]
    pred_high  = daily_forecast["pred_high"]
    pred_low   = daily_forecast["pred_low"]
    p_bull     = daily_forecast.get("p_bull", 0.5)
    f["pred_y_hi"]  = (pred_high - close_asof) / close_asof
    f["pred_y_lo"]  = (close_asof - pred_low)  / close_asof
    f["pred_range"] = f["pred_y_hi"] + f["pred_y_lo"]
    f["pred_skew"]  = f["pred_y_hi"] - f["pred_y_lo"]
    f["p_bull"]     = float(p_bull)

    asof_t = df.index[-1]
    x_row = f.loc[[asof_t], FEATS]
    if x_row.isna().any().any():
        return None
    proba = gbm.predict_proba(x_row)[0]
    cls   = list(gbm.classes_)
    proba_by_class = {c: float(p) for c, p in zip(cls, proba)}
    top_idx = int(np.argmax(proba))
    return dict(
        predicted_class = cls[top_idx],
        probability     = float(proba[top_idx]),
        proba_by_class  = proba_by_class,
        target_date     = target_date,
        as_of_date      = asof_t,
        calibration     = art.get("calibration_meta", {}),
    )


def build_features(df):
    f = pd.DataFrame(index=df.index)
    c, h, l_, v = df["btc_close"], df["btc_high"], df["btc_low"], df["btc_volume"]
    rt = np.log(c).diff()
    for k in [1,2,4,8,12,24,48,72]: f[f"ret_{k}h"] = rt.rolling(k).sum()
    for k in [4,8,24,48]:           f[f"vol_{k}h"] = rt.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h-l_),(h-prev_c).abs(),(l_-prev_c).abs()],axis=1).max(axis=1)
    for k in [4,12,24]: f[f"atr_{k}h"] = tr.rolling(k).mean()/c
    f["range_now"]   = (h-l_)/c
    f["range_ma24"]  = f["range_now"].rolling(24).mean()
    f["range_ma72"]  = f["range_now"].rolling(72).mean()
    f["vol_chg_1"]   = np.log(v).diff()
    f["vol_z_24"]    = (np.log(v)-np.log(v).rolling(24).mean())/np.log(v).rolling(24).std()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain/loss.replace(0,np.nan)
    f["rsi_14"] = 100 - 100/(1+rs)
    ema12=c.ewm(span=12,adjust=False).mean(); ema26=c.ewm(span=26,adjust=False).mean()
    macd=ema12-ema26
    f["macd"]=macd/c; f["macd_hist"]=(macd-macd.ewm(span=9,adjust=False).mean())/c
    ma24=c.rolling(24).mean(); sd24=c.rolling(24).std()
    f["bb24_width"]=(4*sd24)/ma24
    f["dist_hi_24"]  = c/c.rolling(24).max() - 1
    f["dist_lo_24"]  = c/c.rolling(24).min() - 1
    f["dist_hi_168"] = c/c.rolling(168).max() - 1
    for nm in ["eth","spx","ndx","vix","gold","dxy","tnx"]:
        s = df[f"{nm}_close"]
        f[f"{nm}_ret_1h"]  = np.log(s).diff()
        f[f"{nm}_ret_24h"] = np.log(s).diff(24)
        f[f"{nm}_vol_24h"] = np.log(s).diff().rolling(24).std()
    f["btc_eth_corr_24"] = rt.rolling(24).corr(np.log(df["eth_close"]).diff())
    f["fng"]   = df["fng"]
    f["fng_d1"]= df["fng"].diff()
    f["fng_d7"]= df["fng_d7"]
    f["fng_d24"]=df["fng_d24"]
    hr = df.index.hour; dow = df.index.dayofweek
    f["hr_sin"]=np.sin(2*np.pi*hr/24);  f["hr_cos"]=np.cos(2*np.pi*hr/24)
    f["dow_sin"]=np.sin(2*np.pi*dow/7); f["dow_cos"]=np.cos(2*np.pi*dow/7)
    f["weekend"]=(dow>=5).astype(int)
    f["us_open"]=((hr>=13)&(hr<=20)&(dow<5)).astype(int)
    return f

# ─────────────────────────── fetch + predict ──────────────────────────
with st.spinner("Fetching live market data ..."):
    df = fetch_data()
    F  = build_features(df).replace([np.inf,-np.inf], np.nan)
    F  = F[feat_cols]
    # Forward-fill any stale macro features so the latest BTC hour is always
    # usable for inference even when SPX/VIX/TNX haven't ticked recently
    # (weekends, off-hours, holidays). The risk is using slightly stale
    # macro values, which is the right trade-off for a live system.
    F_filled = F.ffill()

# Use the most recent BTC bar where we have at least the core BTC features
valid_mask = F_filled.notna().all(axis=1)
if not valid_mask.any():
    st.error("Not enough recent data to compute features. Try again later.")
    st.stop()
latest_t_global = F_filled.index[valid_mask][-1]
live_spot, live_spot_ts = fetch_live_spot()


# ════════════════════════════════════════════════════════════════════════
# Dashboard renderer — used by both Live and Historical tabs
# ════════════════════════════════════════════════════════════════════════
def render_dashboard(as_of_t, *, is_live, live_spot=None, live_spot_ts=None,
                     hist_picker=None):
    """Render the full dashboard (KPIs + chart + look-back metrics)
    as-of `as_of_t`.  In live mode, `now_utc` is wall-clock and we anchor
    the prediction at the Binance live spot.  In historical mode, `now_utc`
    is the picked timestamp and the anchor is the hourly close at that time."""
    latest_t = as_of_t
    latest_close = float(df.loc[latest_t, "btc_close"])
    next_t = latest_t + pd.Timedelta(hours=1)

    # Daily H/L forecast — labelled by the TARGET date (the bar being predicted).
    # The target bar covers [target 7am CT, target+1 7am CT). Only data through
    # target 7am CT is used (bars with start ≤ target − 1 day).
    #   Live mode   → target = "today_CT" (the date for which 7am CT has most
    #                 recently passed; before 7am CT it stays on the previous day).
    #   Historical  → target = picked_date (the date the user selected).
    if is_live:
        ref_t = datetime.now(timezone.utc)
        target_date = pd.Timestamp((ref_t - timedelta(hours=ANCHOR_HOUR_UTC)).date())
    else:
        picked_date_ct = st.session_state.get("hist_date")
        if picked_date_ct is None:
            ref_t = as_of_t.replace(tzinfo=timezone.utc) if as_of_t.tzinfo is None else as_of_t
            target_date = pd.Timestamp((ref_t - timedelta(hours=ANCHOR_HOUR_UTC)).date())
        else:
            target_date = pd.Timestamp(picked_date_ct)
        # Warn if the picked date is in any model's training window.
        render_replay_in_sample_warning(target_date)
    daily = compute_daily_forecast(target_date.strftime("%Y-%m-%d"))

    # Rolling forecast target (now+1h in live, as_of+1h in historical)
    if is_live:
        now_utc = pd.Timestamp(datetime.now(timezone.utc)).tz_convert(None)
    else:
        now_utc = latest_t
    forecast_target = now_utc + pd.Timedelta(hours=1)

    x_now = F_filled.loc[[latest_t]]
    y_pred = float(model.predict(x_now)[0])

    # Anchor: live spot in live mode, hourly close in historical mode
    if is_live and live_spot is not None:
        anchor_price = live_spot
    else:
        anchor_price = latest_close
    pred_close   = anchor_price * np.exp(y_pred)
    pred_close_up = pred_close * (1 + BAND_PCT)
    pred_close_dn = pred_close * (1 - BAND_PCT)
    expected_ret_pct = (np.exp(y_pred) - 1) * 100
    fng_now = int(df.loc[latest_t, "fng"]) if pd.notna(df.loc[latest_t, "fng"]) else None

    # ─────────────────────────── headline KPIs ────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    if live_spot is not None:
        c1.metric("Live BTC spot (Binance)",
                  f"${live_spot:,.0f}",
                  delta=f"as of {live_spot_ts.strftime('%H:%M:%S')} UTC")
    else:
        c1.metric("Latest BTC close (Yahoo)",
                  f"${latest_close:,.0f}",
                  delta=f"as of {latest_t.strftime('%H:%M')} UTC")
    c2.metric(f"Forecast 1 h from now ({forecast_target.strftime('%H:%M:%S')} UTC)",
              f"${pred_close:,.0f}",
              delta=f"{expected_ret_pct:+.3f}% vs spot")
    c3.metric("Forecast band ±0.5 %",
              f"${pred_close_dn:,.0f} – ${pred_close_up:,.0f}",
              delta=f"width = {2*BAND_PCT*100:.1f} %")
    c4.metric("Fear & Greed (latest daily)",
              f"{fng_now if fng_now is not None else 'n/a'}",
              delta=(f"{df['fng'].diff().iloc[-1]:+.0f} d/d"
                     if fng_now is not None else None))

    # ---------- Daily H/L forecast KPIs (12:00-UTC = 7am-CT bars) ----------
    if daily is not None:
        ws = daily["target_window_start"]; we = daily["target_window_end"]
        td = pd.Timestamp(daily["target_date"])
        st.markdown(
            f"#### 🗓️ Daily H/L forecast for **{td.strftime('%Y-%m-%d')}** — "
            f"window **{ws.strftime('%Y-%m-%d %H:%M')} → {we.strftime('%Y-%m-%d %H:%M')} UTC**  "
            f"(= {td.strftime('%Y-%m-%d')} 7am CT → {(td + pd.Timedelta(days=1)).strftime('%Y-%m-%d')} 7am CT)  "
            f"<small>(uses data through {td.strftime('%Y-%m-%d')} 7am CT — i.e. "
            f"the bar starting {daily['as_of_date'].strftime('%Y-%m-%d')} 12:00 UTC, close "
            f"${daily['close_asof']:,.0f}. Refreshes at 12:00 UTC (7am CT) each day. "
            f"Model: ensemble (Huber+Bayes+GBM-MAE). "
            f"Backtest MAPE H=1.12%, L=1.31%; hit ±1% on 54–60% of test days.)</small>",
            unsafe_allow_html=True,
        )
        d1, d2 = st.columns(2)
        d1.metric("Predicted DAILY HIGH",
                  f"${daily['pred_high']:,.0f}",
                  delta=(f"+{(daily['pred_high']/daily['close_asof']-1)*100:.2f}% vs close"))
        d1.caption(
            f"±1.5 % band ${daily['pred_high']*0.985:,.0f} – ${daily['pred_high']*1.015:,.0f}"
        )
        d2.metric("Predicted DAILY LOW",
                  f"${daily['pred_low']:,.0f}",
                  delta=(f"{(daily['pred_low']/daily['close_asof']-1)*100:.2f}% vs close"))
        d2.caption(
            f"±1.5 % band ${daily['pred_low']*0.985:,.0f} – ${daily['pred_low']*1.015:,.0f}"
        )
        # Direction-head bias (if model artefact has one)
        pb = daily.get("p_bull")
        if pb is not None:
            pred_asym = (daily["pred_high"] - daily["close_asof"]) \
                        - (daily["close_asof"] - daily["pred_low"])
            bias_word = "bullish" if pred_asym > 0 else ("bearish" if pred_asym < 0 else "neutral")
            bias_color = "#1a7f37" if pred_asym > 0 else ("#b91c1c" if pred_asym < 0 else "#555")
            be = daily.get("beta_eff")
            be_str = (f" · β_eff = <b>{be:.2f}</b>" if be is not None else "")
            st.markdown(
                f"<small>🧭 <b>Direction head</b>: "
                f"P(bullish bar) = <b>{pb*100:.1f}%</b> · "
                f"resulting bias = <span style='color:{bias_color}'><b>{bias_word}</b></span> "
                f"(predicted up-tail − down-tail = ${pred_asym:+,.0f})"
                f"{be_str}</small>",
                unsafe_allow_html=True,
            )

    # ────── 3-class day-type classifier (Big Upper / Big Lower / Quiet) ──
    day_type = compute_day_type_forecast(target_date.strftime("%Y-%m-%d"))
    if day_type is not None:
        DT_COLORS  = {"BigUpper": "#16a34a", "BigLower": "#dc2626", "Quiet": "#475569"}
        DT_EMOJI   = {"BigUpper": "🔼",      "BigLower": "🔽",      "Quiet": "▫️"}
        DT_LABEL   = {"BigUpper": "Big Upper Movement",
                      "BigLower": "Big Lower Movement",
                      "Quiet":    "Quiet Movement"}
        pc   = day_type["predicted_class"]
        prob = day_type["probability"]
        probs = day_type["proba_by_class"]
        cal   = day_type["calibration"]
        td    = pd.Timestamp(day_type["target_date"])
        # Confidence-gated guidance — match the artefact's stored selective
        # table to a coarse band so the caption stays honest.
        if   prob >= 0.65: gating = "high confidence"
        elif prob >= 0.55: gating = "moderate confidence"
        elif prob >= 0.45: gating = "low confidence"
        else:              gating = "very low confidence"
        st.markdown(
            f"#### {DT_EMOJI[pc]} Day-type forecast — "
            f"<span style='color:{DT_COLORS[pc]}'>"
            f"<b>{DT_LABEL[pc]}</b></span> "
            f"<small>(confidence <b>{prob*100:.0f}%</b> · {gating})</small>",
            unsafe_allow_html=True,
        )
        # Probability bar: three horizontal segments with widths ∝ probability
        order = ["BigUpper", "BigLower", "Quiet"]
        bar_html = "<div style='display:flex; width:100%; height:24px; border-radius:6px; overflow:hidden; border:1px solid #ddd; font-size:11px; font-weight:600; color:white;'>"
        for k in order:
            pct = probs.get(k, 0.0) * 100
            border = ("3px solid #111" if k == pc else "0")
            bar_html += (
                f"<div title='{DT_LABEL[k]}: {pct:.1f}%' "
                f"style='flex: {max(probs.get(k,0.001), 0.001)}; "
                f"background:{DT_COLORS[k]}; "
                f"display:flex; align-items:center; justify-content:center; "
                f"border-right:{border};'>"
                f"{DT_EMOJI[k]} {pct:.0f}%"
                f"</div>"
            )
        bar_html += "</div>"
        st.markdown(bar_html, unsafe_allow_html=True)
        # Honest caption: test acc + selective accuracy at the relevant band
        test_acc = cal.get("test_accuracy_pct")
        test_n   = cal.get("test_n")
        sel      = cal.get("selective", [])
        # Pick the largest threshold the current prob satisfies
        matched  = max((s for s in sel if prob >= s["thr"]),
                       key=lambda s: s["thr"], default=None)
        sel_note = ""
        if matched is not None:
            sel_note = (f" When the model is at least {matched['thr']:.0%} "
                        f"confident it covers ~{matched['coverage_pct']:.0f}% of "
                        f"days at {matched['accuracy_pct']:.0f}% accuracy on the held-out tail.")
        st.caption(
            f"3-class day-type classifier (GBM, 31 features). Predicts the "
            f"realised next-day H/L bar shape for **{td.strftime('%Y-%m-%d')}**. "
            f"Hold-out unconditional accuracy = {test_acc:.0f}% on n={test_n} days "
            f"(majority baseline ≈ 33% on three balanced classes)."
            + sel_note
        )

    # ────── Historical picker (date strip, calendar, hour slider,
    # bookmarks) rendered RIGHT ABOVE the plots so the user can navigate
    # to a different day without scrolling back up.
    if hist_picker is not None:
        hist_picker()

    # ─────────────────────────── walk-forward look-back ───────────────────
    # Live mode  → last LOOKBACK_HOURS hours up to now.
    # Historical → fixed 24h bar [picked_date 12:00 UTC, +24h), matching the
    # daily model's anchor exactly. That is 7am CDT (summer) / 6am CST (winter)
    # — labelled loosely as "7am CT" per README §"Day-boundary contract".
    # Anchoring at 12:00 UTC (not DST-following local 7am) ensures the hourly
    # chart's 24h window covers the same bar the daily prediction is for.
    if is_live:
        look_idx = F_filled.index[(F_filled.index <= latest_t) & valid_mask][-LOOKBACK_HOURS:]
        win_start_utc = None  # signals to chart code: use look_idx-derived range
        win_end_utc   = None
    else:
        _CT = "America/Chicago"
        picked_date_ct = st.session_state.get("hist_date")
        if picked_date_ct is None:
            picked_date_ct = (latest_t.tz_localize("UTC")
                                       .tz_convert(_CT).tz_localize(None).date())
        # Fixed 12:00 UTC anchor — same boundary as _rebucket_12utc / daily model.
        win_start_utc = pd.Timestamp(picked_date_ct) + pd.Timedelta(hours=ANCHOR_HOUR_UTC)
        win_end_utc   = win_start_utc + pd.Timedelta(days=1)
        # CT-time edges for the chart's x-axis range (DST-correct labels).
        day_start_ct = (win_start_utc.tz_localize("UTC").tz_convert(_CT)
                                     .tz_localize(None))
        day_end_ct   = (win_end_utc.tz_localize("UTC").tz_convert(_CT)
                                   .tz_localize(None))
        look_idx = F_filled.index[(F_filled.index >= win_start_utc) &
                                  (F_filled.index <  win_end_utc) &
                                  valid_mask]
    y_lb = model.predict(F_filled.loc[look_idx])
    close_lb     = df.loc[look_idx, "btc_close"].values
    pred_close_lb = close_lb * np.exp(y_lb)
    pred_up_lb    = close_lb * np.exp(y_lb + 1.96*sigma)
    pred_dn_lb    = close_lb * np.exp(y_lb - 1.96*sigma)
    target_dates_lb = [d + pd.Timedelta(hours=1) for d in look_idx]
    actual_lb = []
    for d in target_dates_lb:
        actual_lb.append(float(df.loc[d, "btc_close"]) if d in df.index else np.nan)
    actual_lb = np.array(actual_lb)

    # back-test metrics on the look-back window
    mask = ~np.isnan(actual_lb)
    if mask.sum() > 5:
        rel = np.abs(pred_close_lb[mask] - actual_lb[mask]) / actual_lb[mask]
        pred_ret_lb = y_lb[mask]
        actual_ret_lb = np.log(actual_lb[mask] / close_lb[mask])
        lb_metrics = {
            "MAPE": rel.mean()*100,
            "hit3":  (rel<=0.03).mean()*100,
            "hit1":  (rel<=0.01).mean()*100,
            "hit0.5":(rel<=0.005).mean()*100,
            "dir_acc": np.mean(np.sign(pred_ret_lb)==np.sign(actual_ret_lb))*100,
        }
    else:
        lb_metrics = None

    # ─────────────────────────── chart ────────────────────────────────────
    fig = go.Figure()
    xt = pd.to_datetime(target_dates_lb)

    # All plot x-axes display US Central time (auto-handles CDT/CST via DST).
    # Source variables stay UTC; we convert only at the plot layer.
    CT_TZ = "America/Chicago"
    def _ct(ts):
        return ts.tz_localize("UTC").tz_convert(CT_TZ).tz_localize(None)
    look_idx_ct        = _ct(look_idx)
    xt_ct              = _ct(xt)
    now_ct             = _ct(pd.DatetimeIndex([now_utc]))[0]
    forecast_target_ct = _ct(pd.DatetimeIndex([forecast_target]))[0]

    # --- ±0.5 % band around PAST PREDICTIONS (not around the actuals) ---
    # Plotted at the target time of each prediction (= look_idx + 1h = xt).
    pred_band_up = pred_close_lb * (1 + BAND_PCT)
    pred_band_dn = pred_close_lb * (1 - BAND_PCT)
    fig.add_trace(go.Scatter(
        x=xt_ct, y=pred_band_up, mode="lines",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=xt_ct, y=pred_band_dn, mode="lines",
        line=dict(color="rgba(0,0,0,0)"), fill="tonexty",
        fillcolor="rgba(65,105,225,0.18)",
        name=f"Pred ±{BAND_PCT*100:.1f}% band",
        hoverinfo="skip",
    ))

    # --- Past actuals (prominent, solid black) ---
    fig.add_trace(go.Scatter(
        x=look_idx_ct, y=close_lb, mode="lines",
        line=dict(color="black", width=2),
        name="Actual close",
        hovertemplate="%{x|%Y-%m-%d %H:%M} CT<br>$%{y:,.0f}<extra></extra>",
    ))

    # --- Past predictions as discrete markers (no lagging line) ---
    # Coloured by directional correctness: green = predicted direction matched
    # the realised direction; red = miscalled; grey = no actual yet.
    mask_r = ~np.isnan(actual_lb)
    pred_dir = np.sign(y_lb)
    actual_ret_lb_full = np.where(mask_r,
                                  np.log(np.where(mask_r, actual_lb, 1) / close_lb),
                                  np.nan)
    actual_dir = np.where(mask_r, np.sign(actual_ret_lb_full), np.nan)
    correct = (pred_dir == actual_dir) & mask_r
    marker_colors = np.where(~mask_r, "lightgrey",
                    np.where(correct, "seagreen", "indianred"))

    fig.add_trace(go.Scatter(
        x=xt_ct, y=pred_close_lb, mode="markers",
        marker=dict(color=marker_colors, size=8,
                    line=dict(width=1, color="white")),
        name="Past hourly predictions (green = correct dir.)",
        customdata=np.column_stack([y_lb*100, actual_lb]),
        hovertemplate=("Past pred for %{x|%Y-%m-%d %H:%M} CT<br>"
                       "Pred close: $%{y:,.0f}<br>"
                       "Pred return: %{customdata[0]:+.3f}%<br>"
                       "Actual close: $%{customdata[1]:,.0f}<extra></extra>"),
    ))

    # --- LIVE FORECAST: prominent zone (rolling 1h-from-now) ---
    fig.add_vrect(
        x0=now_ct, x1=forecast_target_ct + pd.Timedelta(minutes=5),
        fillcolor="khaki", opacity=0.30, line_width=0, layer="below",
    )

    # Connector segment from anchor (live spot or latest close) to the forecast
    fig.add_trace(go.Scatter(
        x=[now_ct, forecast_target_ct],
        y=[anchor_price, pred_close],
        mode="lines",
        line=dict(color="darkorange", width=2),
        showlegend=False, hoverinfo="skip",
    ))

    # Forecast marker: STAR with ±0.5 % error bars at the ROLLING target time
    fig.add_trace(go.Scatter(
        x=[forecast_target_ct], y=[pred_close],
        mode="markers",
        marker=dict(symbol="star", size=14, color="darkorange",
                    line=dict(width=1.5, color="black")),
        error_y=dict(
            type="data", symmetric=False,
            array=[pred_close_up - pred_close],
            arrayminus=[pred_close - pred_close_dn],
            thickness=2.5, width=10, color="darkorange",
        ),
        name=f"🎯 Live rolling forecast → {forecast_target_ct.strftime('%H:%M:%S')} CT",
        hovertemplate=(f"<b>Live rolling 1 h forecast</b><br>"
                       f"For: %{{x|%Y-%m-%d %H:%M:%S}} CT<br>"
                       f"Pred: $%{{y:,.0f}}<br>"
                       f"Band (±{BAND_PCT*100:.1f}%%): ${pred_close_dn:,.0f} – ${pred_close_up:,.0f}"
                       f"<extra></extra>"),
    ))

    # --- Current wall-clock "Now" line ---
    fig.add_vline(x=now_ct, line=dict(color="crimson", width=2, dash="dash"))
    fig.add_annotation(
        x=now_ct, y=1.0, xref="x", yref="paper",
        text=f"<b>Now</b> {now_ct.strftime('%H:%M')} CT",
        showarrow=False, yanchor="bottom", xanchor="center",
        bgcolor="rgba(255,255,255,0.92)", bordercolor="crimson", borderwidth=1,
        font=dict(color="crimson", size=11),
    )

    # --- Daily H/L forecast: full-width flat threshold lines + CI ---
    actual_hi_now = None
    actual_lo_now = None
    if daily is not None:
        wstart = pd.Timestamp(daily["target_window_start"])
        wend   = pd.Timestamp(daily["target_window_end"])

        # Threshold lines: span the FULL x-axis
        fig.add_hline(
            y=daily["pred_high"],
            line=dict(color="green", width=2.5, dash="dot"),
            annotation_text=f"Daily Pred HIGH ${daily['pred_high']:,.0f}",
            annotation_position="top right",
            annotation_font=dict(color="green", size=12),
            annotation_bgcolor="rgba(255,255,255,0.92)",
            annotation_bordercolor="green",
            annotation_borderwidth=1,
        )
        fig.add_hline(
            y=daily["pred_low"],
            line=dict(color="red", width=2.5, dash="dot"),
            annotation_text=f"Daily Pred LOW ${daily['pred_low']:,.0f}",
            annotation_position="bottom right",
            annotation_font=dict(color="red", size=12),
            annotation_bgcolor="rgba(255,255,255,0.92)",
            annotation_bordercolor="red",
            annotation_borderwidth=1,
        )
        # ±2.5% bands around HIGH (green) and LOW (red).  When the ±2.5%
        # zones would overlap, we CLIP each at the midpoint between the two
        # predictions so the green and red never blend into yellow.
        DAILY_BAND_PCT = 0.015
        mid = (daily["pred_high"] + daily["pred_low"]) / 2
        hi_raw_dn = daily["pred_high"] * (1 - DAILY_BAND_PCT)
        hi_raw_up = daily["pred_high"] * (1 + DAILY_BAND_PCT)
        lo_raw_dn = daily["pred_low"]  * (1 - DAILY_BAND_PCT)
        lo_raw_up = daily["pred_low"]  * (1 + DAILY_BAND_PCT)
        hi_band_dn = max(hi_raw_dn, mid)         # clip green at mid
        hi_band_up = hi_raw_up
        lo_band_dn = lo_raw_dn
        lo_band_up = min(lo_raw_up, mid)         # clip red at mid
        if hi_band_up > hi_band_dn:
            fig.add_hrect(
                y0=hi_band_dn, y1=hi_band_up,
                fillcolor="rgba(0,170,0,0.16)", line_width=0, layer="below",
                annotation_text=f"±{DAILY_BAND_PCT*100:.1f}% around HIGH",
                annotation_position="top left",
                annotation_font=dict(color="green", size=10),
            )
        if lo_band_up > lo_band_dn:
            fig.add_hrect(
                y0=lo_band_dn, y1=lo_band_up,
                fillcolor="rgba(220,30,30,0.16)", line_width=0, layer="below",
                annotation_text=f"±{DAILY_BAND_PCT*100:.1f}% around LOW",
                annotation_position="bottom left",
                annotation_font=dict(color="red", size=10),
            )

        # --- Realised daily HIGH / LOW for the displayed bar (historical
        # mode only; the live target bar may still be in progress). Solid
        # lines distinguish realised from the dotted predicted thresholds.
        if not is_live:
            try:
                _daily_raw = _fetch_daily_raw()
                _tgt = pd.Timestamp(daily["target_date"]).normalize()
                if _tgt in _daily_raw.index:
                    _ah = _daily_raw.loc[_tgt, "btc_high"]
                    _al = _daily_raw.loc[_tgt, "btc_low"]
                    if pd.notna(_ah) and pd.notna(_al):
                        actual_hi_now = float(_ah)
                        actual_lo_now = float(_al)
            except Exception:
                pass
        if actual_hi_now is not None:
            fig.add_hline(
                y=actual_hi_now,
                line=dict(color="darkgreen", width=2, dash="solid"),
                annotation_text=f"Actual HIGH ${actual_hi_now:,.0f}",
                annotation_position="top left",
                annotation_font=dict(color="darkgreen", size=11),
                annotation_bgcolor="rgba(255,255,255,0.92)",
                annotation_bordercolor="darkgreen",
                annotation_borderwidth=1,
            )
            fig.add_hline(
                y=actual_lo_now,
                line=dict(color="darkred", width=2, dash="solid"),
                annotation_text=f"Actual LOW ${actual_lo_now:,.0f}",
                annotation_position="bottom left",
                annotation_font=dict(color="darkred", size=11),
                annotation_bgcolor="rgba(255,255,255,0.92)",
                annotation_bordercolor="darkred",
                annotation_borderwidth=1,
            )

    # --- Live spot price marker (Binance, current second) ---
    if live_spot is not None:
        fig.add_trace(go.Scatter(
            x=[now_ct], y=[live_spot], mode="markers",
            marker=dict(symbol="circle", size=11, color="crimson",
                        line=dict(width=1.5, color="white")),
            name=f"Live spot (Binance)",
            hovertemplate=(f"<b>Live BTC spot</b><br>"
                           f"%{{x|%Y-%m-%d %H:%M:%S}} CT<br>"
                           f"$%{{y:,.0f}}<extra></extra>"),
        ))
    # Label the forecast time at the bottom of the khaki zone (rolling target)
    fig.add_annotation(
        x=forecast_target_ct, y=0, xref="x", yref="paper",
        text=f"forecast: <b>{forecast_target_ct.strftime('%H:%M:%S')} CT</b>",
        showarrow=False, yanchor="top", xanchor="center", yshift=-25,
        bgcolor="rgba(255,255,255,0.92)", bordercolor="darkorange", borderwidth=1,
        font=dict(color="darkorange", size=11),
    )

    # x-axis range:
    #   live mode  → last look-back window, padded right for the rolling ⭐
    #   historical → fixed 7am-CT day [day_start_ct, day_end_ct]
    next_t_ct = _ct(pd.DatetimeIndex([next_t]))[0]
    if is_live:
        right_edge = max(next_t_ct, forecast_target_ct) + pd.Timedelta(minutes=30)
        left_edge  = (look_idx_ct[0] - pd.Timedelta(hours=1)) if len(look_idx_ct) else right_edge - pd.Timedelta(hours=LOOKBACK_HOURS)
    else:
        left_edge  = day_start_ct
        right_edge = day_end_ct
    fig.update_xaxes(
        tickformat="%d-%b %H:%M",
        title_text="Time (US Central)",
        title_standoff=12,
        range=[left_edge, right_edge],
    )
    # Bound y-axis tightly to actual data + key reference levels.
    # The daily 95 % CI hrects span ~5-10 % which would otherwise distort the plot.
    y_pts = list(close_lb)
    y_pts.extend([pred_close, pred_close_up, pred_close_dn])
    if live_spot is not None: y_pts.append(live_spot)
    if daily is not None:
        y_pts.extend([daily["pred_high"], daily["pred_low"]])
    if actual_hi_now is not None:
        y_pts.extend([actual_hi_now, actual_lo_now])
    y_min, y_max = min(y_pts), max(y_pts)
    y_pad = max((y_max - y_min) * 0.10, y_max * 0.003)  # ≥0.3% breathing room
    fig.update_yaxes(range=[y_min - y_pad, y_max + y_pad])
    fig.update_layout(
        template="plotly_white", height=600, hovermode="x unified",
        title=dict(
            text=(f"<b>BTC live ROLLING 1 h forecast → "
                  f"{forecast_target_ct.strftime('%H:%M:%S')} CT</b>"
                  "<br><span style='font-size:13px;color:#555'>"
                  f"Refreshes every {REFRESH_SECONDS}s; target slides forward each minute.  "
                  f"Last {LOOKBACK_HOURS}h actuals (black) ±0.5 % shaded.  "
                  f"Past-hour dots: <span style='color:seagreen'>green=correct dir.</span>/"
                  f"<span style='color:indianred'>red=miscalled</span>.  "
                  f"⭐ rolling forecast, anchored at live spot, ±0.5 % band.  "
                  f"Dotted lines = daily H/L threshold (refreshes at 12:00 UTC = 7am CT)."
                  "</span>"),
            x=0.01, xanchor="left", y=0.98, yanchor="top",
        ),
        yaxis_title="BTC / USD",
        margin=dict(t=115, r=210, b=80, l=70),
        legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0, yanchor="top",
                    bgcolor="rgba(255,255,255,0.95)", bordercolor="#ccc",
                    borderwidth=1, font=dict(size=11)),
    )
    st.plotly_chart(fig, use_container_width=True,
                    key=f"chart_hourly_{'live' if is_live else 'hist'}")

    # ═══════ NEW PLOT — Daily H/L: last 7 days predictions + actuals ═══════
    # end_target = the rightmost target date — the bar STARTING on this date
    # is the one highlighted as the current forecast:
    #   Live      → today_CT's bar (starts today 7am CT, ends tomorrow 7am CT).
    #   Historical → bar starting on picked_date (matches the KPI card).
    if is_live:
        end_target = pd.Timestamp(
            (datetime.now(timezone.utc) - timedelta(hours=ANCHOR_HOUR_UTC)).date()
        )
    else:
        picked_date_ct = st.session_state.get("hist_date")
        if picked_date_ct is not None:
            end_target = pd.Timestamp(picked_date_ct)
        else:
            ref = as_of_t.replace(tzinfo=timezone.utc) if as_of_t.tzinfo is None else as_of_t
            end_target = pd.Timestamp((ref - timedelta(hours=ANCHOR_HOUR_UTC)).date())
    series = compute_daily_series(end_target.strftime("%Y-%m-%d"), days_back=7)

    if len(series) > 0:
        st.markdown(
            f"#### 📈 Daily H/L — predictions vs actuals "
            f"(last 7 bars + current target; each bar opens 12:00 UTC = 7am CT, "
            f"highlighted target = **{end_target.strftime('%Y-%m-%d')}**)"
        )
        fig2 = go.Figure()
        # ±2 % uncertainty bands around each predicted line — added first so
        # they render behind the prediction & actual markers.
        DAILY_BAND_PCT = 0.02
        # HIGH ±2% band (green tint)
        fig2.add_trace(go.Scatter(
            x=series["target_date"], y=series["pred_high"] * (1 + DAILY_BAND_PCT),
            mode="lines", line=dict(color="rgba(34,139,34,0)"),
            hoverinfo="skip", showlegend=False,
        ))
        fig2.add_trace(go.Scatter(
            x=series["target_date"], y=series["pred_high"] * (1 - DAILY_BAND_PCT),
            mode="lines", line=dict(color="rgba(34,139,34,0)"),
            fill="tonexty", fillcolor="rgba(34,139,34,0.13)",
            name=f"HIGH ±{DAILY_BAND_PCT*100:.0f}% band", hoverinfo="skip",
        ))
        # LOW ±2% band (red tint)
        fig2.add_trace(go.Scatter(
            x=series["target_date"], y=series["pred_low"] * (1 + DAILY_BAND_PCT),
            mode="lines", line=dict(color="rgba(220,20,60,0)"),
            hoverinfo="skip", showlegend=False,
        ))
        fig2.add_trace(go.Scatter(
            x=series["target_date"], y=series["pred_low"] * (1 - DAILY_BAND_PCT),
            mode="lines", line=dict(color="rgba(220,20,60,0)"),
            fill="tonexty", fillcolor="rgba(220,20,60,0.13)",
            name=f"LOW ±{DAILY_BAND_PCT*100:.0f}% band", hoverinfo="skip",
        ))
        # Predicted HIGH line
        fig2.add_trace(go.Scatter(
            x=series["target_date"], y=series["pred_high"],
            mode="lines+markers",
            line=dict(color="green", width=2.2, dash="dot"),
            marker=dict(size=9, symbol="circle"),
            name="Predicted HIGH",
            hovertemplate=("Bar starts %{x|%Y-%m-%d} 7am CT<br>"
                           "Pred HIGH $%{y:,.0f}<extra></extra>"),
        ))
        # Predicted LOW line
        fig2.add_trace(go.Scatter(
            x=series["target_date"], y=series["pred_low"],
            mode="lines+markers",
            line=dict(color="red", width=2.2, dash="dot"),
            marker=dict(size=9, symbol="circle"),
            name="Predicted LOW",
            hovertemplate=("Bar starts %{x|%Y-%m-%d} 7am CT<br>"
                           "Pred LOW $%{y:,.0f}<extra></extra>"),
        ))
        # Actual HIGH/LOW where realised
        have = series["actual_high"].notna()
        if have.any():
            fig2.add_trace(go.Scatter(
                x=series.loc[have,"target_date"], y=series.loc[have,"actual_high"],
                mode="markers",
                marker=dict(symbol="x-thin", size=13,
                            line=dict(width=3, color="darkgreen")),
                name="Actual HIGH",
                hovertemplate=("Bar %{x|%Y-%m-%d} (7am CT → 7am CT next day)<br>"
                               "Actual HIGH $%{y:,.0f}<extra></extra>"),
            ))
            fig2.add_trace(go.Scatter(
                x=series.loc[have,"target_date"], y=series.loc[have,"actual_low"],
                mode="markers",
                marker=dict(symbol="x-thin", size=13,
                            line=dict(width=3, color="darkred")),
                name="Actual LOW",
                hovertemplate=("Bar %{x|%Y-%m-%d} (7am CT → 7am CT next day)<br>"
                               "Actual LOW $%{y:,.0f}<extra></extra>"),
            ))

        # Direction-correctness per day, computed SEPARATELY for HIGH and LOW.
        # For each day D, compare the day-over-day MOVE of the plotted lines:
        #   HIGH direction realised  = sign(actual_high[D] − actual_high[D-1])
        #   HIGH direction predicted = sign(pred_high[D]   − pred_high[D-1])
        #   LOW  direction realised  = sign(actual_low[D]  − actual_low[D-1])
        #   LOW  direction predicted = sign(pred_low[D]    − pred_low[D-1])
        # Match iff signs agree. This reflects what's visible in the chart:
        # did the predicted line trend the same way as the realised line?
        dir_hit_str = ""
        yest_act_hi  = series["actual_high"].shift(1)
        yest_act_lo  = series["actual_low"].shift(1)
        yest_pred_hi = series["pred_high"].shift(1)
        yest_pred_lo = series["pred_low"].shift(1)
        d_hi_act  = series["actual_high"] - yest_act_hi
        d_lo_act  = series["actual_low"]  - yest_act_lo
        d_hi_pred = series["pred_high"]   - yest_pred_hi
        d_lo_pred = series["pred_low"]    - yest_pred_lo
        have_hi = (yest_act_hi.notna() & series["actual_high"].notna()
                   & yest_pred_hi.notna() & series["pred_high"].notna())
        have_lo = (yest_act_lo.notna() & series["actual_low"].notna()
                   & yest_pred_lo.notna() & series["pred_low"].notna())
        correct_hi = (np.sign(d_hi_act) == np.sign(d_hi_pred)) & have_hi & (np.sign(d_hi_act) != 0)
        correct_lo = (np.sign(d_lo_act) == np.sign(d_lo_pred)) & have_lo & (np.sign(d_lo_act) != 0)
        if have_hi.any() or have_lo.any():
            hits_hi = int(correct_hi[have_hi].sum())
            hits_lo = int(correct_lo[have_lo].sum())
            n_hi = int(have_hi.sum())
            n_lo = int(have_lo.sum())
            dir_hit_str = (
                f"  <span style='color:#1a7f37'><b>HIGH</b> line-trend hit-rate: "
                f"{hits_hi}/{n_hi} = {hits_hi/n_hi*100:.0f}%</span> · "
                f"<span style='color:#b91c1c'><b>LOW</b> line-trend hit-rate: "
                f"{hits_lo}/{n_lo} = {hits_lo/n_lo*100:.0f}%</span>"
                if n_hi and n_lo else dir_hit_str
            )
            # HIGH row (top, green) and LOW row (slightly below, red)
            for i in range(len(series)):
                td_label = series["target_date"].iloc[i].strftime('%Y-%m-%d')
                if bool(have_hi.iloc[i]):
                    ok_h = bool(correct_hi.iloc[i])
                    fig2.add_annotation(
                        x=series["target_date"].iloc[i], y=1.10,
                        xref="x", yref="paper",
                        text=("✓" if ok_h else "✗"),
                        showarrow=False, yanchor="bottom", xanchor="center",
                        font=dict(color="seagreen", size=18, family="monospace"),
                        hovertext=(
                            f"{td_label} — HIGH day-to-day direction<br>"
                            f"Pred line: {'up' if d_hi_pred.iloc[i]>0 else 'down' if d_hi_pred.iloc[i]<0 else 'flat'} "
                            f"(Δ pred_high vs yest pred_high = ${d_hi_pred.iloc[i]:+,.0f})<br>"
                            f"Actual line: {'up' if d_hi_act.iloc[i]>0 else 'down' if d_hi_act.iloc[i]<0 else 'flat'} "
                            f"(Δ actual_high vs yest actual_high = ${d_hi_act.iloc[i]:+,.0f})<br>"
                            f"{'Matched' if ok_h else 'Missed'}"
                        ),
                    )
                if bool(have_lo.iloc[i]):
                    ok_l = bool(correct_lo.iloc[i])
                    fig2.add_annotation(
                        x=series["target_date"].iloc[i], y=1.02,
                        xref="x", yref="paper",
                        text=("✓" if ok_l else "✗"),
                        showarrow=False, yanchor="bottom", xanchor="center",
                        font=dict(color="indianred", size=18, family="monospace"),
                        hovertext=(
                            f"{td_label} — LOW day-to-day direction<br>"
                            f"Pred line: {'up' if d_lo_pred.iloc[i]>0 else 'down' if d_lo_pred.iloc[i]<0 else 'flat'} "
                            f"(Δ pred_low vs yest pred_low = ${d_lo_pred.iloc[i]:+,.0f})<br>"
                            f"Actual line: {'up' if d_lo_act.iloc[i]>0 else 'down' if d_lo_act.iloc[i]<0 else 'flat'} "
                            f"(Δ actual_low vs yest actual_low = ${d_lo_act.iloc[i]:+,.0f})<br>"
                            f"{'Matched' if ok_l else 'Missed'}"
                        ),
                    )

        # Highlight the right-most point (the target forecast)
        last_t = series["target_date"].iloc[-1]
        fig2.add_vrect(x0=last_t - pd.Timedelta(hours=12),
                       x1=last_t + pd.Timedelta(hours=12),
                       fillcolor="khaki", opacity=0.30, line_width=0,
                       layer="below")
        fig2.add_annotation(
            x=last_t, y=1.18, xref="x", yref="paper",
            text=f"<b>target forecast</b><br>{last_t.strftime('%Y-%m-%d')}",
            showarrow=False, yanchor="bottom", xanchor="center",
            bgcolor="rgba(255,255,255,0.92)", bordercolor="goldenrod",
            borderwidth=1, font=dict(color="goldenrod", size=10),
        )

        fig2.update_layout(
            template="plotly_white", height=500, hovermode="x unified",
            title=dict(
                text=("<b>Daily H/L — predictions (dotted) vs actuals (X markers)</b>"
                      "<br><span style='font-size:12px;color:#555'>"
                      "8 target days: last 7 with realised values + target forecast highlighted. "
                      "Top rows: <b style='color:#1a7f37'>green</b> = HIGH line "
                      "day-over-day direction (pred-trend vs actual-trend), "
                      "<b style='color:#b91c1c'>red</b> = LOW line "
                      "day-over-day direction. ✓ both lines moved the same way / ✗ opposed."
                      f"{dir_hit_str}"
                      "</span>"),
                x=0.01, xanchor="left", y=0.97, yanchor="top",
            ),
            yaxis_title="BTC / USD",
            xaxis_title="Target bar start date (US Central, bar opens 7am CT)",
            margin=dict(t=140, r=200, b=60, l=70),
            legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0,
                        yanchor="top", bgcolor="rgba(255,255,255,0.95)",
                        bordercolor="#ccc", borderwidth=1, font=dict(size=11)),
        )
        fig2.update_xaxes(tickformat="%a %d-%b")
        st.plotly_chart(fig2, use_container_width=True,
                        key=f"chart_daily_{'live' if is_live else 'hist'}")

    # ─────────── 7-day close cone (regime-based) chart ────────────────────
    # Past 7 weekly closes anchored at the daily-model target date, plus
    # the +7d prediction and the fixed ±9.7 % band reported in
    # notebooks/btc_7d_close_research.ipynb.
    cone7 = compute_7d_close_cone_forecast(target_date.strftime("%Y-%m-%d"))
    if cone7 is not None:
        hist = cone7["history"]
        ret_pct = (np.exp(cone7["regime_median_logret"]) - 1) * 100
        st.markdown(
            f"#### 📅 7-day close-price cone — regime: **{cone7['regime_label']}**  "
            f"<small>(as-of {pd.Timestamp(cone7['asof_date']).strftime('%Y-%m-%d')} "
            f"close ${cone7['asof_close']:,.0f} → "
            f"forecast {cone7['pred_date'].strftime('%Y-%m-%d')} "
            f"${cone7['pred_close']:,.0f} "
            f"({ret_pct:+.2f}% regime median return), "
            f"band ±{cone7['band_pct']*100:.1f}%)</small>",
            unsafe_allow_html=True,
        )
        hp = cone7.get("hist_preds", pd.DataFrame()).copy()
        band_pct = cone7["band_pct"]
        # In historical replay, when the +7d target already has a realized
        # close, treat it as the *last historical prediction* (with its
        # actual on the realized line) rather than as a future-looking
        # star. This keeps the chart strictly past-only in Historical
        # Replay so every prediction has an actual to compare against.
        in_hist_with_actual = (
            (not is_live) and (cone7.get("actual_pred_close") is not None)
        )
        if in_hist_with_actual:
            hp = pd.concat([
                hp,
                pd.DataFrame([{
                    "anchor_date":  cone7["asof_date"],
                    "target_date":  cone7["pred_date"],
                    "anchor_close": cone7["asof_close"],
                    "pred_close":   cone7["pred_close"],
                    "lower":        cone7["lower"],
                    "upper":        cone7["upper"],
                    "actual_close": cone7["actual_pred_close"],
                    "regime":       cone7["regime"],
                }]),
            ], ignore_index=True)
            hist = pd.concat([
                hist,
                pd.DataFrame({"close": [cone7["actual_pred_close"]]},
                             index=pd.DatetimeIndex([cone7["actual_pred_date"]])),
            ])
        fig3 = go.Figure()

        # Historical ±band — shaded fill across consecutive prediction targets.
        if len(hp) >= 2:
            fig3.add_trace(go.Scatter(
                x=list(hp["target_date"]), y=list(hp["upper"]),
                mode="lines",
                line=dict(color="rgba(37,99,235,0)"),
                hoverinfo="skip", showlegend=False,
            ))
            fig3.add_trace(go.Scatter(
                x=list(hp["target_date"]), y=list(hp["lower"]),
                mode="lines",
                line=dict(color="rgba(37,99,235,0)"),
                fill="tonexty", fillcolor="rgba(147,197,253,0.25)",
                name=f"hist ±{band_pct*100:.1f}% band",
                hoverinfo="skip",
            ))
        # Historical prediction markers + per-point error bars (redundant with
        # the fill but reads clearer for individual points)
        if len(hp):
            fig3.add_trace(go.Scatter(
                x=list(hp["target_date"]), y=list(hp["pred_close"]),
                mode="lines+markers", name="historical predictions",
                line=dict(color="#2563eb", width=1.4, dash="dot"),
                marker=dict(size=10, color="#2563eb", symbol="diamond",
                            line=dict(color="white", width=1)),
                error_y=dict(
                    type="data",
                    array=list(hp["upper"] - hp["pred_close"]),
                    arrayminus=list(hp["pred_close"] - hp["lower"]),
                    color="#93c5fd", thickness=1.2, width=4,
                ),
                hovertemplate=(
                    "target %{x|%Y-%m-%d}<br>"
                    "predicted $%{y:,.0f}<br>"
                    f"band ±{band_pct*100:.1f}%"
                    "<extra></extra>"
                ),
            ))
        # Realized closes (actuals) at each historical target date
        fig3.add_trace(go.Scatter(
            x=list(hist.index), y=list(hist["close"]),
            mode="lines+markers", name="realized close",
            line=dict(color="#1f2937", width=2),
            marker=dict(size=9, color="#1f2937"),
            hovertemplate="actual %{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
        ))
        if is_live:
            # Live mode: forward-looking +7d star (no actual exists yet).
            fig3.add_trace(go.Scatter(
                x=[hist.index[-1], cone7["pred_date"]],
                y=[hist["close"].iloc[-1], cone7["pred_close"]],
                mode="lines",
                line=dict(color="#2563eb", width=2, dash="dot"),
                hoverinfo="skip", showlegend=False,
            ))
            fig3.add_trace(go.Scatter(
                x=[cone7["pred_date"]], y=[cone7["pred_close"]],
                mode="markers+text", name="7-day forecast",
                marker=dict(size=14, color="#2563eb", symbol="star",
                            line=dict(color="white", width=1)),
                text=[f"${cone7['pred_close']:,.0f}"],
                textposition="top center",
                error_y=dict(
                    type="data",
                    array=[cone7["upper"] - cone7["pred_close"]],
                    arrayminus=[cone7["pred_close"] - cone7["lower"]],
                    color="#60a5fa", thickness=2, width=6,
                ),
                hovertemplate=(
                    "forecast %{x|%Y-%m-%d}<br>"
                    f"median ${cone7['pred_close']:,.0f}<br>"
                    f"band ±{band_pct*100:.1f}% "
                    f"→ ${cone7['lower']:,.0f}…${cone7['upper']:,.0f}"
                    "<extra></extra>"
                ),
            ))
            fig3.add_shape(
                type="rect",
                x0=cone7["pred_date"] - pd.Timedelta(days=1),
                x1=cone7["pred_date"] + pd.Timedelta(days=1),
                y0=cone7["lower"], y1=cone7["upper"],
                fillcolor="rgba(147,197,253,0.45)", line=dict(width=0),
                layer="below",
            )
        elif in_hist_with_actual:
            # Historical replay AND the last prediction has a realized
            # actual: highlight that actual with a distinct X marker.
            ap_close = cone7["actual_pred_close"]
            ap_date  = cone7["actual_pred_date"]
            ap_err   = (cone7["pred_close"] - ap_close) / ap_close * 100
            ap_inside = (cone7["lower"] <= ap_close <= cone7["upper"])
            fig3.add_trace(go.Scatter(
                x=[ap_date], y=[ap_close],
                mode="markers+text", name="actual at last prediction",
                marker=dict(size=14, color="#dc2626", symbol="x-thin",
                            line=dict(color="#dc2626", width=4)),
                text=[f"${ap_close:,.0f}"],
                textposition="bottom center",
                hovertemplate=(
                    "actual %{x|%Y-%m-%d}<br>"
                    f"$%{{y:,.0f}}<br>"
                    f"forecast error: {ap_err:+.2f}%<br>"
                    f"{'inside' if ap_inside else 'outside'} "
                    f"±{cone7['band_pct']*100:.1f}% band"
                    "<extra></extra>"
                ),
            ))
        # else: Historical replay where +7d is still unrealized — drop
        # that point entirely (user requested past-only points).
        fig3.update_layout(
            height=380, template="plotly_white",
            yaxis_title="BTC / USD",
            xaxis_title="Date",
            margin=dict(t=40, r=30, b=40, l=70),
            legend=dict(orientation="h", x=0, xanchor="left", y=1.10,
                        yanchor="bottom", bgcolor="rgba(255,255,255,0.95)",
                        bordercolor="#ccc", borderwidth=1, font=dict(size=11)),
        )
        fig3.update_xaxes(tickformat="%d-%b")
        st.plotly_chart(fig3, use_container_width=True,
                        key=f"chart_7d_cone_{'live' if is_live else 'hist'}")
        # Quick accuracy footnote on historical predictions
        accuracy_note = ""
        if len(hp):
            err_pct = (hp["pred_close"] - hp["actual_close"]).abs() / hp["actual_close"] * 100
            within_band = ((hp["actual_close"] >= hp["lower"])
                           & (hp["actual_close"] <= hp["upper"])).mean() * 100
            accuracy_note = (
                f"  Historical (last {len(hp)} weekly targets): MAPE = "
                f"{err_pct.mean():.2f} %; "
                f"{within_band:.0f} % of actuals fell inside the ±{band_pct*100:.1f} % band."
            )
        if is_live:
            legend_blurb = (
                "Diamonds = historical predictions made 7 days *before* each target; "
                "black dots = realized weekly closes; "
                "star = current +7-day forecast with band."
            )
        elif in_hist_with_actual:
            legend_blurb = (
                "Diamonds = historical predictions made 7 days *before* each target; "
                "black dots = realized weekly closes; "
                "red ✕ at the last point = realized close at the most recent "
                "7-day target (Historical Replay shows past predictions only)."
            )
        else:
            legend_blurb = (
                "Diamonds = historical predictions made 7 days *before* each target; "
                "black dots = realized weekly closes. "
                "(Historical Replay shows only past predictions whose target "
                f"date has a realized close — the +7-day target "
                f"{cone7['pred_date'].strftime('%Y-%m-%d')} is still in the future.)"
            )
        st.caption(
            legend_blurb + " The fixed ±9.7 % interval corresponds to ≈ 88 % "
            "empirical coverage on the held-out 8-month tail "
            "(see `notebooks/btc_7d_close_research.ipynb`)." + accuracy_note
        )

    # ─────────────────────── live look-back metrics ───────────────────────
    if lb_metrics:
        st.subheader(f"Live look-back accuracy (last {mask.sum()} realised hours)")
        cols = st.columns(5)
        cols[0].metric("MAPE", f"{lb_metrics['MAPE']:.2f} %")
        cols[1].metric("Hit ±3 %", f"{lb_metrics['hit3']:.1f} %")
        cols[2].metric("Hit ±1 %", f"{lb_metrics['hit1']:.1f} %")
        cols[3].metric("Hit ±0.5 %", f"{lb_metrics['hit0.5']:.1f} %")
        cols[4].metric("Direction acc.", f"{lb_metrics['dir_acc']:.1f} %")

    # ─────────────────────── extra context: features now ──────────────────
    with st.expander("🔍 Latest feature snapshot (top contributors)"):
        imp = A.get("importance", None)
        if imp is not None:
            topf = imp.sort_values(ascending=False).head(15).index.tolist()
            snap = F.loc[latest_t, topf].rename("value").to_frame()
            snap["importance"] = imp.loc[topf].round(5)
            st.dataframe(snap.round(5), use_container_width=True)

    st.markdown("---")
    # Surface data freshness so user can verify the chart is current
    lag_min = (now_utc - latest_t).total_seconds() / 60
    st.caption(
        f"Rolling forecast target: **now + 1 h** → "
        f"**{forecast_target.strftime('%Y-%m-%d %H:%M:%S UTC')}** (slides each refresh).  "
        f"Hourly Yahoo bar latest = {latest_t.strftime('%H:%M UTC')} ({lag_min:.0f} min behind).  "
        f"Live Binance spot = "
        + (f"{live_spot_ts.strftime('%H:%M:%S UTC')}"
           if live_spot_ts else "_unavailable_") + ".  "
        f"Page auto-refresh every **{REFRESH_SECONDS}s**.  "
        f"Last run: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}."
    )



# ════════════════════════════════════════════════════════════════════════
# Tabs: Live | Historical
# ════════════════════════════════════════════════════════════════════════
tab_live, tab_hist = st.tabs(["🔴 Live (rolling now+1h)", "🕒 Historical replay"])

with tab_live:
    render_dashboard(latest_t_global, is_live=True,
                     live_spot=live_spot, live_spot_ts=live_spot_ts)

with tab_hist:
    valid_times = F_filled.index[valid_mask]
    if len(valid_times) < 30:
        st.error("Not enough historical data available yet.")
    else:
        CT_TZ = "America/Chicago"
        min_t = valid_times.min(); max_t = valid_times.max()
        min_t_ct = min_t.tz_localize("UTC").tz_convert(CT_TZ).tz_localize(None)
        max_t_ct = max_t.tz_localize("UTC").tz_convert(CT_TZ).tz_localize(None)
        min_date = min_t_ct.date()
        data_max_date = (max_t_ct - pd.Timedelta(hours=1)).date()
        # Historical replay is for PAST dates only — today's bar is in progress
        # (it hasn't closed at 7am CT next day yet) so we exclude it.
        today_ct = pd.Timestamp.now(tz=CT_TZ).date()
        max_date = min(data_max_date, today_ct - timedelta(days=1))
        if "hist_date" not in st.session_state:
            st.session_state["hist_date"] = max_date

        # Compute slider bounds from the current picked_date (DST-correct).
        # Summer: 7am→7am CDT. Winter: 6am→6am CST. Matches the daily model's
        # 24h bar anchored at 12:00 UTC.
        picked_date = st.session_state["hist_date"]
        import datetime as _dt
        slider_win_start_utc = pd.Timestamp(picked_date) + pd.Timedelta(hours=ANCHOR_HOUR_UTC)
        slider_win_end_utc   = slider_win_start_utc + pd.Timedelta(days=1)
        slider_min = (slider_win_start_utc.tz_localize("UTC")
                      .tz_convert(CT_TZ).tz_localize(None).to_pydatetime())
        slider_max = (slider_win_end_utc.tz_localize("UTC")
                      .tz_convert(CT_TZ).tz_localize(None).to_pydatetime())
        # Sanitize stored hour value if the date changed and it's now out of range.
        prior = st.session_state.get("hist_hour_ts")
        if prior is None or not (slider_min <= prior <= slider_max):
            st.session_state["hist_hour_ts"] = slider_max

        # Pre-compute target_t / actual_t from session state so render_dashboard
        # can render KPIs FIRST, then the picker, then the plots. (The picker's
        # widgets read/write the same session_state keys; on user interaction
        # Streamlit reruns and these recompute consistently for the next pass.)
        picked_t_ct = pd.Timestamp(st.session_state["hist_hour_ts"])
        target_t = (picked_t_ct
                    .tz_localize(CT_TZ, ambiguous=True, nonexistent="shift_forward")
                    .tz_convert("UTC").tz_localize(None))
        avail = valid_times[valid_times <= target_t]
        actual_t = avail[-1] if len(avail) else None
        actual_t_ct = (actual_t.tz_localize("UTC").tz_convert(CT_TZ).tz_localize(None)
                       if actual_t is not None else None)

        # Callbacks (closures over min_date/max_date)
        def _shift_date(delta_days):
            cur = st.session_state.get("hist_date", max_date)
            new = cur + timedelta(days=delta_days)
            st.session_state["hist_date"] = max(min_date, min(max_date, new))

        def _select_date(d):
            st.session_state["hist_date"] = max(min_date, min(max_date, d))

        def _go_to_bookmark(iso_date_str):
            try:
                d = _date.fromisoformat(iso_date_str)
                st.session_state["hist_date"] = max(min_date, min(max_date, d))
            except Exception:
                pass

        bookmarks = load_bookmarks()

        def _hist_picker():
            """Renders date strip, calendar, hour slider, and bookmarks expander.
            Called from inside render_dashboard so it sits right above the plots."""
            st.markdown("#### 🕒 Pick a historical date / hour to replay")

            # ── Day strip: ◀  [-3] [-2] [-1] [SEL] [+1] [+2] [+3]  ▶ ──
            cur_picked = st.session_state.get("hist_date", max_date)
            strip_cols = st.columns([0.4, 1, 1, 1, 1, 1, 1, 1, 0.4])
            with strip_cols[0]:
                st.button("◀", key="hist_prev_day", help="Previous day",
                          on_click=_shift_date, args=(-1,),
                          disabled=(cur_picked <= min_date),
                          use_container_width=True)
            for i, offset in enumerate(range(-3, 4)):
                d = cur_picked + timedelta(days=offset)
                label = d.strftime("%a\n%b %-d")
                in_range = (min_date <= d <= max_date)
                is_selected = (d == cur_picked)
                with strip_cols[i + 1]:
                    st.button(
                        label, key=f"hist_pill_{offset}",
                        help=d.strftime("%Y-%m-%d (US Central)"),
                        on_click=_select_date, args=(d,),
                        disabled=(not in_range),
                        type=("primary" if is_selected else "secondary"),
                        use_container_width=True,
                    )
            with strip_cols[8]:
                st.button("▶", key="hist_next_day", help="Next day",
                          on_click=_shift_date, args=(1,),
                          disabled=(cur_picked >= max_date),
                          use_container_width=True)

            # ── Calendar + hour slider side-by-side ──
            cal_col, slider_col = st.columns([1, 2])
            with cal_col:
                st.date_input(
                    "Or pick from calendar (CT)",
                    min_value=min_date, max_value=max_date,
                    key="hist_date",
                )
            with slider_col:
                st.slider(
                    "Hour (US Central) — 7am picked → 7am next day",
                    min_value=slider_min, max_value=slider_max,
                    step=_dt.timedelta(hours=1),
                    format="MMM D, HH:mm",
                    key="hist_hour_ts",
                )

            # As-of caption (snap if needed)
            if actual_t is not None:
                if actual_t != target_t:
                    st.caption(f"⚠️ Snapped to **{actual_t_ct} CT** ({actual_t} UTC) "
                               f"— picked {picked_t_ct} CT")
                else:
                    st.caption(f"As-of: **{actual_t_ct} CT** ({actual_t} UTC)")

            # ── Bookmarks (collapsed by default since panel is mid-page) ──
            with st.expander(
                f"🔖 Bookmarks  ({sum(len(v) for v in bookmarks.values())} saved "
                f"across {len(bookmarks)} categor{'y' if len(bookmarks)==1 else 'ies'})",
                expanded=False,
            ):
                bk_browse, bk_save = st.columns([1.2, 1])
                with bk_browse:
                    st.markdown("**Jump to a bookmarked date**")
                    if not bookmarks:
                        st.caption("_No bookmarks yet. Save the current date on the right._")
                    else:
                        cat = st.selectbox("Category", sorted(bookmarks.keys()),
                                           key="bk_cat_pick")
                        entries = bookmarks.get(cat, [])
                        if entries:
                            opt_labels = [
                                f"{e['date']}" + (f" — {e['label']}" if e.get("label") else "")
                                for e in entries
                            ]
                            idx = st.selectbox(
                                "Date", range(len(entries)),
                                format_func=lambda i: opt_labels[i],
                                key="bk_date_pick",
                            )
                            sel = entries[idx]
                            b1, b2 = st.columns([1, 1])
                            with b1:
                                st.button("Go to this date", key="bk_go",
                                          on_click=_go_to_bookmark, args=(sel["date"],),
                                          type="primary", use_container_width=True)
                            with b2:
                                st.button("🗑 Delete", key="bk_del",
                                          on_click=lambda c=cat, d=sel["date"]: delete_bookmark(c, d),
                                          use_container_width=True)
                with bk_save:
                    st.markdown("**Save current selection**")
                    cur_d = st.session_state.get("hist_date", max_date)
                    st.caption(f"Current date: **{cur_d.isoformat()}** (CT)")
                    cats = sorted(bookmarks.keys())
                    cat_choice = st.selectbox(
                        "Category", options=cats + ["➕ New category…"],
                        key="bk_save_cat",
                    )
                    if cat_choice == "➕ New category…":
                        new_cat = st.text_input("New category name", key="bk_new_cat",
                                                placeholder="e.g. Macro events")
                        final_cat = (new_cat or "").strip()
                    else:
                        final_cat = cat_choice
                    bk_label = st.text_input("Optional label", key="bk_label",
                                             placeholder="e.g. FOMC meeting")
                    if st.button("💾 Save bookmark", key="bk_save_btn",
                                 disabled=not final_cat, use_container_width=True):
                        add_bookmark(final_cat, cur_d, bk_label.strip())
                        st.success(f"Saved **{cur_d.isoformat()}** to *{final_cat}*")
                        st.rerun()

        if actual_t is None:
            st.error(f"No data available at or before {picked_t_ct} CT "
                     f"(= {target_t} UTC). Pick a different date/hour below.")
            _hist_picker()
        else:
            render_dashboard(actual_t, is_live=False,
                             live_spot=None, live_spot_ts=None,
                             hist_picker=_hist_picker)
# ─────────────────────── timer-driven re-run ──────────────────────────
time.sleep(REFRESH_SECONDS)
st.rerun()
