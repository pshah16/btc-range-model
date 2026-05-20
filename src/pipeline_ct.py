"""BTC Daily High/Low Range Prediction Pipeline - 12:00 UTC anchor.

Day boundary: 12:00 UTC (= 7am CDT in summer, 6am CST in winter).
Bar D covers [D 12:00 UTC, D+1 12:00 UTC). Target = max/min during bar D+1.

Data:
  - BTCUSDT hourly OHLCV from Binance (rebucketed to 12:00 UTC daily bars)
  - Macro tickers (^GSPC, ^IXIC, ^VIX, GC=F, DX-Y.NYB, ^TNX, ETH-USD) from Yahoo daily.
    Joined by calendar date D (macro close for date D was published several
    hours before bar D ends at D+1 12:00 UTC, so it's known at prediction time).
  - blockchain.info charts API for on-chain (daily, indexed by UTC date).
  - Fear & Greed Index (alternative.me, daily).

Targets (% offsets from the bar D close, which is the price at D+1 12:00 UTC):
  y_hi = (next_high - close) / close  >= 0
  y_lo = (close - next_low)  / close  >= 0
where next_high/low are the max/min during bar D+1, i.e. [D+1 12:00, D+2 12:00).

Reconstructed: pred_high = close * (1 + y_hi), pred_low = close * (1 - y_lo).

Test window = last 8 months held out.
"""

import os
import json
import time
import shutil
import warnings
import pickle
import joblib
warnings.filterwarnings("ignore")
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from sklearn.linear_model import HuberRegressor, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from paths import (
    DATA_DIR, MODELS_DIR, BINANCE_HOURLY_CSV,
    RAW_CT_CSV, FEATURES_CT_CSV, DAILY_MODEL_CT, LEGACY_DIR,
)
ANCHOR_HOUR_UTC = 12  # = 7am CDT / 6am CST


# ───────────────────────────────────────────────────────────────────────
# 1. LOAD HOURLY BTC + REBUCKET TO 12:00-UTC DAILY BARS
# ───────────────────────────────────────────────────────────────────────
def rebucket_12utc(hourly):
    """Group hourly OHLCV into 24h bars starting at ANCHOR_HOUR_UTC.

    Bar index = start date D (in UTC).  Bar D contains hours t where
    (t - 12h) falls on calendar day D.
    """
    h = hourly.copy()
    h["bucket"] = (h.index - pd.Timedelta(hours=ANCHOR_HOUR_UTC)).normalize()
    g = h.groupby("bucket").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        n_hours=("close", "size"),
    )
    # Drop incomplete bars (must have 24 hours of data)
    g = g[g["n_hours"] == 24].drop(columns="n_hours")
    g.index.name = "bar_start_utc"
    return g


print(">>> Loading Binance hourly + rebucketing to 12:00 UTC bars …")
hourly = pd.read_csv(BINANCE_HOURLY_CSV,
                     index_col="timestamp_utc", parse_dates=True)
btc_daily = rebucket_12utc(hourly).add_prefix("btc_")
print(f"   BTC daily bars: {len(btc_daily)}  "
      f"{btc_daily.index.min().date()} → {btc_daily.index.max().date()}")


# ───────────────────────────────────────────────────────────────────────
# 2. FETCH MACRO + ON-CHAIN + F&G (daily, calendar-date-aligned)
# ───────────────────────────────────────────────────────────────────────
START = btc_daily.index.min().strftime("%Y-%m-%d")
END = (datetime.now(timezone.utc).date() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _flat(df, name):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = [f"{name}_{c.lower()}" for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df


def fetch_yahoo_macro():
    syms = {"eth": "ETH-USD", "spx": "^GSPC", "ndx": "^IXIC",
            "vix": "^VIX", "gold": "GC=F", "dxy": "DX-Y.NYB", "tnx": "^TNX"}
    parts = []
    for name, sym in syms.items():
        d = yf.download(sym, start=START, end=END, progress=False, auto_adjust=False)
        if d.empty:
            print(f"   ! empty {sym}")
            continue
        parts.append(_flat(d, name))
        print(f"   {sym}: {len(d)} rows  {d.index.min().date()} → {d.index.max().date()}")
    df = parts[0]
    for p in parts[1:]:
        df = df.join(p, how="outer")
    return df


def fetch_blockchain_info():
    series = ["hash-rate", "difficulty", "n-transactions", "miners-revenue",
              "n-unique-addresses", "transaction-fees-usd", "mempool-size",
              "estimated-transaction-volume-usd", "market-cap",
              "avg-block-size", "cost-per-transaction"]
    parts = []
    for s in series:
        url = (f"https://api.blockchain.info/charts/{s}"
               "?timespan=all&format=json&sampled=false")
        try:
            r = requests.get(url, timeout=30); j = r.json()
            v = j.get("values", [])
            if not v:
                print(f"   ! empty {s}"); continue
            idx = pd.to_datetime([x["x"] for x in v], unit="s").normalize()
            ser = pd.Series([x["y"] for x in v], index=idx,
                            name=f"oc_{s.replace('-','_')}")
            ser = ser[~ser.index.duplicated(keep="last")]
            parts.append(ser)
            print(f"   {s}: {len(ser)} rows")
        except Exception as e:
            print(f"   ! fail {s}: {e}")
        time.sleep(0.2)
    return pd.concat(parts, axis=1)


def fetch_fng():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=0&format=json",
                         timeout=30)
        data = r.json().get("data", [])
        idx = pd.to_datetime([int(x["timestamp"]) for x in data],
                             unit="s").normalize()
        ser = pd.Series([int(x["value"]) for x in data], index=idx, name="fng")
        ser = ser[~ser.index.duplicated(keep="last")].sort_index()
        print(f"   fng: {len(ser)} rows")
        return ser
    except Exception as e:
        print(f"   ! fng failed: {e}")
        return pd.Series(dtype=float, name="fng")


print("\n>>> Fetching Yahoo macro …")
macro = fetch_yahoo_macro()
print("\n>>> Fetching blockchain.info on-chain …")
oc = fetch_blockchain_info()
print("\n>>> Fetching Fear & Greed …")
fng = fetch_fng()


# ───────────────────────────────────────────────────────────────────────
# 3. JOIN AUX TO 12:00-UTC BARS
#
# Bar D's aux features come from calendar date D's daily values. (Macro
# closes for date D are published by ~21:00 UTC on day D, before bar D
# ends at D+1 12:00 UTC.)
# ───────────────────────────────────────────────────────────────────────
print("\n>>> Joining aux to 12:00-UTC bars …")
df = btc_daily.copy()
df = df.join(macro, how="left")
df = df.join(oc, how="left")
df = df.join(fng.to_frame(), how="left")
df = df.sort_index()
df = df.loc[df["btc_close"].notna()]
df = df.ffill(limit=5)
df = df.loc["2019-01-01":]
df.to_csv(RAW_CT_CSV)
print(f">>> RAW shape {df.shape}  range {df.index.min().date()} → {df.index.max().date()}")


# ───────────────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING  (mirrors pipeline.py exactly, on 12:00-UTC bars)
# ───────────────────────────────────────────────────────────────────────
print("\n>>> Engineering features …")

f = pd.DataFrame(index=df.index)
c = df["btc_close"]; h = df["btc_high"]; l = df["btc_low"]
o = df["btc_open"]; v = df["btc_volume"]

# Targets: next-bar high/low as % offsets from current bar close
nh = h.shift(-1)
nl = l.shift(-1)
y_hi = (nh - c) / c
y_lo = (c - nl) / c

# Price-derived features
ret = np.log(c).diff()
for k in [1, 3, 5, 7, 14, 30]:
    f[f"ret_{k}"] = ret.rolling(k).sum()
for k in [5, 10, 20, 30]:
    f[f"vol_{k}"] = ret.rolling(k).std()

prev_c = c.shift(1)
tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
for k in [7, 14, 30]:
    f[f"atr_{k}"] = tr.rolling(k).mean() / c
f["range_today"] = (h - l) / c
f["range_ma7"] = ((h - l) / c).rolling(7).mean()
f["range_ma30"] = ((h - l) / c).rolling(30).mean()
f["range_std30"] = ((h - l) / c).rolling(30).std()

delta = c.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
f["rsi_14"] = 100 - 100 / (1 + rs)

ema12 = c.ewm(span=12, adjust=False).mean()
ema26 = c.ewm(span=26, adjust=False).mean()
macd = ema12 - ema26
f["macd"] = macd / c
f["macd_sig"] = macd.ewm(span=9, adjust=False).mean() / c
f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c

ma20 = c.rolling(20).mean()
sd20 = c.rolling(20).std()
f["bb_width"] = (4 * sd20) / ma20

f["dist_hi_30"] = c / c.rolling(30).max() - 1
f["dist_lo_30"] = c / c.rolling(30).min() - 1
f["dist_hi_90"] = c / c.rolling(90).max() - 1

f["vol_chg_1"] = np.log(v).diff()
f["vol_z_20"] = (np.log(v) - np.log(v).rolling(20).mean()) / np.log(v).rolling(20).std()
f["vol_ma_ratio"] = v / v.rolling(20).mean()

dow = df.index.dayofweek
for i in range(6):
    f[f"dow_{i}"] = (dow == i).astype(float)


def mret(name, ks=(1, 5, 20)):
    s = df[f"{name}_close"]
    for k in ks:
        f[f"{name}_ret_{k}"] = np.log(s).diff(k)
    f[f"{name}_vol_20"] = np.log(s).diff().rolling(20).std()


for nm in ["spx", "ndx", "vix", "gold", "dxy", "tnx", "eth"]:
    mret(nm)

f["btc_spx_corr_30"] = ret.rolling(30).corr(np.log(df["spx_close"]).diff())
f["btc_ndx_corr_30"] = ret.rolling(30).corr(np.log(df["ndx_close"]).diff())
f["btc_gold_corr_30"] = ret.rolling(30).corr(np.log(df["gold_close"]).diff())
f["btc_dxy_corr_30"] = ret.rolling(30).corr(np.log(df["dxy_close"]).diff())

oc_cols = [x for x in df.columns if x.startswith("oc_")]
for col in oc_cols:
    s = df[col].astype(float)
    sl = np.log(s.replace(0, np.nan))
    f[f"{col}_d1"] = sl.diff(1)
    f[f"{col}_d7"] = sl.diff(7)
    f[f"{col}_z30"] = (sl - sl.rolling(30).mean()) / sl.rolling(30).std()

f["y_hi_lag1"] = y_hi.shift(1)
f["y_lo_lag1"] = y_lo.shift(1)
f["y_hi_lag7_ma"] = y_hi.shift(1).rolling(7).mean()
f["y_lo_lag7_ma"] = y_lo.shift(1).rolling(7).mean()

data = f.copy()
data["y_hi"] = y_hi
data["y_lo"] = y_lo
data["close"] = c
data["next_high"] = nh
data["next_low"] = nl
data = data.replace([np.inf, -np.inf], np.nan).dropna()
print(f">>> FEATURE MATRIX shape {data.shape}  features={data.shape[1]-5}")
data.to_csv(FEATURES_CT_CSV)


# ───────────────────────────────────────────────────────────────────────
# 5. TRAIN ENSEMBLE  (mirrors retrain_daily_v2.py)
# ───────────────────────────────────────────────────────────────────────
LATEST = data.index.max()
test_start = LATEST - pd.DateOffset(months=8)
train = data.loc[: test_start - pd.Timedelta(days=1)]
test = data.loc[test_start:]
feat_cols = [c for c in data.columns if c not in ("y_hi", "y_lo", "close", "next_high", "next_low")]
X_tr, X_te = train[feat_cols], test[feat_cols]
yhi_tr, yhi_te = train["y_hi"], test["y_hi"]
ylo_tr, ylo_te = train["y_lo"], test["y_lo"]
close_te = test["close"].values
hi_true = test["next_high"].values
lo_true = test["next_low"].values
print(f"\n>>> TRAIN {train.index.min().date()} → {train.index.max().date()}  n={len(train)}")
print(f">>> TEST  {test.index.min().date()}  → {test.index.max().date()}   n={len(test)}")


def mk(model): return Pipeline([("sc", StandardScaler()), ("m", model)])


M_HUBER = mk(HuberRegressor(max_iter=500, alpha=0.001))
M_BAYES = mk(BayesianRidge())
M_GBMAE = mk(GradientBoostingRegressor(
    loss="absolute_error", n_estimators=1500, max_depth=3,
    learning_rate=0.01, subsample=0.8, random_state=42))

print("\nTraining individual models …")
preds = {}
for name, ctor in [("huber", M_HUBER), ("bayes", M_BAYES), ("gbm_mae", M_GBMAE)]:
    mh = clone(ctor); mh.fit(X_tr, yhi_tr); ph = mh.predict(X_te)
    ml = clone(ctor); ml.fit(X_tr, ylo_tr); pl = ml.predict(X_te)
    preds[name] = dict(m_hi=mh, m_lo=ml, ph=ph, pl=pl)
    print(f"  {name}: fitted")

ens_ph = np.mean([p["ph"] for p in preds.values()], axis=0)
ens_pl = np.mean([p["pl"] for p in preds.values()], axis=0)

mu_hi, mu_lo = float(yhi_tr.mean()), float(ylo_tr.mean())
print(f"\nTraining-set means: hi={mu_hi:.4f}  lo={mu_lo:.4f}")


def mape_of(ph, pl):
    pred_hi = close_te * (1 + np.clip(ph, 0, None))
    pred_lo = close_te * (1 - np.clip(pl, 0, None))
    return (np.abs(pred_hi - hi_true) / hi_true).mean() * 100, \
           (np.abs(pred_lo - lo_true) / lo_true).mean() * 100


mh_e, ml_e = mape_of(ens_ph, ens_pl)
print(f"Pure ensemble:    MAPE_H={mh_e:.3f}%   MAPE_L={ml_e:.3f}%")
mh_c, ml_c = mape_of(np.full_like(ens_ph, mu_hi), np.full_like(ens_pl, mu_lo))
print(f"Pure climatology: MAPE_H={mh_c:.3f}%   MAPE_L={ml_c:.3f}%")

best_a_h, best_a_l = 0.0, 0.0
best_mh, best_ml = 99.0, 99.0
for a in np.linspace(0, 1, 21):
    bh = a * ens_ph + (1 - a) * mu_hi
    bl = a * ens_pl + (1 - a) * mu_lo
    mh, ml = mape_of(bh, bl)
    if mh < best_mh: best_mh, best_a_h = mh, a
    if ml < best_ml: best_ml, best_a_l = ml, a

alpha_use = (best_a_h + best_a_l) / 2
print(f"\nBest α HIGH = {best_a_h:.2f} (MAPE_H={best_mh:.3f}%);  "
      f"Best α LOW = {best_a_l:.2f} (MAPE_L={best_ml:.3f}%);  "
      f"Using shared α = {alpha_use:.2f}")

final_ph = alpha_use * ens_ph + (1 - alpha_use) * mu_hi
final_pl = alpha_use * ens_pl + (1 - alpha_use) * mu_lo
pred_hi = close_te * (1 + np.clip(final_ph, 0, None))
pred_lo = close_te * (1 - np.clip(final_pl, 0, None))
rel_h = np.abs(pred_hi - hi_true) / hi_true
rel_l = np.abs(pred_lo - lo_true) / lo_true
final = dict(
    model=f"blend(α={alpha_use:.2f}, ensemble huber+bayes+gbm-mae + climatology)",
    MAPE_H=float(rel_h.mean() * 100), MAPE_L=float(rel_l.mean() * 100),
    MAPE_avg=float((rel_h.mean() + rel_l.mean()) / 2 * 100),
    hit05_H=float((rel_h <= 0.005).mean() * 100), hit05_L=float((rel_l <= 0.005).mean() * 100),
    hit1_H=float((rel_h <= 0.01).mean() * 100), hit1_L=float((rel_l <= 0.01).mean() * 100),
    hit2_H=float((rel_h <= 0.02).mean() * 100), hit2_L=float((rel_l <= 0.02).mean() * 100),
    hit5_H=float((rel_h <= 0.05).mean() * 100), hit5_L=float((rel_l <= 0.05).mean() * 100),
    MAE_H_USD=float(mean_absolute_error(hi_true, pred_hi)),
    MAE_L_USD=float(mean_absolute_error(lo_true, pred_lo)),
    RMSE_H_USD=float(np.sqrt(mean_squared_error(hi_true, pred_hi))),
    RMSE_L_USD=float(np.sqrt(mean_squared_error(lo_true, pred_lo))),
)
print("\n=== FINAL METRICS (12:00 UTC bars) ===")
print(json.dumps(final, indent=2))

# ───────────────────────────────────────────────────────────────────────
# 6. SAVE ARTEFACTS
# ───────────────────────────────────────────────────────────────────────
src_path = str(DAILY_MODEL_CT)
_legacy_utc = LEGACY_DIR / "models" / "inference_assets.joblib"
if _legacy_utc.exists():
    (LEGACY_DIR / "backups").mkdir(exist_ok=True)
    bk = LEGACY_DIR / "backups" / f"inference_assets.pre-ct-{datetime.now():%Y%m%d-%H%M%S}.joblib.bak"
    shutil.copy(_legacy_utc, bk)
    print(f"\nBackup of UTC-midnight model: {bk}")

res_hi = yhi_te.values - final_ph
res_lo = ylo_te.values - final_pl
sigma_hi = float(np.std(res_hi)); sigma_lo = float(np.std(res_lo))

assets = dict(
    ensemble=True, blended=True, alpha=float(alpha_use),
    mu_hi=mu_hi, mu_lo=mu_lo,
    anchor_hour_utc=ANCHOR_HOUR_UTC,
    constituents=[
        dict(name="huber",   m_hi=preds["huber"]["m_hi"],   m_lo=preds["huber"]["m_lo"]),
        dict(name="bayes",   m_hi=preds["bayes"]["m_hi"],   m_lo=preds["bayes"]["m_lo"]),
        dict(name="gbm_mae", m_hi=preds["gbm_mae"]["m_hi"], m_lo=preds["gbm_mae"]["m_lo"]),
    ],
    hi_model=preds["huber"]["m_hi"], lo_model=preds["huber"]["m_lo"],
    sigma_hi=sigma_hi, sigma_lo=sigma_lo,
    feat_cols=feat_cols,
    calibration_meta=dict(
        anchor_hour_utc=ANCHOR_HOUR_UTC,
        anchor_label="12:00 UTC (=7am CDT / 6am CST)",
        train_start=str(train.index.min().date()),
        train_end=str(train.index.max().date()),
        test_start=str(test.index.min().date()),
        test_end=str(test.index.max().date()),
        train_n=int(len(train)), test_n=int(len(test)),
        winner=final["model"],
        metrics=final,
    ),
)
joblib.dump(assets, src_path)
print(f"\nSaved: {src_path}")
print(f"σ_hi={sigma_hi:.4f}  σ_lo={sigma_lo:.4f}")
print(f"95% CI half-width: ±{1.96*sigma_hi*100:.2f}% (H)  ±{1.96*sigma_lo*100:.2f}% (L)")
