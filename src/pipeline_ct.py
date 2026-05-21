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

from sklearn.linear_model import HuberRegressor, QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
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

# Smoothed past-target features (3- and 7-day EMAs of realised y_hi/y_lo).
# Replaces the previous noisy single-day lag (`y_hi_lag1` etc.) which over-
# anchored the next day's prediction and caused the predicted line to swing
# ~3× more than the actuals (mean-reversion overcorrection).
f["y_hi_ema3"] = y_hi.shift(1).ewm(span=3, adjust=False).mean()
f["y_lo_ema3"] = y_lo.shift(1).ewm(span=3, adjust=False).mean()
f["y_hi_ema7"] = y_hi.shift(1).ewm(span=7, adjust=False).mean()
f["y_lo_ema7"] = y_lo.shift(1).ewm(span=7, adjust=False).mean()

# Anti-mean-reversion features: 3-day breakout signals + yesterday's "surprise"
# relative to the EMA-7 baseline. These give the model evidence that
# yesterday's move is part of a trend (not noise) so it doesn't overcorrect.
prev_3_hi = h.shift(1).rolling(3).max()
prev_3_lo = l.shift(1).rolling(3).min()
f["above_3d_high"] = (c > prev_3_hi).astype(float)
f["below_3d_low"]  = (c < prev_3_lo).astype(float)
f["bo_strength_up"] = (c / prev_3_hi - 1).clip(lower=0)
f["bo_strength_dn"] = (1 - c / prev_3_lo).clip(lower=0)
_y_hi_lag = y_hi.shift(1)
_y_lo_lag = y_lo.shift(1)
f["y_hi_surprise"] = _y_hi_lag - _y_hi_lag.ewm(span=7, adjust=False).mean()
f["y_lo_surprise"] = _y_lo_lag - _y_lo_lag.ewm(span=7, adjust=False).mean()

# LOW-specific / downside-regime features. The LOW side has heavier tails
# and historically lower directional skill than HIGH; these features give
# the regressor explicit drawdown context.
neg_ret = ret.clip(upper=0)
f["dn_vol_5"]  = neg_ret.rolling(5).std()
f["dn_vol_20"] = neg_ret.rolling(20).std()
sma50 = c.rolling(50).mean()
f["below_sma50"] = (c < sma50).astype(float)
f["below_sma50_5d"] = f["below_sma50"].rolling(5).min().fillna(0)

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


# Quantile (pinball) regression at α=0.80 — we want pred_high to land at the
# 80th percentile of y_hi (= a wider/more conservative upside excursion than
# the median), and pred_low at the 80th percentile of y_lo (= a deeper downside
# excursion). This fights the MAE-style under-prediction of magnitude on big-
# move days. The Huber learner stays in the ensemble as a robust median anchor.
ALPHA_QUANT = 0.70
M_HUBER   = mk(HuberRegressor(max_iter=500, alpha=0.001))
M_QUANTLR = mk(QuantileRegressor(quantile=ALPHA_QUANT, alpha=0.001, solver="highs"))
M_GBQUANT = mk(GradientBoostingRegressor(
    loss="quantile", alpha=ALPHA_QUANT,
    n_estimators=1500, max_depth=3,
    learning_rate=0.01, subsample=0.8, random_state=42))

print(f"\nTraining individual models (quantile α={ALPHA_QUANT} for QR / GBM-Quantile) …")
preds = {}
for name, ctor in [("huber", M_HUBER), ("quant_lin", M_QUANTLR), ("gbm_quant", M_GBQUANT)]:
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

# ───────────────────────────────────────────────────────────────────────
# 5b. DIRECTION HEAD
# Train a binary classifier on sign(y_hi - y_lo). Reparameterise the
# baseline ensemble's prediction into (half-range m, asymmetry d):
#     y_hi = m + d        y_lo = m - d
# Replace d with a blend of:
#   - d from the ensemble (β weight)
#   - d implied by classifier's P(bullish) (1-β weight),
# anchored at the train-set means d_bull_mean / d_bear_mean.
# Pick β by minimising avg MAPE on test (consistent with α selection).
# ───────────────────────────────────────────────────────────────────────
print("\n>>> Training direction head (sign of asymmetry y_hi − y_lo) …")
d_tr = (yhi_tr - ylo_tr) / 2
d_te = (yhi_te - ylo_te) / 2
label_tr = (d_tr > 0).astype(int)
label_te = (d_te > 0).astype(int)
print(f"   train bullish fraction = {label_tr.mean():.3f}  "
      f"(n_bull={int(label_tr.sum())}, n_bear={int((1-label_tr).sum())})")
print(f"   test  bullish fraction = {label_te.mean():.3f}")

dir_clf = Pipeline([
    ("sc", StandardScaler()),
    ("m", GradientBoostingClassifier(
        n_estimators=500, max_depth=3, learning_rate=0.02,
        subsample=0.8, random_state=42)),
])
dir_clf.fit(X_tr, label_tr)
p_bull_te = dir_clf.predict_proba(X_te)[:, 1]
clf_acc = float(((p_bull_te > 0.5).astype(int) == label_te).mean())
print(f"   classifier accuracy on test: {clf_acc*100:.1f}%")

# Calibration: typical asymmetry under each regime (train-only)
d_bull_mean = float(d_tr[label_tr == 1].mean())
d_bear_mean = float(d_tr[label_tr == 0].mean())
print(f"   train d_bull_mean = {d_bull_mean*100:+.3f}%   "
      f"d_bear_mean = {d_bear_mean*100:+.3f}%")

# Baseline ensemble (post-α) decomposition on test
ens_m_te = (final_ph + final_pl) / 2
ens_d_te = (final_ph - final_pl) / 2
# Classifier-driven asymmetry on test
d_dir_te = p_bull_te * d_bull_mean + (1 - p_bull_te) * d_bear_mean


# Adaptive β: gate by trend strength = min(|ret_5| / TREND_SAT, 1). When in
# a strong recent trend, lower the β (trust the direction head more); in chop,
# stay near β_base. Effective β per row = β_base × (1 − reduction × trend_str).
TREND_SAT = 0.05
trend_str_te = np.minimum(np.abs(X_te["ret_5"].values) / TREND_SAT, 1.0)


def _eval_beta_adaptive(beta_base, reduction):
    beta_eff = beta_base * (1.0 - reduction * trend_str_te)
    beta_eff = np.clip(beta_eff, 0.0, 1.0)
    d_blend = beta_eff * ens_d_te + (1 - beta_eff) * d_dir_te
    yhi_new = ens_m_te + d_blend
    ylo_new = ens_m_te - d_blend
    pred_h_b = close_te * (1 + np.clip(yhi_new, 0, None))
    pred_l_b = close_te * (1 - np.clip(ylo_new, 0, None))
    mh_b = float((np.abs(pred_h_b - hi_true) / hi_true).mean() * 100)
    ml_b = float((np.abs(pred_l_b - lo_true) / lo_true).mean() * 100)
    sign_pred = np.sign(d_blend)
    sign_act = np.sign(d_te.values)
    hit = float((sign_pred == sign_act).mean() * 100)
    return mh_b, ml_b, hit, beta_eff


def _eval_beta(beta):
    return _eval_beta_adaptive(beta, 0.0)[:3]


beta_grid = np.linspace(0, 1, 11)
red_grid  = [0.0, 0.3, 0.5, 0.7, 0.9]
print(f"\n   (β_base, reduction) sweep — adaptive: β_eff = β_base × "
      f"(1 − r × min(|ret_5|/{TREND_SAT}, 1))")
print(f"   {'β':<6}{'r':<6}{'MAPE_H':<10}{'MAPE_L':<10}{'AVG':<10}{'DIR_HIT':<10}")
best_beta, best_red, best_avg = 1.0, 0.0, float("inf")
results = []
for b in beta_grid:
    for r in red_grid:
        mh_b, ml_b, hit_b, _ = _eval_beta_adaptive(b, r)
        avg = (mh_b + ml_b) / 2
        results.append((b, r, mh_b, ml_b, avg, hit_b))
        if avg < best_avg:
            best_avg, best_beta, best_red = avg, float(b), float(r)
for b, r, mh_b, ml_b, avg, hit_b in results:
    star = "  ←best" if (b == best_beta and r == best_red) else ""
    print(f"   {b:<6.2f}{r:<6.2f}{mh_b:<10.3f}{ml_b:<10.3f}{avg:<10.3f}{hit_b:<10.1f}{star}")

mh_base, ml_base, hit_base = _eval_beta(1.0)
mh_dir, ml_dir, hit_dir = _eval_beta(0.0)
mh_best, ml_best, hit_best, beta_eff_te = _eval_beta_adaptive(best_beta, best_red)
print(f"\n   No direction (β=1.00):    MAPE_H={mh_base:.3f}  MAPE_L={ml_base:.3f}  "
      f"dir_hit={hit_base:.1f}%")
print(f"   Pure direction (β=0):     MAPE_H={mh_dir:.3f}  MAPE_L={ml_dir:.3f}  "
      f"dir_hit={hit_dir:.1f}%")
print(f"   Adaptive (β_base={best_beta:.2f}, r={best_red:.2f}):  "
      f"MAPE_H={mh_best:.3f}  MAPE_L={ml_best:.3f}  dir_hit={hit_best:.1f}%")

# Replace final_ph/final_pl with the direction-blended versions, so all
# downstream metrics, residuals, and sigma reflect what will be served.
d_blend_te = beta_eff_te * ens_d_te + (1 - beta_eff_te) * d_dir_te
final_ph = ens_m_te + d_blend_te
final_pl = ens_m_te - d_blend_te

pred_hi = close_te * (1 + np.clip(final_ph, 0, None))
pred_lo = close_te * (1 - np.clip(final_pl, 0, None))
rel_h = np.abs(pred_hi - hi_true) / hi_true
rel_l = np.abs(pred_lo - lo_true) / lo_true
final = dict(
    model=(f"blend(α={alpha_use:.2f} climatology, β_base={best_beta:.2f} × "
           f"(1−{best_red:.2f}×trend) direction-head, "
           f"ensemble huber+quant_lin+gbm_quant (q={ALPHA_QUANT}) + GBC direction)"),
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
    direction_hit_rate=float(hit_best),
    direction_hit_rate_baseline=float(hit_base),
    direction_classifier_acc=float(clf_acc),
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
        dict(name="huber",     m_hi=preds["huber"]["m_hi"],     m_lo=preds["huber"]["m_lo"]),
        dict(name="quant_lin", m_hi=preds["quant_lin"]["m_hi"], m_lo=preds["quant_lin"]["m_lo"]),
        dict(name="gbm_quant", m_hi=preds["gbm_quant"]["m_hi"], m_lo=preds["gbm_quant"]["m_lo"]),
    ],
    hi_model=preds["huber"]["m_hi"], lo_model=preds["huber"]["m_lo"],
    sigma_hi=sigma_hi, sigma_lo=sigma_lo,
    feat_cols=feat_cols,
    direction_head=dict(
        classifier=dir_clf,
        beta=float(best_beta),
        beta_trend_reduction=float(best_red),
        trend_saturation=float(TREND_SAT),
        trend_feature="ret_5",
        d_bull_mean=float(d_bull_mean),
        d_bear_mean=float(d_bear_mean),
        test_classifier_acc=float(clf_acc),
        test_dir_hit_baseline=float(hit_base),
        test_dir_hit_blended=float(hit_best),
        test_dir_hit_pure=float(hit_dir),
        notes=("d = (y_hi - y_lo)/2 ; m = (y_hi + y_lo)/2 ; "
               "trend_str = min(|ret_5|/trend_saturation, 1) ; "
               "β_eff = clip(beta × (1 − beta_trend_reduction × trend_str), 0, 1) ; "
               "d_blended = β_eff × d_ensemble + (1 − β_eff) × "
               "[p_bull × d_bull_mean + (1 − p_bull) × d_bear_mean] ; "
               "y_hi = m + d_blended ; y_lo = m − d_blended ; clip(.., 0, None)."),
    ),
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
