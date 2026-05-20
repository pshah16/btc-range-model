"""
BTC Daily High/Low Range Prediction Pipeline.

Data:
  - Yahoo Finance (BTC-USD, ^GSPC, ^IXIC, ^VIX, GC=F, DX-Y.NYB, ^TNX, ETH-USD)
  - blockchain.info charts API (hash rate, active addresses, miner revenue, etc.)

Targets (predicted as % offsets from today's close):
  hi_pct = (next_high - close) / close   >= 0
  lo_pct = (close - next_low)  / close   >= 0
Then reconstruct: pred_high = close * (1 + hi_pct), pred_low = close * (1 - lo_pct).

Test window = last 8 months. Models trained on data before that.
"""

import os, json, time, warnings, pickle
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUT = "/home/jovyan/btc-range-model"
os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------------------- #
# 1. DATA FETCH
# --------------------------------------------------------------------------- #
START = "2019-01-01"
TODAY = datetime.utcnow().date()
END   = TODAY.strftime("%Y-%m-%d")

def _flat(df, name):
    """Flatten yfinance multiindex to simple columns prefixed by name."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = [f"{name}_{c.lower()}" for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df

def fetch_yahoo():
    syms = {
        "btc":  "BTC-USD",
        "eth":  "ETH-USD",
        "spx":  "^GSPC",
        "ndx":  "^IXIC",
        "vix":  "^VIX",
        "gold": "GC=F",
        "dxy":  "DX-Y.NYB",
        "tnx":  "^TNX",
    }
    parts = []
    for name, sym in syms.items():
        d = yf.download(sym, start=START, end=END, progress=False, auto_adjust=False)
        if d.empty:
            print(f"  ! empty {sym}")
            continue
        parts.append(_flat(d, name))
        print(f"  {sym}: {len(d)} rows  {d.index.min().date()} -> {d.index.max().date()}")
    df = parts[0]
    for p in parts[1:]:
        df = df.join(p, how="outer")
    return df

def fetch_blockchain_info():
    """Fetch on-chain series from blockchain.info charts API (no key)."""
    series = [
        "hash-rate", "difficulty", "n-transactions", "miners-revenue",
        "n-unique-addresses", "transaction-fees-usd", "mempool-size",
        "estimated-transaction-volume-usd", "market-cap",
        "avg-block-size", "cost-per-transaction",
    ]
    parts = []
    for s in series:
        url = ("https://api.blockchain.info/charts/" + s
               + "?timespan=all&format=json&sampled=false")
        try:
            r = requests.get(url, timeout=30)
            j = r.json()
            v = j.get("values", [])
            if not v:
                print(f"  ! empty {s}"); continue
            idx = pd.to_datetime([x["x"] for x in v], unit="s").normalize()
            vals = [x["y"] for x in v]
            ser = pd.Series(vals, index=idx, name=f"oc_{s.replace('-','_')}")
            ser = ser[~ser.index.duplicated(keep="last")]
            parts.append(ser)
            print(f"  {s}: {len(ser)} rows  {ser.index.min().date()} -> {ser.index.max().date()}")
        except Exception as e:
            print(f"  ! fail {s}: {e}")
        time.sleep(0.2)
    return pd.concat(parts, axis=1)

print(">>> Fetching Yahoo data ...")
mkt = fetch_yahoo()
print(">>> Fetching blockchain.info on-chain ...")
oc = fetch_blockchain_info()

df = mkt.join(oc, how="left")
df = df.sort_index()
df = df.loc[df["btc_close"].notna()]  # keep BTC trading days (every day for crypto)
df = df.ffill(limit=5)                 # macro fwd-fill across crypto weekends/holidays
df = df.loc["2019-01-01":]
df.to_csv(f"{OUT}/raw.csv")
print(f">>> RAW shape {df.shape}  range {df.index.min().date()} -> {df.index.max().date()}")
print(df.tail(2).T.head(15))

# --------------------------------------------------------------------------- #
# 2. FEATURE ENGINEERING
# --------------------------------------------------------------------------- #
print("\n>>> Engineering features ...")

f = pd.DataFrame(index=df.index)
c = df["btc_close"]; h = df["btc_high"]; l = df["btc_low"]
o = df["btc_open"]; v = df["btc_volume"]

# --- Targets: next-day high/low as % offsets from today's close ---
nh = h.shift(-1)
nl = l.shift(-1)
y_hi = (nh - c) / c    # positive
y_lo = (c - nl) / c    # positive
y_range = (nh - nl) / c

# --- Price-derived features ---
ret = np.log(c).diff()
for k in [1, 3, 5, 7, 14, 30]:
    f[f"ret_{k}"] = ret.rolling(k).sum()
for k in [5, 10, 20, 30]:
    f[f"vol_{k}"] = ret.rolling(k).std()

# True range / ATR
prev_c = c.shift(1)
tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
for k in [7, 14, 30]:
    f[f"atr_{k}"] = tr.rolling(k).mean() / c
# Today's range (very strong predictor of tomorrow's range)
f["range_today"] = (h - l) / c
f["range_ma7"]   = ((h - l) / c).rolling(7).mean()
f["range_ma30"]  = ((h - l) / c).rolling(30).mean()
f["range_std30"] = ((h - l) / c).rolling(30).std()

# RSI(14)
delta = c.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
f["rsi_14"] = 100 - 100 / (1 + rs)

# MACD
ema12 = c.ewm(span=12, adjust=False).mean()
ema26 = c.ewm(span=26, adjust=False).mean()
macd = ema12 - ema26
f["macd"]      = macd / c
f["macd_sig"]  = macd.ewm(span=9, adjust=False).mean() / c
f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c

# Bollinger band width
ma20 = c.rolling(20).mean()
sd20 = c.rolling(20).std()
f["bb_width"] = (4 * sd20) / ma20

# Distance from recent extremes
f["dist_hi_30"] = c / c.rolling(30).max() - 1
f["dist_lo_30"] = c / c.rolling(30).min() - 1
f["dist_hi_90"] = c / c.rolling(90).max() - 1

# Volume features
f["vol_chg_1"]  = np.log(v).diff()
f["vol_z_20"]   = (np.log(v) - np.log(v).rolling(20).mean()) / np.log(v).rolling(20).std()
f["vol_ma_ratio"] = v / v.rolling(20).mean()

# Day-of-week (BTC weekend behavior is real)
dow = df.index.dayofweek
for i in range(6):
    f[f"dow_{i}"] = (dow == i).astype(float)

# --- Macro / cross-market features ---
def mret(name, ks=(1, 5, 20)):
    s = df[f"{name}_close"]
    for k in ks:
        f[f"{name}_ret_{k}"] = np.log(s).diff(k)
    f[f"{name}_vol_20"] = np.log(s).diff().rolling(20).std()

for nm in ["spx", "ndx", "vix", "gold", "dxy", "tnx", "eth"]:
    mret(nm)

# BTC-equity correlation rolling
f["btc_spx_corr_30"] = ret.rolling(30).corr(np.log(df["spx_close"]).diff())
f["btc_ndx_corr_30"] = ret.rolling(30).corr(np.log(df["ndx_close"]).diff())
f["btc_gold_corr_30"] = ret.rolling(30).corr(np.log(df["gold_close"]).diff())
f["btc_dxy_corr_30"] = ret.rolling(30).corr(np.log(df["dxy_close"]).diff())

# --- On-chain features (deltas + z-scores so they're stationary) ---
oc_cols = [x for x in df.columns if x.startswith("oc_")]
for col in oc_cols:
    s = df[col].astype(float)
    sl = np.log(s.replace(0, np.nan))
    f[f"{col}_d1"]  = sl.diff(1)
    f[f"{col}_d7"]  = sl.diff(7)
    f[f"{col}_z30"] = (sl - sl.rolling(30).mean()) / sl.rolling(30).std()

# Target lag (helps: range is autocorrelated)
f["y_hi_lag1"] = y_hi.shift(1)
f["y_lo_lag1"] = y_lo.shift(1)
f["y_hi_lag7_ma"] = y_hi.shift(1).rolling(7).mean()
f["y_lo_lag7_ma"] = y_lo.shift(1).rolling(7).mean()

# Combine
data = f.copy()
data["y_hi"]    = y_hi
data["y_lo"]    = y_lo
data["close"]   = c
data["next_high"] = nh
data["next_low"]  = nl
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()
print(f">>> FEATURE MATRIX shape {data.shape}  features={data.shape[1]-5}")

data.to_csv(f"{OUT}/features.csv")

# --------------------------------------------------------------------------- #
# 3. TRAIN / TEST SPLIT  (last 8 months held out)
# --------------------------------------------------------------------------- #
test_start = pd.Timestamp(TODAY) - pd.DateOffset(months=8)
train = data.loc[:test_start - pd.Timedelta(days=1)]
test  = data.loc[test_start:]
print(f"\n>>> TRAIN {train.index.min().date()} -> {train.index.max().date()}  n={len(train)}")
print(f">>> TEST  {test.index.min().date()}  -> {test.index.max().date()}   n={len(test)}")

feat_cols = [c for c in data.columns if c not in ("y_hi","y_lo","close","next_high","next_low")]
X_tr, X_te = train[feat_cols], test[feat_cols]
yhi_tr, yhi_te = train["y_hi"], test["y_hi"]
ylo_tr, ylo_te = train["y_lo"], test["y_lo"]

# --------------------------------------------------------------------------- #
# 4. MODELS
# --------------------------------------------------------------------------- #
print("\n>>> Training models ...")

def make_pipeline(model):
    return Pipeline([("sc", StandardScaler()), ("m", model)])

models = {
    "ridge": lambda: make_pipeline(RidgeCV(alphas=np.logspace(-3, 3, 13))),
    "gbm":   lambda: make_pipeline(GradientBoostingRegressor(
                n_estimators=600, max_depth=3, learning_rate=0.03,
                subsample=0.8, random_state=42)),
    "rf":    lambda: make_pipeline(RandomForestRegressor(
                n_estimators=400, max_depth=None, min_samples_leaf=5,
                n_jobs=-1, random_state=42)),
}

def fit_eval(name, mk, X_tr, y_tr, X_te, y_te, tag):
    m = mk()
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    r2   = r2_score(y_te, pred)
    mae  = mean_absolute_error(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    print(f"  {tag}/{name}: R2={r2:.3f}  MAE={mae:.4f}  RMSE={rmse:.4f}")
    return m, pred

results = {}
for name in models:
    mh, ph = fit_eval(name, models[name], X_tr, yhi_tr, X_te, yhi_te, "HI")
    ml, pl = fit_eval(name, models[name], X_tr, ylo_tr, X_te, ylo_te, "LO")
    results[name] = (mh, ml, ph, pl)

# --------------------------------------------------------------------------- #
# 5. EVALUATION:  reconstruct high/low and compute hit rates
# --------------------------------------------------------------------------- #
print("\n>>> Evaluation on last 8 months ...")
close_te = test["close"].values
high_true = test["next_high"].values
low_true  = test["next_low"].values

def evaluate(ph, pl):
    pred_hi = close_te * (1 + np.clip(ph, 0, None))
    pred_lo = close_te * (1 - np.clip(pl, 0, None))
    mae_h = mean_absolute_error(high_true, pred_hi)
    mae_l = mean_absolute_error(low_true,  pred_lo)
    mape_h = np.mean(np.abs(pred_hi - high_true) / high_true) * 100
    mape_l = np.mean(np.abs(pred_lo - low_true)  / low_true)  * 100
    rmse_h = np.sqrt(mean_squared_error(high_true, pred_hi))
    rmse_l = np.sqrt(mean_squared_error(low_true,  pred_lo))
    hit5_h = np.mean(np.abs(pred_hi - high_true) / high_true <= 0.05) * 100
    hit5_l = np.mean(np.abs(pred_lo - low_true)  / low_true  <= 0.05) * 100
    hit10_h = np.mean(np.abs(pred_hi - high_true) / high_true <= 0.10) * 100
    hit10_l = np.mean(np.abs(pred_lo - low_true)  / low_true  <= 0.10) * 100
    # Predicted range MAPE
    pred_range = pred_hi - pred_lo
    true_range = high_true - low_true
    mape_range = np.mean(np.abs(pred_range - true_range) / true_range) * 100
    hit5_range = np.mean(np.abs(pred_range - true_range) / true_range <= 0.05) * 100
    return dict(mae_h=mae_h, mae_l=mae_l, mape_h=mape_h, mape_l=mape_l,
                rmse_h=rmse_h, rmse_l=rmse_l, hit5_h=hit5_h, hit5_l=hit5_l,
                hit10_h=hit10_h, hit10_l=hit10_l,
                mape_range=mape_range, hit5_range=hit5_range,
                pred_hi=pred_hi, pred_lo=pred_lo)

rep = {}
for name, (mh, ml, ph, pl) in results.items():
    rep[name] = evaluate(ph, pl)
    print(f"  {name}: MAPE_high={rep[name]['mape_h']:.2f}%  MAPE_low={rep[name]['mape_l']:.2f}%  "
          f"hit±5%(H/L)={rep[name]['hit5_h']:.1f}/{rep[name]['hit5_l']:.1f}  "
          f"hit±10%(H/L)={rep[name]['hit10_h']:.1f}/{rep[name]['hit10_l']:.1f}  "
          f"MAPE_range={rep[name]['mape_range']:.2f}%  hit5_range={rep[name]['hit5_range']:.1f}%")

# Best model = lowest avg MAPE across high+low
best = min(rep.keys(), key=lambda k: rep[k]["mape_h"] + rep[k]["mape_l"])
print(f"\n>>> BEST: {best}")

# --------------------------------------------------------------------------- #
# 6. FEATURE IMPORTANCE + VARIANCE EXPLAINED
# --------------------------------------------------------------------------- #
print("\n>>> Feature importance (permutation, HI target, on test set) ...")
mh_best = results[best][0]
# permutation on a subset for speed
perm_hi = permutation_importance(mh_best, X_te, yhi_te, n_repeats=5,
                                  random_state=42, n_jobs=-1)
imp_hi = pd.Series(perm_hi.importances_mean, index=feat_cols).sort_values(ascending=False)
ml_best = results[best][1]
perm_lo = permutation_importance(ml_best, X_te, ylo_te, n_repeats=5,
                                  random_state=42, n_jobs=-1)
imp_lo = pd.Series(perm_lo.importances_mean, index=feat_cols).sort_values(ascending=False)
imp_combined = (imp_hi.rank(ascending=False) + imp_lo.rank(ascending=False)).sort_values()

print("Top 20 features (combined rank):")
print(imp_combined.head(20))

# PCA variance explained by top-20 features (standardized)
top20 = imp_combined.head(20).index.tolist()
sc = StandardScaler().fit(X_tr[top20])
pca = PCA().fit(sc.transform(X_tr[top20]))
cum_var = np.cumsum(pca.explained_variance_ratio_)
print("\nPCA cumulative variance (top-20 features):")
for i, v in enumerate(cum_var, 1):
    print(f"  PC{i:2d}: {v*100:5.2f}%")

# Save artifacts
out = dict(rep=rep, imp_hi=imp_hi, imp_lo=imp_lo, imp_combined=imp_combined,
           cum_var=cum_var, feat_cols=feat_cols, best=best,
           test_index=test.index, high_true=high_true, low_true=low_true,
           close_te=close_te)
with open(f"{OUT}/artifacts.pkl", "wb") as fp:
    pickle.dump(out, fp)
print(f"\n>>> Saved artifacts to {OUT}/artifacts.pkl")
