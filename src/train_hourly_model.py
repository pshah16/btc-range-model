"""Hourly BTC next-hour close prediction.

Target: log-return r_{t+1} = log(C_{t+1}/C_t)   →  pred_close = C_t * exp(r_pred)
±3 % accuracy: |pred_close - actual_close|/actual_close ≤ 0.03

Honest framing: hourly BTC returns are typically <1 %, so ±3 % hit-rate is
trivially ~99 %. We also report ±1 %, ±0.5 %, direction accuracy, R² on return.
"""
import warnings, time, joblib, json
warnings.filterwarnings("ignore")
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.linear_model    import RidgeCV
from sklearn.ensemble        import GradientBoostingRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection      import permutation_importance

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from paths import RAW_HOURLY_CSV, HOURLY_MODEL

# ── 1. FETCH ───────────────────────────────────────────────────────────── #
def _flat(df, name):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open","High","Low","Close","Volume"]].copy()
    df.columns = [f"{name}_{c.lower()}" for c in df.columns]
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

print("Fetching hourly Yahoo data ...")
SYMS = {"btc":"BTC-USD","eth":"ETH-USD","spx":"^GSPC","ndx":"^IXIC",
        "vix":"^VIX","gold":"GC=F","dxy":"DX-Y.NYB","tnx":"^TNX"}
parts = {}
for name, sym in SYMS.items():
    d = yf.download(sym, period="2y", interval="60m",
                    progress=False, auto_adjust=False)
    d = _flat(d, name)
    parts[name] = d
    print(f"  {sym:10s} rows={len(d):6d}  {d.index.min()} -> {d.index.max()}")

# Master hourly grid = BTC's range, on hour boundaries (UTC, hour:00)
btc = parts["btc"]
grid = pd.date_range(btc.index.min().floor("h"), btc.index.max().floor("h"),
                     freq="h")
print(f"\nMaster grid: {len(grid)} hours  {grid.min()} -> {grid.max()}")

# Resample each source to hourly (last value within each hour), then ffill
# with enough headroom for weekends/holidays (168h = 1 week of hours)
df = pd.DataFrame(index=grid)
for name, d in parts.items():
    agg = {f"{name}_open":"first", f"{name}_high":"max", f"{name}_low":"min",
           f"{name}_close":"last", f"{name}_volume":"last"}
    d = d.resample("h").agg(agg)
    d[f"{name}_volume"] = d[f"{name}_volume"].replace(0, np.nan)
    d = d.reindex(grid)
    ffill_lim = 4 if name in ("btc","eth") else 168  # 1 week for macro
    d = d.ffill(limit=ffill_lim)
    df = df.join(d)
print(f"Joined hourly frame: {df.shape}")
print(f"  NaN counts (top 10): {df.isna().sum().sort_values(ascending=False).head(10).to_dict()}")

# Fear & Greed (daily) → forward-fill to hourly
#
# IMPORTANT (causality): alternative.me publishes one F&G value per UTC
# date and *updates it throughout the day* (the latest record carries a
# `time_until_update` countdown). That means the value stamped at
# 2026-05-21 00:00 UTC is recomputed during the day with information from
# later hours of 2026-05-21.  Using it at hour 14:00 UTC of the same date
# would be look-ahead.
#
# Fix: lag the F&G series by 1 day so each hour t on calendar date D uses
# the value finalised for date D-1 (which became immutable when the next
# day's record was created).
print("\nFetching Fear & Greed daily ...")
r = requests.get("https://api.alternative.me/fng/?limit=0", timeout=20).json()
fng = pd.DataFrame(r["data"])
fng["dt"]    = pd.to_datetime(fng["timestamp"].astype(int), unit="s").dt.normalize()
fng["value"] = fng["value"].astype(int)
fng = fng[["dt","value"]].sort_values("dt").drop_duplicates("dt").set_index("dt")
fng = fng.shift(1, freq="D")   # ← 1-day causal lag (anti-leak)
# Forward-fill to hourly grid
fng_hourly = fng.reindex(grid.normalize()).ffill()
fng_hourly.index = grid
df["fng"]     = fng_hourly["value"].values
df["fng_d7"]  = pd.Series(df["fng"].values, index=grid).diff(24*7).values
df["fng_d24"] = pd.Series(df["fng"].values, index=grid).diff(24).values
print(f"  FNG joined (lagged 1d). Latest={df['fng'].iloc[-1]}  "
      f"range=[{df['fng'].min()}, {df['fng'].max()}]")

# Drop rows where BTC is missing (rare)
df = df.dropna(subset=["btc_close"])
print(f"After drop NA BTC: {df.shape}")

# Save raw hourly snapshot for the app
df.to_csv(RAW_HOURLY_CSV)

# ── 2. FEATURES ───────────────────────────────────────────────────────── #
print("\nEngineering features ...")
f = pd.DataFrame(index=df.index)
c = df["btc_close"]; h = df["btc_high"]; l_ = df["btc_low"]
o = df["btc_open"];  v = df["btc_volume"]

# Returns at multiple horizons
rt = np.log(c).diff()
for k in [1,2,4,8,12,24,48,72]:
    f[f"ret_{k}h"] = rt.rolling(k).sum()
# Realized vol
for k in [4,8,24,48]:
    f[f"vol_{k}h"] = rt.rolling(k).std()
# True range / ATR
prev_c = c.shift(1)
tr = pd.concat([(h-l_),(h-prev_c).abs(),(l_-prev_c).abs()],axis=1).max(axis=1)
for k in [4,12,24]:
    f[f"atr_{k}h"] = tr.rolling(k).mean()/c
# Range
f["range_now"]   = (h-l_)/c
f["range_ma24"]  = f["range_now"].rolling(24).mean()
f["range_ma72"]  = f["range_now"].rolling(72).mean()
# Volume
f["vol_chg_1"]   = np.log(v).diff()
f["vol_z_24"]    = (np.log(v)-np.log(v).rolling(24).mean()) / np.log(v).rolling(24).std()
# RSI(14) on hourly
delta = c.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
f["rsi_14"] = 100 - 100/(1+rs)
# MACD on hourly (faster spans)
ema12 = c.ewm(span=12,adjust=False).mean()
ema26 = c.ewm(span=26,adjust=False).mean()
macd  = ema12 - ema26
f["macd"]      = macd/c
f["macd_hist"] = (macd - macd.ewm(span=9,adjust=False).mean())/c
# Bollinger band width (24h)
ma24 = c.rolling(24).mean(); sd24 = c.rolling(24).std()
f["bb24_width"] = (4*sd24)/ma24
# Distance from recent extremes
f["dist_hi_24"]  = c / c.rolling(24).max() - 1
f["dist_lo_24"]  = c / c.rolling(24).min() - 1
f["dist_hi_168"] = c / c.rolling(168).max() - 1   # weekly hi

# Cross-market features (per-asset return + level vs recent mean)
for nm in ["eth","spx","ndx","vix","gold","dxy","tnx"]:
    s = df[f"{nm}_close"]
    f[f"{nm}_ret_1h"]  = np.log(s).diff()
    f[f"{nm}_ret_24h"] = np.log(s).diff(24)
    f[f"{nm}_vol_24h"] = np.log(s).diff().rolling(24).std()
# BTC-ETH 24h correlation
f["btc_eth_corr_24"] = rt.rolling(24).corr(np.log(df["eth_close"]).diff())

# Sentiment
f["fng"]    = df["fng"]
f["fng_d1"] = df["fng"].diff()         # day-over-day F&G change
f["fng_d7"] = df["fng_d7"]
f["fng_d24"]= df["fng_d24"]

# Calendar features (cyclic encoding for hour-of-day & day-of-week)
hr = df.index.hour
dow = df.index.dayofweek
f["hr_sin"]  = np.sin(2*np.pi*hr/24);  f["hr_cos"]  = np.cos(2*np.pi*hr/24)
f["dow_sin"] = np.sin(2*np.pi*dow/7);  f["dow_cos"] = np.cos(2*np.pi*dow/7)
f["weekend"] = (dow >= 5).astype(int)
f["us_open"] = ((hr >= 13) & (hr <= 20) & (dow < 5)).astype(int)  # 13-20 UTC

# Target: next-hour log return
y = rt.shift(-1)
data = f.copy()
data["y_ret"]     = y
data["close"]     = c
data["next_close"]= c.shift(-1)
data = data.replace([np.inf,-np.inf], np.nan)
print("Top 15 features by NaN count (before dropna):")
print(data.isna().sum().sort_values(ascending=False).head(15).to_string())
data = data.dropna()
print(f"Feature matrix: {data.shape}   features={data.shape[1]-3}")

# ── 3. TRAIN / VAL / TEST  SPLIT  (with 1-hour embargo for shift(-1) target) #
#
#   TRAIN  → fit each candidate model
#   VAL    → pick the winner (no peeking at TEST)
#   TEST   → final, untouched evaluation
#
# Embargo = 1 hour: the last training row's target is the first val row's
# close, so we drop that row to keep the splits causally disjoint.
TODAY        = pd.Timestamp(datetime.now(timezone.utc).date())
TEST_DAYS    = 45
VAL_DAYS     = 15
EMBARGO_HRS  = 1   # = forecast horizon

test_start  = TODAY      - pd.Timedelta(days=TEST_DAYS)
val_start   = test_start - pd.Timedelta(days=VAL_DAYS)
train_end   = val_start  - pd.Timedelta(hours=EMBARGO_HRS)
val_end     = test_start - pd.Timedelta(hours=EMBARGO_HRS)

train = data.loc[: train_end]
val   = data.loc[val_start: val_end]
test  = data.loc[test_start:]
feat_cols = [c for c in data.columns if c not in ("y_ret","close","next_close")]
print(f"TRAIN  {train.index.min()} → {train.index.max()}  n={len(train)}")
print(f"VAL    {val.index.min()}   → {val.index.max()}    n={len(val)}  "
      f"(embargo {EMBARGO_HRS}h before val_start)")
print(f"TEST   {test.index.min()}  → {test.index.max()}   n={len(test)}  "
      f"(embargo {EMBARGO_HRS}h before test_start)")

X_tr, X_va, X_te = train[feat_cols], val[feat_cols], test[feat_cols]
y_tr,  y_va,  y_te = train["y_ret"], val["y_ret"], test["y_ret"]
close_va, next_va = val["close"].values,  val["next_close"].values
close_te, next_close_true = test["close"].values, test["next_close"].values

# ── 4. EVAL HELPERS ───────────────────────────────────────────────────── #
def _evaluate(name, y_true, close_, next_, pred_ret):
    pred_close = close_ * np.exp(pred_ret)
    rel = np.abs(pred_close - next_) / next_
    direction_acc = np.mean(np.sign(pred_ret) == np.sign(y_true)) * 100
    return dict(
        model      = name,
        MAPE_pct   = float(rel.mean() * 100),
        hit3_pct   = float((rel <= 0.03).mean() * 100),
        hit1_pct   = float((rel <= 0.01).mean() * 100),
        hit05_pct  = float((rel <= 0.005).mean() * 100),
        dir_acc_pct= float(direction_acc),
        R2_return  = float(r2_score(y_true, pred_ret)),
        MAE_USD    = float(mean_absolute_error(next_, pred_close)),
        RMSE_USD   = float(np.sqrt(mean_squared_error(next_, pred_close))),
    )
eval_va = lambda name, p: _evaluate(name, y_va.values, close_va, next_va,  p)
eval_te = lambda name, p: _evaluate(name, y_te.values, close_te, next_close_true, p)

# Sanity baselines (reported on VAL — for context, not selection)
print("\n>>> VAL BASELINES (for context):")
for r in (
    eval_va("A. zero return (close→close)",  np.zeros_like(y_va.values)),
    eval_va(f"B. const mean return ({y_tr.mean():.6f})",
            np.full_like(y_va.values, y_tr.mean())),
    eval_va("C. last-1h return persistence", X_va["ret_1h"].values),
):
    print(f"  {r['model']:38s}  MAPE={r['MAPE_pct']:5.2f}%  "
          f"dir_acc={r['dir_acc_pct']:5.2f}%  R2={r['R2_return']:6.3f}")

# ── 5. TRAIN MODELS, PICK ON VAL ─────────────────────────────────────── #
def mk(model):
    return Pipeline([("sc", StandardScaler()), ("m", model)])

MODELS = {
    "ridge": mk(RidgeCV(alphas=np.logspace(-3,3,13))),
    "gbm":   mk(GradientBoostingRegressor(
                n_estimators=800, max_depth=3, learning_rate=0.02,
                subsample=0.8, random_state=42)),
}
print("\n>>> MODELS — train on TRAIN, score on VAL (selection); TEST untouched:")
val_results = {}
for name, m in MODELS.items():
    m.fit(X_tr, y_tr)
    pred_va = m.predict(X_va)
    r = eval_va(name, pred_va)
    print(f"  {name:6s} VAL  MAPE={r['MAPE_pct']:5.2f}%  "
          f"dir_acc={r['dir_acc_pct']:5.2f}%  R2={r['R2_return']:6.3f}")
    val_results[name] = dict(model=m, pred_va=pred_va, metrics_va=r)

best = max(val_results.keys(), key=lambda k: val_results[k]["metrics_va"]["dir_acc_pct"])
print(f"\n>>> SELECTED on VAL by dir_acc: {best}")
mh = val_results[best]["model"]

# Final UNBIASED evaluation on TEST.
pred_te = mh.predict(X_te)
test_metrics = eval_te(best, pred_te)
print(f"\n>>> {best:6s} TEST (unbiased)  MAPE={test_metrics['MAPE_pct']:5.2f}%  "
      f"hit3={test_metrics['hit3_pct']:5.1f}%  hit1={test_metrics['hit1_pct']:5.1f}%  "
      f"hit0.5={test_metrics['hit05_pct']:4.1f}%  dir_acc={test_metrics['dir_acc_pct']:5.2f}%  "
      f"R2={test_metrics['R2_return']:6.3f}")

# σ fit on VAL residuals (not TEST) → CI is unbiased w.r.t. the test set
sigma = float(np.std(y_va.values - val_results[best]["pred_va"]))
print(f"σ (return space, from VAL) = {sigma:.5f}  →  "
      f"95% half-width ≈ ±{1.96*sigma*100:.2f}% of close")

# ── 6. BIG-MOVE DAY EVALUATION ────────────────────────────────────────── #
print("\n>>> Big-move day analysis ...")
test_close = test["close"]
# Daily realized return on the test set
daily_close = test_close.resample("D").last()
daily_open  = test_close.resample("D").first()
daily_ret   = ((daily_close - daily_open) / daily_open) * 100
big_thresh = 3.0   # daily move ≥ 3%
big_days = daily_ret.index[daily_ret.abs() >= big_thresh]
print(f"  Days with |daily move| ≥ {big_thresh}%: {len(big_days)} of {len(daily_ret)} test days")
if len(big_days):
    print(f"  Top 5 big-move days:")
    for d in daily_ret.abs().sort_values(ascending=False).head(5).index:
        print(f"     {d.date()}  {daily_ret.loc[d]:+5.2f}%")

# Subset metrics on big-move hours (hours within big-move days)
big_mask = test.index.normalize().isin(big_days)
if big_mask.any():
    nc = next_close_true[big_mask]
    cc = close_te[big_mask]
    pc = cc * np.exp(pred_te[big_mask])
    rel = np.abs(pc - nc) / nc
    da = np.mean(np.sign(pred_te[big_mask]) == np.sign(y_te.values[big_mask])) * 100
    print(f"\n>>> BIG-MOVE HOURS  (n={big_mask.sum()}):")
    print(f"  MAPE={rel.mean()*100:5.2f}%  hit3={ (rel<=0.03).mean()*100:5.1f}%  "
          f"hit1={(rel<=0.01).mean()*100:5.1f}%  hit0.5={(rel<=0.005).mean()*100:5.1f}%  "
          f"dir_acc={da:5.2f}%")

# ── 7. PERMUTATION FEATURE IMPORTANCE ─────────────────────────────────── #
print("\n>>> Permutation importance (test, n_repeats=5) ...")
perm = permutation_importance(mh, X_te, y_te, n_repeats=5, random_state=42, n_jobs=-1)
imp = pd.Series(perm.importances_mean, index=feat_cols).sort_values(ascending=False)
print(imp.head(15).to_string())

# ── 8. SAVE ───────────────────────────────────────────────────────────── #
joblib.dump(dict(
    model=mh, sigma=sigma, feat_cols=feat_cols,
    best_name=best,
    fng_baseline=int(df["fng"].iloc[-1]),
    train_start=str(train.index.min()),
    train_end  =str(train.index.max()),
    val_start  =str(val.index.min()),
    val_end    =str(val.index.max()),
    test_start =str(test.index.min()),
    test_end   =str(test.index.max()),
    embargo_hours=int(EMBARGO_HRS),
    metrics_val =val_results[best]["metrics_va"],
    metrics_test=test_metrics,
    importance=imp,
    big_days=[str(d.date()) for d in big_days],
    tuning_notes=("Model picked on VAL by dir_acc; σ fit on VAL residuals; "
                  "TEST untouched until final report. F&G lagged by 1 day "
                  "to avoid intraday update leakage."),
), HOURLY_MODEL)
print(f"\nSaved inference_assets_hourly.joblib  (best={best})")
