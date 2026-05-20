"""Build btc_inference_ct.ipynb — focused next-bar H/L inference notebook
for the 12:00-UTC (7am-CT) day boundary.

Mirrors the live UI logic in btc_hourly_app.py:
  - Fetch BTC hourly from Binance, rebucket to 12:00-UTC daily bars
  - Join macro (Yahoo daily) + on-chain (blockchain.info) + F&G aligned by
    calendar date
  - Build features per pipeline_ct.py
  - Apply inference_assets_ct.joblib ensemble to predict the next bar's
    H/L = today 7am CT -> tomorrow 7am CT

The original btc_inference.ipynb is unchanged and remains the home for the
7-day-window and per-horizon models (still UTC-midnight anchored).
"""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
def md(t): cells.append(new_markdown_cell(t))
def code(s): cells.append(new_code_cell(s))

md(r"""# BTC daily H/L inference — **7am Central Time** day boundary

This notebook predicts BTC's high and low for the window
**today 7am CT → tomorrow 7am CT**, i.e. the 24-hour bar starting at
the most recent 12:00 UTC.

- Day boundary: **fixed 12:00 UTC** (= 7am CDT in summer, 6am CST in winter).
- Model: ensemble (Huber + BayesianRidge + GBM-MAE), trained by `pipeline_ct.py`
  with the last 8 months held out (test MAPE H≈1.12%, L≈1.31%).
- Data:
  - BTC hourly OHLCV from Binance, rebucketed into 12:00→12:00 UTC bars
  - Macro (SPX/NDX/VIX/Gold/DXY/TNX/ETH) daily from Yahoo, joined by calendar date
  - On-chain daily from blockchain.info, joined by calendar date

Run with **Kernel → Restart & Run All**.
""")

# ────────────────────────────────────────────────────────────────────── #
md("## 1 · Setup")
code(r"""
import os, time, warnings, joblib, requests
warnings.filterwarnings("ignore")
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook"
TEMPLATE = "plotly_white"

ANCHOR_HOUR_UTC = 12  # 7am CDT / 6am CST

A = joblib.load("/home/jovyan/btc-range-model/models/inference_assets_ct.joblib")
for k, v in A["calibration_meta"].items():
    print(f"  {k}: {v}")
print(f"  σ_hi={A['sigma_hi']:.4f}  σ_lo={A['sigma_lo']:.4f}  "
      f"95% half-width ≈ ±{1.96*A['sigma_hi']*100:.2f}%/±{1.96*A['sigma_lo']*100:.2f}%")
""")

# ────────────────────────────────────────────────────────────────────── #
md("## 2 · Fetch BTC hourly from Binance + rebucket to 12:00-UTC bars")
code(r"""
def fetch_binance_hourly(days_back=400):
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    cursor = end_ms - days_back * 86400_000
    rows = []
    while cursor < end_ms:
        params = dict(symbol="BTCUSDT", interval="1h",
                      startTime=cursor, limit=1000)
        r = requests.get("https://api.binance.com/api/v3/klines",
                         params=params, timeout=30)
        batch = r.json()
        if not batch: break
        rows.extend(batch)
        cursor = batch[-1][0] + 3600_000
        time.sleep(0.1)
    cols = ["open_time","open","high","low","close","volume",
            "close_time","qv","n","tb","tq","ig"]
    df = pd.DataFrame(rows, columns=cols)
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df = df.set_index("ts")[["open","high","low","close","volume"]]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def rebucket_12utc(hourly):
    h = hourly.copy()
    h["bucket"] = (h.index - pd.Timedelta(hours=ANCHOR_HOUR_UTC)).normalize()
    g = h.groupby("bucket").agg(
        open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last"),
        volume=("volume","sum"), n_hours=("close","size"),
    )
    g = g[g["n_hours"] == 24].drop(columns="n_hours")
    g.index.name = "bar_start"
    return g


btc_hourly = fetch_binance_hourly(days_back=400)
btc_daily = rebucket_12utc(btc_hourly).add_prefix("btc_")
print(f"BTC hourly:   {len(btc_hourly):,} rows  ({btc_hourly.index.min()} → {btc_hourly.index.max()})")
print(f"12:00-UTC bars: {len(btc_daily)}    "
      f"latest complete bar starts {btc_daily.index.max().date()} 12:00 UTC")
""")

# ────────────────────────────────────────────────────────────────────── #
md("## 3 · Fetch macro + on-chain")
code(r"""
START = (btc_daily.index.min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
END = (datetime.now(timezone.utc).date() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

def _flat(df, name):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open","High","Low","Close","Volume"]].copy()
    df.columns = [f"{name}_{c.lower()}" for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df

YAHOO = {"eth":"ETH-USD","spx":"^GSPC","ndx":"^IXIC",
         "vix":"^VIX","gold":"GC=F","dxy":"DX-Y.NYB","tnx":"^TNX"}
parts = []
for name, sym in YAHOO.items():
    d = yf.download(sym, start=START, end=END, progress=False, auto_adjust=False)
    parts.append(_flat(d, name))
mkt = parts[0]
for p in parts[1:]:
    mkt = mkt.join(p, how="outer")

ONCHAIN = ["hash-rate","difficulty","n-transactions","miners-revenue",
           "n-unique-addresses","transaction-fees-usd","mempool-size",
           "estimated-transaction-volume-usd","market-cap","avg-block-size",
           "cost-per-transaction"]
oc_parts = []
for s in ONCHAIN:
    j = requests.get(f"https://api.blockchain.info/charts/{s}?timespan=all&format=json&sampled=false",
                     timeout=30).json()
    idx = pd.to_datetime([x["x"] for x in j["values"]], unit="s").normalize()
    ser = pd.Series([x["y"] for x in j["values"]], index=idx,
                    name=f"oc_{s.replace('-','_')}")
    ser = ser[~ser.index.duplicated(keep="last")]
    oc_parts.append(ser); time.sleep(0.15)
oc = pd.concat(oc_parts, axis=1)

df = btc_daily.join(mkt, how="left").join(oc, how="left").sort_index()
df = df.loc[df["btc_close"].notna()].ffill(limit=5)
print(f"Joined data: {df.shape}   latest bar: {df.index.max().date()}")
""")

# ────────────────────────────────────────────────────────────────────── #
md("## 4 · Build features (mirrors `pipeline_ct.py`)")
code(r"""
def build_features(df):
    f = pd.DataFrame(index=df.index)
    c, h, l_, v = df["btc_close"], df["btc_high"], df["btc_low"], df["btc_volume"]
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
    f["macd_hist"] = (macd - macd.ewm(span=9,adjust=False).mean())/c
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
        s = df[col].astype(float); sl = np.log(s.replace(0, np.nan))
        f[f"{col}_d1"]  = sl.diff(1)
        f[f"{col}_d7"]  = sl.diff(7)
        f[f"{col}_z30"] = (sl - sl.rolling(30).mean())/sl.rolling(30).std()
    nh, nl = h.shift(-1), l_.shift(-1)
    y_hi = (nh-c)/c; y_lo = (c-nl)/c
    f["y_hi_lag1"]    = y_hi.shift(1)
    f["y_lo_lag1"]    = y_lo.shift(1)
    f["y_hi_lag7_ma"] = y_hi.shift(1).rolling(7).mean()
    f["y_lo_lag7_ma"] = y_lo.shift(1).rolling(7).mean()
    return f.replace([np.inf,-np.inf], np.nan)


F_all = build_features(df)
feat_cols = A["feat_cols"]
F = F_all[feat_cols].dropna()
print(f"Feature matrix: {F.shape}  feature_count={len(feat_cols)}")
""")

# ────────────────────────────────────────────────────────────────────── #
md("## 5 · Predict — next bar's H/L (today 7am CT → tomorrow 7am CT)")
code(r"""
asof = F.index[-1]
close_asof = float(df.loc[asof, "btc_close"])
target_window_start = asof + pd.Timedelta(days=1, hours=ANCHOR_HOUR_UTC)
target_window_end   = asof + pd.Timedelta(days=2, hours=ANCHOR_HOUR_UTC)

row = F.loc[[asof]]
yhi_list = [float(c["m_hi"].predict(row)[0]) for c in A["constituents"]]
ylo_list = [float(c["m_lo"].predict(row)[0]) for c in A["constituents"]]
yhi, ylo = float(np.mean(yhi_list)), float(np.mean(ylo_list))
# (training found α = 1.0, so no climatology blend; preserve the form anyway)
if A.get("blended") and A.get("alpha", 1.0) < 1.0:
    a = float(A["alpha"])
    yhi = a*yhi + (1-a)*float(A["mu_hi"])
    ylo = a*ylo + (1-a)*float(A["mu_lo"])
sh, sl = float(A["sigma_hi"]), float(A["sigma_lo"])

clip0 = lambda x: max(float(x), 0.0)
pred_high   = close_asof * (1 + clip0(yhi))
pred_low    = close_asof * (1 - clip0(ylo))
hi_ci_up    = close_asof * (1 + yhi + 1.96*sh)
hi_ci_dn    = close_asof * (1 + clip0(yhi - 1.96*sh))
lo_ci_up    = close_asof * (1 - clip0(ylo - 1.96*sl))
lo_ci_dn    = close_asof * (1 - clip0(ylo + 1.96*sl))

print(f"As-of bar start (UTC):  {asof.strftime('%Y-%m-%d 12:00')}  "
      f"(= 7am CT on {asof.strftime('%Y-%m-%d')})")
print(f"As-of close:            ${close_asof:,.2f}")
print(f"Target window UTC:      {target_window_start} → {target_window_end}")
print(f"Target window local:    today 7am CT → tomorrow 7am CT")
print()
print(f"Predicted HIGH:  ${pred_high:>10,.0f}   "
      f"(+{(pred_high/close_asof-1)*100:.2f}% vs close)")
print(f"  95% CI:        ${hi_ci_dn:>10,.0f} – ${hi_ci_up:,.0f}")
print(f"Predicted LOW:   ${pred_low:>10,.0f}   "
      f"({(pred_low/close_asof-1)*100:.2f}% vs close)")
print(f"  95% CI:        ${lo_ci_dn:>10,.0f} – ${lo_ci_up:,.0f}")
print(f"Implied RANGE:   ${pred_high-pred_low:>10,.0f}   "
      f"({(pred_high-pred_low)/close_asof*100:.2f}% of close)")
""")

# ────────────────────────────────────────────────────────────────────── #
md("## 6 · Visualise — last 14 bars + forecast")
code(r"""
look = 14
recent = df.iloc[-look:]
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=recent.index,
    open=recent["btc_open"], high=recent["btc_high"],
    low=recent["btc_low"],   close=recent["btc_close"],
    name="BTC 12:00-UTC bars",
    increasing_line_color="#1a9e4f", decreasing_line_color="#d23535",
))

# Forecast HIGH/LOW with CI bands at the target window
target_x = target_window_start + (target_window_end - target_window_start)/2
fig.add_trace(go.Scatter(
    x=[target_x], y=[pred_high], mode="markers",
    marker=dict(symbol="diamond", size=14, color="#1a9e4f",
                line=dict(color="black", width=1)),
    name="Predicted HIGH",
    error_y=dict(type="data",
                 array=[hi_ci_up - pred_high],
                 arrayminus=[pred_high - hi_ci_dn],
                 color="#1a9e4f", thickness=2, width=8),
    hovertemplate=f"Target {target_window_start:%Y-%m-%d %H:%M} → {target_window_end:%H:%M} UTC<br>"
                  f"Pred HIGH $%{{y:,.0f}}<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=[target_x], y=[pred_low], mode="markers",
    marker=dict(symbol="diamond", size=14, color="#d23535",
                line=dict(color="black", width=1)),
    name="Predicted LOW",
    error_y=dict(type="data",
                 array=[lo_ci_up - pred_low],
                 arrayminus=[pred_low - lo_ci_dn],
                 color="#d23535", thickness=2, width=8),
    hovertemplate=f"Target {target_window_start:%Y-%m-%d %H:%M} → {target_window_end:%H:%M} UTC<br>"
                  f"Pred LOW $%{{y:,.0f}}<extra></extra>",
))
fig.add_vrect(x0=target_window_start, x1=target_window_end,
              fillcolor="khaki", opacity=0.30, line_width=0, layer="below",
              annotation_text="forecast window<br>(today 7am CT → tomorrow 7am CT)",
              annotation_position="top left",
              annotation=dict(font=dict(size=11, color="#666")))
fig.update_layout(
    template=TEMPLATE,
    title=f"BTC — next-bar H/L forecast (12:00 UTC anchored)  "
          f"issued from bar starting {asof.strftime('%Y-%m-%d 12:00')} UTC",
    xaxis_title="UTC", yaxis_title="USD",
    xaxis_rangeslider_visible=False,
    height=520,
)
fig.show()
""")

# ────────────────────────────────────────────────────────────────────── #
md(r"""## Notes

* **Day boundary.** All bars start at 12:00 UTC = 7am CDT (summer) / 6am CST (winter).
  We chose a fixed UTC anchor over DST-following so every bar is exactly 24h —
  no 23/25h edge cases twice a year.
* **Macro and on-chain alignment.** Bar D's auxiliary features are taken from
  calendar date D. Macro closes for date D are published by ~21:00 UTC on day D
  (well before bar D's end at D+1 12:00 UTC), and blockchain.info's daily
  aggregate for date D is similarly available.
* **What this model does NOT cover.** The 7-day-window model
  (`inference_assets_7d.joblib`) and per-horizon (`inference_assets_horizon.joblib`)
  models in `btc_inference.ipynb` are still anchored at **UTC midnight** and
  would need a parallel retraining if you want them on the 7am-CT boundary.
""")

# ────────────────────────────────────────────────────────────────────── #
nb = new_notebook(cells=cells)
nb.metadata.kernelspec = dict(display_name="Python 3", language="python", name="python3")
nb.metadata.language_info = dict(name="python", version="3.x",
                                 pygments_lexer="ipython3",
                                 mimetype="text/x-python",
                                 file_extension=".py")

out = "/home/jovyan/btc-range-model/notebooks/btc_inference_ct.ipynb"
with open(out, "w") as f:
    nbformat.write(nb, f)
print(f"Wrote {out}  ({len(cells)} cells)")
