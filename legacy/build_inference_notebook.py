"""Build the combined 1-day + 7-day daily-inference notebook with PLOTLY plots.

Plot A — 1-day: last 7 days candlesticks + predictions + tomorrow's forecast.
Plot B — 7-day: 30-day backtest + FORWARD 7-day forecast band (extends x-axis to
          show the actual forecast window past `latest_complete`).
All plots interactive (zoom, pan, hover tooltips).
"""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
def md(t):   cells.append(new_markdown_cell(t))
def code(s): cells.append(new_code_cell(s))

md(r"""# Bitcoin Daily H/L Inference — 1-day + 7-day forecasts

**Interactive notebook.** Plots are rendered with Plotly — zoom, pan and hover
over points to see exact values. Run with **Kernel → Restart & Run All**.

1. **Next-day** H/L (Ridge model) with 95 % CI.
2. **Next-7-day window** max H & min L (RF model) with 95 % CI.
3. Two interactive backtest plots (one per model) + the 7-day forward forecast envelope.
""")

# ────────────────────────────────────────────────────────────────────── #
md("## 1 · Setup & load BOTH models")
code(r"""
import os, time, warnings, joblib
warnings.filterwarnings("ignore")
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests, yfinance as yf

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = "notebook"      # safe default for nbconvert + JupyterLab
TEMPLATE = "plotly_white"

A1 = joblib.load("/home/jovyan/btc-range-model/inference_assets.joblib")
A7 = joblib.load("/home/jovyan/btc-range-model/inference_assets_7d.joblib")
AH = joblib.load("/home/jovyan/btc-range-model/inference_assets_horizon.joblib")

print("1-day model")
for k,v in A1["calibration_meta"].items(): print(f"   {k}: {v}")
print(f"   σ_hi={A1['sigma_hi']:.4f}  σ_lo={A1['sigma_lo']:.4f}  "
      f"95% half-width ≈ ±{1.96*A1['sigma_hi']*100:.2f}%/±{1.96*A1['sigma_lo']*100:.2f}%\n")
print("7-day window model")
for k,v in A7["calibration_meta"].items(): print(f"   {k}: {v}")
print(f"   σ_hi={A7['sigma_hi']:.4f}  σ_lo={A7['sigma_lo']:.4f}  "
      f"95% half-width ≈ ±{1.96*A7['sigma_hi']*100:.2f}%/±{1.96*A7['sigma_lo']*100:.2f}%\n")
print(f"Per-horizon (k=1..7) models — for daily forward predictions")
for m in AH["test_metrics"]:
    print(f"   k={m['k']}d  MAPE H/L={m['MAPE_H']:.2f}/{m['MAPE_L']:.2f}%  "
          f"σ_hi={m['sigma_hi']:.4f}  σ_lo={m['sigma_lo']:.4f}")
""")

# ────────────────────────────────────────────────────────────────────── #
md("## 2 · Fetch latest data")
code(r"""
START = "2019-01-01"
NOW_UTC = datetime.now(timezone.utc)
END = (NOW_UTC.date() + pd.Timedelta(days=1).to_pytimedelta()).strftime("%Y-%m-%d")
print(f"Run time UTC = {NOW_UTC.isoformat(timespec='seconds')}")

def _flat(df, name):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open","High","Low","Close","Volume"]].copy()
    df.columns = [f"{name}_{c.lower()}" for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df

YAHOO = {"btc":"BTC-USD","eth":"ETH-USD","spx":"^GSPC","ndx":"^IXIC",
         "vix":"^VIX","gold":"GC=F","dxy":"DX-Y.NYB","tnx":"^TNX"}
parts=[]
for name,sym in YAHOO.items():
    d = yf.download(sym, start=START, end=END, progress=False, auto_adjust=False)
    parts.append(_flat(d,name))
mkt = parts[0]
for p in parts[1:]: mkt = mkt.join(p,how="outer")

ONCHAIN = ["hash-rate","difficulty","n-transactions","miners-revenue",
           "n-unique-addresses","transaction-fees-usd","mempool-size",
           "estimated-transaction-volume-usd","market-cap","avg-block-size",
           "cost-per-transaction"]
oc_parts=[]
for s in ONCHAIN:
    j = requests.get(f"https://api.blockchain.info/charts/{s}?timespan=all&format=json&sampled=false",
                     timeout=30).json()
    idx = pd.to_datetime([x["x"] for x in j["values"]], unit="s").normalize()
    ser = pd.Series([x["y"] for x in j["values"]], index=idx,
                    name=f"oc_{s.replace('-','_')}")
    ser = ser[~ser.index.duplicated(keep="last")]
    oc_parts.append(ser); time.sleep(0.2)
oc = pd.concat(oc_parts,axis=1)

df = mkt.join(oc,how="left").sort_index()
df = df.loc[df["btc_close"].notna()]
df = df.ffill(limit=5).loc["2019-01-01":]
print(f"Joined data: {df.shape}   latest row: {df.index.max().date()}")
""")

# ────────────────────────────────────────────────────────────────────── #
md("## 3 · Latest complete day → prediction targets")
code(r"""
oc_cols = [c for c in df.columns if c.startswith("oc_")]
complete = df["btc_close"].notna() & df[oc_cols].notna().all(axis=1)
latest_complete = df.index[complete].max()
pred_1d_date  = latest_complete + pd.Timedelta(days=1)
pred_7d_start = latest_complete + pd.Timedelta(days=1)
pred_7d_end   = latest_complete + pd.Timedelta(days=7)

print(f"Latest complete data day : {latest_complete.date()}  (close = ${df.loc[latest_complete,'btc_close']:,.0f})")
print(f"NEXT-DAY target          : {pred_1d_date.date()}")
print(f"NEXT-7-DAY window target : {pred_7d_start.date()}  →  {pred_7d_end.date()}")
""")

# ────────────────────────────────────────────────────────────────────── #
md("## 4 · Build features")
code(r"""
def build_features(df):
    f = pd.DataFrame(index=df.index)
    c, h, l_, v = df["btc_close"], df["btc_high"], df["btc_low"], df["btc_volume"]
    ret = np.log(c).diff()
    for k in [1,3,5,7,14,30]: f[f"ret_{k}"] = ret.rolling(k).sum()
    for k in [5,10,20,30]:    f[f"vol_{k}"] = ret.rolling(k).std()
    prev_c = c.shift(1)
    tr = pd.concat([(h-l_),(h-prev_c).abs(),(l_-prev_c).abs()],axis=1).max(axis=1)
    for k in [7,14,30]: f[f"atr_{k}"] = tr.rolling(k).mean() / c
    f["range_today"] = (h-l_)/c
    f["range_ma7"]   = ((h-l_)/c).rolling(7).mean()
    f["range_ma30"]  = ((h-l_)/c).rolling(30).mean()
    f["range_std30"] = ((h-l_)/c).rolling(30).std()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0,np.nan)
    f["rsi_14"] = 100 - 100/(1+rs)
    ema12 = c.ewm(span=12,adjust=False).mean(); ema26 = c.ewm(span=26,adjust=False).mean()
    macd = ema12 - ema26
    f["macd"]      = macd / c
    f["macd_sig"]  = macd.ewm(span=9,adjust=False).mean() / c
    f["macd_hist"] = (macd - macd.ewm(span=9,adjust=False).mean()) / c
    ma20=c.rolling(20).mean(); sd20=c.rolling(20).std()
    f["bb_width"]   = (4*sd20)/ma20
    f["dist_hi_30"] = c / c.rolling(30).max() - 1
    f["dist_lo_30"] = c / c.rolling(30).min() - 1
    f["dist_hi_90"] = c / c.rolling(90).max() - 1
    f["vol_chg_1"]    = np.log(v).diff()
    f["vol_z_20"]     = (np.log(v)-np.log(v).rolling(20).mean())/np.log(v).rolling(20).std()
    f["vol_ma_ratio"] = v / v.rolling(20).mean()
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
        s = df[col].astype(float); sl = np.log(s.replace(0,np.nan))
        f[f"{col}_d1"]  = sl.diff(1)
        f[f"{col}_d7"]  = sl.diff(7)
        f[f"{col}_z30"] = (sl - sl.rolling(30).mean())/sl.rolling(30).std()
    nh,nl = h.shift(-1), l_.shift(-1)
    y_hi=(nh-c)/c; y_lo=(c-nl)/c
    f["y_hi_lag1"]    = y_hi.shift(1)
    f["y_lo_lag1"]    = y_lo.shift(1)
    f["y_hi_lag7_ma"] = y_hi.shift(1).rolling(7).mean()
    f["y_lo_lag7_ma"] = y_lo.shift(1).rolling(7).mean()
    return f

F  = build_features(df).replace([np.inf,-np.inf], np.nan)
F1 = F[A1["feat_cols"]]
F7 = F[A7["feat_cols"]]
print(f"Feature matrix: {F.shape}")
""")

# ────────────────────────────────────────────────────────────────────── #
md("## 5 · Walk-forward predictions for last 30 as-of days")
code(r"""
LOOKBACK = 30
win_idx = df.index[(df.index >= latest_complete - pd.Timedelta(days=LOOKBACK-1)) &
                    (df.index <= latest_complete)]
win_idx = [d for d in win_idx if F.loc[d].notna().all()]
closes  = df.loc[win_idx, "btc_close"]

# 1-day
yhi1 = A1["hi_model"].predict(F1.loc[win_idx])
ylo1 = A1["lo_model"].predict(F1.loc[win_idx])
pred_hi1 = closes.values * (1 + np.clip(yhi1, 0, None))
pred_lo1 = closes.values * (1 - np.clip(ylo1, 0, None))
band_hi1_up = closes.values * (1 + yhi1 + 1.96*A1["sigma_hi"])
band_hi1_dn = closes.values * (1 + np.clip(yhi1 - 1.96*A1["sigma_hi"], 0, None))
band_lo1_up = closes.values * (1 - np.clip(ylo1 - 1.96*A1["sigma_lo"], 0, None))
band_lo1_dn = closes.values * (1 - np.clip(ylo1 + 1.96*A1["sigma_lo"], 0, None))
target_1d_dates = [d + pd.Timedelta(days=1) for d in win_idx]
real_hi1 = np.array([df.loc[d,"btc_high"] if d in df.index else np.nan for d in target_1d_dates])
real_lo1 = np.array([df.loc[d,"btc_low"]  if d in df.index else np.nan for d in target_1d_dates])

# 7-day
yhi7 = A7["hi_model"].predict(F7.loc[win_idx])
ylo7 = A7["lo_model"].predict(F7.loc[win_idx])
pred_hi7 = closes.values * (1 + np.clip(yhi7, 0, None))
pred_lo7 = closes.values * (1 - np.clip(ylo7, 0, None))
band_hi7_up = closes.values * (1 + yhi7 + 1.96*A7["sigma_hi"])
band_hi7_dn = closes.values * (1 + np.clip(yhi7 - 1.96*A7["sigma_hi"], 0, None))
band_lo7_up = closes.values * (1 - np.clip(ylo7 - 1.96*A7["sigma_lo"], 0, None))
band_lo7_dn = closes.values * (1 - np.clip(ylo7 + 1.96*A7["sigma_lo"], 0, None))
target_7d_start = [d + pd.Timedelta(days=1) for d in win_idx]
target_7d_end   = [d + pd.Timedelta(days=7) for d in win_idx]

real_hi7, real_lo7 = [], []
for s, e in zip(target_7d_start, target_7d_end):
    w = df.loc[(df.index >= s) & (df.index <= e)]
    if len(w) >= 7 and not w["btc_high"].isna().any():
        real_hi7.append(w["btc_high"].max()); real_lo7.append(w["btc_low"].min())
    else:
        real_hi7.append(np.nan); real_lo7.append(np.nan)
real_hi7 = np.array(real_hi7); real_lo7 = np.array(real_lo7)

print(f"Generated {len(win_idx)} as-of forecasts.")
print(f"  1-day realised days   : {np.sum(~np.isnan(real_hi1))} / {len(real_hi1)}")
print(f"  7-day realised windows: {np.sum(~np.isnan(real_hi7))} / {len(real_hi7)}")
""")

# ────────────────────────────────────────────────────────────────────── #
md("## 6 · 🎯 Headline forecasts")
code(r"""
asof    = win_idx[-1]
asof_p  = closes.iloc[-1]
print("=" * 78)
print(f"  AS OF CLOSE: {asof.date()}   BTC = ${asof_p:,.0f}")
print("=" * 78)
print()
print(f"  🟢 1-DAY FORECAST   target = {pred_1d_date.date()}")
print(f"     HIGH  ${pred_hi1[-1]:>10,.0f}    95% CI [${band_hi1_dn[-1]:,.0f},  ${band_hi1_up[-1]:,.0f}]")
print(f"     LOW   ${pred_lo1[-1]:>10,.0f}    95% CI [${band_lo1_dn[-1]:,.0f},  ${band_lo1_up[-1]:,.0f}]")
print(f"     RANGE ${pred_hi1[-1]-pred_lo1[-1]:>10,.0f}   ({(pred_hi1[-1]-pred_lo1[-1])/asof_p*100:.2f}% of close)")
print()
print(f"  🔵 7-DAY-WINDOW FORECAST   window = {pred_7d_start.date()}  →  {pred_7d_end.date()}")
print(f"     MAX HIGH  ${pred_hi7[-1]:>10,.0f}   95% CI [${band_hi7_dn[-1]:,.0f},  ${band_hi7_up[-1]:,.0f}]")
print(f"     MIN LOW   ${pred_lo7[-1]:>10,.0f}   95% CI [${band_lo7_dn[-1]:,.0f},  ${band_lo7_up[-1]:,.0f}]")
print(f"     RANGE     ${pred_hi7[-1]-pred_lo7[-1]:>10,.0f}   ({(pred_hi7[-1]-pred_lo7[-1])/asof_p*100:.2f}% of close)")
print("=" * 78)
""")

# ────────────────────────────────────────────────────────────────────── #
md(r"""## 7 · PLOT A — 1-day model: last 7 days backtest + tomorrow's forecast

Interactive: zoom by dragging, hover for exact values, click legend to hide series.
""")
code(r"""
# Last 8 entries (7 backtest days + tomorrow)
n_show = 8
idx = list(range(len(win_idx)-n_show, len(win_idx)))
xt = pd.to_datetime([target_1d_dates[i] for i in idx])

ph_hi  = np.array([pred_hi1[i] for i in idx])
ph_lo  = np.array([pred_lo1[i] for i in idx])
bh_up  = np.array([band_hi1_up[i] for i in idx])
bh_dn  = np.array([band_hi1_dn[i] for i in idx])
bl_up  = np.array([band_lo1_up[i] for i in idx])
bl_dn  = np.array([band_lo1_dn[i] for i in idx])
r_hi   = np.array([real_hi1[i] for i in idx])
r_lo   = np.array([real_lo1[i] for i in idx])
asof_c = np.array([closes.iloc[i] for i in idx])

fig = go.Figure()

# Realised OHLC candles for days that have actuals
mask = ~np.isnan(r_hi)
cand_x = xt[mask]
cand_o = [df.loc[d,"btc_open"]  for d,m in zip(xt,mask) if m]
cand_h = [df.loc[d,"btc_high"]  for d,m in zip(xt,mask) if m]
cand_l = [df.loc[d,"btc_low"]   for d,m in zip(xt,mask) if m]
cand_c = [df.loc[d,"btc_close"] for d,m in zip(xt,mask) if m]
fig.add_trace(go.Candlestick(
    x=cand_x, open=cand_o, high=cand_h, low=cand_l, close=cand_c,
    increasing_line_color="#666", decreasing_line_color="#aaa",
    increasing_fillcolor="rgba(120,120,120,0.55)",
    decreasing_fillcolor="rgba(170,170,170,0.55)",
    name="Realised OHLC", showlegend=True,
))

# HIGH 95% CI band
fig.add_trace(go.Scatter(x=xt, y=bh_up, mode="lines",
                         line=dict(color="rgba(0,0,0,0)"),
                         showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=xt, y=bh_dn, mode="lines",
                         line=dict(color="rgba(0,0,0,0)"),
                         fill="tonexty", fillcolor="rgba(0,160,0,0.18)",
                         name="HIGH 95% CI", hoverinfo="skip"))
# LOW 95% CI band
fig.add_trace(go.Scatter(x=xt, y=bl_up, mode="lines",
                         line=dict(color="rgba(0,0,0,0)"),
                         showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=xt, y=bl_dn, mode="lines",
                         line=dict(color="rgba(0,0,0,0)"),
                         fill="tonexty", fillcolor="rgba(200,0,0,0.18)",
                         name="LOW 95% CI", hoverinfo="skip"))

# Predicted HIGH line
fig.add_trace(go.Scatter(
    x=xt, y=ph_hi, mode="lines+markers",
    line=dict(color="green", width=2.4),
    marker=dict(size=10, symbol="circle"),
    name="Pred HIGH",
    hovertemplate="Pred HIGH<br>Date: %{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
))
# Predicted LOW line
fig.add_trace(go.Scatter(
    x=xt, y=ph_lo, mode="lines+markers",
    line=dict(color="red", width=2.4),
    marker=dict(size=10, symbol="circle"),
    name="Pred LOW",
    hovertemplate="Pred LOW<br>Date: %{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
))

# Actual H/L (X markers)
fig.add_trace(go.Scatter(
    x=xt[mask], y=r_hi[mask], mode="markers",
    marker=dict(symbol="x-thin", size=14, line=dict(width=3, color="darkgreen")),
    name="Actual HIGH",
    hovertemplate="Actual HIGH<br>Date: %{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=xt[mask], y=r_lo[mask], mode="markers",
    marker=dict(symbol="x-thin", size=14, line=dict(width=3, color="darkred")),
    name="Actual LOW",
    hovertemplate="Actual LOW<br>Date: %{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
))

# Highlight tomorrow
tom = pd.to_datetime(pred_1d_date)
fig.add_vrect(x0=tom - pd.Timedelta(hours=12), x1=tom + pd.Timedelta(hours=12),
              fillcolor="khaki", opacity=0.35, line_width=0, layer="below")
fig.add_annotation(x=tom, y=ph_hi[-1], text=f"<b>FORECAST<br>{tom.date()}</b>",
                   showarrow=True, arrowhead=2, ax=80, ay=-40,
                   font=dict(color="#b8860b", size=12))

fig.update_xaxes(rangeslider_visible=False, tickformat="%a %d-%b")
fig.update_layout(
    title=f"PLOT A · 1-DAY model — issued {asof.date()}, target {tom.date()}<br>"
          f"<sup>Last 7 days backtest + tomorrow's forecast (interactive)</sup>",
    template=TEMPLATE,
    yaxis_title="BTC / USD", hovermode="x unified",
    height=560, legend=dict(orientation="h", y=1.08),
)
fig.show()
""")

# ────────────────────────────────────────────────────────────────────── #
md(r"""## 8 · PLOT B — 7-day-window model + daily forward H/L forecasts

The x-axis extends 7 days past `latest_complete` so you can see today's
**day-by-day** forecast for the next 7 days, with each day's CI band fanning
out as horizon grows.

* **Time-series part** (left of the divider) — for each historical as-of date, the
  point shows the 7-day-window MAX-H / MIN-L prediction issued that day. X-marks
  are the realised window max-H / min-L (only present for as-of dates whose 7-day
  window has fully expired).
* **Forecast part** (right of the divider, shaded khaki) — daily H/L predictions
  generated by 7 **per-horizon models** (k=1..7-days ahead, ridge regressions
  trained on the same features as the 1-day model). The CI bands widen as
  horizon grows: ±3.2 % at k=1 → ±13 % at k=7.
* **Black dotted close line** = BTC close on each as-of date (context).
""")
code(r"""
xs = pd.to_datetime(win_idx)
have_r = ~np.isnan(real_hi7)

# ── Forward daily predictions using the per-horizon models ──
F_horizon = F[AH["feat_cols"]]
feat_today = F_horizon.loc[[latest_complete]]
hi_models = AH["hi_models"]; lo_models = AH["lo_models"]
sig_hi    = AH["sigma_hi"];  sig_lo    = AH["sigma_lo"]
horizons  = AH["horizons"]   # [1..7]

fwd_dates = [latest_complete + pd.Timedelta(days=k) for k in horizons]
fwd_pred_hi = np.zeros(len(horizons)); fwd_pred_lo = np.zeros(len(horizons))
fwd_ci_hi_up = np.zeros(len(horizons)); fwd_ci_hi_dn = np.zeros(len(horizons))
fwd_ci_lo_up = np.zeros(len(horizons)); fwd_ci_lo_dn = np.zeros(len(horizons))
for i, k in enumerate(horizons):
    ph = float(hi_models[i].predict(feat_today)[0])
    pl = float(lo_models[i].predict(feat_today)[0])
    fwd_pred_hi[i]  = asof_p * (1 + max(ph, 0))
    fwd_pred_lo[i]  = asof_p * (1 - max(pl, 0))
    fwd_ci_hi_up[i] = asof_p * (1 + ph + 1.96*sig_hi[i])
    fwd_ci_hi_dn[i] = asof_p * (1 + max(ph - 1.96*sig_hi[i], 0))
    fwd_ci_lo_up[i] = asof_p * (1 - max(pl - 1.96*sig_lo[i], 0))
    fwd_ci_lo_dn[i] = asof_p * (1 - max(pl + 1.96*sig_lo[i], 0))

# Print summary
fwd_df = pd.DataFrame({
    "k (days ahead)":   horizons,
    "date":             [d.date() for d in fwd_dates],
    "Pred HIGH":        fwd_pred_hi.round(0).astype(int),
    "HIGH 95% lo":      fwd_ci_hi_dn.round(0).astype(int),
    "HIGH 95% hi":      fwd_ci_hi_up.round(0).astype(int),
    "Pred LOW":         fwd_pred_lo.round(0).astype(int),
    "LOW 95% lo":       fwd_ci_lo_dn.round(0).astype(int),
    "LOW 95% hi":       fwd_ci_lo_up.round(0).astype(int),
})
print(f"Daily forward forecasts from {latest_complete.date()} (close = ${asof_p:,.0f}):")
print(fwd_df.to_string(index=False))
""")

code(r"""
# ── Build Plot B ──
fig = go.Figure()

# Backtest CI bands (the 7-day-window aggregate model)
fig.add_trace(go.Scatter(x=xs, y=band_hi7_up, mode="lines",
                         line=dict(color="rgba(0,0,0,0)"),
                         showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=xs, y=band_hi7_dn, mode="lines",
                         line=dict(color="rgba(0,0,0,0)"),
                         fill="tonexty", fillcolor="rgba(0,160,0,0.13)",
                         name="Backtest MAX-H 95% CI", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=xs, y=band_lo7_up, mode="lines",
                         line=dict(color="rgba(0,0,0,0)"),
                         showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=xs, y=band_lo7_dn, mode="lines",
                         line=dict(color="rgba(0,0,0,0)"),
                         fill="tonexty", fillcolor="rgba(200,0,0,0.13)",
                         name="Backtest MIN-L 95% CI", hoverinfo="skip"))

# Backtest predicted lines
fig.add_trace(go.Scatter(
    x=xs, y=pred_hi7, mode="lines+markers",
    line=dict(color="green", width=2), marker=dict(size=7),
    name="Backtest Pred MAX-H",
    customdata=np.stack([np.array(target_7d_start), np.array(target_7d_end)], axis=-1),
    hovertemplate=("Pred MAX-H (7d window)<br>"
                   "Issued: %{x|%Y-%m-%d}<br>"
                   "$%{y:,.0f}<br>"
                   "Window: %{customdata[0]|%Y-%m-%d} → %{customdata[1]|%Y-%m-%d}"
                   "<extra></extra>"),
))
fig.add_trace(go.Scatter(
    x=xs, y=pred_lo7, mode="lines+markers",
    line=dict(color="red", width=2), marker=dict(size=7),
    name="Backtest Pred MIN-L",
    customdata=np.stack([np.array(target_7d_start), np.array(target_7d_end)], axis=-1),
    hovertemplate=("Pred MIN-L (7d window)<br>"
                   "Issued: %{x|%Y-%m-%d}<br>"
                   "$%{y:,.0f}<br>"
                   "Window: %{customdata[0]|%Y-%m-%d} → %{customdata[1]|%Y-%m-%d}"
                   "<extra></extra>"),
))

# Realised X markers
fig.add_trace(go.Scatter(
    x=xs[have_r], y=real_hi7[have_r], mode="markers",
    marker=dict(symbol="x-thin", size=13, line=dict(width=3, color="darkgreen")),
    name="Realised MAX-H",
    hovertemplate="Realised MAX-H<br>As-of: %{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=xs[have_r], y=real_lo7[have_r], mode="markers",
    marker=dict(symbol="x-thin", size=13, line=dict(width=3, color="darkred")),
    name="Realised MIN-L",
    hovertemplate="Realised MIN-L<br>As-of: %{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
))

# BTC close
fig.add_trace(go.Scatter(
    x=xs, y=closes.values, mode="lines",
    line=dict(color="black", width=1.2, dash="dot"),
    name="BTC close (as-of)",
    hovertemplate="Close<br>%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
))

# ───── FORWARD ZONE: daily forecasts from per-horizon models ─────
# Shaded khaki rectangle + vertical divider
fig.add_vrect(x0=latest_complete, x1=fwd_dates[-1],
              fillcolor="khaki", opacity=0.18, line_width=0, layer="below")
fig.add_vline(x=latest_complete, line=dict(color="goldenrod", width=2, dash="dash"))

# Anchor the forward predictions at today's close so the line is continuous
fwd_dates_full = [latest_complete] + fwd_dates
fwd_hi_full    = [asof_p] + list(fwd_pred_hi)
fwd_lo_full    = [asof_p] + list(fwd_pred_lo)
fwd_hi_up_full = [asof_p] + list(fwd_ci_hi_up)
fwd_hi_dn_full = [asof_p] + list(fwd_ci_hi_dn)
fwd_lo_up_full = [asof_p] + list(fwd_ci_lo_up)
fwd_lo_dn_full = [asof_p] + list(fwd_ci_lo_dn)

# CI fans (filled scatter areas)
fig.add_trace(go.Scatter(
    x=fwd_dates_full + fwd_dates_full[::-1],
    y=fwd_hi_up_full + fwd_hi_dn_full[::-1],
    fill="toself", fillcolor="rgba(0,160,0,0.20)",
    line=dict(color="rgba(0,0,0,0)"),
    name="Forward HIGH 95% CI", hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=fwd_dates_full + fwd_dates_full[::-1],
    y=fwd_lo_up_full + fwd_lo_dn_full[::-1],
    fill="toself", fillcolor="rgba(200,0,0,0.20)",
    line=dict(color="rgba(0,0,0,0)"),
    name="Forward LOW 95% CI", hoverinfo="skip",
))

# Forward forecast lines + markers (DAY-BY-DAY)
hover_hi = [f"Forward Pred HIGH<br>{d.date()} (k=+{k}d)<br>$%{{y:,.0f}}"
            f"<br>95% CI [${lo:,.0f}, ${up:,.0f}]"
            for d, k, lo, up in zip(fwd_dates, horizons, fwd_ci_hi_dn, fwd_ci_hi_up)]
hover_lo = [f"Forward Pred LOW<br>{d.date()} (k=+{k}d)<br>$%{{y:,.0f}}"
            f"<br>95% CI [${lo:,.0f}, ${up:,.0f}]"
            for d, k, lo, up in zip(fwd_dates, horizons, fwd_ci_lo_dn, fwd_ci_lo_up)]

fig.add_trace(go.Scatter(
    x=fwd_dates_full, y=fwd_hi_full, mode="lines+markers",
    line=dict(color="green", width=3),
    marker=dict(size=10, symbol="diamond", line=dict(width=1.5, color="white")),
    name="Forward Pred HIGH (per-horizon)",
    customdata=[[k, lo, up] for k, lo, up in zip([0]+list(horizons),
                                                  [asof_p]+list(fwd_ci_hi_dn),
                                                  [asof_p]+list(fwd_ci_hi_up))],
    hovertemplate=("Forward HIGH<br>"
                   "%{x|%Y-%m-%d} (k=+%{customdata[0]}d)<br>"
                   "$%{y:,.0f}<br>"
                   "95% CI [$%{customdata[1]:,.0f}, $%{customdata[2]:,.0f}]"
                   "<extra></extra>"),
))
fig.add_trace(go.Scatter(
    x=fwd_dates_full, y=fwd_lo_full, mode="lines+markers",
    line=dict(color="red", width=3),
    marker=dict(size=10, symbol="diamond", line=dict(width=1.5, color="white")),
    name="Forward Pred LOW (per-horizon)",
    customdata=[[k, lo, up] for k, lo, up in zip([0]+list(horizons),
                                                  [asof_p]+list(fwd_ci_lo_dn),
                                                  [asof_p]+list(fwd_ci_lo_up))],
    hovertemplate=("Forward LOW<br>"
                   "%{x|%Y-%m-%d} (k=+%{customdata[0]}d)<br>"
                   "$%{y:,.0f}<br>"
                   "95% CI [$%{customdata[1]:,.0f}, $%{customdata[2]:,.0f}]"
                   "<extra></extra>"),
))

# Annotation in forecast zone
fig.add_annotation(
    x=fwd_dates[3], y=max(fwd_ci_hi_up)*1.005, yshift=8,
    text=f"<b>Forecast window</b><br>{fwd_dates[0].date()} → {fwd_dates[-1].date()}",
    showarrow=False, font=dict(color="#b8860b", size=11),
    bgcolor="rgba(255,255,255,0.7)",
)

fig.update_xaxes(tickformat="%d-%b",
                 title="Date (as-of for backtest, calendar for forecast zone)")
fig.update_layout(
    title=f"PLOT B · 7-DAY-WINDOW backtest + DAILY forward forecasts — issued {asof.date()}<br>"
          f"<sup>Forward zone uses per-horizon models (k=1..7), CI widens with horizon</sup>",
    template=TEMPLATE, yaxis_title="BTC / USD",
    hovermode="x unified", height=680,
    legend=dict(orientation="h", y=-0.18, yanchor="top"),
)
fig.show()
""")

# ────────────────────────────────────────────────────────────────────── #
md("## 9 · Live look-back accuracy")
code(r"""
mask = ~np.isnan(real_hi1)
if mask.any():
    rh, rl = real_hi1[mask], real_lo1[mask]
    ph, pl = pred_hi1[mask], pred_lo1[mask]
    print(f"1-DAY model — last {mask.sum()} realised days")
    print(f"  MAPE  H/L : {np.mean(np.abs(ph-rh)/rh)*100:5.2f}% / {np.mean(np.abs(pl-rl)/rl)*100:5.2f}%")
    print(f"  Hit ±5%   : {np.mean(np.abs(ph-rh)/rh<=.05)*100:5.1f}% / {np.mean(np.abs(pl-rl)/rl<=.05)*100:5.1f}%")
    print(f"  95% CI cov: {np.mean((rh>=np.array(band_hi1_dn)[mask])&(rh<=np.array(band_hi1_up)[mask]))*100:5.1f}% (H) "
          f"{np.mean((rl>=np.array(band_lo1_dn)[mask])&(rl<=np.array(band_lo1_up)[mask]))*100:5.1f}% (L)")

mask7 = ~np.isnan(real_hi7)
if mask7.any():
    rh7, rl7 = real_hi7[mask7], real_lo7[mask7]
    ph7, pl7 = pred_hi7[mask7], pred_lo7[mask7]
    print(f"\n7-DAY model — last {mask7.sum()} realised windows")
    print(f"  MAPE  H/L : {np.mean(np.abs(ph7-rh7)/rh7)*100:5.2f}% / {np.mean(np.abs(pl7-rl7)/rl7)*100:5.2f}%")
    print(f"  Hit ±5%   : {np.mean(np.abs(ph7-rh7)/rh7<=.05)*100:5.1f}% / {np.mean(np.abs(pl7-rl7)/rl7<=.05)*100:5.1f}%")
    print(f"  Hit ±10%  : {np.mean(np.abs(ph7-rh7)/rh7<=.10)*100:5.1f}% / {np.mean(np.abs(pl7-rl7)/rl7<=.10)*100:5.1f}%")
""")

# ────────────────────────────────────────────────────────────────────── #
md(r"""## 10 · Caveats

* **1-day model** — R² ≈ 0.10, 95 % CI ±3.2/±3.7 %, usually hits ±5 % on 95-100 % of days.
* **7-day model** — R² is negative; outputs are climatological. CI is wide. Useful as a *range envelope*, not a tight price target.
* The 7-day plot can only show realised X marks for as-of dates whose 7-day window has fully expired.
* On-chain feeds must be published before inference (≈ UTC 03:00). The notebook waits for the latest complete day.
* The forward forecast in Plot B uses **7 per-horizon ridge models** (k=1..7), each trained to predict H_{t+k} and L_{t+k} from features at time t. CI half-widths grow from ±3.2 % at k=1 to ±13 % at k=7, reflecting the rapid decay of predictability. The on-screen MAX-H / MIN-L over the 7-day window can be read as max / min of the 7 daily HIGH and LOW points.
""")

nb = new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec":{"name":"python3","display_name":"Python 3"},
                  "language_info":{"name":"python","version":"3.14"}}
with open("/home/jovyan/btc-range-model/btc_inference.ipynb","w") as fp:
    nbformat.write(nb, fp)
print("Wrote btc_inference.ipynb (plotly, with forward 7-day forecast in Plot B)")
