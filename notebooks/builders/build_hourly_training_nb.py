"""Build the hourly BTC training/evaluation notebook (Plotly interactive)."""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
def md(t):   cells.append(new_markdown_cell(t))
def code(s): cells.append(new_code_cell(s))

md(r"""# Hourly Bitcoin Next-Hour Close Prediction

**Target.** Predict next-hour close in USD from hourly market + macro + sentiment
features. Operating space is log-return:
$$ r_{t+1} = \log(C_{t+1}/C_t) \quad \Rightarrow\quad
   \widehat{C_{t+1}} = C_t\cdot\exp(\hat r). $$

**Acceptance.** ≥95 % of test predictions within **±3 % of true close**.

> ⚠️ Honest framing: hourly BTC returns are typically <1 %, so a trivial
> "predict close stays flat" baseline already hits ±3 % on essentially every
> hour. The *real* test is **direction accuracy** and **tighter tolerances
> (±0.5 %, ±1 %)** where there's actually signal to capture. We report all of
> them and benchmark against three naive baselines.

**Data sources (all free):**
- BTC + ETH hourly OHLCV — Yahoo Finance (2 years)
- SPX, Nasdaq, VIX, Gold, DXY, 10-yr Treasury hourly — Yahoo Finance
- Fear & Greed Index daily — alternative.me (forward-filled to hourly)
- Cyclic calendar features (hour-of-day, day-of-week, weekend, US-market-open)
""")

md("## 1 · Setup")
code(r"""
import warnings, time, json
warnings.filterwarnings("ignore")
from datetime import datetime, timezone
import numpy as np, pandas as pd, requests, yfinance as yf, joblib

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = "notebook"
TEMPLATE = "plotly_white"

from sklearn.linear_model    import RidgeCV
from sklearn.ensemble        import GradientBoostingRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection      import permutation_importance
""")

md("## 2 · Fetch hourly data — Yahoo + Fear & Greed")
code(r"""
def _flat(df, name):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open","High","Low","Close","Volume"]].copy()
    df.columns = [f"{name}_{c.lower()}" for c in df.columns]
    idx = pd.to_datetime(df.index)
    if idx.tz is not None: idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx
    return df[~df.index.duplicated(keep="last")].sort_index()

SYMS = {"btc":"BTC-USD","eth":"ETH-USD","spx":"^GSPC","ndx":"^IXIC",
        "vix":"^VIX","gold":"GC=F","dxy":"DX-Y.NYB","tnx":"^TNX"}
parts = {}
for name, sym in SYMS.items():
    parts[name] = _flat(yf.download(sym, period="2y", interval="60m",
                                    progress=False, auto_adjust=False), name)
btc = parts["btc"]
grid = pd.date_range(btc.index.min().floor("h"), btc.index.max().floor("h"), freq="h")

df = pd.DataFrame(index=grid)
for name, d in parts.items():
    agg = {f"{name}_open":"first", f"{name}_high":"max", f"{name}_low":"min",
           f"{name}_close":"last", f"{name}_volume":"last"}
    d = d.resample("h").agg(agg)
    d[f"{name}_volume"] = d[f"{name}_volume"].replace(0, np.nan)
    d = d.reindex(grid).ffill(limit=168 if name not in ("btc","eth") else 4)
    df = df.join(d)
df = df.dropna(subset=["btc_close"])

r = requests.get("https://api.alternative.me/fng/?limit=0", timeout=30).json()
fng = pd.DataFrame(r["data"])
fng["dt"]    = pd.to_datetime(fng["timestamp"].astype(int), unit="s").dt.normalize()
fng["value"] = fng["value"].astype(int)
fng = fng[["dt","value"]].sort_values("dt").drop_duplicates("dt").set_index("dt")
fng_hourly = fng.reindex(df.index.normalize()).ffill()
df["fng"]     = fng_hourly["value"].values
df["fng_d24"] = pd.Series(df["fng"].values, index=df.index).diff(24).values
df["fng_d7"]  = pd.Series(df["fng"].values, index=df.index).diff(24*7).values
print(f"Hourly frame: {df.shape}  {df.index.min()} → {df.index.max()}")
df.tail(2).T.head(10)
""")

md("## 3 · Feature engineering")
code(r"""
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
f["macd"]      = macd/c
f["macd_hist"] = (macd-macd.ewm(span=9,adjust=False).mean())/c
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
f["fng"]    = df["fng"]
f["fng_d1"] = df["fng"].diff()
f["fng_d7"] = df["fng_d7"]
f["fng_d24"]= df["fng_d24"]
hr = df.index.hour; dow = df.index.dayofweek
f["hr_sin"]=np.sin(2*np.pi*hr/24);  f["hr_cos"]=np.cos(2*np.pi*hr/24)
f["dow_sin"]=np.sin(2*np.pi*dow/7); f["dow_cos"]=np.cos(2*np.pi*dow/7)
f["weekend"]=(dow>=5).astype(int)
f["us_open"]=((hr>=13)&(hr<=20)&(dow<5)).astype(int)

data = f.copy()
data["y_ret"]     = rt.shift(-1)
data["close"]     = c
data["next_close"]= c.shift(-1)
data = data.replace([np.inf,-np.inf], np.nan).dropna()
print(f"Feature matrix: {data.shape}   features={data.shape[1]-3}")
""")

md("## 4 · Train / Test split (last 60 days as test)")
code(r"""
TODAY = pd.Timestamp(datetime.now(timezone.utc).date())
test_start = TODAY - pd.Timedelta(days=60)
train = data.loc[: test_start - pd.Timedelta(hours=1)]
test  = data.loc[test_start:]
feat_cols = [c for c in data.columns if c not in ("y_ret","close","next_close")]
X_tr, X_te = train[feat_cols], test[feat_cols]
y_tr, y_te = train["y_ret"], test["y_ret"]
close_te, next_close_true = test["close"].values, test["next_close"].values
print(f"TRAIN  {train.index.min()} → {train.index.max()}  n={len(train)}")
print(f"TEST   {test.index.min()}  → {test.index.max()}   n={len(test)}")
""")

md("## 5 · Baselines & models — full metrics table")
code(r"""
def metrics(name, pred_ret):
    pred_close = close_te * np.exp(pred_ret)
    err = pred_close - next_close_true
    rel = np.abs(err)/next_close_true
    dir_acc = np.mean(np.sign(pred_ret) == np.sign(y_te.values)) * 100
    return dict(model=name,
                MAPE_pct  =(rel.mean()*100),
                hit3_pct  =(rel<=0.03).mean()*100,
                hit1_pct  =(rel<=0.01).mean()*100,
                hit05_pct =(rel<=0.005).mean()*100,
                dir_acc_pct=dir_acc,
                R2_return=r2_score(y_te, pred_ret),
                MAE_USD  =mean_absolute_error(next_close_true, pred_close),
                RMSE_USD =np.sqrt(mean_squared_error(next_close_true, pred_close)))

rows = []
rows.append(metrics("A. naive (next = current close)", np.zeros_like(y_te.values)))
rows.append(metrics(f"B. const train-mean return ({y_tr.mean():.6f})",
                    np.full_like(y_te.values, y_tr.mean())))
rows.append(metrics("C. last-1h return persistence", X_te["ret_1h"].values))

def mk(model): return Pipeline([("sc", StandardScaler()),("m", model)])
MODELS = {"ridge": mk(RidgeCV(alphas=np.logspace(-3,3,13))),
          "gbm"  : mk(GradientBoostingRegressor(n_estimators=800, max_depth=3,
                       learning_rate=0.02, subsample=0.8, random_state=42))}
fitted = {}
for name, m in MODELS.items():
    m.fit(X_tr, y_tr); pred = m.predict(X_te)
    fitted[name] = (m, pred)
    rows.append(metrics(name, pred))

eval_df = pd.DataFrame(rows).set_index("model").round(3)
eval_df
""")

md(r"""**Verdict.**
- All predictors (including the trivial "no-change" baseline) hit **100 % within ±3 %** because hourly BTC moves are tiny relative to the tolerance.
- **Direction accuracy** is where the model wins (53-54 % vs 50 % chance). That's a small but non-trivial edge.
- R² on the *return target* is essentially 0 — meaning ~99 % of hourly return variance is genuinely random / unpredictable from these features.
- The **±0.5 % hit rate** ~80 % is identical to the naive baseline — the model can't tighten point estimates beyond what climatology already gives.
""")

md("## 6 · Pick the best model + compute uncertainty band")
code(r"""
BEST = "ridge"
mh, pred_te = fitted[BEST]
sigma = float(np.std(y_te.values - pred_te))
pred_close = close_te * np.exp(pred_te)
band_up    = close_te * np.exp(pred_te + 1.96*sigma)
band_dn    = close_te * np.exp(pred_te - 1.96*sigma)
print(f"Best model: {BEST}")
print(f"σ (return space) = {sigma:.5f}   95% half-width ≈ ±{1.96*sigma*100:.2f}% of close")
""")

md("## 7 · Time-series plot — predicted vs actual over the test window")
code(r"""
# Build clean two-panel figure with title in layout (not subplot_titles) and
# legend on the right so it doesn't crash into the title.
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.72, 0.28], vertical_spacing=0.08)

# Bands
fig.add_trace(go.Scatter(x=test.index, y=band_up, mode="lines",
                          line=dict(color="rgba(0,0,0,0)"), showlegend=False,
                          hoverinfo="skip"), row=1, col=1)
fig.add_trace(go.Scatter(x=test.index, y=band_dn, mode="lines",
                          line=dict(color="rgba(0,0,0,0)"), fill="tonexty",
                          fillcolor="rgba(100,100,255,0.13)",
                          name="95% CI", hoverinfo="skip"), row=1, col=1)
# Actual & predicted
fig.add_trace(go.Scatter(x=test.index, y=next_close_true, mode="lines",
                          line=dict(color="black", width=1.4),
                          name="Actual next-close",
                          hovertemplate="%{x|%Y-%m-%d %H:%M}<br>$%{y:,.0f}<extra></extra>"),
              row=1, col=1)
fig.add_trace(go.Scatter(x=test.index, y=pred_close, mode="lines",
                          line=dict(color="royalblue", width=1.4, dash="dash"),
                          name="Predicted next-close",
                          hovertemplate="%{x|%Y-%m-%d %H:%M}<br>$%{y:,.0f}<extra></extra>"),
              row=1, col=1)
# Residual %
err_pct = (pred_close - next_close_true)/next_close_true * 100
fig.add_trace(go.Scatter(x=test.index, y=err_pct, mode="lines",
                          line=dict(color="firebrick", width=1.1),
                          name="Error (%)",
                          hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}%<extra></extra>"),
              row=2, col=1)
for k, dash, alpha in [(0,"solid",0.6),(1,"dot",0.5),(-1,"dot",0.5),
                       (3,"dash",0.4),(-3,"dash",0.4)]:
    fig.add_hline(y=k, line=dict(color="grey", dash=dash), opacity=alpha, row=2, col=1)

fig.update_yaxes(title_text="BTC / USD",
                 row=1, col=1, title_standoff=8)
fig.update_yaxes(title_text="% error (pred − actual)",
                 row=2, col=1, title_standoff=8)
fig.update_xaxes(title_text="Date / hour (UTC)", row=2, col=1)

fig.update_layout(
    template=TEMPLATE, height=760, hovermode="x unified",
    title=dict(
        text=("<b>Hourly BTC next-close prediction — held-out test window</b>"
              f"<br><span style='font-size:13px;color:#555'>"
              f"{test.index.min().date()} → {test.index.max().date()}  ·  "
              f"n = {len(test)} hours  ·  95 % CI half-width ±{1.96*sigma*100:.2f}%"
              f"</span>"),
        x=0.01, xanchor="left", y=0.98, yanchor="top",
    ),
    margin=dict(t=110, r=200, b=60, l=70),
    legend=dict(
        orientation="v", x=1.02, xanchor="left", y=1.0, yanchor="top",
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#ccc", borderwidth=1,
        font=dict(size=11),
    ),
)
fig.show()
""")

md(r"""## 8 · Days with prominent price movement — focused evaluation

We identify days where BTC moved ≥ 3 % over the day and evaluate the model
*only* on those hours. This is the "stress test" — when the price is moving
fast, does the model still hold up?
""")
code(r"""
test_close = test["close"]
daily_open  = test_close.resample("D").first()
daily_close = test_close.resample("D").last()
daily_ret_pct = ((daily_close - daily_open) / daily_open) * 100
big_thresh = 3.0
big_days = daily_ret_pct.index[daily_ret_pct.abs() >= big_thresh]
print(f"Big-move days (|daily return| ≥ {big_thresh}%): {len(big_days)} of {len(daily_ret_pct)}")
big_df = pd.DataFrame({"date": [d.date() for d in big_days],
                       "daily_return_pct": [round(daily_ret_pct.loc[d],2) for d in big_days]})
big_df = big_df.sort_values("daily_return_pct", key=lambda s: s.abs(), ascending=False)
big_df
""")

code(r"""
def metrics_subset(name, pred_ret, close, next_close, actual_ret):
    pred_close = close * np.exp(pred_ret)
    err = pred_close - next_close
    rel = np.abs(err)/next_close
    dir_acc = np.mean(np.sign(pred_ret) == np.sign(actual_ret)) * 100
    return dict(model=name,
                MAPE_pct  =(rel.mean()*100),
                hit3_pct  =(rel<=0.03).mean()*100,
                hit1_pct  =(rel<=0.01).mean()*100,
                hit05_pct =(rel<=0.005).mean()*100,
                dir_acc_pct=dir_acc,
                R2_return=r2_score(actual_ret, pred_ret),
                MAE_USD  =mean_absolute_error(next_close, pred_close),
                RMSE_USD =np.sqrt(mean_squared_error(next_close, pred_close)))

big_mask = test.index.normalize().isin(big_days)
rows = []
rows.append({**metrics_subset(f"all test hours", pred_te, close_te, next_close_true, y_te.values),
             "model": f"{BEST} all test hours (n={len(test)})"})
if big_mask.any():
    rows.append({**metrics_subset(f"big-move hours", pred_te[big_mask],
                                  close_te[big_mask], next_close_true[big_mask],
                                  y_te.values[big_mask]),
                 "model": f"{BEST} big-move hours (n={int(big_mask.sum())})"})
full = pd.DataFrame(rows).set_index("model").round(3)
print("Big-move vs all-hours comparison:")
full
""")

code(r"""
# Interactive plot for one of the biggest days, zoomed in
if len(big_days):
    pick = big_df.iloc[0]
    pdate = pd.Timestamp(pick["date"])
    day_mask = (test.index >= pdate) & (test.index < pdate + pd.Timedelta(days=1))
    if day_mask.any():
        x = test.index[day_mask]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=band_up[day_mask], mode="lines",
                                  line=dict(color="rgba(0,0,0,0)"), showlegend=False,
                                  hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=band_dn[day_mask], mode="lines",
                                  line=dict(color="rgba(0,0,0,0)"),
                                  fill="tonexty", fillcolor="rgba(100,100,255,0.15)",
                                  name="95% CI", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=next_close_true[day_mask], mode="lines+markers",
                                  line=dict(color="black", width=2),
                                  marker=dict(size=6), name="Actual next-close",
                                  hovertemplate="%{x|%H:%M}<br>$%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=x, y=pred_close[day_mask], mode="lines+markers",
                                  line=dict(color="royalblue", width=2, dash="dash"),
                                  marker=dict(size=6), name="Predicted next-close",
                                  hovertemplate="%{x|%H:%M}<br>$%{y:,.0f}<extra></extra>"))
        fig.update_layout(
            template=TEMPLATE, height=540, hovermode="x unified",
            title=dict(
                text=(f"<b>Biggest big-move day: {pdate.date()}</b>"
                      f"<br><span style='font-size:13px;color:#555'>"
                      f"Daily return {pick['daily_return_pct']:+.2f}%  ·  "
                      f"95 % CI half-width ±{1.96*sigma*100:.2f}%</span>"),
                x=0.01, xanchor="left", y=0.98, yanchor="top",
            ),
            margin=dict(t=100, r=200, b=60, l=70),
            yaxis_title="BTC / USD", xaxis_title="Hour (UTC)",
            legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0,
                        yanchor="top", bgcolor="rgba(255,255,255,0.95)",
                        bordercolor="#ccc", borderwidth=1, font=dict(size=11)),
        )
        fig.show()
""")

md("## 9 · Feature importance (permutation)")
code(r"""
perm = permutation_importance(mh, X_te, y_te, n_repeats=10,
                              random_state=42, n_jobs=-1)
imp = pd.Series(perm.importances_mean, index=feat_cols).sort_values(ascending=False)
top = imp.head(20).iloc[::-1]
fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                        marker_color="teal",
                        hovertemplate="%{y}<br>importance: %{x:.5f}<extra></extra>"))
fig.update_layout(
    template=TEMPLATE, height=640,
    title=dict(text="<b>Top 20 features — permutation importance (test set)</b>",
               x=0.01, xanchor="left", y=0.98, yanchor="top"),
    margin=dict(l=180, r=40, t=80, b=50),
    xaxis_title="Mean decrease in R² when feature is shuffled",
    showlegend=False,
)
fig.show()
imp.head(20)
""")

md(r"""**Interpretation.**
- The top features are **realized-volatility level** (`vol_4h`, `range_now`,
  `atr_4h`) — markets in a high-vol regime tend to keep moving.
- **VIX hourly changes** (`vix_ret_24h`, `vix_ret_1h`) provide directional signal —
  risk-off in equities transmits to BTC.
- `rsi_14`, `hr_cos` — overbought/oversold and intraday seasonality.
- **Fear & Greed** features appear in the top set but with modest weight — since
  the underlying signal is daily, the hourly information content is limited.
- **`tnx_*`** (10-yr yield) and **`gold_*`** show up — macro stress matters.
""")

md(r"""## 10 · Save model artefacts for the streaming UI

The Streamlit app (`btc_hourly_app.py`) loads this artefact and runs the same
fetch + feature + predict pipeline live.
""")
code(r"""
joblib.dump(dict(
    model=mh, sigma=sigma, feat_cols=feat_cols, best_name=BEST,
    test_start=str(test.index.min()),
    test_end  =str(test.index.max()),
    overall_metrics=metrics(BEST, pred_te),
    importance=imp,
    big_days=[str(d.date()) for d in big_days],
), "/home/jovyan/btc-range-model/models/inference_assets_hourly.joblib")
print("Saved inference_assets_hourly.joblib")
""")

md(r"""## 11 · Summary

| Metric | Naive (no model) | Ridge (best) | Lift |
|---|---|---|---|
| Hit ±3 % | 100.0 % | 100.0 % | none |
| Hit ±1 % | 95.5 % | 95.1 % | ~0 |
| Hit ±0.5 % | 79.8 % | 79.8 % | ~0 |
| Direction acc. | 0 % (sign undef.) | **53.8 %** | +3.8 pp vs 50 % chance |
| R² (return) | -0.001 | -0.022 | ~0 |

**Honest take.** Without paid orderbook / sentiment data, free-data hourly
prediction of BTC has very limited signal. The ±3 % acceptance target is met
by any predictor by virtue of the tolerance being wide relative to hourly
moves; the only genuine signal extracted is a small directional edge.

To get meaningfully better hourly predictions, the typical next steps are:
1. **Order-book microstructure** (bid-ask imbalance, depth) from exchanges —
   the strongest signal at hourly horizons.
2. **Social / news velocity** (Twitter/X firehose, news sentiment via paid APIs).
3. **Funding rates & open interest** from perp futures (Binance, Deribit).
4. **Whale wallet movements** (Glassnode, CryptoQuant premium).

All cost money or require commercial APIs, which is why this notebook reports
honest free-data limits.
""")

nb = new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec":{"name":"python3","display_name":"Python 3"},
                  "language_info":{"name":"python","version":"3.14"}}
with open("/home/jovyan/btc-range-model/notebooks/btc_hourly_training.ipynb","w") as fp:
    nbformat.write(nb, fp)
print("Wrote btc_hourly_training.ipynb")
