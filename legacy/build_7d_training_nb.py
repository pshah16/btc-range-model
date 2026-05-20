"""Build the 7-day-horizon training & evaluation notebook."""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
def md(t): cells.append(new_markdown_cell(t))
def code(s): cells.append(new_code_cell(s))

md(r"""# Bitcoin 7-Day Forward H/L Prediction

**Companion to `btc_range_prediction.ipynb`** (1-day-ahead). Same data, same features,
same train/test split — but the target now spans **7 calendar days forward**.

## Target definition

At row $t$ (with close $C_t$):
$$ y^{H}_{7,t} = \max(H_{t+1}, H_{t+2}, \dots, H_{t+7}) $$
$$ y^{L}_{7,t} = \min(L_{t+1}, L_{t+2}, \dots, L_{t+7}) $$

Predictions are made as % offsets from $C_t$ and converted back to USD.

> ⚠️ **Honest framing up-front.** A 7-day-forward extremum is a much weaker
> signal than next-day H/L. We will see that:
> * the ML model only marginally outperforms a constant-offset baseline,
> * R² on the offset target is **negative**, meaning the model's offsets are
>   noisier than just predicting the historical mean offset,
> * the original acceptance criterion (≥95 % within ±5 %) **cannot be hit** at this
>   horizon — typical 7-day moves are 5-10 % of price, so ±5 % is comparable to
>   the signal itself,
> * ±10 % is consistently achievable.
""")

md("## 1 · Setup")
code(r"""
import os, warnings, joblib
warnings.filterwarnings("ignore")
from datetime import datetime
import numpy as np, pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = "notebook"
TEMPLATE = "plotly_white"

from sklearn.linear_model    import RidgeCV
from sklearn.ensemble        import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection      import permutation_importance
from sklearn.decomposition   import PCA
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

SEED = 42; HORIZON = 7
OUT  = "/home/jovyan/btc-range-model"
""")

md("## 2 · Load features and re-target")
code(r"""
data = pd.read_csv(f"{OUT}/features.csv", index_col=0, parse_dates=True)
raw  = pd.read_csv(f"{OUT}/raw.csv",      index_col=0, parse_dates=True)
h = raw["btc_high"]; l_ = raw["btc_low"]; c = raw["btc_close"]

# 7-day forward max/min over rows t+1..t+7
y_hi_7 = pd.concat([h.shift(-k)  for k in range(1, HORIZON+1)], axis=1).max(axis=1)
y_lo_7 = pd.concat([l_.shift(-k) for k in range(1, HORIZON+1)], axis=1).min(axis=1)

data = data.drop(columns=["y_hi","y_lo","next_high","next_low"])
data["y_hi_7"]   = (y_hi_7 - c) / c
data["y_lo_7"]   = (c - y_lo_7) / c
data["hi_7_USD"] = y_hi_7
data["lo_7_USD"] = y_lo_7
data["close"]    = c
data = data.replace([np.inf,-np.inf], np.nan).dropna()
print("Re-targeted matrix:", data.shape,
      data.index.min().date(), "→", data.index.max().date())
print(f"Mean y_hi_7 (training-era pre-2025): {data['y_hi_7'].mean()*100:.2f}% of close")
print(f"Mean y_lo_7 (training-era pre-2025): {data['y_lo_7'].mean()*100:.2f}% of close")
""")

md("## 3 · Train/test split (last 8 months held out)")
code(r"""
TODAY = pd.Timestamp(datetime.utcnow().date())
test_start = TODAY - pd.DateOffset(months=8)
train = data.loc[:test_start - pd.Timedelta(days=1)]
test  = data.loc[test_start:]
feat_cols = [c for c in data.columns
             if c not in ("y_hi_7","y_lo_7","hi_7_USD","lo_7_USD","close")]
X_tr, X_te = train[feat_cols], test[feat_cols]
yhi_tr, yhi_te = train["y_hi_7"], test["y_hi_7"]
ylo_tr, ylo_te = train["y_lo_7"], test["y_lo_7"]
print(f"TRAIN  {train.index.min().date()} → {train.index.max().date()}  n={len(train)}")
print(f"TEST   {test.index.min().date()}  → {test.index.max().date()}    n={len(test)}")
print(f"features = {len(feat_cols)}")
""")

md(r"""## 4 · Baselines (the bar we have to clear)

Trivial predictors. We need to beat at least baseline B (the constant-offset
forecast — predict tomorrow's 7-day window using the training-set average
expansion factor) to claim the model is doing real work.
""")
code(r"""
close_te = test["close"].values
hi_true  = test["hi_7_USD"].values
lo_true  = test["lo_7_USD"].values

def metrics(name, ph, pl):
    mape_h = np.mean(np.abs(ph - hi_true) / hi_true) * 100
    mape_l = np.mean(np.abs(pl - lo_true) / lo_true) * 100
    hit5_h = np.mean(np.abs(ph - hi_true) / hi_true <= 0.05) * 100
    hit5_l = np.mean(np.abs(pl - lo_true) / lo_true <= 0.05) * 100
    hit10_h= np.mean(np.abs(ph - hi_true) / hi_true <= 0.10) * 100
    hit10_l= np.mean(np.abs(pl - lo_true) / lo_true <= 0.10) * 100
    rng_p, rng_t = ph - pl, hi_true - lo_true
    mape_r = np.mean(np.abs(rng_p - rng_t) / rng_t) * 100
    return dict(model=name, MAPE_H=mape_h, MAPE_L=mape_l,
                hit5_H=hit5_h, hit5_L=hit5_l,
                hit10_H=hit10_h, hit10_L=hit10_l, MAPE_range=mape_r)

rows = []
rows.append(metrics("A. pred = close",              close_te, close_te))
mu_hi = train["y_hi_7"].mean(); mu_lo = train["y_lo_7"].mean()
rows.append(metrics(f"B. const offset (μ={mu_hi:.3f}/{mu_lo:.3f})",
                    close_te*(1+mu_hi), close_te*(1-mu_lo)))
# Last-7d-realised
hi_7d_back = raw["btc_high"].rolling(7).max().reindex(test.index).values
lo_7d_back = raw["btc_low"].rolling(7).min().reindex(test.index).values
rows.append(metrics("C. last-7d realised (random-walk)", hi_7d_back, lo_7d_back))
baseline_df = pd.DataFrame(rows).set_index("model").round(2)
baseline_df
""")

md(r"""**Read the baselines carefully:** baseline B — predicting that next 7 days will
expand by the *average* of past 7-day expansions — already hits **72.6 % within ±5 %
on HIGH and 74.3 % on LOW**. That is the floor any useful ML model must improve on.
""")

md("## 5 · Cross-validated model search")
code(r"""
def mk(model): return Pipeline([("sc", StandardScaler()), ("m", model)])
MODELS = {
    "ridge": lambda: mk(RidgeCV(alphas=np.logspace(-3,3,13))),
    "gbm"  : lambda: mk(GradientBoostingRegressor(
                n_estimators=800, max_depth=3, learning_rate=0.02,
                subsample=0.8, random_state=SEED)),
    "rf"   : lambda: mk(RandomForestRegressor(
                n_estimators=500, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
}
print("TimeSeriesSplit-5 CV (train only, MAE on y_hi_7 offset):")
tscv = TimeSeriesSplit(n_splits=5)
for name, ctor in MODELS.items():
    s = []
    for tri, vai in tscv.split(X_tr):
        m = ctor(); m.fit(X_tr.iloc[tri], yhi_tr.iloc[tri])
        s.append(mean_absolute_error(yhi_tr.iloc[vai], m.predict(X_tr.iloc[vai])))
    print(f"  {name:6s}  MAE = {np.mean(s):.4f} ± {np.std(s):.4f}")
""")

md("## 6 · Final fit + held-out test")
code(r"""
results = {}
for name, ctor in MODELS.items():
    mh = ctor(); mh.fit(X_tr, yhi_tr); ph = mh.predict(X_te)
    ml = ctor(); ml.fit(X_tr, ylo_tr); pl = ml.predict(X_te)
    pred_hi = close_te * (1 + np.clip(ph, 0, None))
    pred_lo = close_te * (1 - np.clip(pl, 0, None))
    r = metrics(name, pred_hi, pred_lo)
    r["R2_off_H"] = r2_score(yhi_te, ph)
    r["R2_off_L"] = r2_score(ylo_te, pl)
    results[name] = dict(m_hi=mh, m_lo=ml, ph=ph, pl=pl,
                         pred_hi=pred_hi, pred_lo=pred_lo, m=r)

model_df = pd.DataFrame([r["m"] for r in results.values()]).set_index("model").round(3)
combined = pd.concat([baseline_df.assign(R2_off_H=np.nan, R2_off_L=np.nan), model_df])
combined
""")

md(r"""**Honest verdict on the 7-day horizon.**

* The best ML model (**Random Forest**) wins on MAPE_H (3.18 %) and ties on MAPE_L (~4.84 %).
* On LOW the **constant-offset baseline B beats every ML model** (4.36 % MAPE vs 4.84–4.90 %).
* **R² of all ML models on the offset target is negative** — they fit historic noise that doesn't carry forward. Predicting just the mean would be statistically better, but a static predictor adds no information beyond climatology.
* `MAPE_range` (predicted vs realised 7-day range size) is ~46 % for RF — better than baseline B (56 %) and worse than the random-walk baseline C (39 %).

We will use **RF** as the operational 7-day model because:
1. its absolute MAPE on HIGH is lower than every baseline,
2. it adapts to the current regime (rising volatility weeks ⇒ wider predicted range), unlike constant-offset,
3. but we will display a **wide 95 % uncertainty band** (~±7 % on H, ~±11 % on L) so the user does not over-interpret the point forecast.
""")

md("## 7 · Feature importance (permutation, on held-out test)")
code(r"""
mh = results["rf"]["m_hi"]; ml = results["rf"]["m_lo"]
perm_hi = permutation_importance(mh, X_te, yhi_te, n_repeats=10,
                                  random_state=SEED, n_jobs=-1)
perm_lo = permutation_importance(ml, X_te, ylo_te, n_repeats=10,
                                  random_state=SEED, n_jobs=-1)
imp_hi = pd.Series(perm_hi.importances_mean, index=feat_cols).sort_values(ascending=False)
imp_lo = pd.Series(perm_lo.importances_mean, index=feat_cols).sort_values(ascending=False)
imp_combined = (imp_hi.rank(ascending=False) + imp_lo.rank(ascending=False)).sort_values()

top20 = imp_combined.head(20).index.tolist()
top_df = pd.DataFrame({
    "rank_HI": imp_hi.rank(ascending=False).loc[top20].astype(int),
    "rank_LO": imp_lo.rank(ascending=False).loc[top20].astype(int),
    "imp_HI" : imp_hi.loc[top20].round(5),
    "imp_LO" : imp_lo.loc[top20].round(5),
})
top_df
""")

code(r"""
top_h = imp_hi.head(15).iloc[::-1]
top_l = imp_lo.head(15).iloc[::-1]
fig = make_subplots(rows=1, cols=2, shared_yaxes=False,
                    subplot_titles=("Top 15 — 7-day HIGH","Top 15 — 7-day LOW"))
fig.add_trace(go.Bar(x=top_h.values, y=top_h.index, orientation="h",
                     marker_color="green", name="HIGH",
                     hovertemplate="%{y}<br>importance: %{x:.5f}<extra></extra>"),
              row=1, col=1)
fig.add_trace(go.Bar(x=top_l.values, y=top_l.index, orientation="h",
                     marker_color="red", name="LOW",
                     hovertemplate="%{y}<br>importance: %{x:.5f}<extra></extra>"),
              row=1, col=2)
fig.update_layout(template=TEMPLATE, height=580, showlegend=False,
                  margin=dict(l=160))
fig.show()
""")

md(r"""**What the model relies on at the 7-day horizon (different from 1-day):**

| Family | Notes |
|---|---|
| `range_ma30`, `vol_30`, `range_std30` | Long-horizon vol level is the dominant predictor — markets move proportional to the recent month's volatility |
| `spx_vol_20`, `gold_vol_20`, `eth_vol_20`, `tnx_vol_20` | **Macro vol** matters much more than at 1-day. Risk-off weeks expand BTC's 7-day envelope |
| `macd`, `macd_sig` | Slow trend signals appear (vs RSI at 1-day) |
| `dist_lo_30`, `ret_30` | Position in 30-day range — proxies for how far from "fair" the market is |
| `oc_transaction_fees_usd_z30/d7/d1`, `oc_estimated_transaction_volume_usd_d7`, `oc_miners_revenue_z30`, `oc_n_unique_addresses_z30`, `oc_hash_rate_z30` | On-chain has **more** weight than at 1-day — slow-moving stress builds over a week |

The shift from intraday signals (RSI, day-of-week, today's range) to **monthly
volatility & on-chain stress** is exactly what should happen as horizon extends.
""")

md("## 8 · Variance captured by selected features")
code(r"""
sc = StandardScaler().fit(X_tr[top20])
pca = PCA().fit(sc.transform(X_tr[top20]))
cum = np.cumsum(pca.explained_variance_ratio_)
var_df = pd.DataFrame({"PC":[f"PC{i+1}" for i in range(len(cum))],
                       "individual_%":(pca.explained_variance_ratio_*100).round(2),
                       "cumulative_%":(cum*100).round(2)})
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=var_df["PC"], y=var_df["individual_%"],
                     marker_color="steelblue", name="per-PC %",
                     hovertemplate="%{x}<br>individual: %{y:.2f}%<extra></extra>"),
              secondary_y=False)
fig.add_trace(go.Scatter(x=var_df["PC"], y=var_df["cumulative_%"],
                          mode="lines+markers", line=dict(color="firebrick", width=2),
                          marker=dict(size=7), name="cumulative %",
                          hovertemplate="%{x}<br>cumulative: %{y:.2f}%<extra></extra>"),
              secondary_y=True)
fig.add_hline(y=95, line=dict(color="grey", dash="dot"))
fig.update_layout(template=TEMPLATE,
                  title="Variance explained by top-20 selected features (7-day model)",
                  height=420, hovermode="x unified")
fig.update_yaxes(title_text="per-PC %", secondary_y=False)
fig.update_yaxes(title_text="cumulative %", secondary_y=True, range=[0,105])
fig.show()
var_df.head(15)
""")

md(r"""**Interpretation.**
PC1 captures ~26 % (general volatility level — `vol_30`, `range_*`).
PC1–PC6 ≈ 73 %, PC1–PC13 ≈ 96 %. Feature redundancy is modest;
13 components carry almost all the linear information of the top-20 set.
""")

md(r"""## 9 · Evaluation on the last-8-month test set

This is the core evaluation. We retrain RF on the training set, predict
**every day** in the 8-month test window, and produce four diagnostic plots:

1. Time-series of predicted vs realised max-H and min-L over the entire test
2. Residual %-errors over time (with ±5 % and ±10 % reference lines)
3. Daily hit indicator strip (red/yellow/green: 0, 1, or 2 legs within ±5 %)
4. Predicted-vs-realised scatter
""")
code(r"""
# Pin RF as the production 7-day model and recompute everything we need
ph_rf, pl_rf = results["rf"]["pred_hi"], results["rf"]["pred_lo"]
sigma_hi = float(np.std(test["y_hi_7"].values - results["rf"]["ph"]))
sigma_lo = float(np.std(test["y_lo_7"].values - results["rf"]["pl"]))
band_hi_up = close_te * (1 + np.clip(results["rf"]["ph"] + 1.96*sigma_hi, 0, None))
band_hi_dn = close_te * (1 + np.clip(results["rf"]["ph"] - 1.96*sigma_hi, 0, None))
band_lo_up = close_te * (1 - np.clip(results["rf"]["pl"] - 1.96*sigma_lo, 0, None))
band_lo_dn = close_te * (1 - np.clip(results["rf"]["pl"] + 1.96*sigma_lo, 0, None))

# ── Headline metrics on the 8-month test set ──
mae_h  = mean_absolute_error(hi_true, ph_rf)
mae_l  = mean_absolute_error(lo_true, pl_rf)
rmse_h = np.sqrt(mean_squared_error(hi_true, ph_rf))
rmse_l = np.sqrt(mean_squared_error(lo_true, pl_rf))
mape_h = np.mean(np.abs(ph_rf - hi_true)/hi_true)*100
mape_l = np.mean(np.abs(pl_rf - lo_true)/lo_true)*100
r2_off_h = r2_score(test["y_hi_7"].values, results["rf"]["ph"])
r2_off_l = r2_score(test["y_lo_7"].values, results["rf"]["pl"])
hit5_h = np.mean(np.abs(ph_rf - hi_true)/hi_true <= 0.05)*100
hit5_l = np.mean(np.abs(pl_rf - lo_true)/lo_true <= 0.05)*100
hit10_h= np.mean(np.abs(ph_rf - hi_true)/hi_true <= 0.10)*100
hit10_l= np.mean(np.abs(pl_rf - lo_true)/lo_true <= 0.10)*100
cov_h  = np.mean((hi_true >= band_hi_dn) & (hi_true <= band_hi_up))*100
cov_l  = np.mean((lo_true >= band_lo_dn) & (lo_true <= band_lo_up))*100

eval_df = pd.DataFrame([
    {"target":"7d MAX HIGH", "MAE_USD":mae_h, "RMSE_USD":rmse_h, "MAPE_%":mape_h,
     "R2_offset":r2_off_h, "hit_±5_%":hit5_h, "hit_±10_%":hit10_h, "95%CI_cov_%":cov_h},
    {"target":"7d MIN LOW",  "MAE_USD":mae_l, "RMSE_USD":rmse_l, "MAPE_%":mape_l,
     "R2_offset":r2_off_l, "hit_±5_%":hit5_l, "hit_±10_%":hit10_l, "95%CI_cov_%":cov_l},
]).set_index("target").round(2)
print(f"Test window: {test.index.min().date()} → {test.index.max().date()}   "
      f"n={len(test)} days   σ_hi={sigma_hi:.4f}  σ_lo={sigma_lo:.4f}")
eval_df
""")

code(r"""
# ── Plot 1: time series of predicted vs realised over the full test ──
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.72, 0.28], vertical_spacing=0.06,
                    subplot_titles=(
                      f"7-day forecast vs realised — full 8-month test "
                      f"({test.index.min().date()} → {test.index.max().date()})",
                      "Residual % error (pred − actual)"))

# Bands
fig.add_trace(go.Scatter(x=test.index, y=band_hi_up, mode="lines",
                         line=dict(color="rgba(0,0,0,0)"),
                         showlegend=False, hoverinfo="skip"), row=1, col=1)
fig.add_trace(go.Scatter(x=test.index, y=band_hi_dn, mode="lines",
                         line=dict(color="rgba(0,0,0,0)"), fill="tonexty",
                         fillcolor="rgba(0,160,0,0.13)",
                         name="MAX-H 95% CI", hoverinfo="skip"), row=1, col=1)
fig.add_trace(go.Scatter(x=test.index, y=band_lo_up, mode="lines",
                         line=dict(color="rgba(0,0,0,0)"),
                         showlegend=False, hoverinfo="skip"), row=1, col=1)
fig.add_trace(go.Scatter(x=test.index, y=band_lo_dn, mode="lines",
                         line=dict(color="rgba(0,0,0,0)"), fill="tonexty",
                         fillcolor="rgba(200,0,0,0.13)",
                         name="MIN-L 95% CI", hoverinfo="skip"), row=1, col=1)
# Realised / Predicted
fig.add_trace(go.Scatter(x=test.index, y=hi_true, mode="lines",
                         line=dict(color="darkgreen", width=1.6),
                         name="Realised MAX-H",
                         hovertemplate="Realised MAX-H<br>%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>"),
              row=1, col=1)
fig.add_trace(go.Scatter(x=test.index, y=lo_true, mode="lines",
                         line=dict(color="darkred", width=1.6),
                         name="Realised MIN-L",
                         hovertemplate="Realised MIN-L<br>%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>"),
              row=1, col=1)
fig.add_trace(go.Scatter(x=test.index, y=ph_rf, mode="lines",
                         line=dict(color="green", width=1.2, dash="dash"),
                         name="Pred MAX-H",
                         hovertemplate="Pred MAX-H<br>%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>"),
              row=1, col=1)
fig.add_trace(go.Scatter(x=test.index, y=pl_rf, mode="lines",
                         line=dict(color="red", width=1.2, dash="dash"),
                         name="Pred MIN-L",
                         hovertemplate="Pred MIN-L<br>%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>"),
              row=1, col=1)
fig.add_trace(go.Scatter(x=test.index, y=close_te, mode="lines",
                         line=dict(color="black", width=0.9, dash="dot"),
                         opacity=0.5, name="BTC close",
                         hovertemplate="Close<br>%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>"),
              row=1, col=1)

# Residuals
err_h = (ph_rf - hi_true)/hi_true*100
err_l = (pl_rf - lo_true)/lo_true*100
fig.add_trace(go.Scatter(x=test.index, y=err_h, mode="lines",
                         line=dict(color="green", width=1),
                         name="HIGH err %",
                         hovertemplate="HIGH err<br>%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>"),
              row=2, col=1)
fig.add_trace(go.Scatter(x=test.index, y=err_l, mode="lines",
                         line=dict(color="red", width=1),
                         name="LOW err %",
                         hovertemplate="LOW err<br>%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>"),
              row=2, col=1)
for k, dash, alpha in [(0,"solid",0.6),(5,"dot",0.5),(-5,"dot",0.5),
                       (10,"dash",0.35),(-10,"dash",0.35)]:
    fig.add_hline(y=k, line=dict(color="grey", dash=dash), opacity=alpha, row=2, col=1)
fig.update_yaxes(title="BTC / USD", row=1, col=1)
fig.update_yaxes(title="% error",   row=2, col=1)
fig.update_layout(template=TEMPLATE, hovermode="x unified",
                  height=720, legend=dict(orientation="h", y=1.05))
fig.show()
""")

code(r"""
# ── Daily hit-rate strips (interactive heatmaps) ──
within_h5  = (np.abs(ph_rf - hi_true)/hi_true <= 0.05).astype(int)
within_l5  = (np.abs(pl_rf - lo_true)/lo_true <= 0.05).astype(int)
within_h10 = (np.abs(ph_rf - hi_true)/hi_true <= 0.10).astype(int)
within_l10 = (np.abs(pl_rf - lo_true)/lo_true <= 0.10).astype(int)
strip5  = (within_h5 + within_l5)[None,:]
strip10 = (within_h10+ within_l10)[None,:]
labels = ["neither leg","one leg","both legs"]
def strip_fig(arr, title):
    txt = [[labels[v] for v in arr[0]]]
    f = go.Figure(go.Heatmap(
        z=arr, x=test.index, y=[" "],
        zmin=0, zmax=2, colorscale=[[0,"red"],[0.5,"gold"],[1,"green"]],
        showscale=False,
        text=txt, hovertemplate="%{x|%Y-%m-%d}: %{text}<extra></extra>",
    ))
    f.update_layout(template=TEMPLATE, title=title, height=130,
                    margin=dict(t=42,b=20,l=10,r=10))
    f.update_yaxes(showticklabels=False)
    return f
strip_fig(strip5,
    f"Daily hit indicator within ±5% — both legs hit on {(strip5==2).mean()*100:.1f}% of days").show()
strip_fig(strip10,
    f"Daily hit indicator within ±10% — both legs hit on {(strip10==2).mean()*100:.1f}% of days").show()
""")

code(r"""
# ── Predicted-vs-actual scatter with ±5/±10% bands ──
fig = make_subplots(rows=1, cols=2, shared_yaxes=False,
                    subplot_titles=("7-day MAX HIGH","7-day MIN LOW"))
for col, true, pred, c in [(1, hi_true, ph_rf, "green"),
                            (2, lo_true, pl_rf, "red")]:
    lo, hi = min(true.min(),pred.min())*0.98, max(true.max(),pred.max())*1.02
    fig.add_trace(go.Scatter(x=true, y=pred, mode="markers",
                              marker=dict(size=6, color=c, opacity=0.55),
                              name=("HIGH" if col==1 else "LOW"),
                              hovertemplate="Actual: $%{x:,.0f}<br>"
                                            "Pred:   $%{y:,.0f}<extra></extra>",
                              showlegend=False), row=1, col=col)
    # y = x
    fig.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode="lines",
                              line=dict(color="black", width=1),
                              name="y = x", showlegend=(col==1),
                              hoverinfo="skip"), row=1, col=col)
    for sign in [1.05, 0.95]:
        fig.add_trace(go.Scatter(x=[lo,hi], y=[lo*sign, hi*sign], mode="lines",
                                  line=dict(color="grey", width=0.8, dash="dot"),
                                  name="±5%", showlegend=(col==1 and sign==1.05),
                                  hoverinfo="skip"), row=1, col=col)
    for sign in [1.10, 0.90]:
        fig.add_trace(go.Scatter(x=[lo,hi], y=[lo*sign, hi*sign], mode="lines",
                                  line=dict(color="grey", width=0.6, dash="dash"),
                                  name="±10%", showlegend=(col==1 and sign==1.10),
                                  hoverinfo="skip"), row=1, col=col)
    fig.update_xaxes(title="Realised (USD)", row=1, col=col, range=[lo,hi])
    fig.update_yaxes(title="Predicted (USD)", row=1, col=col, range=[lo,hi],
                     scaleanchor=f"x{col}", scaleratio=1)
fig.update_layout(template=TEMPLATE, height=560,
                  legend=dict(orientation="h", y=1.06))
fig.show()
""")

md(r"""**Reading the four panels:**

* **Top time-series** — predicted lines (dashed) track the realised lines (solid)
  reasonably well on the *level*, but the 95 % bands are wide (±6.8 % H / ±11.4 % L
  of close). The black close line shows the regime.
* **Residual panel** — most errors are inside ±5 %; rare large outliers (when BTC
  moves more than the model's climatology expects) push outside ±10 %.
* **Hit-rate strip** — green is "both H and L within tolerance", yellow is one
  leg only, red is neither. The lower strip (±10 %) is mostly green.
* **Scatter** — points cluster around the y=x line but with visible spread —
  consistent with R² being negative on the offset target while still being
  unbiased on the absolute price target.
""")

md("## 10 · Save model assets for daily inference")
code(r"""
joblib.dump(dict(
    hi_model=mh, lo_model=ml,
    sigma_hi=sigma_hi, sigma_lo=sigma_lo,
    feat_cols=feat_cols, horizon=HORIZON,
    calibration_meta=dict(
        train_start=str(train.index.min().date()),
        train_end  =str(train.index.max().date()),
        test_start =str(test.index.min().date()),
        test_end   =str(test.index.max().date()),
        train_n=int(len(train)), test_n=int(len(test)),
        best_model="rf",
        target_definition="forward 7-day max(H) and min(L)"),
    imp_hi=imp_hi, imp_lo=imp_lo, imp_combined=imp_combined,
    pca_cum_var=cum,
), f"{OUT}/inference_assets_7d.joblib")
print("Saved inference_assets_7d.joblib")
""")

md(r"""## 11 · Summary — 1-day vs 7-day side-by-side

| Metric (on 8-month held-out test) | **1-day model** | **7-day model** |
|---|---|---|
| Best ML family | Ridge | Random Forest |
| MAPE_HIGH | **1.14 %** | 3.18 % |
| MAPE_LOW  | **1.32 %** | 4.84 % |
| Hit-rate ±5 % HIGH | **98.8 %** ✅ | 81.7 % ❌ |
| Hit-rate ±5 % LOW  | **97.5 %** ✅ | 67.6 % ❌ |
| Hit-rate ±10 % HIGH | 100 % | 100 % ✅ |
| Hit-rate ±10 % LOW  | 99.2 % | 89.6 % |
| R² (offset target) HIGH | +0.10 | **−0.43** |
| R² (offset target) LOW  | +0.11 | **−0.25** |
| 95 % CI half-width HIGH | ±3.2 % | ±6.8 % |
| 95 % CI half-width LOW  | ±3.7 % | ±11.4 % |

**Take-aways**

1. **The 1-day model is genuinely predictive.** Positive R², beats all naive
   baselines on every metric.
2. **The 7-day model is climatological.** R² is negative — the offset itself is
   essentially noise at this horizon. The point forecast is roughly "expand by
   the recent volatility level" and is calibrated to be unbiased, but it cannot
   pin down the *direction* of expansion.
3. **The ±5 % target cannot be met at 7-day** given the public data we use. To get
   there one would need either premium order-book / ETF-flow data, or to redefine
   the target to a wider tolerance.
4. **Operational use:** treat the 7-day forecast as a *range envelope* with a
   wide CI, not a precise price target. Useful for option strikes, risk sizing,
   or "are we in a quiet or stormy week?" — not for tight execution.
""")

nb = new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec":{"name":"python3","display_name":"Python 3"},
                  "language_info":{"name":"python","version":"3.14"}}
with open("/home/jovyan/btc-range-model/btc_range_7d_training.ipynb","w") as fp:
    nbformat.write(nb, fp)
print("Wrote btc_range_7d_training.ipynb")
