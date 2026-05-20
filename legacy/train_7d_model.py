"""Train a 7-day-out BTC range model.

Targets (rolling forward window of next 7 calendar days):
  y_hi_7 = max(H_{t+1}, ..., H_{t+7})   →  offset = (y_hi_7 - C_t)/C_t  (≥0)
  y_lo_7 = min(L_{t+1}, ..., L_{t+7})   →  offset = (C_t - y_lo_7)/C_t  (≥0)

Same feature set as the 1-day model (103 features).
Same train/test split (last 8 months held out).
Same metrics: MAPE, hit ±5%, R², plus baseline comparisons.
"""
import joblib, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from sklearn.linear_model    import RidgeCV
from sklearn.ensemble        import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection      import permutation_importance
from sklearn.decomposition   import PCA
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

OUT = "/home/jovyan/btc-range-model"
SEED = 42
HORIZON = 7

# ── 1. Load existing features + raw close to recompute targets ────────────
data = pd.read_csv(f"{OUT}/features.csv", index_col=0, parse_dates=True)
raw  = pd.read_csv(f"{OUT}/raw.csv",      index_col=0, parse_dates=True)
h = raw["btc_high"]; l_ = raw["btc_low"]; c = raw["btc_close"]

# 7-day forward max/min  (rows t+1 .. t+7)
y_hi_7 = pd.concat([h.shift(-k) for k in range(1, HORIZON+1)], axis=1).max(axis=1)
y_lo_7 = pd.concat([l_.shift(-k) for k in range(1, HORIZON+1)], axis=1).min(axis=1)

# Replace target columns
data = data.drop(columns=["y_hi","y_lo","next_high","next_low"])
data["y_hi_7"]    = (y_hi_7 - c) / c        # offset, ≥0
data["y_lo_7"]    = (c - y_lo_7) / c        # offset, ≥0
data["hi_7_USD"]  = y_hi_7                  # absolute target
data["lo_7_USD"]  = y_lo_7
data["close"]     = c
data = data.replace([np.inf,-np.inf], np.nan).dropna()
print("Data after re-targeting:", data.shape,
      data.index.min().date(), "→", data.index.max().date())

# ── 2. Train/test split (last 8 months held out) ─────────────────────────
TODAY = pd.Timestamp(datetime.utcnow().date())
test_start = TODAY - pd.DateOffset(months=8)
train = data.loc[:test_start - pd.Timedelta(days=1)]
test  = data.loc[test_start:]
print(f"TRAIN {train.index.min().date()} → {train.index.max().date()}  n={len(train)}")
print(f"TEST  {test.index.min().date()}  → {test.index.max().date()}   n={len(test)}")

# Features = same 103
feat_cols = [c for c in data.columns
             if c not in ("y_hi_7","y_lo_7","hi_7_USD","lo_7_USD","close")]
X_tr, X_te = train[feat_cols], test[feat_cols]
yhi_tr, yhi_te = train["y_hi_7"], test["y_hi_7"]
ylo_tr, ylo_te = train["y_lo_7"], test["y_lo_7"]

# ── 3. Baselines ─────────────────────────────────────────────────────────
close_te = test["close"].values
hi_true = test["hi_7_USD"].values
lo_true = test["lo_7_USD"].values

def metrics(name, ph, pl):
    mape_h = np.mean(np.abs(ph - hi_true) / hi_true) * 100
    mape_l = np.mean(np.abs(pl - lo_true) / lo_true) * 100
    hit5_h = np.mean(np.abs(ph - hi_true) / hi_true <= 0.05) * 100
    hit5_l = np.mean(np.abs(pl - lo_true) / lo_true <= 0.05) * 100
    hit10_h = np.mean(np.abs(ph - hi_true) / hi_true <= 0.10) * 100
    hit10_l = np.mean(np.abs(pl - lo_true) / lo_true <= 0.10) * 100
    rng_p = ph - pl; rng_t = hi_true - lo_true
    mape_r = np.mean(np.abs(rng_p - rng_t) / rng_t) * 100
    print(f"  {name:38s}  MAPE_H={mape_h:5.2f}%  MAPE_L={mape_l:5.2f}%  "
          f"hit5(H/L)={hit5_h:5.1f}/{hit5_l:5.1f}  "
          f"hit10(H/L)={hit10_h:5.1f}/{hit10_l:5.1f}  MAPE_range={mape_r:5.1f}%")
    return dict(mape_h=mape_h, mape_l=mape_l, hit5_h=hit5_h, hit5_l=hit5_l,
                hit10_h=hit10_h, hit10_l=hit10_l, mape_r=mape_r)

print("\n>>> BASELINES on test ({} days):".format(len(test)))
metrics("A. pred = close (no range)", close_te, close_te)
mu_hi_train = train["y_hi_7"].mean(); mu_lo_train = train["y_lo_7"].mean()
metrics(f"B. const train-mean offset (μ_h={mu_hi_train:.3f}, μ_l={mu_lo_train:.3f})",
        close_te*(1+mu_hi_train), close_te*(1-mu_lo_train))
# Random-walk: predict last-7-days realised H/L (uses today's window)
hi_7d_back = raw["btc_high"].rolling(7).max().reindex(test.index).values
lo_7d_back = raw["btc_low"].rolling(7).min().reindex(test.index).values
metrics("C. random-walk (last-7d realised H/L)", hi_7d_back, lo_7d_back)

# ── 4. Train candidate models ────────────────────────────────────────────
def mk_pipe(model): return Pipeline([("sc", StandardScaler()), ("m", model)])
MODELS = {
    "ridge": lambda: mk_pipe(RidgeCV(alphas=np.logspace(-3,3,13))),
    "gbm"  : lambda: mk_pipe(GradientBoostingRegressor(
                n_estimators=800, max_depth=3, learning_rate=0.02,
                subsample=0.8, random_state=SEED)),
    "rf"   : lambda: mk_pipe(RandomForestRegressor(
                n_estimators=500, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
}

print("\n>>> TimeSeriesSplit-5 CV (train only, MAE on y_hi_7):")
tscv = TimeSeriesSplit(n_splits=5)
for name, mk in MODELS.items():
    s = []
    for tri, vai in tscv.split(X_tr):
        m = mk(); m.fit(X_tr.iloc[tri], yhi_tr.iloc[tri])
        s.append(mean_absolute_error(yhi_tr.iloc[vai], m.predict(X_tr.iloc[vai])))
    print(f"  {name:6s}  MAE = {np.mean(s):.4f} ± {np.std(s):.4f}")

print("\n>>> MODEL PERFORMANCE on held-out 8-month test:")
results = {}
for name, mk in MODELS.items():
    mh = mk(); mh.fit(X_tr, yhi_tr); ph = mh.predict(X_te)
    ml = mk(); ml.fit(X_tr, ylo_tr); pl = ml.predict(X_te)
    pred_hi = close_te * (1 + np.clip(ph, 0, None))
    pred_lo = close_te * (1 - np.clip(pl, 0, None))
    r = metrics(name, pred_hi, pred_lo)
    r["r2_hi_offset"] = r2_score(yhi_te, ph)
    r["r2_lo_offset"] = r2_score(ylo_te, pl)
    results[name] = dict(m_hi=mh, m_lo=ml, ph=ph, pl=pl, metrics=r,
                         pred_hi=pred_hi, pred_lo=pred_lo)

best = min(results.keys(), key=lambda k: results[k]["metrics"]["mape_h"]
                                       + results[k]["metrics"]["mape_l"])
print(f"\n>>> BEST: {best}")
print(f"  R² offset (high): {results[best]['metrics']['r2_hi_offset']:.3f}")
print(f"  R² offset (low) : {results[best]['metrics']['r2_lo_offset']:.3f}")

# ── 5. Feature importance (permutation) ──────────────────────────────────
print("\n>>> Permutation feature importance on test (top 20 combined):")
mh = results[best]["m_hi"]; ml = results[best]["m_lo"]
perm_hi = permutation_importance(mh, X_te, yhi_te, n_repeats=8,
                                  random_state=SEED, n_jobs=-1)
perm_lo = permutation_importance(ml, X_te, ylo_te, n_repeats=8,
                                  random_state=SEED, n_jobs=-1)
imp_hi = pd.Series(perm_hi.importances_mean, index=feat_cols).sort_values(ascending=False)
imp_lo = pd.Series(perm_lo.importances_mean, index=feat_cols).sort_values(ascending=False)
imp_combined = (imp_hi.rank(ascending=False) + imp_lo.rank(ascending=False)).sort_values()
print(imp_combined.head(20).to_string())

# ── 6. PCA variance explained by top-20 features ─────────────────────────
top20 = imp_combined.head(20).index.tolist()
sc = StandardScaler().fit(X_tr[top20])
pca = PCA().fit(sc.transform(X_tr[top20]))
cum = np.cumsum(pca.explained_variance_ratio_)
print("\n>>> PCA cumulative variance (top-20 features):")
for i, v in enumerate(cum, 1):
    print(f"  PC{i:2d}: {v*100:5.2f}%")

# ── 7. Residual std for inference band ───────────────────────────────────
res_hi = test["y_hi_7"].values - results[best]["ph"]
res_lo = test["y_lo_7"].values - results[best]["pl"]
sigma_hi = float(np.std(res_hi)); sigma_lo = float(np.std(res_lo))
print(f"\nResidual std (offset space): σ_hi={sigma_hi:.4f}  σ_lo={sigma_lo:.4f}")
print(f"95% CI ≈ ±{1.96*sigma_hi*100:.2f}% (H)  ±{1.96*sigma_lo*100:.2f}% (L) of close")

# ── 8. Save artifacts ────────────────────────────────────────────────────
joblib.dump(dict(
    hi_model=mh, lo_model=ml,
    sigma_hi=sigma_hi, sigma_lo=sigma_lo,
    feat_cols=feat_cols,
    horizon=HORIZON,
    calibration_meta=dict(
        train_start=str(train.index.min().date()),
        train_end  =str(train.index.max().date()),
        test_start =str(test.index.min().date()),
        test_end   =str(test.index.max().date()),
        train_n=int(len(train)), test_n=int(len(test)),
        best_model=best,
        target_definition="forward 7-day max(H) and min(L)",
    ),
    imp_hi=imp_hi, imp_lo=imp_lo, imp_combined=imp_combined,
    pca_cum_var=cum,
), f"{OUT}/inference_assets_7d.joblib")
print(f"\n>>> Saved inference_assets_7d.joblib  (best={best})")
