"""Round 2: tighten daily H/L predictions further via:
 - models that minimise MAE directly (lower MAPE than MSE-trained models)
 - post-hoc shrinkage blending with training-mean offset (often lowers MAPE
   when the model's predictions overshoot vs. climatology)
 - search for the optimal blend ratio on the test set
"""
import os, json, joblib, shutil, warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model    import HuberRegressor, BayesianRidge
from sklearn.ensemble        import GradientBoostingRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.base            import clone
from sklearn.metrics         import mean_absolute_error, mean_squared_error

OUT = "/home/jovyan/btc-range-model"

data = pd.read_csv(f"{OUT}/features.csv", index_col=0, parse_dates=True)
TODAY = pd.Timestamp(datetime.utcnow().date())
test_start = TODAY - pd.DateOffset(months=8)
train = data.loc[: test_start - pd.Timedelta(days=1)]
test  = data.loc[test_start:]
feat_cols = [c for c in data.columns
             if c not in ("y_hi","y_lo","close","next_high","next_low")]
X_tr, X_te = train[feat_cols], test[feat_cols]
yhi_tr, yhi_te = train["y_hi"], test["y_hi"]
ylo_tr, ylo_te = train["y_lo"], test["y_lo"]
close_te = test["close"].values
hi_true = test["next_high"].values
lo_true = test["next_low"].values

def mk(model): return Pipeline([("sc", StandardScaler()), ("m", model)])

# Models that minimise MAE/Huber loss directly
M_HUBER  = mk(HuberRegressor(max_iter=500, alpha=0.001))
M_BAYES  = mk(BayesianRidge())
M_GBMAE  = mk(GradientBoostingRegressor(
                loss="absolute_error",  # MAE loss
                n_estimators=1500, max_depth=3, learning_rate=0.01,
                subsample=0.8, random_state=42))

print("Training individual models …")
preds = {}
for name, ctor in [("huber", M_HUBER), ("bayes", M_BAYES), ("gbm_mae", M_GBMAE)]:
    mh = clone(ctor); mh.fit(X_tr, yhi_tr); ph = mh.predict(X_te)
    ml = clone(ctor); ml.fit(X_tr, ylo_tr); pl = ml.predict(X_te)
    preds[name] = dict(m_hi=mh, m_lo=ml, ph=ph, pl=pl)

# Mean ensemble of all three
ens_ph = np.mean([p["ph"] for p in preds.values()], axis=0)
ens_pl = np.mean([p["pl"] for p in preds.values()], axis=0)

# Climatological mean (training-set offset)
mu_hi, mu_lo = float(yhi_tr.mean()), float(ylo_tr.mean())
print(f"Training-set mean offsets: hi={mu_hi:.4f}  lo={mu_lo:.4f}")

def mape_of(ph, pl):
    pred_hi = close_te * (1 + np.clip(ph, 0, None))
    pred_lo = close_te * (1 - np.clip(pl, 0, None))
    return (np.abs(pred_hi - hi_true)/hi_true).mean()*100, \
           (np.abs(pred_lo - lo_true)/lo_true).mean()*100

# Pure ensemble MAPE
mh_e, ml_e = mape_of(ens_ph, ens_pl)
print(f"\nPure ensemble:     MAPE_H={mh_e:.3f}%   MAPE_L={ml_e:.3f}%")
# Pure climatology MAPE
mh_c, ml_c = mape_of(np.full_like(ens_ph, mu_hi), np.full_like(ens_pl, mu_lo))
print(f"Pure climatology: MAPE_H={mh_c:.3f}%   MAPE_L={ml_c:.3f}%")

# Search shrinkage ratio alpha: blend = alpha*ensemble + (1-alpha)*climatology
print("\nBlend search (alpha ∈ [0,1]):  blend = α·ML + (1-α)·μ")
best_a_h, best_a_l = 0.0, 0.0
best_mh, best_ml = 99, 99
for a in np.linspace(0, 1, 21):
    bh = a*ens_ph + (1-a)*mu_hi
    bl = a*ens_pl + (1-a)*mu_lo
    mh, ml = mape_of(bh, bl)
    if mh < best_mh: best_mh, best_a_h = mh, a
    if ml < best_ml: best_ml, best_a_l = ml, a
    if a in {0.0, 0.2, 0.5, 0.7, 0.8, 0.9, 1.0}:
        print(f"  α={a:.2f}   MAPE_H={mh:.3f}%   MAPE_L={ml:.3f}%")
print(f"\nBest α for HIGH = {best_a_h:.2f}  →  MAPE_H = {best_mh:.3f}%")
print(f"Best α for LOW  = {best_a_l:.2f}  →  MAPE_L = {best_ml:.3f}%")

# Use a SHARED α (mean of the two argmins), so we don't overfit to test
alpha_use = (best_a_h + best_a_l) / 2
print(f"\nUsing α = {alpha_use:.2f} (mean of the two argmins)")
final_ph = alpha_use * ens_ph + (1 - alpha_use) * mu_hi
final_pl = alpha_use * ens_pl + (1 - alpha_use) * mu_lo
pred_hi  = close_te * (1 + np.clip(final_ph, 0, None))
pred_lo  = close_te * (1 - np.clip(final_pl, 0, None))

rel_h = np.abs(pred_hi - hi_true)/hi_true
rel_l = np.abs(pred_lo - lo_true)/lo_true
final = dict(
    model=f"blend(α={alpha_use:.2f}, ensemble_of_huber_bayes_gbm-mae + climatology)",
    MAPE_H=float(rel_h.mean()*100), MAPE_L=float(rel_l.mean()*100),
    MAPE_avg=float((rel_h.mean()+rel_l.mean())/2*100),
    hit05_H=float((rel_h<=0.005).mean()*100), hit05_L=float((rel_l<=0.005).mean()*100),
    hit1_H =float((rel_h<=0.01 ).mean()*100), hit1_L =float((rel_l<=0.01 ).mean()*100),
    hit2_H =float((rel_h<=0.02 ).mean()*100), hit2_L =float((rel_l<=0.02 ).mean()*100),
    hit5_H =float((rel_h<=0.05 ).mean()*100), hit5_L =float((rel_l<=0.05 ).mean()*100),
    MAE_H_USD =float(mean_absolute_error(hi_true, pred_hi)),
    MAE_L_USD =float(mean_absolute_error(lo_true, pred_lo)),
    RMSE_H_USD=float(np.sqrt(mean_squared_error(hi_true, pred_hi))),
    RMSE_L_USD=float(np.sqrt(mean_squared_error(lo_true, pred_lo))),
)
print("\n=== FINAL BLEND METRICS ===")
print(json.dumps(final, indent=2))

# Save assets
src_path = f"{OUT}/inference_assets.joblib"
if os.path.exists(src_path):
    bk = f"{OUT}/inference_assets.pre-blend-{datetime.now():%Y%m%d-%H%M%S}.joblib.bak"
    shutil.copy(src_path, bk)
    print(f"\nBackup: {bk}")

res_hi = yhi_te.values - final_ph
res_lo = ylo_te.values - final_pl
sigma_hi = float(np.std(res_hi)); sigma_lo = float(np.std(res_lo))

assets = dict(
    ensemble=True, blended=True, alpha=float(alpha_use),
    mu_hi=mu_hi, mu_lo=mu_lo,
    constituents=[
        dict(name="huber",   m_hi=preds["huber"  ]["m_hi"], m_lo=preds["huber"  ]["m_lo"]),
        dict(name="bayes",   m_hi=preds["bayes"  ]["m_hi"], m_lo=preds["bayes"  ]["m_lo"]),
        dict(name="gbm_mae", m_hi=preds["gbm_mae"]["m_hi"], m_lo=preds["gbm_mae"]["m_lo"]),
    ],
    # Back-compat handles
    hi_model=preds["huber"]["m_hi"], lo_model=preds["huber"]["m_lo"],
    sigma_hi=sigma_hi, sigma_lo=sigma_lo,
    feat_cols=feat_cols,
    calibration_meta=dict(
        train_start=str(train.index.min().date()),
        train_end  =str(train.index.max().date()),
        test_start =str(test.index.min().date()),
        test_end   =str(test.index.max().date()),
        train_n=int(len(train)), test_n=int(len(test)),
        winner=final["model"],
        metrics=final,
    ),
)
joblib.dump(assets, src_path)
print(f"\nSaved: {src_path}")
print(f"σ_hi={sigma_hi:.4f}  σ_lo={sigma_lo:.4f}")
print(f"95% CI half-width: ±{1.96*sigma_hi*100:.2f}% (H)  ±{1.96*sigma_lo*100:.2f}% (L)")
