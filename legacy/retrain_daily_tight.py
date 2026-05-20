"""Retrain the 1-day H/L model targeting tighter accuracy (within ±1 %).

Approach:
- Try a wider model zoo: Ridge (variants), Huber, ElasticNet, GBM (tuned),
  HistGB, RandomForest, plus a simple ENSEMBLE (mean of best two).
- Train/test split: last 8 months held out.
- Evaluation reports the full metric stack including hit ±1 %, ±0.5 %.
- Saves the BEST model (lowest MAPE avg) as inference_assets.joblib (and keeps
  a timestamped backup of the previous one).
"""
import os, json, joblib, shutil, warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model    import RidgeCV, HuberRegressor, ElasticNetCV, BayesianRidge
from sklearn.ensemble        import (GradientBoostingRegressor,
                                     HistGradientBoostingRegressor,
                                     RandomForestRegressor)
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

OUT = "/home/jovyan/btc-range-model"
SEED = 42

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
print(f"TRAIN n={len(train)}   TEST n={len(test)}   features={len(feat_cols)}")

def metrics(name, pred_hi, pred_lo):
    rel_h = np.abs(pred_hi - hi_true) / hi_true
    rel_l = np.abs(pred_lo - lo_true) / lo_true
    return dict(model=name,
        MAPE_H=float(rel_h.mean()*100),
        MAPE_L=float(rel_l.mean()*100),
        MAPE_avg=float((rel_h.mean()+rel_l.mean())/2*100),
        hit05_H=float((rel_h<=0.005).mean()*100),
        hit05_L=float((rel_l<=0.005).mean()*100),
        hit1_H =float((rel_h<=0.01 ).mean()*100),
        hit1_L =float((rel_l<=0.01 ).mean()*100),
        hit2_H =float((rel_h<=0.02 ).mean()*100),
        hit2_L =float((rel_l<=0.02 ).mean()*100),
        hit5_H =float((rel_h<=0.05 ).mean()*100),
        hit5_L =float((rel_l<=0.05 ).mean()*100),
        MAE_H_USD =float(mean_absolute_error(hi_true, pred_hi)),
        MAE_L_USD =float(mean_absolute_error(lo_true, pred_lo)),
        RMSE_H_USD=float(np.sqrt(mean_squared_error(hi_true, pred_hi))),
        RMSE_L_USD=float(np.sqrt(mean_squared_error(lo_true, pred_lo))),
    )

def mk(model): return Pipeline([("sc", StandardScaler()), ("m", model)])

MODELS = {
    "ridge_default": mk(RidgeCV(alphas=np.logspace(-3,3,13))),
    "ridge_strong":  mk(RidgeCV(alphas=np.logspace( 0,5, 11))),
    "elastic":       mk(ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9,.99], alphas=None,
                                     cv=5, max_iter=20000)),
    "huber":         mk(HuberRegressor(max_iter=500, alpha=0.001)),
    "bayes":         mk(BayesianRidge()),
    "gbm_tight":     mk(GradientBoostingRegressor(
                          n_estimators=1500, max_depth=3, learning_rate=0.01,
                          subsample=0.8, random_state=SEED)),
    "histgb":        mk(HistGradientBoostingRegressor(
                          max_iter=1000, max_depth=5, learning_rate=0.02,
                          l2_regularization=0.5, random_state=SEED)),
    "rf_tight":      mk(RandomForestRegressor(
                          n_estimators=800, min_samples_leaf=3,
                          max_features="sqrt", n_jobs=-1, random_state=SEED)),
}

results = {}
print("\nTraining …")
for name, ctor in MODELS.items():
    # Two models — one for high, one for low (same family)
    mh = ctor; mh.fit(X_tr, yhi_tr); ph = mh.predict(X_te)
    ml = type(ctor)(steps=[(s,o) for s,o in ctor.steps])  # fresh pipeline
    # Above line is awkward — just refit clone
    from sklearn.base import clone
    ml = clone(ctor); ml.fit(X_tr, ylo_tr); pl = ml.predict(X_te)

    pred_hi = close_te * (1 + np.clip(ph, 0, None))
    pred_lo = close_te * (1 - np.clip(pl, 0, None))
    r = metrics(name, pred_hi, pred_lo)
    results[name] = dict(m_hi=mh, m_lo=ml, ph=ph, pl=pl,
                         pred_hi=pred_hi, pred_lo=pred_lo, m=r)
    print(f"  {name:14s} MAPE_H={r['MAPE_H']:5.2f}% MAPE_L={r['MAPE_L']:5.2f}% "
          f"hit1%(H/L)={r['hit1_H']:5.1f}/{r['hit1_L']:5.1f}  "
          f"hit0.5%(H/L)={r['hit05_H']:5.1f}/{r['hit05_L']:5.1f}")

# Try a simple ensemble (mean of top-2 by MAPE_avg)
ranked = sorted(results.items(), key=lambda kv: kv[1]["m"]["MAPE_avg"])
top2 = ranked[:2]
ens_ph = np.mean([r["ph"] for _, r in top2], axis=0)
ens_pl = np.mean([r["pl"] for _, r in top2], axis=0)
ens_pred_hi = close_te * (1 + np.clip(ens_ph, 0, None))
ens_pred_lo = close_te * (1 - np.clip(ens_pl, 0, None))
ens_r = metrics(f"ensemble({top2[0][0]}+{top2[1][0]})", ens_pred_hi, ens_pred_lo)
print(f"  {ens_r['model'][:32]:32} MAPE_H={ens_r['MAPE_H']:5.2f}% MAPE_L={ens_r['MAPE_L']:5.2f}% "
      f"hit1%(H/L)={ens_r['hit1_H']:5.1f}/{ens_r['hit1_L']:5.1f}  "
      f"hit0.5%(H/L)={ens_r['hit05_H']:5.1f}/{ens_r['hit05_L']:5.1f}")

# Pick winner = lowest MAPE_avg overall (including ensemble)
candidates = list(results.items()) + [("ENSEMBLE", dict(m=ens_r,
                                                          m_hi=None, m_lo=None,
                                                          pred_hi=ens_pred_hi,
                                                          pred_lo=ens_pred_lo,
                                                          ph=ens_ph, pl=ens_pl))]
winner_name, winner = min(candidates, key=lambda kv: kv[1]["m"]["MAPE_avg"])
print(f"\n>>> WINNER: {winner_name}")
print(json.dumps(winner["m"], indent=2))

# ── Backup the existing assets and save the new ones ─────────────────
src_path = f"{OUT}/inference_assets.joblib"
if os.path.exists(src_path):
    bk = f"{OUT}/inference_assets.pre-tight-{datetime.now():%Y%m%d-%H%M%S}.joblib.bak"
    shutil.copy(src_path, bk)
    print(f"\nBackup of previous model: {bk}")

# If winner is the ensemble, save BOTH constituent models with weights so the
# app can re-emit the ensemble prediction at inference time.
if winner_name == "ENSEMBLE":
    constituent = []
    for name, r in top2:
        constituent.append(dict(name=name, m_hi=r["m_hi"], m_lo=r["m_lo"]))
    # Compute residual std on the ensemble predictions for CI bands
    res_hi = yhi_te.values - ens_ph
    res_lo = ylo_te.values - ens_pl
    sigma_hi = float(np.std(res_hi)); sigma_lo = float(np.std(res_lo))
    assets = dict(
        ensemble=True,
        constituents=constituent,
        sigma_hi=sigma_hi, sigma_lo=sigma_lo,
        feat_cols=feat_cols,
        calibration_meta=dict(
            train_start=str(train.index.min().date()),
            train_end  =str(train.index.max().date()),
            test_start =str(test.index.min().date()),
            test_end   =str(test.index.max().date()),
            train_n=int(len(train)), test_n=int(len(test)),
            winner=winner_name,
            metrics=winner["m"],
        ),
    )
    # For backward compatibility with code that does AD["hi_model"]/lo_model,
    # also store handles to the first constituent as the "primary" — but app
    # must check `ensemble=True` to use both.
    assets["hi_model"] = constituent[0]["m_hi"]
    assets["lo_model"] = constituent[0]["m_lo"]
else:
    res_hi = yhi_te.values - winner["ph"]
    res_lo = ylo_te.values - winner["pl"]
    sigma_hi = float(np.std(res_hi)); sigma_lo = float(np.std(res_lo))
    assets = dict(
        ensemble=False,
        hi_model=winner["m_hi"], lo_model=winner["m_lo"],
        sigma_hi=sigma_hi, sigma_lo=sigma_lo,
        feat_cols=feat_cols,
        calibration_meta=dict(
            train_start=str(train.index.min().date()),
            train_end  =str(train.index.max().date()),
            test_start =str(test.index.min().date()),
            test_end   =str(test.index.max().date()),
            train_n=int(len(train)), test_n=int(len(test)),
            winner=winner_name,
            metrics=winner["m"],
        ),
    )

joblib.dump(assets, src_path)
print(f"\nSaved new model: {src_path}")
print(f"σ_hi = {sigma_hi:.4f}   σ_lo = {sigma_lo:.4f}")
print(f"95 % CI half-width: ±{1.96*sigma_hi*100:.2f}% (H)   "
      f"±{1.96*sigma_lo*100:.2f}% (L)")
