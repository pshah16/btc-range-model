"""Train the production model and save everything needed for daily inference.

Saves:
  - hi_model, lo_model            (sklearn Pipeline = scaler + RidgeCV)
  - sigma_hi, sigma_lo            (residual std on held-out test, in %-offset space)
  - feat_cols                     (ordered feature names — STRICT ordering for inference)
  - calibration_meta              (train range, test range, residual sample size)
"""
import joblib, numpy as np, pandas as pd
from datetime import datetime
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

OUT = "/home/jovyan/btc-range-model"
data = pd.read_csv(f"{OUT}/features.csv", index_col=0, parse_dates=True)

TODAY = pd.Timestamp(datetime.utcnow().date())
test_start = TODAY - pd.DateOffset(months=8)
train = data.loc[: test_start - pd.Timedelta(days=1)]
test  = data.loc[test_start:]
feat_cols = [c for c in data.columns
             if c not in ("y_hi","y_lo","close","next_high","next_low")]

def mk():
    return Pipeline([("sc", StandardScaler()),
                     ("m",  RidgeCV(alphas=np.logspace(-3,3,13)))])

hi_model = mk().fit(train[feat_cols], train["y_hi"])
lo_model = mk().fit(train[feat_cols], train["y_lo"])

# Residual std on TEST window (last 8 months — recent calibration)
res_hi = test["y_hi"].values - hi_model.predict(test[feat_cols])
res_lo = test["y_lo"].values - lo_model.predict(test[feat_cols])
sigma_hi = float(np.std(res_hi))
sigma_lo = float(np.std(res_lo))

assets = dict(
    hi_model=hi_model, lo_model=lo_model,
    sigma_hi=sigma_hi, sigma_lo=sigma_lo,
    feat_cols=feat_cols,
    calibration_meta=dict(
        train_start=str(train.index.min().date()),
        train_end=str(train.index.max().date()),
        test_start=str(test.index.min().date()),
        test_end=str(test.index.max().date()),
        train_n=int(len(train)), test_n=int(len(test)),
        residual_n=int(len(res_hi)),
        residual_mean_hi=float(np.mean(res_hi)),
        residual_mean_lo=float(np.mean(res_lo)),
    ),
)
joblib.dump(assets, f"{OUT}/inference_assets.joblib")
print("Saved inference_assets.joblib")
print(f"  sigma_hi = {sigma_hi:.4f}  (±1.96σ ≈ ±{1.96*sigma_hi*100:.2f}% of close)")
print(f"  sigma_lo = {sigma_lo:.4f}  (±1.96σ ≈ ±{1.96*sigma_lo*100:.2f}% of close)")
print(f"  features = {len(feat_cols)}")
print(f"  train: {assets['calibration_meta']['train_start']} → {assets['calibration_meta']['train_end']}")
print(f"  test : {assets['calibration_meta']['test_start']} → {assets['calibration_meta']['test_end']}")
