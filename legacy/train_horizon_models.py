"""Train per-horizon (k=1..7) daily H and L models for forward 7-day predictions.

At time t, predict H_{t+k} and L_{t+k} for each k ∈ {1..7}.
Features at time t are the same 103 features used by the 1-day model.

Saves inference_assets_horizon.joblib with:
  hi_models[k-1], lo_models[k-1]     for k = 1..7
  sigma_hi[k-1],  sigma_lo[k-1]      residual std at each horizon
  feat_cols, calibration_meta
"""
import joblib, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from sklearn.linear_model    import RidgeCV
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import mean_absolute_error

OUT = "/home/jovyan/btc-range-model"
HORIZONS = list(range(1, 8))   # 1..7

data = pd.read_csv(f"{OUT}/features.csv", index_col=0, parse_dates=True)
raw  = pd.read_csv(f"{OUT}/raw.csv",      index_col=0, parse_dates=True)
h = raw["btc_high"]; l_ = raw["btc_low"]; c = raw["btc_close"]

# Drop the 1-day targets — we'll build per-horizon targets below
data = data.drop(columns=["y_hi","y_lo","next_high","next_low"], errors="ignore")
feat_cols = [col for col in data.columns if col != "close"]
if "close" not in data.columns: data["close"] = c
print(f"Feature matrix: {data.shape}   features={len(feat_cols)}")

# Build per-horizon targets in offset space
for k in HORIZONS:
    data[f"y_hi_k{k}"] = (h.shift(-k) - c) / c     # H at t+k vs C at t
    data[f"y_lo_k{k}"] = (c - l_.shift(-k)) / c    # L at t+k vs C at t
    data[f"hi_k{k}_USD"] = h.shift(-k)
    data[f"lo_k{k}_USD"] = l_.shift(-k)

data = data.replace([np.inf,-np.inf], np.nan).dropna()
print(f"After per-horizon target build: {data.shape}")

# Train/test split — same as before
TODAY = pd.Timestamp(datetime.utcnow().date())
test_start = TODAY - pd.DateOffset(months=8)
train = data.loc[: test_start - pd.Timedelta(days=1)]
test  = data.loc[test_start:]
print(f"TRAIN {train.index.min().date()} → {train.index.max().date()}  n={len(train)}")
print(f"TEST  {test.index.min().date()}  → {test.index.max().date()}   n={len(test)}\n")

def mk():
    return Pipeline([("sc", StandardScaler()),
                     ("m",  RidgeCV(alphas=np.logspace(-3, 3, 13)))])

hi_models, lo_models = [], []
sigma_hi, sigma_lo = [], []
test_metrics = []

for k in HORIZONS:
    y_hi_train = train[f"y_hi_k{k}"]
    y_lo_train = train[f"y_lo_k{k}"]
    y_hi_test  = test[f"y_hi_k{k}"]
    y_lo_test  = test[f"y_lo_k{k}"]
    hi_true_usd = test[f"hi_k{k}_USD"].values
    lo_true_usd = test[f"lo_k{k}_USD"].values
    close_te = test["close"].values

    mh = mk().fit(train[feat_cols], y_hi_train)
    ml = mk().fit(train[feat_cols], y_lo_train)

    ph = mh.predict(test[feat_cols])
    pl = ml.predict(test[feat_cols])
    pred_hi = close_te * (1 + np.clip(ph, 0, None))
    pred_lo = close_te * (1 - np.clip(pl, 0, None))

    res_hi = y_hi_test.values - ph
    res_lo = y_lo_test.values - pl
    s_hi = float(np.std(res_hi)); s_lo = float(np.std(res_lo))

    mape_h = np.mean(np.abs(pred_hi - hi_true_usd)/hi_true_usd)*100
    mape_l = np.mean(np.abs(pred_lo - lo_true_usd)/lo_true_usd)*100
    hit5_h = np.mean(np.abs(pred_hi - hi_true_usd)/hi_true_usd <= 0.05)*100
    hit5_l = np.mean(np.abs(pred_lo - lo_true_usd)/lo_true_usd <= 0.05)*100

    hi_models.append(mh); lo_models.append(ml)
    sigma_hi.append(s_hi); sigma_lo.append(s_lo)
    test_metrics.append(dict(k=k, MAPE_H=mape_h, MAPE_L=mape_l,
                              hit5_H=hit5_h, hit5_L=hit5_l,
                              sigma_hi=s_hi, sigma_lo=s_lo))
    print(f"  k={k}d  MAPE_H={mape_h:5.2f}%  MAPE_L={mape_l:5.2f}%  "
          f"hit5_H={hit5_h:5.1f}  hit5_L={hit5_l:5.1f}  "
          f"σ_hi={s_hi:.4f}  σ_lo={s_lo:.4f}")

joblib.dump(dict(
    hi_models=hi_models, lo_models=lo_models,
    sigma_hi=sigma_hi, sigma_lo=sigma_lo,
    feat_cols=feat_cols, horizons=HORIZONS,
    calibration_meta=dict(
        train_start=str(train.index.min().date()),
        train_end  =str(train.index.max().date()),
        test_start =str(test.index.min().date()),
        test_end   =str(test.index.max().date()),
        train_n=int(len(train)), test_n=int(len(test)),
        model_family="RidgeCV per horizon",
    ),
    test_metrics=test_metrics,
), f"{OUT}/inference_assets_horizon.joblib")
print(f"\nSaved inference_assets_horizon.joblib (7 horizon models)")
