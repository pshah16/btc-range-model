"""Compare the model to two naive baselines to show real lift.

Baseline A — 'no model': pred_high = pred_low = close (degenerate, but if 95%
hit comes purely from price proximity, we'll see it here).
Baseline B — 'historical avg range': pred_hi_off = mean(y_hi_train),
                                     pred_lo_off = mean(y_lo_train).
"""
import pickle, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error
OUT = "/home/jovyan/btc-range-model"

data = pd.read_csv(f"{OUT}/features.csv", index_col=0, parse_dates=True)
test_start = data.index.max() - pd.DateOffset(months=8)
train = data.loc[:test_start - pd.Timedelta(days=1)]
test  = data.loc[test_start:]

close = test["close"].values
hi_true = test["next_high"].values
lo_true = test["next_low"].values

def report(name, pred_hi, pred_lo):
    mape_h = np.mean(np.abs(pred_hi - hi_true) / hi_true) * 100
    mape_l = np.mean(np.abs(pred_lo - lo_true) / lo_true) * 100
    hit5_h = np.mean(np.abs(pred_hi - hi_true) / hi_true <= 0.05) * 100
    hit5_l = np.mean(np.abs(pred_lo - lo_true) / lo_true <= 0.05) * 100
    pred_rng = pred_hi - pred_lo; true_rng = hi_true - lo_true
    mape_r = np.mean(np.abs(pred_rng - true_rng) / true_rng) * 100
    print(f"{name:30s}  MAPE_H={mape_h:5.2f}%  MAPE_L={mape_l:5.2f}%  "
          f"hit5_H={hit5_h:5.1f}%  hit5_L={hit5_l:5.1f}%  MAPE_range={mape_r:5.1f}%")

# A: predict high = low = close
report("A. close==high==low",      close, close)

# B: predict offsets = training mean
mu_hi = train["y_hi"].mean(); mu_lo = train["y_lo"].mean()
report(f"B. const offset (h={mu_hi:.3f}, l={mu_lo:.3f})",
       close*(1+mu_hi), close*(1-mu_lo))

# C: yesterday's high/low (random walk)
report("C. random walk (today's H/L)",
       test["close"].values + (test["close"].values * 0),  # placeholder
       test["close"].values + (test["close"].values * 0))
# Better C: use today's high & today's low directly
yest_hi = test["close"].values * (1 + 0)  # need today's H/L from raw
raw = pd.read_csv(f"{OUT}/raw.csv", index_col=0, parse_dates=True)
test_raw = raw.loc[test.index]
report("C. yesterday H/L (random walk)",
       test_raw["btc_high"].values, test_raw["btc_low"].values)

# D: trailing-7-day H/L average offsets
hi_off = train["y_hi"].rolling(7).mean().iloc[-1]
lo_off = train["y_lo"].rolling(7).mean().iloc[-1]
# better: per-row trailing-7 from the lagged target feature
report("D. 7d-trailing avg offset",
       close*(1 + test["y_hi_lag7_ma"].values),
       close*(1 - test["y_lo_lag7_ma"].values))
