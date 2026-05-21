"""Train the 7-day close-price *regime cone* and save its artefact.

Approach (chosen after the research in
notebooks/btc_7d_close_research.ipynb):

  • Regime indicator: range_ma30 (30-day average daily range, already in
    the CT pipeline's feature matrix).
  • Bin training-set days into terciles of range_ma30.
  • For each tercile, record the empirical quantiles of the 7-day
    forward log-return.
  • At inference, classify today's range_ma30 into a regime, take the
    regime's median forward log-return → multiply against today's close,
    and apply a fixed ±9.7 % band (the average empirical [q10, q90]
    half-width reported in the research notebook, which corresponds to
    ~84 % coverage on the held-out 8-month tail).

Training data: data/raw_ct.csv + data/features_ct.csv (12:00-UTC anchored).
Hold-out: last 8 months — same convention as the rest of the repo.
"""
import sys, joblib
from datetime import datetime
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from paths import RAW_CT_CSV, FEATURES_CT_CSV, MODELS_DIR

HORIZON = 7
N_REGIMES = 3
BAND_PCT  = 0.097     # ±9.7 % — reported uncertainty band from the research notebook

raw = pd.read_csv(RAW_CT_CSV, index_col=0, parse_dates=True)
fts = pd.read_csv(FEATURES_CT_CSV, index_col=0, parse_dates=True)
c   = raw["btc_close"]

# 7-day-ahead log-return target. Drop any pre-existing 'close' col from
# features_ct.csv to avoid the join collision.
fts2 = fts.drop(columns=[col for col in ("close",) if col in fts.columns])
data = fts2.join(c.rename("close"), how="left")
data["y_logret_7"] = np.log(c.shift(-HORIZON) / c)
data = data.replace([np.inf,-np.inf], np.nan).dropna(subset=["range_ma30","y_logret_7"])
print(f"Data: {data.shape}   {data.index.min().date()} → {data.index.max().date()}")

# Hold-out with HORIZON-sized embargo so the last training row's target
# (which uses log(c[t+HORIZON]/c[t])) does not contain any test-window
# prices.  Without this, regime quantiles fitted on the training set would
# absorb up to HORIZON days of the test period.
TODAY        = pd.Timestamp(datetime.utcnow().date())
EMBARGO_DAYS = HORIZON   # = 7
test_start   = TODAY      - pd.DateOffset(months=8)
train_end    = test_start - pd.Timedelta(days=EMBARGO_DAYS)
train = data.loc[: train_end]
test  = data.loc[test_start:]
print(f"  embargo of {EMBARGO_DAYS} days between train_end={train_end.date()} "
      f"and test_start={test_start.date()}")

# Tercile edges from training set
edges = np.quantile(train["range_ma30"], [1/3, 2/3])
print(f"  tercile edges (range_ma30): {edges.round(4)}")

def to_regime(x):
    return np.searchsorted(edges, np.asarray(x, dtype=float), side="right")

train_regimes = to_regime(train["range_ma30"])
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
regime_stats = {}
for r in range(N_REGIMES):
    sub = train.loc[train_regimes == r, "y_logret_7"].values
    regime_stats[r] = {
        **{q: float(np.quantile(sub, q)) for q in QUANTILES},
        "mean": float(sub.mean()),
        "std":  float(sub.std()),
        "n":    int(len(sub)),
    }
print("Regime stats:")
print(pd.DataFrame(regime_stats).T.round(4))

# Quick eval on hold-out: empirical coverage of the fixed ±BAND_PCT band
test_regimes = to_regime(test["range_ma30"])
median_logret = np.array([regime_stats[r][0.50] for r in test_regimes])
band_lo = np.log(1 - BAND_PCT) + median_logret    # log-space lower band
band_hi = np.log(1 + BAND_PCT) + median_logret    # log-space upper band
inside = (test["y_logret_7"].values >= band_lo) & (test["y_logret_7"].values <= band_hi)
coverage_pct = 100 * inside.mean()
print(f"  held-out coverage of ±{BAND_PCT*100:.1f}% band around regime median: {coverage_pct:.1f}%")

art = dict(
    regime_feature   = "range_ma30",
    regime_edges     = edges.tolist(),
    regime_stats     = regime_stats,
    band_pct         = BAND_PCT,
    horizon_days     = HORIZON,
    quantiles_stored = QUANTILES,
    calibration_meta = dict(
        train_start = str(train.index.min().date()),
        train_end   = str(train.index.max().date()),
        test_start  = str(test.index.min().date()),
        test_end    = str(test.index.max().date()),
        train_n     = int(len(train)),
        test_n      = int(len(test)),
        embargo_days = int(EMBARGO_DAYS),
        method      = "range_ma30 tercile regime cone; median forward log-return",
        held_out_band_coverage_pct = float(coverage_pct),
    ),
)
OUT = MODELS_DIR / "inference_assets_7d_cone.joblib"
joblib.dump(art, OUT)
print(f"Saved {OUT}")
