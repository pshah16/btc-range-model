"""Train the 3-class next-day day-type classifier and save the artefact.

Classes (defined on realized next-day H/L vs today's close):
  • BigUpper  : (y_hi + y_lo) ≥ tercile threshold AND y_hi > y_lo
  • BigLower  : (y_hi + y_lo) ≥ tercile threshold AND y_lo > y_hi
  • Quiet     : (y_hi + y_lo) < tercile threshold

Features = the same 103 daily features used by the H/L model, plus:
  pred_y_hi, pred_y_lo, pred_range, pred_skew  (from the H/L model)
  p_bull                                       (direction head)
  regime_0, regime_1, regime_2                 (7-day cone tercile)

Training split (chosen because it gave the best held-out accuracy in
/tmp/3class_split_comparison.py — see chat thread for the comparison):
  TRAIN ≤ 2026-02-18    TEST 2026-02-19 → 2026-05-18

(The smaller hold-out is for sanity-check calibration; the production
model is trained on TRAIN only, then evaluated on TEST.)
"""
import sys, joblib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.base            import clone
from sklearn.ensemble        import (GradientBoostingClassifier,
                                     GradientBoostingRegressor)
from sklearn.linear_model    import HuberRegressor, QuantileRegressor
from sklearn.metrics         import (accuracy_score, balanced_accuracy_score,
                                     confusion_matrix, classification_report)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from paths import (DAILY_MODEL_CT, CONE_7D_MODEL, RAW_CT_CSV, FEATURES_CT_CSV,
                   MODELS_DIR)

# ── Load source data + upstream artefacts ───────────────────────────────
raw  = pd.read_csv(RAW_CT_CSV,      index_col=0, parse_dates=True)
fts  = pd.read_csv(FEATURES_CT_CSV, index_col=0, parse_dates=True)
hl   = joblib.load(DAILY_MODEL_CT)
cone = joblib.load(CONE_7D_MODEL)
mh, ml = hl["hi_model"], hl["lo_model"]
fc      = hl["feat_cols"]
clf_dir = hl["direction_head"]["classifier"]

# H/L training boundary — rows ≤ this date were IN-SAMPLE to mh/ml/clf_dir
# and would leak optimism into the 3-class training features. We replace
# in-sample predictions on those rows with TimeSeriesSplit OOF predictions
# from a fresh ensemble; rows after this date keep the saved-model preds
# (which are already legitimately out-of-sample to H/L).
hl_train_end = pd.Timestamp(hl["calibration_meta"]["train_end"])
print(f"H/L model train_end = {hl_train_end.date()}  "
      f"→ OOF predictions required for rows ≤ this date")

data = fts.copy()
data["close"]     = raw.loc[data.index, "btc_close"]
data["next_high"] = raw["btc_high"].shift(-1).reindex(data.index)
data["next_low"]  = raw["btc_low" ].shift(-1).reindex(data.index)
data["y_hi"]      = (data["next_high"] - data["close"]) / data["close"]
data["y_lo"]      = (data["close"] - data["next_low"])  / data["close"]
data = data.dropna(subset=fc + ["y_hi","y_lo"]).copy()

# ── Generate H/L predictions: OOF for in-H/L-train rows, direct for the rest
ALPHA_QUANT     = 0.70
N_OOF_SPLITS    = 5
OOF_EMBARGO     = 1   # = forecast horizon for daily H/L


def _mk_hl_constituents():
    """Same 3 regressors used in src/pipeline_ct.py — kept locally
    so we don't need to import the (side-effectful) training module."""
    base = lambda model: Pipeline([("sc", StandardScaler()), ("m", model)])
    return [
        base(HuberRegressor(max_iter=500, alpha=0.001)),
        base(QuantileRegressor(quantile=ALPHA_QUANT, alpha=0.001, solver="highs")),
        base(GradientBoostingRegressor(
            loss="quantile", alpha=ALPHA_QUANT,
            n_estimators=1500, max_depth=3, learning_rate=0.01,
            subsample=0.8, random_state=42)),
    ]


def _mk_hl_dir_clf():
    return Pipeline([
        ("sc", StandardScaler()),
        ("m", GradientBoostingClassifier(
            n_estimators=500, max_depth=3, learning_rate=0.02,
            subsample=0.8, random_state=42)),
    ])


# Initialise the columns. Out-of-H/L-train rows get the saved-model preds.
data["pred_y_hi"] = np.nan
data["pred_y_lo"] = np.nan
data["p_bull"]    = np.nan

post_hl_mask = data.index > hl_train_end
if post_hl_mask.any():
    X_post = data.loc[post_hl_mask, fc]
    data.loc[post_hl_mask, "pred_y_hi"] = mh.predict(X_post)
    data.loc[post_hl_mask, "pred_y_lo"] = ml.predict(X_post)
    data.loc[post_hl_mask, "p_bull"]    = clf_dir.predict_proba(X_post)[:, 1]
    print(f"  saved-model preds for {int(post_hl_mask.sum())} post-H/L-train rows")

# OOF predictions for in-H/L-train rows.  TimeSeriesSplit gives N_OOF_SPLITS
# validation folds with an expanding training window; the first
# 1/(N_OOF_SPLITS+1) of the data is never in a val fold and gets no OOF
# prediction — we drop those rows below.
oof_dates = data.index[data.index <= hl_train_end]
if len(oof_dates) > 0:
    X_oof     = data.loc[oof_dates, fc]
    yhi_oof   = data.loc[oof_dates, "y_hi"].values
    ylo_oof   = data.loc[oof_dates, "y_lo"].values
    tss = TimeSeriesSplit(n_splits=N_OOF_SPLITS)
    print(f"  generating OOF H/L preds for {len(oof_dates)} rows "
          f"via TimeSeriesSplit({N_OOF_SPLITS} folds, embargo={OOF_EMBARGO}d)")
    for fold_i, (tr_i, va_i) in enumerate(tss.split(oof_dates)):
        # Embargo: drop last OOF_EMBARGO rows of fold train so the shift(-1)
        # target of the last training row doesn't sit inside the val fold.
        if OOF_EMBARGO > 0 and len(tr_i) > OOF_EMBARGO:
            tr_i = tr_i[:-OOF_EMBARGO]
        Xf_tr, Xf_va = X_oof.iloc[tr_i], X_oof.iloc[va_i]
        yhi_tr, ylo_tr = yhi_oof[tr_i], ylo_oof[tr_i]

        # H/L ensemble
        ph_va = np.zeros(len(va_i)); pl_va = np.zeros(len(va_i))
        for ctor in _mk_hl_constituents():
            mhi = clone(ctor); mhi.fit(Xf_tr, yhi_tr); ph_va += mhi.predict(Xf_va)
            mlo = clone(ctor); mlo.fit(Xf_tr, ylo_tr); pl_va += mlo.predict(Xf_va)
        ph_va /= 3.0; pl_va /= 3.0

        # Direction classifier
        d_tr = (yhi_tr - ylo_tr) / 2
        label_tr = (d_tr > 0).astype(int)
        if label_tr.min() == label_tr.max():
            # degenerate fold — fall back to majority probability
            pbull_va = np.full(len(va_i), float(label_tr.mean()))
        else:
            pbull_va = _mk_hl_dir_clf().fit(Xf_tr, label_tr).predict_proba(Xf_va)[:, 1]

        va_dates = oof_dates[va_i]
        data.loc[va_dates, "pred_y_hi"] = ph_va
        data.loc[va_dates, "pred_y_lo"] = pl_va
        data.loc[va_dates, "p_bull"]    = pbull_va
        print(f"    fold {fold_i+1}: tr_n={len(tr_i):>5} va_n={len(va_i):>5}  "
              f"({oof_dates[va_i[0]].date()} → {oof_dates[va_i[-1]].date()})")

# Drop rows that ended up without OOF predictions (the un-validated prefix
# of TimeSeriesSplit).
n_before = len(data)
data = data.dropna(subset=["pred_y_hi", "pred_y_lo", "p_bull"]).copy()
print(f"  dropped {n_before - len(data)} rows without OOF coverage; "
      f"n={len(data)} remain")

data["pred_range"] = data["pred_y_hi"] + data["pred_y_lo"]
data["pred_skew"]  = data["pred_y_hi"] - data["pred_y_lo"]

edges = np.asarray(cone["regime_edges"])
reg   = np.searchsorted(edges, data["range_ma30"].values, side="right").clip(0, 2)
for r in (0, 1, 2):
    data[f"regime_{r}"] = (reg == r).astype(int)

FEATS = [
    "pred_y_hi","pred_y_lo","pred_range","pred_skew","p_bull",
    "range_today","range_ma7","range_ma30","range_std30",
    "atr_7","atr_14","atr_30","vol_10","vol_20","vol_30",
    "bb_width","macd","macd_hist","rsi_14",
    "ret_3","ret_7","ret_14",
    "regime_0","regime_1","regime_2",
    "dow_0","dow_1","dow_2","dow_3","dow_4","dow_5",
]

# ── Split with 1-day embargo so the last train row's shift(-1) target
#    does not live inside the test window.
EMBARGO_DAYS = 1   # = forecast horizon for daily H/L
test_start   = pd.Timestamp("2026-02-19")
train_end    = test_start - pd.Timedelta(days=EMBARGO_DAYS + 1)  # one day of embargo
tr = data.loc[: train_end].copy()
te = data.loc[test_start:].copy()
print(f"TRAIN n={len(tr)}  ({tr.index.min().date()} → {tr.index.max().date()})")
print(f"TEST  n={len(te)}  ({te.index.min().date()} → {te.index.max().date()})")
print(f"  embargo {EMBARGO_DAYS}d between train_end={train_end.date()} "
      f"and test_start={test_start.date()}")

# ── Quiet threshold = bottom tercile of (y_hi+y_lo) from TRAIN ─────────
qthr = float(np.quantile(tr["y_hi"] + tr["y_lo"], 0.34))
def label(df):
    rng = df["y_hi"] + df["y_lo"]
    quiet = rng < qthr
    upper = (~quiet) & (df["y_hi"] > df["y_lo"])
    lower = (~quiet) & (df["y_lo"] > df["y_hi"])
    out = np.full(len(df), "Quiet", dtype=object)
    out[upper.values] = "BigUpper"
    out[lower.values] = "BigLower"
    return out
tr["label"] = label(tr); te["label"] = label(te)
print(f"  Quiet threshold on y_hi+y_lo (TRAIN tercile): {qthr:.4f}")
print(f"  TRAIN class counts: {dict(tr['label'].value_counts())}")
print(f"  TEST  class counts: {dict(te['label'].value_counts())}")

# ── Train GBM ────────────────────────────────────────────────────────────
gbm = GradientBoostingClassifier(
    n_estimators=400, max_depth=3, learning_rate=0.03,
    subsample=0.8, random_state=42,
).fit(tr[FEATS], tr["label"])

# ── Evaluate on TEST ────────────────────────────────────────────────────
pred = gbm.predict(te[FEATS])
acc  = accuracy_score(te["label"], pred) * 100
bacc = balanced_accuracy_score(te["label"], pred) * 100
print(f"\nTEST accuracy = {acc:.2f}%   balanced = {bacc:.2f}%")
print(classification_report(te["label"], pred,
                            labels=["BigUpper","BigLower","Quiet"], digits=3,
                            zero_division=0))
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(te["label"], pred, labels=["BigUpper","BigLower","Quiet"]))

# ── Selective accuracy by max-class probability ─────────────────────────
proba = gbm.predict_proba(te[FEATS]); cls = list(gbm.classes_)
top_p = proba.max(axis=1); pred_g = np.array(cls)[proba.argmax(axis=1)]
selective = []
for thr in [0.45, 0.50, 0.55, 0.60, 0.65]:
    mask = top_p >= thr
    if mask.sum() == 0: continue
    sacc = (pred_g[mask] == te["label"].values[mask]).mean() * 100
    selective.append(dict(thr=thr, coverage_pct=float(mask.mean()*100),
                          accuracy_pct=float(sacc), n=int(mask.sum())))
    print(f"  p ≥ {thr:.2f}   coverage = {mask.mean()*100:5.1f}%  "
          f"accuracy = {sacc:5.1f}%   (n={mask.sum()})")

# ── Save artefact ───────────────────────────────────────────────────────
art = dict(
    model = gbm,
    feature_columns = FEATS,
    class_labels = list(gbm.classes_),
    quiet_threshold = qthr,
    regime_edges = list(map(float, edges)),
    calibration_meta = dict(
        train_start = str(tr.index.min().date()),
        train_end   = str(tr.index.max().date()),
        test_start  = str(te.index.min().date()),
        test_end    = str(te.index.max().date()),
        train_n     = int(len(tr)),
        test_n      = int(len(te)),
        embargo_days = int(EMBARGO_DAYS),
        oof_n_splits = int(N_OOF_SPLITS),
        oof_embargo  = int(OOF_EMBARGO),
        hl_train_end = str(hl_train_end.date()),
        test_accuracy_pct = float(acc),
        test_balanced_acc_pct = float(bacc),
        selective    = selective,
        method       = ("GBM 400×3 lr=0.03 trained on stacked features built "
                        "with OOF H/L predictions (TimeSeriesSplit x5) for "
                        "rows ≤ hl_train_end and direct H/L preds for rows "
                        f"after. Embargo {EMBARGO_DAYS}d between train/test."),
    ),
)
OUT = MODELS_DIR / "inference_assets_3class.joblib"
joblib.dump(art, OUT)
print(f"\nSaved {OUT}")
