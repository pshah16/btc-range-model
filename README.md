# btc-range-model

A four-model Bitcoin forecasting stack served through a Streamlit dashboard:

| Model | What it predicts | Cadence | Horizon |
|---|---|---|---|
| **Daily H/L (7am-CT)** | Next 24-hour bar's `high` and `low` | Refreshes once per day at 12:00 UTC (= 7am CDT / 6am CST) | 1 bar (24 h) ahead |
| **Hourly close** | Next-hour BTC closing price | Refreshes every 60 s in the live tab | 1 hour ahead |
| **7-day close cone** | Median + ±9.7 % band for the close 7 days out | Refreshes once per day | 7 days ahead |
| **3-class day-type** | Next-day bar shape: `BigUpper` / `BigLower` / `Quiet` | Refreshes once per day | 1 bar ahead |

All four are linear-or-shallow learners trained on a mix of BTC price/volume
history, cross-market macro indicators, on-chain blockchain metrics, and a
sentiment index. The Streamlit UI surfaces their predictions side-by-side with
realised price action so accuracy is observable in real time.

> **Anti-leakage methodology** (May 2026 retrain). Every model uses a strict
> chronological **train / val / test** split with a per-horizon embargo gap; α,
> β, model winner, and σ are all selected on **val** with **test untouched**
> until final reporting. The Fear & Greed feed — which alternative.me updates
> intraday on the current day's record — is lagged 1 day for the hourly model.
> The 3-class day-type model uses **TimeSeriesSplit-5 out-of-fold H/L
> predictions** for rows ≤ the H/L training cutoff to remove stacked in-sample
> contamination. See [§Leakage audit & fixes](#leakage-audit--fixes).

---

## Repository layout

```
btc-range-model/
├── README.md                ← this file
├── requirements.txt         ← Python deps
├── paths.py                 ← single source of truth for all file paths
├── .gitignore
│
├── app/
│   └── btc_hourly_app.py    ← the Streamlit UI (entrypoint)
│
├── src/                     ← active training & data-fetch code
│   ├── pipeline_ct.py       ← end-to-end daily 7am-CT pipeline (rebucket → features → train → save)
│   ├── train_hourly_model.py← hourly close-price model training
│   └── fetch_binance_hourly.py ← one-shot Binance hourly history fetcher
│
├── notebooks/
│   ├── btc_inference_ct.ipynb   ← 7am-CT daily H/L inference walk-through
│   ├── btc_hourly_training.ipynb← hourly model training walk-through
│   └── builders/                 ← scripts that re-generate the notebooks above
│       ├── build_inference_ct_notebook.py
│       └── build_hourly_training_nb.py
│
├── models/                   ← trained artefacts loaded at inference time
│   ├── inference_assets_ct.joblib       ← active daily H/L model (7am-CT)
│   └── inference_assets_hourly.joblib   ← active hourly close model
│
├── data/                     ← input & engineered data (mostly cached / regenerable)
│   ├── binance_hourly_btc.csv   ← full BTCUSDT hourly history (2017-08 → now)
│   ├── raw_hourly.csv            ← Yahoo BTC-USD hourly + macro (rolling 2y)
│   ├── raw_ct.csv                ← joined 12:00-UTC daily bars
│   └── features_ct.csv           ← engineered features matrix
│
├── artifacts/                ← training-time diagnostics
│   └── artifacts.pkl
│
├── runtime/                  ← UI-mutable state (persists across restarts)
│   └── bookmarks.json
│
├── tests/                    ← lightweight smoke tests for external data feeds
│
└── legacy/                   ← UTC-midnight pipeline, kept for audit (see legacy/README.md)
    ├── pipeline.py, retrain_daily_*, train_horizon_models.py, train_7d_model.py …
    ├── btc_range_prediction.ipynb, btc_inference.ipynb, …
    ├── models/  (inference_assets.joblib, _7d.joblib, _horizon.joblib)
    ├── data/    (raw.csv, features.csv)
    └── backups/ (.joblib.bak files from each retrain)
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Launch the dashboard
streamlit run app/btc_hourly_app.py

# Re-train the four models.
# Order matters: pipeline_ct → cone → 3-class (the latter two read the H/L
# artefact). Hourly is independent.
python src/pipeline_ct.py            # daily H/L  → models/inference_assets_ct.joblib
python src/train_7d_close_cone.py    # 7-day cone → models/inference_assets_7d_cone.joblib
python src/train_3class_day_type.py  # day-type   → models/inference_assets_3class.joblib
python src/train_hourly_model.py     # hourly     → models/inference_assets_hourly.joblib

# Pull a fresh full BTC hourly history (writes data/binance_hourly_btc.csv)
python src/fetch_binance_hourly.py

# Re-generate notebooks from their builder scripts
python notebooks/builders/build_inference_ct_notebook.py
python notebooks/builders/build_hourly_training_nb.py
```

The Streamlit app is the public-facing surface; the `src/` scripts and notebooks
are training utilities. Everything reads paths from `paths.py`.

---

## Model 1 — Daily H/L (7am-CT day boundary)

### Day-boundary contract

A "day" in this model is a **24-hour bar starting at 12:00 UTC** =
- **7:00 AM CDT** (US Central Daylight Time, March → November), or
- **6:00 AM CST** (US Central Standard Time, November → March).

Fixed UTC anchor (not DST-following) so every bar is exactly 24 hours — no
23/25 h edge cases. The model predicts the **next** such bar's high and low.

### Pipeline (`src/pipeline_ct.py`)

```
Binance hourly OHLCV  ─┐
                       ├─► rebucket into 12:00→12:00 UTC daily bars
Yahoo daily macro     ─┤   (open=first, high=max, low=min, close=last, vol=sum)
(SPX, NDX, VIX, GOLD, │
DXY, TNX, ETH)        │
                       │
blockchain.info       ─┤── join on calendar date D ─► engineered features
on-chain (11 series)  │
                       │
Fear & Greed Index    ─┘

Target (per bar D):
  y_hi = (next_high - close) / close   ≥ 0
  y_lo = (close - next_low ) / close   ≥ 0
  Reconstruct: pred_high = close·(1 + y_hi),  pred_low = close·(1 - y_lo)

Model:
  Ensemble of 3 regressors (all wrapped in StandardScaler):
    1. HuberRegressor                (linear, robust)
    2. QuantileRegressor   (q=0.70, linear quantile)
    3. GradientBoostingRegressor     (loss="quantile", q=0.70,
                                      1500 trees, depth 3, lr 0.01)
  Pure mean of the three predictions, then a shrinkage blend with the
  training-set climatological mean offset (μ_hi, μ_lo) using α.

Direction head:
  GradientBoostingClassifier on sign(y_hi − y_lo). The post-α ensemble is
  reparameterised into (half-range m, asymmetry d); d is blended with a
  classifier-driven d using an adaptive β_eff = β_base × (1 − r × trend_str),
  where trend_str = min(|ret_5|/0.05, 1).

Tuning protocol (anti-leak):
  TRAIN  → fit ensemble + direction classifier
  VAL    → pick α and (β_base, r) by validation MAPE
  TEST   → final, untouched evaluation (never seen during tuning)
  Embargo of 1 day between train↔val and val↔test (= forecast horizon)
  so a training row's shift(-1) target cannot live in the next split.

Current split (artefact dated 2026-05-21):
  TRAIN  2019-04-01 → 2025-09-17   (n=2342)
  VAL    2025-09-18 → 2025-11-17   (n=61)
  TEST   2025-11-18 → 2026-05-18   (n=182)
  Selected: α = 1.00, β_base = 1.00, r = 0.30.
```

### Features — full inventory (103 total)

Grouped by family. Suffixes: `_n` = lag in days, `_dN` = log-difference over N days, `_zN` = N-day rolling z-score. Notation `c` = BTC close, `h` = high, `l` = low, `v` = volume.

#### BTC price action (38 features)

| # | Feature | Definition | Why it matters |
|---|---|---|---|
| 1–6 | `ret_1, ret_3, ret_5, ret_7, ret_14, ret_30` | Sum of daily log-returns over the last 1/3/5/7/14/30 bars | Momentum signal across multiple horizons; range autocorrelates with recent direction |
| 7–10 | `vol_5, vol_10, vol_20, vol_30` | Rolling std-dev of log-returns over 5/10/20/30 bars | Realized volatility regime. Range scales with vol. |
| 11–13 | `atr_7, atr_14, atr_30` | ATR (true range)/`c` averaged over 7/14/30 days | Volatility in price-units, normalized by current price. Range = ATR scaled. |
| 14 | `range_today` | `(h - l) / c` | Current bar's range — the single strongest predictor of tomorrow's range |
| 15 | `range_ma7` | 7-day average of `range_today` | Range mean-reversion baseline |
| 16 | `range_ma30` | 30-day average of `range_today` | Long-run baseline; comparison to `range_ma7` reveals regime shift |
| 17 | `range_std30` | 30-day std-dev of `range_today` | Is volatility of volatility rising? Spike before regime breaks. |
| 18 | `rsi_14` | RSI on 14 closes | Overbought/oversold; affects asymmetry of tomorrow's range |
| 19 | `macd` | (EMA12 − EMA26) / `c` | Trend strength |
| 20 | `macd_sig` | EMA9 of MACD, /`c` | Trend-direction signal line |
| 21 | `macd_hist` | MACD − MACD signal | Trend-acceleration |
| 22 | `bb_width` | 4·σ20 / MA20 (Bollinger band width) | Compression → expansion volatility cycle |
| 23 | `dist_hi_30` | `c / max(c, 30) − 1` | How far below the 30-day high (≤ 0). Sets ceiling pressure. |
| 24 | `dist_lo_30` | `c / min(c, 30) − 1` | How far above the 30-day low (≥ 0). Sets floor pressure. |
| 25 | `dist_hi_90` | `c / max(c, 90) − 1` | 90-day-extreme proximity; longer-horizon trend context |
| 26 | `vol_chg_1` | `log(v) − log(v.shift(1))` | 1-day volume change |
| 27 | `vol_z_20` | z-score of `log(v)` over 20 days | Standardised volume surprise |
| 28 | `vol_ma_ratio` | `v / MA20(v)` | Volume vs recent average |
| 29–34 | `dow_0…dow_5` | Day-of-week one-hots (Mon=0…Sat=5) | BTC has real weekend behavior; Sun is the reference category |

#### Cross-asset macro (28 features)

Each of 7 macro symbols contributes 4 features:

| Symbol | Source | Features |
|---|---|---|
| `spx` (S&P 500) | Yahoo `^GSPC` | `spx_ret_1`, `spx_ret_5`, `spx_ret_20`, `spx_vol_20` |
| `ndx` (Nasdaq Composite) | Yahoo `^IXIC` | `ndx_ret_1`, `ndx_ret_5`, `ndx_ret_20`, `ndx_vol_20` |
| `vix` (CBOE Volatility) | Yahoo `^VIX` | `vix_ret_1`, `vix_ret_5`, `vix_ret_20`, `vix_vol_20` |
| `gold` (Gold futures) | Yahoo `GC=F` | `gold_ret_1`, `gold_ret_5`, `gold_ret_20`, `gold_vol_20` |
| `dxy` (US Dollar Index) | Yahoo `DX-Y.NYB` | `dxy_ret_1`, `dxy_ret_5`, `dxy_ret_20`, `dxy_vol_20` |
| `tnx` (10-yr Treasury yield) | Yahoo `^TNX` | `tnx_ret_1`, `tnx_ret_5`, `tnx_ret_20`, `tnx_vol_20` |
| `eth` (Ether) | Yahoo `ETH-USD` | `eth_ret_1`, `eth_ret_5`, `eth_ret_20`, `eth_vol_20` |

Each `*_ret_k` is `log(close[t]) − log(close[t-k])` and each `*_vol_20` is the std-dev of the corresponding 1-day return series over 20 days. **Rationale:** crypto correlates positively with risk-on (SPX/NDX), negatively with USD strength (DXY) and yields (TNX), and shows regime-dependent links to VIX and Gold.

Plus 4 cross-asset rolling correlations:

| Feature | Definition |
|---|---|
| `btc_spx_corr_30` | 30-day Pearson corr between BTC and SPX daily log-returns |
| `btc_ndx_corr_30` | …BTC vs NDX |
| `btc_gold_corr_30` | …BTC vs Gold |
| `btc_dxy_corr_30` | …BTC vs DXY |

#### On-chain (33 features)

Pulled from `blockchain.info`'s public charts API. 11 series × 3 transforms each:

| Series | Plain meaning |
|---|---|
| `oc_hash_rate` | Network total hash rate (security / miner commitment) |
| `oc_difficulty` | Mining difficulty (lags hash rate by epoch length) |
| `oc_n_transactions` | Daily tx count |
| `oc_miners_revenue` | Total daily revenue to miners (block subsidy + fees) |
| `oc_n_unique_addresses` | Daily active addresses — adoption proxy |
| `oc_transaction_fees_usd` | Aggregate fees paid (mempool pressure) |
| `oc_mempool_size` | Pending tx queue size |
| `oc_estimated_transaction_volume_usd` | On-chain dollar throughput |
| `oc_market_cap` | Daily market cap |
| `oc_avg_block_size` | Mean block size |
| `oc_cost_per_transaction` | Network cost / tx — efficiency measure |

For each series `S` the three features are:

| Feature suffix | Formula | Captures |
|---|---|---|
| `_d1` | `log(S[t]) − log(S[t-1])` | 1-day delta — surprise / event-driven moves |
| `_d7` | `log(S[t]) − log(S[t-7])` | Week-over-week trend |
| `_z30` | z-score of `log(S)` over 30 days | Standardised deviation from monthly norm |

#### Smoothed past-target features (4 features) + anti-mean-reversion (6 features) + downside regime (4 features)

| Family | Feature | Definition |
|---|---|---|
| Smoothed lag | `y_hi_ema3`, `y_hi_ema7`, `y_lo_ema3`, `y_lo_ema7` | EMA (span 3, 7) of `y_hi.shift(1)` / `y_lo.shift(1)`. Smoother than a raw lag → less mean-reversion overshoot. |
| Breakout signals | `above_3d_high`, `below_3d_low`, `bo_strength_up`, `bo_strength_dn` | Binary + magnitude versions of `close vs max/min(high, 3) / min/max(low, 3)`. |
| "Surprise" | `y_hi_surprise`, `y_lo_surprise` | `y_hi.shift(1) − ema7(y_hi.shift(1))` — how much yesterday's excursion differed from the recent baseline. |
| Downside | `dn_vol_5`, `dn_vol_20`, `below_sma50`, `below_sma50_5d` | Downside-only realised vol (5/20-day) + flags for "close below SMA50" (today, and ≥5 of last 5 days). |

These give the model a memory of how its own target has been behaving — range
is autocorrelated, so yesterday's realised range is informative — without the
single-day lag's tendency to over-anchor and oscillate.

#### Top contributors

Permutation importance for the current daily H/L artefact is not persisted in
the joblib; the per-feature ranking from prior training runs (price-action
dominant: `range_today`, `atr_7`, `dist_hi_30`, day-of-week effects, and
`oc_transaction_fees_usd_z30` as the strongest on-chain contributor) is
qualitatively unchanged with the new train/val/test methodology. The dominance
of price-action and self-lag families confirms range is mostly autoregressive;
macro and on-chain provide secondary refinement.

### Held-out test metrics  (honest — α/β tuned on **val**, test untouched)

Test window: **2025-11-18 → 2026-05-18** (182 days; 6 months). Validation
window: 2025-09-18 → 2025-11-17 (61 days). Training: 2,342 days from
2019-04-01.

| Metric | HIGH | LOW |
|---|---:|---:|
| **MAPE** | **1.32 %** | **1.30 %** |
| MAE (USD) | $1,049 | $988 |
| RMSE (USD) | $1,291 | $1,475 |
| Hit ±0.5 % | 20.9 % | 23.6 % |
| Hit ±1 % | 45.1 % | 52.2 % |
| Hit ±2 % | 79.7 % | 87.9 % |
| Hit ±5 % | 100.0 % | 97.3 % |
| σ (residual, return space) | 0.0161 | 0.0186 |
| 95 % CI half-width | ±3.16 % | ±3.64 % |
| Direction hit-rate (sign of y_hi − y_lo) | colspan=2 | **50.0 %** |
| Direction classifier accuracy (TEST) | colspan=2 | 53.3 % |

> **Honest framing.** The previously-reported figures of MAPE 1.12 % / 1.31 %
> and **53.8 %** direction hit-rate were inflated by hyperparameter tuning on
> the test set. With α and β picked on a held-out val slice, MAPE_H lands at
> **~1.32 %** and the direction-head's apparent edge **collapses to coin-flip
> on test (50.0 %)**, even though the underlying classifier still gets 53.3 %
> on test alone. The model's real value is **price-magnitude bracketing**, not
> direction calling — use it as a volatility cone, not a directional signal.
>
> Compared with the training-set climatology baseline (val MAPE 1.37 % / 1.78 %),
> the ensemble still extracts ~5–15 % of structure above pure mean-reversion
> on val; α=1.00 means the artefact uses no climatology shrinkage.

### Inference contract (used by `app/`)

- `compute_daily_forecast(asof_date_iso)` (in `app/btc_hourly_app.py`) re-builds
  the same features from the latest 12:00-UTC bars, truncates to bars with
  start date ≤ `asof_date_iso`, runs the ensemble, and returns
  `(pred_high, pred_low, CI bands, target_window_start, target_window_end)`.
- The "as-of bar" is automatically the latest *completed* 12:00-UTC bar; the
  target window is the next 24-h bar after that.
- Cached for 24 h per as-of-date key (the date itself is in the cache key, so
  it rolls over automatically at each 12:00 UTC).

---

## Model 2 — Hourly close

### What it predicts

Next-hour log-return of `BTC-USD` (Yahoo) close, anchored at the most recent
completed hourly bar.  At inference, the predicted log-return is multiplied
against the **live Binance spot** (refreshed every 30 s) to produce the
"1-hour-from-now" price point on the chart.

### Pipeline (`src/train_hourly_model.py`)

```
Yahoo BTC-USD 1h bars  ─┐
+ ETH-USD, SPX, NDX,   │
  VIX, GOLD, DXY, TNX  ─┤ all resampled to a common hourly grid
+ alternative.me F&G   ─┘ shifted 1 day, then forward-filled

Target:  y = log(close[t+1] / close[t])     (i.e. one-hour log-return)

Models compared:
  • RidgeCV (linear, regularisation chosen by CV)
  • GradientBoostingRegressor (squared loss, 800 trees, depth 3, lr 0.02)

Tuning protocol (anti-leak):
  TRAIN  → fit each candidate model
  VAL    → pick the winner by validation direction-accuracy; fit σ on VAL residuals
  TEST   → final, untouched evaluation
  Embargo of 1 hour between train↔val and val↔test (= forecast horizon).

Current split (artefact dated 2026-05-21):
  TRAIN  2024-05-28 22h UTC → 2026-03-21 02h UTC   (n=6487)
  VAL    2026-03-22 22h UTC → 2026-04-05 23h UTC   (n=139)
  TEST   2026-04-06 00h UTC → 2026-05-21 21h UTC   (n=339)
  Winner: gbm.

F&G causality fix:
  alternative.me publishes one record per UTC date but re-computes the
  current-day record *throughout* that day. Using the value stamped at
  D 00:00 UTC at hour D 14:00 UTC leaks information from later than 14:00.
  The series is therefore lagged by 1 day before joining (hours of D use
  D-1's finalised value). Applied identically in the live `fetch_data()`.
```

### Features — full inventory (59 total)

Suffixes: `_Nh` = lag in hours, `_N` = lookback window in hours. `c` = BTC close, `h/l` = hourly high/low, `v` = hourly volume.

#### BTC price action (29 features)

| # | Feature | Definition | Why it matters |
|---|---|---|---|
| 1–8 | `ret_1h, ret_2h, ret_4h, ret_8h, ret_12h, ret_24h, ret_48h, ret_72h` | Sum of hourly log-returns over last 1/2/4/8/12/24/48/72 hours | Momentum across multiple horizons; hourly returns are short-memory but multi-hour aggregates carry signal |
| 9–12 | `vol_4h, vol_8h, vol_24h, vol_48h` | Rolling std-dev of hourly log-returns over 4/8/24/48 hours | Vol regime classifier |
| 13–15 | `atr_4h, atr_12h, atr_24h` | True-range/`c` averaged over 4/12/24 hours | Bar-size volatility; complements return-std |
| 16 | `range_now` | `(h - l) / c` current hour | Current hour's bar-size |
| 17 | `range_ma24` | 24-hour MA of `range_now` | Day-scale range baseline |
| 18 | `range_ma72` | 72-hour MA of `range_now` | 3-day range baseline |
| 19 | `vol_chg_1` | `log(v) − log(v.shift(1))` | Hour-over-hour volume change |
| 20 | `vol_z_24` | 24-hour z-score of `log(v)` | Standardised volume surprise |
| 21 | `rsi_14` | 14-hour RSI | Overbought/oversold |
| 22 | `macd` | (EMA12 − EMA26) / `c` | Trend strength |
| 23 | `macd_hist` | MACD − signal-line MACD | Trend acceleration |
| 24 | `bb24_width` | 24-hour Bollinger band width | Volatility compression / expansion |
| 25 | `dist_hi_24` | `c / max(c, 24h) − 1` | Distance from 24h high (≤ 0) |
| 26 | `dist_lo_24` | `c / min(c, 24h) − 1` | Distance from 24h low (≥ 0) |
| 27 | `dist_hi_168` | `c / max(c, 168h) − 1` | Distance from week's high — longer-cycle context |

#### Cross-asset macro (21 features)

Same 7 macro symbols as the daily model (`spx`, `ndx`, `vix`, `gold`, `dxy`, `tnx`, `eth`), each with 3 features per symbol:

| Feature suffix | Definition | Purpose |
|---|---|---|
| `<sym>_ret_1h` | 1-hour log-return | Synchronous co-move |
| `<sym>_ret_24h` | 24-hour log-return | Daily-cycle co-move |
| `<sym>_vol_24h` | 24-hour rolling std of returns | Cross-asset vol context |

So 21 features total: `eth_ret_1h, eth_ret_24h, eth_vol_24h, spx_ret_1h, …, tnx_vol_24h`.

**Caveat:** the daily macro feeds from Yahoo only update once a day. Within a UTC day the daily-cycle values are constant for all 24 hours — they add **inter-day** information but no **intraday** signal.

Plus one cross-asset correlation:

| Feature | Definition |
|---|---|
| `btc_eth_corr_24` | 24-hour rolling Pearson corr between BTC and ETH hourly log-returns. ETH is the only crypto with a true hourly feed alongside BTC, so this is the cleanest cross-crypto regime indicator. |

#### Sentiment (4 features) — Fear & Greed Index

Source: `alternative.me/fng` (daily; forward-filled to every hour within the day).

| Feature | Definition |
|---|---|
| `fng` | Current F&G value (0=extreme fear, 100=extreme greed) |
| `fng_d1` | 1-hour change (essentially 0 within a day; non-zero at day rollover) |
| `fng_d7` | 7-day change |
| `fng_d24` | 24-hour change |

#### Time-of-day / calendar (5 features)

| Feature | Definition | Purpose |
|---|---|---|
| `hr_sin` | `sin(2π · hour / 24)` | Cyclical hour-of-day encoding |
| `hr_cos` | `cos(2π · hour / 24)` | (pair with `hr_sin` to embed hour on the unit circle) |
| `dow_sin` | `sin(2π · dayofweek / 7)` | Cyclical day-of-week |
| `dow_cos` | `cos(2π · dayofweek / 7)` | (pair with `dow_sin`) |
| `weekend` | 1 if Saturday or Sunday else 0 | BTC volume/vol regime shifts on weekends (no equity markets) |
| `us_open` | 1 if the hour overlaps the NYSE session, else 0 | Liquidity surge during US trading hours |

(`us_open` counts as the 5th when including the `weekend` flag, totaling 6 calendar features — see `feat_cols` for the exact set; the model uses 5 distinct calendar features after redundancy was pruned.)

#### Top contributors (permutation importance on TEST, n_repeats=5)

From the current artefact (gbm, 2026-05-21 retrain):

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `tnx_ret_1h` | +0.0221 |
| 2 | `vix_ret_24h` | +0.0089 |
| 3 | `ret_4h` | +0.0064 |
| 4 | `vix_ret_1h` | +0.0056 |
| 5 | `btc_eth_corr_24` | +0.0052 |
| 6 | `eth_vol_24h` | +0.0044 |
| 7 | `spx_ret_1h` | +0.0040 |
| 8 | `vol_48h` | +0.0037 |
| 9 | `tnx_ret_24h` | +0.0036 |
| 10 | `dxy_ret_24h` | +0.0030 |
| 11 | `ndx_vol_24h` | +0.0027 |
| 12 | `eth_ret_24h` | +0.0026 |
| 13 | `vol_z_24` | +0.0024 |
| 14 | `vol_4h` | +0.0022 |
| 15 | `dxy_ret_1h` | +0.0016 |

**Reading:** the top driver is **the 1-hour change in the 10-year Treasury
yield** (`tnx_ret_1h`), followed by the **VIX 24-hour return** — i.e., short-
horizon rates and risk-off pressure dominate. Cross-asset features (TNX, VIX,
SPX, ETH, DXY, GOLD) take 10 of the top 15 slots; BTC-internal features
(`ret_4h`, `vol_48h`, `vol_z_24`, `vol_4h`) take the rest. The hourly model
is best read as a **macro-vol regime classifier** routed through Bitcoin
returns, not a BTC-autoregressive model.

### Held-out metrics  (val picks the winner, test is unbiased)

Test window: **2026-04-06 → 2026-05-21** (≈ 1,080 hours of hourly data).
Validation window: 2026-03-22 → 2026-04-05 (≈ 340 hours).
Winning model: **GradientBoostingRegressor on log-returns** (picked by
VAL direction-accuracy; ridge was second).

| Metric | VAL (used for selection) | TEST (unbiased) |
|---|---:|---:|
| MAPE on next-hour close | 0.40 % | **0.29 %** |
| MAE (USD) | $271 | **$220** |
| RMSE (USD) | $381 | $317 |
| Hit ±0.5 % | 73.4 % | 83.8 % |
| Hit ±1 % | 91.4 % | 96.2 % |
| Hit ±3 % | 100.0 % | 100.0 % |
| **Direction accuracy** | 51.1 % | **50.15 %** |
| R² on log-return | −0.05 | +0.007 |
| σ (residual, log-return space) | 0.0056 | — |
| 95 % CI half-width on predicted close (from VAL σ) | colspan=2 | **±1.09 %** |

> **Honest framing.** The previously-reported **53.8 % direction accuracy was
> a test-selection artifact** — when we pick the winning model on TEST, the
> winner's TEST score is by construction optimistic. Picking on VAL instead
> gives a TEST direction accuracy of **50.15 %**, indistinguishable from a
> coin-flip. ±3 % "accuracy" is also trivially ~100 % at a 1-hour horizon
> since hourly BTC moves rarely exceed 3 %.
>
> What the hourly model *can* honestly offer is a **tight 1-hour volatility
> cone** — the 95 % CI half-width of ±1.09 % brackets the expected return
> reasonably well — and a **slight magnitude advantage** (0.29 % MAPE
> vs ~0.39 % for the zero-return baseline). Use it as a vol-cone forecaster,
> *not* as a directional signal.

### Inference contract (used by `app/`)

- Streamlit caches `fetch_data()` (Yahoo hourly + F&G) for 300 s.
- Each refresh tick: build features for the latest hour, predict next-hour
  log-return `y`, multiply against `live_spot` (Binance) to get the dollar
  forecast point. Display ±0.5 % band around it.
- A 24-hour walk-forward look-back is recomputed every refresh so accuracy
  metrics on the last realised hours are visible.

---

## Model 3 — 7-day close cone

### What it predicts

The expected close 7 days after the as-of bar, with a fixed-width
uncertainty band that reflects empirical regime-conditioned dispersion of
7-day forward returns.

### Pipeline (`src/train_7d_close_cone.py`)

```
features_ct.csv (12:00-UTC daily features matrix)
+ raw_ct.csv    (BTC close)

Target:  y_logret_7 = log(close[t+7] / close[t])

Regime classifier:
  Bin training-set days into terciles of range_ma30 (30-day avg of (h-l)/c).
  Three regimes: low / mid / high realised-vol.

For each regime r:
  Record empirical quantiles {10, 25, 50, 75, 90} of y_logret_7 on TRAIN.

Forecast at time t:
  regime(t) = tercile of range_ma30(t)
  pred_close = close(t) * exp(regime_stats[regime(t)][0.50])
  band       = pred_close × (1 ± 0.097)        # fixed ±9.7 %

Embargo: 7 days (= forecast horizon) between train_end and test_start so
the last train rows' shift(-7) target does not contain any test-window
prices.
```

### Current artefact (2026-05-21 retrain)

| Field | Value |
|---|---|
| Train | 2019-04-01 → 2025-09-14   (n=2339) |
| Test  | 2025-09-21 → 2026-05-12   (n=234) |
| Embargo | 7 days |
| Tercile edges (range_ma30) | 0.0379 / 0.0506 |
| Held-out coverage of ±9.7 % band | **88.0 %** |
| Band width (fixed) | ±9.7 % around regime median |

Regime medians (forward 7-day log-return) on TRAIN:

| Regime | Label | Median log-ret | Median % equiv | n_train |
|---:|---|---:|---:|---:|
| 0 | low vol  | +0.0105 | +1.06 % | 780 |
| 1 | mid vol  | −0.0025 | −0.25 % | 779 |
| 2 | high vol | +0.0130 | +1.31 % | 780 |

(The 90th-percentile up-move is meaningfully different across regimes:
+10.7 % in low-vol, +9.0 % in mid-vol, +15.1 % in high-vol regimes — so the
band's *coverage* changes by regime even though its *width* is fixed.)

### Inference contract (used by `app/`)

`compute_7d_close_cone_forecast(asof_date_iso)` in `app/btc_hourly_app.py`:
- classifies the as-of bar's `range_ma30` into a regime,
- multiplies the as-of close by `exp(regime_stats[regime][0.50])`,
- applies a fixed ±9.7 % band on the resulting price,
- also returns the last 7 weekly close observations and their 7-day-back
  predictions so the historical-replay tab can plot realised closes vs the
  prior week's cone forecasts.

---

## Model 4 — 3-class day-type classifier

### What it predicts

Categorical label for the next 12:00-UTC bar's shape:

| Class | Definition |
|---|---|
| `BigUpper` | (y_hi + y_lo) ≥ TRAIN tercile threshold **and** y_hi > y_lo |
| `BigLower` | (y_hi + y_lo) ≥ TRAIN tercile threshold **and** y_lo > y_hi |
| `Quiet`    | (y_hi + y_lo) < TRAIN tercile threshold |

### Pipeline (`src/train_3class_day_type.py`)

```
Features = the 103 daily features in features_ct.csv
         + 5 model-derived features:
              pred_y_hi, pred_y_lo, pred_range, pred_skew, p_bull
         + 3 cone-regime one-hots (regime_0 / regime_1 / regime_2)
         → 31 total

Classifier: GradientBoostingClassifier (400 trees, depth 3, lr 0.03).

Anti-leak: TimeSeriesSplit-5 OOF for the H/L-derived features
  • Rows ≤ H/L train_end were IN-SAMPLE to the saved H/L ensemble +
    direction classifier; if we used those directly as features, the
    3-class model would learn to over-trust pred_y_hi / pred_y_lo / p_bull
    because they match y_hi / y_lo / sign too closely on train.
  • Instead, TimeSeriesSplit(5) refits the full H/L ensemble + direction
    classifier on each fold's growing training window, then predicts on
    the val fold. That gives **honest out-of-fold** H/L predictions for
    every in-H/L-train row. Rows after H/L train_end keep the saved-model
    preds (which are already legitimately out-of-sample to H/L).
  • The first ~1/(N_splits+1) of the data has no val-fold assignment in
    TimeSeriesSplit and is dropped from 3-class training.
  • A 1-day embargo separates 3-class TRAIN from TEST.

Quiet threshold: 34th percentile of (y_hi + y_lo) on TRAIN only.
```

### Current artefact (2026-05-21 retrain)

| Field | Value |
|---|---|
| Train | 2020-05-06 → 2026-02-17   (n=2,103, after dropping the 392 un-OOF'd prefix rows) |
| Test  | 2026-02-19 → 2026-05-18   (n=89) |
| Embargo | 1 day |
| OOF strategy | TimeSeriesSplit(5), embargo 1 day per fold |
| H/L train_end (boundary for OOF vs direct) | 2025-09-17 |
| Quiet threshold (y_hi + y_lo) on TRAIN | 0.0289 |
| TRAIN class balance | Quiet 715 / BigUpper 705 / BigLower 683 |
| TEST class balance | Quiet 35 / BigUpper 29 / BigLower 25 |

### Held-out test metrics

| Metric | Value |
|---|---:|
| **Unconditional accuracy** | **52.8 %** (vs 33 % majority baseline) |
| Balanced accuracy | 49.6 % |
| F1 — Quiet | 0.697 |
| F1 — BigUpper | 0.318 |
| F1 — BigLower | 0.400 |

Selective accuracy by top-class probability:

| p ≥ | Coverage | Accuracy | n |
|---:|---:|---:|---:|
| 0.45 | 74.2 % | 56.1 % | 66 |
| 0.50 | 57.3 % | 58.8 % | 51 |
| **0.55** | **43.8 %** | **69.2 %** | **39** |
| 0.60 | 23.6 % | 76.2 % | 21 |
| 0.65 | 15.7 % | 64.3 % | 14 |

> **Reading.** The 3-class model is *not* useful when forced to label every
> day (52.8 % overall) — but it earns most of its accuracy on the days where
> it's confident. At `p ≥ 0.55` it covers ~44 % of days at ~69 % accuracy;
> at `p ≥ 0.60` it covers ~24 % of days at ~76 %. The Quiet class is by far
> the easiest to call; both Big-move classes have weak recall.

### Inference contract (used by `app/`)

`compute_day_type_forecast(target_date_iso)` in `app/btc_hourly_app.py`:
- pulls the same 31 features at the as-of bar (using the H/L model's
  predictions and the cone's regime one-hots),
- runs the GBM classifier,
- returns the top class plus full probability vector for the pill display.

---

## UI architecture (`app/btc_hourly_app.py`)

### Components

```
streamlit page
├── sidebar
│   ├── "Refresh now" button  (clears all @st.cache_data)
│   └── auto-refresh note
│
├── headline KPIs (4 columns)
│   ├── Live BTC spot (Binance)         ← refreshed every 30 s
│   ├── 1-hour forecast (price + return%)
│   ├── 0.5 % forecast band
│   └── Fear & Greed (latest daily)
│
├── Daily H/L card
│   ├── target window (UTC + "= today/tomorrow 7am CT")
│   ├── Predicted DAILY HIGH  (+ ±2.5 % band)
│   └── Predicted DAILY LOW   (+ ±2.5 % band)
│
├── Hourly chart  (Plotly, fig)
│   ├── last 24h actual close (black line)
│   ├── past hourly predictions (color-coded by direction accuracy)
│   ├── ±0.5 % band around each past prediction
│   ├── live ⭐ rolling 1-hour forecast (with error bars)
│   ├── live spot dot
│   ├── "Now" vertical line + label (CT)
│   └── horizontal daily H/L threshold lines  (+ ±2.5 % shaded bands)
│
└── Daily H/L vs actuals chart  (Plotly, fig2)
    ├── last 7 bars' (HIGH/LOW) predictions (dotted lines)
    ├── realised HIGH/LOW (X markers)
    └── next-day forecast (rightmost, highlighted khaki)
```

The page has **two tabs** sharing the same render code:

| Tab | "Now" anchor | Hourly chart window | Daily H/L target |
|---|---|---|---|
| **🔴 Live** | wall-clock `datetime.now(UTC)` | rolling last 24 h ending at the latest valid hour | tomorrow's bar (7am-CT-anchored) |
| **🕒 Historical replay** | user-picked CT timestamp (date strip + datetime slider) | fixed 24-h CT day `[7am picked_date → 7am picked_date+1]` | bar starting on picked_date + 1 |

### Historical-replay extras

- **7-day pill strip** — clickable buttons for picked − 3 … + 3 days. Re-centers on selection.
- **◀ / ▶ arrows** — shift selected date by ±1 day.
- **Calendar picker** — st.date_input for arbitrary date selection across the data window.
- **Datetime slider** — 25-tick CT slider from 7am picked_date through 7am next day, controls the as-of moment within the bar.
- **🔖 Bookmarks panel** — categorize and persist favorite dates. Backed by `runtime/bookmarks.json` so they survive restarts.

### How the UI talks to the ML models

```
                           ┌──────────────────────────────────────────┐
                           │ app/btc_hourly_app.py  (Streamlit)        │
                           └──────────────────────────────────────────┘
                                          │
                                          │ on each rerun
                                          ▼
       ┌──────────────────────────────────────────────────────────────┐
       │  fetch_data()              live spot                          │
       │  (Yahoo BTC-USD hourly,    fetch_live_spot()                  │
       │   ETH, SPX, NDX, VIX,      (Binance ticker API,               │
       │   GOLD, DXY, TNX  +        BTCUSDT, 30 s cache)               │
       │   alternative.me F&G,                                          │
       │   300 s cache)                                                 │
       └──────────────────────────────────────────────────────────────┘
                  │                                  │
                  ▼                                  │
       ┌────────────────────────────┐                │
       │ build_features() (hourly)  │                │
       │ produces 103-col matrix    │                │
       └────────────────────────────┘                │
                  │                                  │
                  ▼                                  ▼
       ┌────────────────────────────┐    ┌─────────────────────────────┐
       │ models/                    │    │ price anchoring             │
       │ inference_assets_hourly    │    │ (multiply log-return        │
       │ .joblib  (Ridge)           │    │  prediction by live spot)   │
       │ predict y = log-return     │    │                             │
       └────────────────────────────┘    └─────────────────────────────┘
                  │                                  │
                  └──────────────┬───────────────────┘
                                 ▼
                  ⭐ next-hour rolling forecast point + ±0.5 % band

       ┌──────────────────────────────────────────────────────────────┐
       │  _fetch_daily_raw()                                            │
       │  • BTC hourly from data/binance_hourly_btc.csv  (full history) │
       │  • +top-up from Binance klines API since CSV's last hour       │
       │  • rebucket to 12:00 UTC daily bars                            │
       │  • join macro (Yahoo daily) + on-chain (blockchain.info)       │
       │    on calendar date                                            │
       │  21 600 s cache                                                │
       └──────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
       ┌────────────────────────────────────────────────────────────────┐
       │ compute_daily_forecast(asof_date_iso)                          │
       │  • re-engineer the same 103 features as pipeline_ct.py         │
       │  • truncate to bars ≤ asof_date                                │
       │  • load models/inference_assets_ct.joblib                      │
       │  • run ensemble (Huber + Bayes + GBM-MAE), mean + climatology  │
       │  • return pred_high, pred_low, CI bands, target window         │
       │  86 400 s cache, keyed by ISO date string                      │
       └────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                  Daily H/L KPI card + horizontal threshold lines on fig
                  + 7-day series (compute_daily_series → 8 calls × cache)
```

Key design choice: **all model loading and feature engineering happens inside
the Streamlit process, not in a separate inference service.** The "API" is
implicit — three functions (`load_assets`, `compute_daily_forecast`,
`compute_daily_series`) consume joblib artefacts and return DataFrames /
dicts that the chart code renders. There's no FastAPI/gRPC layer. This is
intentional: the app is a single-user analytics dashboard, not a high-QPS
prediction service.

### Where each price on the chart comes from

| Surface | Source feed | Pair | Frequency |
|---|---|---|---|
| Live spot KPI + crimson dot | Binance public ticker | BTC/**USDT** | 30 s |
| Hourly black "Actual close" line | Yahoo (yfinance) | BTC/**USD** | 60-min bars, 300 s cache |
| Daily H/L bars | Binance hourly → rebucketed | BTC/**USDT** | 24 h, 6 h cache |
| 1-hour rolling ⭐ forecast | Hourly model × Binance spot | — | every rerun |

The two feeds (Yahoo BTC-USD and Binance BTC/USDT) differ by a few basis
points (USDT premium/discount + slight latency). On the chart, the live red
dot occasionally sits slightly off the Yahoo hourly close line — that's the
feed gap, not a bug.

---

## Limitations & known caveats

1. **Direction edge is coin-flip out-of-sample.** With α, β, and the model
   winner all picked on a VAL slice instead of TEST, both the daily H/L
   model's and the hourly model's directional accuracy lands at ~50 % on
   TEST. The earlier 53–54 % figures were a test-tuning artefact. **Use
   these models for magnitude/volatility bracketing, not direction calling.**
2. **The 3-class model is most useful in selective mode.** Its 52.8 %
   unconditional accuracy is barely above the 33 % majority baseline, but
   restricting to days where `p_top ≥ 0.55` yields ~69 % accuracy at ~44 %
   coverage. Treat the pill colour as advisory; trust the probability bar.
3. **Yahoo hourly history is capped at ~2 years.** The historical-replay
   tab's earliest selectable date floats forward as time passes.
4. **Hourly model's macro features are daily-resolution.** Treasury yields,
   VIX, etc. are constant within a UTC day — they add directional signal
   over a week but no *intraday* signal.
5. **The 7am-CT boundary uses a fixed 12:00 UTC anchor**, not a DST-following
   "always 7 AM Central" rule. The chart label always says "7 am CT", but
   in the November–March window the actual local time the bar starts is
   6 AM CST. This was a deliberate trade — uniform 24-h bars beat one 23/25 h
   bar per year.
6. **`live_spot` is BTC/USDT, not BTC/USD.** The ~5 bp basis is rarely
   material but worth knowing.
7. **σ_lo > σ_hi on daily H/L** (0.0186 vs 0.0161 in the current artefact).
   BTC drawdowns are heavier-tailed than rallies in this training window,
   so the LOW prediction has a slightly wider 95 % CI half-width (±3.64 %)
   than the HIGH (±3.16 %).
8. **In-sample replay banner.** When a user picks a historical date inside
   any model's training window, the app shows a yellow `st.warning`
   explaining the predictions for that date are memorisation, not honest
   forecasts. Pick a date after each model's `train_end` (see each model's
   section) to see genuine out-of-sample behaviour.

---

## Leakage audit & fixes

Following a leakage audit in May 2026, the training and inference code paths
were hardened against five classes of issue. Snapshot of the fixes:

| Issue | Where | Fix |
|---|---|---|
| Hyperparameters (α, β, model winner, σ) tuned on TEST | `src/pipeline_ct.py`, `src/train_hourly_model.py` | New TRAIN / VAL / TEST split. Tuning lives on VAL; TEST untouched until final report. |
| No embargo at train/test boundary (target shifted forward) | All 4 training scripts | Embargo of `horizon` rows between splits: 1 day (daily H/L, 3-class), 1 hour (hourly), 7 days (cone). |
| Stacked in-sample H/L predictions used as features for 3-class | `src/train_3class_day_type.py` | TimeSeriesSplit-5 with 1-day per-fold embargo generates OOF H/L predictions for rows ≤ H/L train_end; rows after use direct preds. |
| Historical-replay tab silently shows in-sample fit for dates inside train windows | `app/btc_hourly_app.py` | Yellow `st.warning` banner naming each affected model and its `train_end` whenever the picked date is in-sample. |
| Fear & Greed index is updated *intraday* on its current-day record (alternative.me) — using it at hour h on date D leaks information from later than h | `src/train_hourly_model.py` + `app/btc_hourly_app.py` (`fetch_data`) | F&G series is shifted forward by 1 day before joining, so hours of date D always use date D-1's finalised value. (Daily H/L is unaffected: its bar D ends at D+1 12:00 UTC, by which time D's F&G is finalised.) |

What changed in the headline metrics after these fixes (daily H/L and hourly):

| Metric | Old (test-tuned, no embargo) | New (val-tuned, embargoed) |
|---|---:|---:|
| Daily H/L MAPE_H | 1.12 % | **1.32 %** |
| Daily H/L MAPE_L | 1.31 % | **1.30 %** |
| Daily H/L direction hit-rate | 53.8 % | **50.0 %** |
| Hourly MAPE | 0.32 % | **0.29 %** |
| Hourly direction accuracy | 53.8 % | **50.15 %** |
| Hourly 95 % CI half-width | ±0.91 % | **±1.09 %** |

The MAPE numbers moved by ≤ 0.2 percentage points (the magnitude estimates
were honest); the direction-accuracy numbers collapsed by ~3.8 pp because
they were almost entirely test-tuning inflation.

---

## Legacy / archived models

The `legacy/` directory contains the previous-generation UTC-midnight pipeline
plus two ancillary models (7-day window max/min; per-horizon k=1..7) that the
current dashboard does not use. See [`legacy/README.md`](legacy/README.md)
for details and why they were superseded.

Backtest summary (UTC-midnight legacy vs 7am-CT current):

| Model | Methodology | MAPE H | MAPE L |
|---|---|---:|---:|
| UTC-midnight (legacy)         | test-tuned α, no embargo | 1.09 % | 1.28 % |
| 7am-CT (previous test-tuned)  | test-tuned α/β, no embargo | 1.12 % | 1.31 % |
| **7am-CT (current honest)**   | **val-tuned α/β, 1-day embargo** | **1.32 %** | **1.30 %** |

The current artefact's MAPE_H is ~0.2 pp higher because the previous figure
was an in-grid best on TEST; honestly-tuned, the model performs roughly the
same as legacy. The CT boundary remains preferred for semantics (uniform
24-h bars, no DST 23/25-h edge case), not for predictive lift.
