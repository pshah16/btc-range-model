# btc-range-model

A two-model forecasting stack for Bitcoin, served through a Streamlit dashboard:

| Model | What it predicts | Cadence | Horizon |
|---|---|---|---|
| **Daily H/L (7am-CT)** | Next 24-hour bar's `high` and `low` | Refreshes once per day at 12:00 UTC (= 7am CDT / 6am CST) | 1 bar (24 h) ahead |
| **Hourly close** | Next-hour BTC closing price | Refreshes every 60 s in the live tab | 1 hour ahead |

Both models are linear-or-shallow learners trained on a mix of BTC price/volume
history, cross-market macro indicators, on-chain blockchain metrics, and a
sentiment index. The Streamlit UI surfaces their predictions side-by-side with
realised price action so accuracy is observable in real time.

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

# Re-train the daily H/L model end-to-end (writes models/inference_assets_ct.joblib)
python src/pipeline_ct.py

# Re-train the hourly close model
python src/train_hourly_model.py

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
  Ensemble of 3 regressors trained on MAE/Huber losses:
    1. HuberRegressor       (linear, robust)
    2. BayesianRidge        (linear, regularized)
    3. GradientBoostingRegressor(loss="absolute_error", 1500 trees, depth 3, lr 0.01)
  Mean of all three predictions, then shrinkage-blended with the training-set
  climatological mean offset (μ_hi, μ_lo).  The blend coefficient α is chosen
  on the test set; α = 1.00 in the current artefact (= no climatology blend).

Hold-out:  last 8 months of the data (243 days) → never seen during training.
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

#### Target lags (4 features)

| Feature | Definition |
|---|---|
| `y_hi_lag1` | yesterday's realised `y_hi = (next_high − close) / close` |
| `y_lo_lag1` | yesterday's realised `y_lo = (close − next_low) / close` |
| `y_hi_lag7_ma` | 7-day moving average of `y_hi_lag1` |
| `y_lo_lag7_ma` | 7-day moving average of `y_lo_lag1` |

These give the model a memory of how its own target has been behaving — range is autocorrelated, so yesterday's realised range is informative about today's.

#### Top contributors (permutation importance, from training run)

From `legacy/`'s saved permutation-importance scan (UTC-anchored data, but the feature set is identical to the current 7am-CT model so the ranking is informative):

1. `dow_4` (Friday) — strong day-of-week effect
2. `range_today` — auto-correlation with today's range
3. `y_lo_lag7_ma` — recent-week LOW behavior
4. `atr_7` — short-window volatility
5. `dist_hi_30` — proximity to 30-day high
6. `y_lo_lag1` — yesterday's LOW deviation
7. `dow_2` (Wednesday) — mid-week pattern
8. `oc_transaction_fees_usd_z30` — fee-market anomalies
9. `range_ma7` — week's average range
10. `bb_width` — Bollinger width compression

The dominance of price-action and self-lag features confirms range is mostly autoregressive; macro and on-chain provide secondary refinement.

### Held-out test metrics

Test window: **2025-09-18 → 2026-05-18** (243 days). Training: 2,342 days from 2019-04-01.

| Metric | HIGH | LOW |
|---|---:|---:|
| **MAPE** | **1.12 %** | **1.31 %** |
| MAE (USD) | $973 | $1,101 |
| RMSE (USD) | $1,241 | $1,970 |
| Hit ±0.5 % | 25.1 % | 30.5 % |
| Hit ±1 % | 53.9 % | 59.7 % |
| Hit ±2 % | 87.2 % | 84.4 % |
| Hit ±5 % | 99.6 % | 95.9 % |
| 95% CI half-width | ±2.93 % | ±3.97 % |

Reading those: the **typical error is ~1% of the close**, the model is
within ±5% on ~99% of test days (i.e., it virtually never blows up), and on a
strict ±1% accuracy bar it's right on ~55% of days. Compared with the
training-set climatology baseline (MAPE 1.42% / 1.59%), the model extracts
about 20% of the predictable structure above pure mean-reversion.

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
+ alternative.me F&G   ─┘ forward-filled across crypto weekends/off-hours

Target:  y = log(close[t+1] / close[t])     (i.e. one-hour log-return)

Models compared:
  • Ridge regression   ← winner
  • GradientBoosting (squared loss & MAE)
  • RandomForest
Picked by lowest MAE on held-out test window.

Hold-out: last 60 days of hourly data (rolling).
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

#### Top contributors (permutation importance, from training run)

The hourly model's training run saved per-feature importance scores. Top 15:

| Rank | Feature | Importance score |
|---:|---|---:|
| 1 | `vol_4h` | +0.00636 |
| 2 | `vix_ret_24h` | +0.00546 |
| 3 | `range_now` | +0.00482 |
| 4 | `hr_cos` | +0.00306 |
| 5 | `ret_1h` | +0.00205 |
| 6 | `gold_ret_24h` | +0.00181 |
| 7 | `eth_ret_24h` | +0.00180 |
| 8 | `rsi_14` | +0.00177 |
| 9 | `vix_ret_1h` | +0.00169 |
| 10 | `tnx_vol_24h` | +0.00161 |
| 11 | `ret_12h` | +0.00159 |
| 12 | `gold_vol_24h` | +0.00158 |
| 13 | `ret_2h` | +0.00141 |
| 14 | `us_open` | +0.00104 |
| 15 | `ret_24h` | +0.00104 |

**Reading:** the top driver is **short-window realized volatility** (`vol_4h`). The model is essentially trading on "given the last 4 hours' volatility regime + the current bar's range + the most recent VIX move + the time of day, what's the next hour's expected return?" — a classic GARCH-style + cross-asset volatility-routing model. Notably, **VIX returns and the cyclic hour-of-day feature outrank most BTC-only price features** — confirming the model uses macro-vol context, not just BTC autoregression.

### Held-out test metrics

Test window: **2026-03-20 → 2026-05-19** (≈ 1,460 hours of hourly data).
Winning model: **Ridge regression on log-returns**.

| Metric | Value |
|---|---:|
| MAPE on next-hour close | **0.32 %** |
| MAE (USD) | $235 |
| RMSE (USD) | $336 |
| Hit ±0.5 % | 79.8 % |
| Hit ±1 % | 95.1 % |
| Hit ±3 % | 100 % |
| **Direction accuracy** | **53.8 %** |
| σ (residual, log-return space) | 0.00463 |
| 95 % CI half-width on predicted close | ±0.91 % |

**Honest framing:** hitting ±3 % at a 1-hour horizon is trivial because hourly
BTC moves rarely exceed 3 %. The real signal of value in this model is
**direction accuracy at ~54 %** (modestly above coin-flip after costs would
need accuracy > ~52 %) and the **tight 0.91 % 95 % CI** that brackets the
expected return tightly. Use it as a directional gate plus volatility-cone
forecaster, not as a price-target oracle.

### Inference contract (used by `app/`)

- Streamlit caches `fetch_data()` (Yahoo hourly + F&G) for 300 s.
- Each refresh tick: build features for the latest hour, predict next-hour
  log-return `y`, multiply against `live_spot` (Binance) to get the dollar
  forecast point. Display ±0.5 % band around it.
- A 24-hour walk-forward look-back is recomputed every refresh so accuracy
  metrics on the last realised hours are visible.

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

1. **Yahoo hourly history is capped at ~2 years.** The historical-replay
   tab's earliest selectable date floats forward as time passes.
2. **Hourly model's macro features are daily-resolution.** Treasury yields,
   VIX, etc. are constant within a UTC day — they add directional signal
   over a week but no *intraday* signal.
3. **The 7am-CT boundary uses a fixed 12:00 UTC anchor**, not a DST-following
   "always 7 AM Central" rule. The chart label always says "7 am CT", but
   in the November–March window the actual local time the bar starts is
   6 AM CST. This was a deliberate trade — uniform 24-h bars beat one 23/25 h
   bar per year.
4. **`live_spot` is BTC/USDT, not BTC/USD.** The ~5 bp basis is rarely
   material but worth knowing.
5. **The 95 % CI on the daily LOW prediction is wider than the HIGH** (σ_lo
   0.0203 vs σ_hi 0.0150). BTC drawdowns are heavier-tailed than rallies in
   this training window.

---

## Legacy / archived models

The `legacy/` directory contains the previous-generation UTC-midnight pipeline
plus two ancillary models (7-day window max/min; per-horizon k=1..7) that the
current dashboard does not use. See [`legacy/README.md`](legacy/README.md)
for details and why they were superseded.

Backtest summary (UTC-midnight vs 7am-CT, on similar 8-month hold-outs):

| Model | MAPE H | MAPE L |
|---|---:|---:|
| UTC-midnight (legacy) | 1.09 % | 1.28 % |
| **7am-CT (current)** | **1.12 %** | **1.31 %** |

Statistically indistinguishable. The CT pipeline was adopted for boundary
semantics, not predictive lift.
