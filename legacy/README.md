# Legacy artefacts (UTC-midnight pipeline)

This directory holds an older generation of the daily H/L pipeline that was
**anchored to UTC-midnight** bars, plus its associated 7-day-window and
per-horizon models. They were superseded on **2026-05-20** by the
7am-Central-Time pipeline at `src/pipeline_ct.py`.

The files here are kept for **reproducibility and audit**, not for ongoing
operation. The hardcoded paths inside these scripts still point to the
*old* flat-directory layout (`/home/jovyan/btc-range-model/raw.csv`, etc.)
and will not work without manual edits — by design. If you need to re-run
any of them, copy the script back to the repo root and put the CSVs back
alongside, or rewrite the paths.

## What's here

```
legacy/
├── pipeline.py                          UTC-midnight daily training pipeline
├── retrain_daily_v2.py                  Ensemble blend retrain (UTC)
├── retrain_daily_tight.py               Tighter MAPE retrain (UTC)
├── train_horizon_models.py              k=1..7 per-horizon models (UTC)
├── train_7d_model.py                    7-day-window max/min model (UTC)
├── baseline.py                          Climatology baseline (UTC)
├── save_inference_assets.py             First-version artifact saver (UTC)
├── build_notebook.py                    Builder for btc_range_prediction.ipynb
├── build_inference_notebook.py          Builder for btc_inference.ipynb
├── build_7d_training_nb.py              Builder for btc_range_7d_training.ipynb
├── btc_range_prediction.ipynb           UTC daily training write-up
├── btc_range_7d_training.ipynb          UTC 7-day window training write-up
├── btc_inference.ipynb                  UTC daily + 7-day + horizon inference
├── models/
│   ├── inference_assets.joblib          UTC daily H/L model (Huber+Bayes+GBM-MAE)
│   ├── inference_assets_7d.joblib       UTC 7-day window model
│   └── inference_assets_horizon.joblib  Per-horizon (k=1..7) models
├── data/
│   ├── raw.csv                          UTC-midnight bars + macro + on-chain
│   └── features.csv                     Engineered features for UTC pipeline
└── backups/
    └── inference_assets.pre-*.joblib.bak   joblib backups from each retrain run
```

## Why a CT pipeline replaced this

The UTC-midnight day boundary didn't reflect when most US-based traders
think about "today's daily range". The current pipeline (`src/pipeline_ct.py`)
re-buckets BTC hourly data into 24h bars starting at **12:00 UTC = 7 AM CDT
(/ 6 AM CST)** and trains the same Huber+Bayes+GBM-MAE ensemble on those bars.

Backtest MAPE was statistically indistinguishable between the two pipelines
(UTC: H 1.09% / L 1.28%; CT: H 1.12% / L 1.31%), so the change was purely
about boundary semantics, not predictive lift.
