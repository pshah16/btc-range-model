"""Central path registry for the btc-range-model repo.

Every script that touches files imports its paths from here. To relocate the
project, only this file needs to change.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Directories
APP_DIR        = ROOT / "app"
SRC_DIR        = ROOT / "src"
NOTEBOOKS_DIR  = ROOT / "notebooks"
MODELS_DIR     = ROOT / "models"
DATA_DIR       = ROOT / "data"
ARTIFACTS_DIR  = ROOT / "artifacts"
RUNTIME_DIR    = ROOT / "runtime"
TESTS_DIR      = ROOT / "tests"
LEGACY_DIR     = ROOT / "legacy"

# Active model artefacts (7am-CT daily + hourly close + 7-day cone + 3-class)
DAILY_MODEL_CT  = MODELS_DIR / "inference_assets_ct.joblib"
HOURLY_MODEL    = MODELS_DIR / "inference_assets_hourly.joblib"
CONE_7D_MODEL   = MODELS_DIR / "inference_assets_7d_cone.joblib"
DAY_TYPE_MODEL  = MODELS_DIR / "inference_assets_3class.joblib"

# Data
BINANCE_HOURLY_CSV  = DATA_DIR / "binance_hourly_btc.csv"
RAW_CT_CSV          = DATA_DIR / "raw_ct.csv"
RAW_HOURLY_CSV      = DATA_DIR / "raw_hourly.csv"
FEATURES_CT_CSV     = DATA_DIR / "features_ct.csv"

# Runtime / user-mutable state
BOOKMARKS_FILE = RUNTIME_DIR / "bookmarks.json"

# Training-time artefacts (feature importance, PCA, etc.)
ARTIFACTS_PKL = ARTIFACTS_DIR / "artifacts.pkl"

# Legacy (UTC-midnight) artefacts — kept for reference / reproducibility
LEGACY_DAILY_MODEL    = LEGACY_DIR / "models" / "inference_assets.joblib"
LEGACY_7D_MODEL       = LEGACY_DIR / "models" / "inference_assets_7d.joblib"
LEGACY_HORIZON_MODEL  = LEGACY_DIR / "models" / "inference_assets_horizon.joblib"
LEGACY_RAW_CSV        = LEGACY_DIR / "data" / "raw.csv"
LEGACY_FEATURES_CSV   = LEGACY_DIR / "data" / "features.csv"
