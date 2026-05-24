"""Root-level entry point for Streamlit Community Cloud.

Streamlit Community Cloud looks for `streamlit_app.py` in the repo root by
default.  The real app lives at app/btc_hourly_app.py — this shim delegates
to it via runpy so that __file__ inside the target script resolves correctly
(paths.py is imported relative to the repo root from there).
"""
import runpy
from pathlib import Path

_app = Path(__file__).parent / "app" / "btc_hourly_app.py"
runpy.run_path(str(_app), run_name="__main__")
