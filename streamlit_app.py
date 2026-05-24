"""Root-level Streamlit entry-point.

Streamlit Community Cloud expects the main file at the repo root named
``streamlit_app.py``.  The real application lives at ``app/btc_hourly_app.py``
(so it can import from ``paths.py`` and share model artefacts cleanly).

This shim delegates to the real app via ``runpy.run_path``, which preserves
``__file__`` as the *actual* path of ``btc_hourly_app.py`` — required because
that module resolves the repo root with ``Path(__file__).parent.parent``.
"""
import runpy
from pathlib import Path

_app = Path(__file__).resolve().parent / "app" / "btc_hourly_app.py"
runpy.run_path(str(_app), run_name="__main__")
