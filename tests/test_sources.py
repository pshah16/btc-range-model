"""Quick connectivity check for all free data sources."""
import sys, json
import pandas as pd
import yfinance as yf
import requests

print("== Yahoo Finance ==")
try:
    btc = yf.download("BTC-USD", start="2025-01-01", end="2025-02-01",
                      progress=False, auto_adjust=False)
    print("BTC-USD rows:", len(btc), "cols:", list(btc.columns.get_level_values(0) if hasattr(btc.columns, 'get_level_values') else btc.columns)[:6])
    print(btc.tail(2))
except Exception as e:
    print("FAIL:", e)

print("\n== blockchain.info charts (no key) ==")
url = "https://api.blockchain.info/charts/hash-rate?timespan=30days&format=json&sampled=false"
try:
    r = requests.get(url, timeout=20)
    j = r.json()
    print("series:", j.get("name"), "points:", len(j.get("values", [])))
    print("first:", j["values"][0], "last:", j["values"][-1])
except Exception as e:
    print("FAIL:", e)

print("\n== Yahoo macro symbols ==")
for sym in ["^GSPC", "DX-Y.NYB", "GC=F", "^VIX", "^TNX", "^IXIC", "ETH-USD"]:
    try:
        d = yf.download(sym, start="2025-01-01", end="2025-01-15",
                        progress=False, auto_adjust=False)
        print(f"{sym}: {len(d)} rows")
    except Exception as e:
        print(f"{sym}: FAIL {e}")
