"""Fetch full BTCUSDT hourly history from Binance public API.

Saves to binance_hourly_btc.csv with columns:
    timestamp_utc, open, high, low, close, volume

Coverage: 2017-08-17 (Binance launch) -> now, hourly.
"""
import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from paths import BINANCE_HOURLY_CSV as _CSV
OUT = str(_CSV)
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
START_MS = int(datetime(2017, 8, 17, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
END_MS = int(datetime.now(timezone.utc).timestamp() * 1000)
LIMIT = 1000
URL = "https://api.binance.com/api/v3/klines"


def fetch():
    rows = []
    cursor = START_MS
    while cursor < END_MS:
        params = dict(symbol=SYMBOL, interval=INTERVAL,
                      startTime=cursor, limit=LIMIT)
        for attempt in range(5):
            try:
                r = requests.get(URL, params=params, timeout=30)
                r.raise_for_status()
                batch = r.json()
                break
            except Exception as e:
                print(f"  retry {attempt+1}: {e}")
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError("Binance fetch failed after retries")

        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        cursor = last_open + 3600_000
        time.sleep(0.2)
        if len(rows) % 10000 == 0:
            ts = datetime.fromtimestamp(cursor/1000, tz=timezone.utc)
            print(f"  fetched {len(rows):>7} rows  cursor={ts}")

    print(f"  total rows: {len(rows)}")
    return rows


def to_dataframe(rows):
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_base", "taker_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df["timestamp_utc"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    df = df[["timestamp_utc", "open", "high", "low", "close", "volume"]].copy()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df.set_index("timestamp_utc").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def main():
    print(">>> Fetching BTCUSDT hourly from Binance ...")
    rows = fetch()
    df = to_dataframe(rows)
    print(f">>> Range: {df.index.min()} -> {df.index.max()}   ({len(df)} hours)")
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    missing = full.difference(df.index)
    print(f">>> Missing hours: {len(missing)}")
    df.to_csv(OUT)
    print(f">>> Saved {OUT}")


if __name__ == "__main__":
    main()
