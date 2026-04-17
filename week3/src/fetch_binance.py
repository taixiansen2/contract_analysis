"""拉 Binance SOLUSDC 1s K 线覆盖某 UTC 天。

Binance `/api/v3/klines` 每次最多 1000 条，interval=1s ⇒ 1000s/次，86400s 需 87 次请求。
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone

import httpx

from .config import data_paths, utc_day_bounds

ENDPOINT = "https://api.binance.com/api/v3/klines"
SYMBOL = "SOLUSDC"
INTERVAL = "1s"
LIMIT = 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    args = ap.parse_args()
    paths = data_paths(args.date)
    start_ts, end_ts = utc_day_bounds(args.date)
    start_ms = start_ts * 1000
    end_ms = end_ts * 1000

    print(f"[binance] {datetime.fromtimestamp(start_ts, tz=timezone.utc)} ~ {datetime.fromtimestamp(end_ts, tz=timezone.utc)}")
    cur = start_ms
    total = 0
    with httpx.Client(timeout=30) as client, open(paths["binance_csv"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["open_time_ms", "open", "high", "low", "close", "volume", "close_time_ms",
             "quote_volume", "trades", "taker_base_vol", "taker_quote_vol"]
        )
        while cur < end_ms:
            r = client.get(
                ENDPOINT,
                params={"symbol": SYMBOL, "interval": INTERVAL, "startTime": cur, "endTime": end_ms, "limit": LIMIT},
            )
            if r.status_code == 429:
                print("  [binance] 429, sleep 10s")
                time.sleep(10)
                continue
            r.raise_for_status()
            data = r.json()
            if not data:
                break
            for k in data:
                writer.writerow(k[:11])
            total += len(data)
            last_open = data[-1][0]
            cur = last_open + 1000  # 1s 后
            if total % 10000 == 0 or len(data) < LIMIT:
                print(f"  [binance] rows={total} cur={datetime.fromtimestamp(cur/1000, tz=timezone.utc)}")
            if len(data) < LIMIT:
                break
    print(f"[binance] wrote {total} rows → {paths['binance_csv']}")


if __name__ == "__main__":
    main()
