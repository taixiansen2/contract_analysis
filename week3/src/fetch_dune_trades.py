"""用 Dune API 拉 Orca + Raydium 的 SOL/USDC 成交记录（某 UTC 天）。

做法：
  1. 调用 POST /v1/query/{id}/execute 创建一次执行（query 是我们现场用 /v1/query 提交的 SQL）。
     为了不依赖预先存在的 query_id，走 Dune 的"即席查询 via create query + execute"流程。
  2. 轮询 status，完成后分页 /results 下载。
  3. 存 data/raw/orca_raydium_sol_usdc_YYYYMMDD.csv。

若未提供 DUNE_API_KEY，脚本会直接提示并退出。
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone

import httpx

from .config import DUNE_API_KEY, data_paths, utc_day_bounds

DUNE_API = "https://api.dune.com/api/v1"

SQL_TEMPLATE = """
SELECT
    block_time,
    block_slot,
    tx_id,
    project,
    version,
    trader_id,
    token_bought_symbol,
    token_sold_symbol,
    token_bought_mint_address,
    token_sold_mint_address,
    token_bought_amount,
    token_sold_amount,
    amount_usd
FROM dex_solana.trades
WHERE block_time >= from_unixtime({start_ts})
  AND block_time <  from_unixtime({end_ts})
  AND project IN {projects}
  AND (
      (token_bought_mint_address = 'So11111111111111111111111111111111111111112'
       AND token_sold_mint_address = 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v')
   OR (token_bought_mint_address = 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'
       AND token_sold_mint_address = 'So11111111111111111111111111111111111111112')
  )
ORDER BY block_time
""".strip()


def _headers():
    return {"X-Dune-API-Key": DUNE_API_KEY, "Content-Type": "application/json"}


def create_query(sql: str, client: httpx.Client) -> int:
    r = client.post(
        f"{DUNE_API}/query",
        headers=_headers(),
        json={
            "query_sql": sql,
            "name": "week3_exp0_orca_raydium_sol_usdc",
            "is_private": True,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["query_id"]


def execute_query(query_id: int, client: httpx.Client) -> str:
    r = client.post(
        f"{DUNE_API}/query/{query_id}/execute",
        headers=_headers(),
        json={"performance": "medium"},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["execution_id"]


def poll_status(exec_id: str, client: httpx.Client, max_wait: int = 1800) -> str:
    start = time.time()
    last_state = None
    while time.time() - start < max_wait:
        r = client.get(f"{DUNE_API}/execution/{exec_id}/status", headers=_headers(), timeout=60)
        r.raise_for_status()
        data = r.json()
        state = data.get("state")
        if state != last_state:
            print(f"  [dune] state={state} elapsed={time.time()-start:.0f}s")
            last_state = state
        if state == "QUERY_STATE_COMPLETED":
            return state
        if state in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"):
            raise RuntimeError(f"Dune execution ended in {state}: {data}")
        time.sleep(5)
    raise TimeoutError(f"Dune execution {exec_id} not completed within {max_wait}s")


def download_results(exec_id: str, client: httpx.Client, out_csv):
    offset = 0
    limit = 1000
    writer = None
    total = 0
    headers_written = False
    with open(out_csv, "w", newline="") as f:
        while True:
            r = client.get(
                f"{DUNE_API}/execution/{exec_id}/results",
                headers=_headers(),
                params={"limit": limit, "offset": offset},
                timeout=120,
            )
            r.raise_for_status()
            j = r.json()
            rows = (j.get("result") or {}).get("rows") or []
            if not rows:
                break
            if not headers_written:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                headers_written = True
            for row in rows:
                writer.writerow(row)
            total += len(rows)
            print(f"  [dune] downloaded {total} rows (offset {offset})")
            if len(rows) < limit:
                break
            offset += limit
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--hours", type=int, default=None, help="只抓 date 00:00 起的 N 小时（可选）")
    ap.add_argument("--out-suffix", default="", help="输出 CSV 文件名后缀，避免覆盖")
    ap.add_argument("--include-humidifi", action="store_true", help="把 humidifi 也加进 project 过滤，便于 cross-check / 做 tx_id 源")
    ap.add_argument("--write-humidifi-sigs", default=None, help="把 project=humidifi 的 tx_id 去重后写到指定文件（每行一个签名）")
    args = ap.parse_args()
    if not DUNE_API_KEY:
        print("DUNE_API_KEY not set; aborting.", file=sys.stderr)
        sys.exit(1)
    paths = data_paths(args.date)
    start_ts, end_ts = utc_day_bounds(args.date)
    if args.hours:
        end_ts = start_ts + args.hours * 3600
    if args.out_suffix:
        p = paths["orca_raydium_csv"]
        paths["orca_raydium_csv"] = p.with_name(p.stem + args.out_suffix + p.suffix)
    projects = ("orca", "whirlpool", "raydium")
    if args.include_humidifi:
        projects = projects + ("humidifi",)
    projects_sql = "(" + ", ".join(f"'{p}'" for p in projects) + ")"
    sql = SQL_TEMPLATE.format(start_ts=start_ts, end_ts=end_ts, projects=projects_sql)
    print(f"[dune] window {datetime.fromtimestamp(start_ts, tz=timezone.utc)} ~ {datetime.fromtimestamp(end_ts, tz=timezone.utc)}")
    with httpx.Client() as client:
        qid = create_query(sql, client)
        print(f"[dune] query_id={qid}")
        exec_id = execute_query(qid, client)
        print(f"[dune] exec_id={exec_id}")
        poll_status(exec_id, client)
        n = download_results(exec_id, client, paths["orca_raydium_csv"])
    print(f"[dune] {n} rows → {paths['orca_raydium_csv']}")

    if args.write_humidifi_sigs:
        import csv as _csv
        seen: set[str] = set()
        with open(paths["orca_raydium_csv"], newline="") as f:
            r = _csv.DictReader(f)
            for row in r:
                if row.get("project") == "humidifi":
                    sig = row.get("tx_id")
                    if sig:
                        seen.add(sig)
        out = args.write_humidifi_sigs
        with open(out, "w") as f:
            for s in sorted(seen):
                f.write(s + "\n")
        print(f"[dune] wrote {len(seen)} humidifi tx_ids → {out}")


if __name__ == "__main__":
    main()
