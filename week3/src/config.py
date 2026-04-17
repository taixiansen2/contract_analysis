"""Week 3 · 实验 0 常量。"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

WEEK3_DIR = Path(__file__).resolve().parent.parent
load_dotenv(WEEK3_DIR / ".env")

HELIUS_RPC_URL = os.environ.get(
    "HELIUS_RPC_URL",
    "https://mainnet.helius-rpc.com/?api-key=281ca7a6-5b58-4818-b226-53c63164e699",
)
DUNE_API_KEY = os.environ.get("DUNE_API_KEY", "")

HUMIDIFI_PROGRAM = "9H6tua7jkLhdm3w8BvgpTn5LZNU7g4ZynDmCiNN3q6Rp"

# Week 2 逆向结论：3 账户 + 65 字节输入 = state_update (oracle update)
ORACLE_UPDATE_DATA_LEN = 65
ORACLE_UPDATE_ACCOUNT_COUNT = 3

# SPL token mints
SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

# Jito tip 账户（8 个，SOL 转账到这些地址视为 bundle tip）
JITO_TIP_ACCOUNTS = {
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
    "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
    "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
    "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDe9B",
    "ADuUkR4vqLUMWXxW9gh6D6L8pivKeVQqvpQfWdkGjLCQ",
    "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
    "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
}


def utc_day_bounds(date_str: str) -> tuple[int, int]:
    """Return (start_ts, end_ts) unix seconds for full UTC day."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start = int(dt.timestamp())
    end = start + 86400
    return start, end


def data_paths(date_str: str) -> dict[str, Path]:
    d = date_str.replace("-", "")
    raw = WEEK3_DIR / "data" / "raw"
    proc = WEEK3_DIR / "data" / "processed"
    reports = WEEK3_DIR / "reports"
    raw.mkdir(parents=True, exist_ok=True)
    proc.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    return {
        "humidifi_raw": raw / f"humidifi_txs_{d}.jsonl.gz",
        "humidifi_failed": raw / f"humidifi_failed_{d}.jsonl",
        "humidifi_sigs": raw / f"humidifi_sigs_{d}.jsonl",
        "oracle_csv": proc / f"humidifi_oracle_updates_{d}.csv",
        "swap_csv": proc / f"humidifi_swaps_{d}.csv",
        "orca_raydium_csv": raw / f"orca_raydium_sol_usdc_{d}.csv",
        "binance_csv": raw / f"binance_sol_usdc_{d}.csv",
        "analysis_csv": proc / f"analysis_table_{d}.csv",
        "report_md": reports / f"{d}-Week3实验0数据准备报告.md",
    }
