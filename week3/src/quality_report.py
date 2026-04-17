"""生成数据质量报告 md。

覆盖：
  - 样本量（按 AMM 分组 + HumidiFi status 细分）
  - Revert 比例、tip 分布、CU 分布
  - Binance 秒级同秒匹配率
  - HumidiFi swap 识别率（raw 总 tx / parsed oracle / parsed swap）
  - HumidiFi state_account 归集
  - 时间窗口 & 抓取耗时（由 pilot 运行日志手工补）
"""
from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime, timezone

import duckdb
import pandas as pd

from .config import data_paths, utc_day_bounds


def _fmt_int(x):
    if x is None:
        return "-"
    return f"{int(x):,}"


def _fmt_pct(x):
    if x is None:
        return "-"
    return f"{x*100:.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--raw", action="append", default=None, help="额外 raw jsonl.gz 列表（含 swap 路径、oracle 路径等）；默认只算 paths.humidifi_raw")
    ap.add_argument("--start-ts", type=int, default=None)
    ap.add_argument("--end-ts", type=int, default=None)
    args = ap.parse_args()
    paths = data_paths(args.date)
    start_ts, end_ts = utc_day_bounds(args.date)
    if args.start_ts is not None:
        start_ts = args.start_ts
    if args.end_ts is not None:
        end_ts = args.end_ts

    con = duckdb.connect()

    from pathlib import Path as _P
    raw_candidates = [paths["humidifi_raw"]]
    if args.raw:
        raw_candidates = [_P(p) for p in args.raw]
    raw_candidates = [p for p in raw_candidates if p.exists()]

    total_raw = 0
    missing_raw = 0
    for rp in raw_candidates:
        with gzip.open(rp, "rt") as f:
            for line in f:
                total_raw += 1
                try:
                    if json.loads(line).get("missing"):
                        missing_raw += 1
                except Exception:
                    continue

    # Oracle
    oracle_stats = {"total": 0, "success": 0, "revert": 0}
    state_counts = []
    tip_q = cu_q = None
    if paths["oracle_csv"].exists():
        o = con.execute(f"SELECT * FROM read_csv_auto('{paths['oracle_csv']}', HEADER=TRUE)").df()
        oracle_stats["total"] = len(o)
        oracle_stats["success"] = int((o["status"] == "success").sum())
        oracle_stats["revert"] = int((o["status"] == "revert").sum())
        state_counts = (
            o.groupby("state_account").size().sort_values(ascending=False).head(10).reset_index(name="count")
        )
        tip_q = o["tip_amount"].describe(percentiles=[0.5, 0.9, 0.99]).to_dict()
        cu_q = o["cu_consumed"].describe(percentiles=[0.5, 0.9, 0.99]).to_dict()

    # Swap
    swap_stats = {"total": 0}
    swap_cu_q = swap_fee_q = None
    if paths["swap_csv"].exists():
        s = con.execute(f"SELECT * FROM read_csv_auto('{paths['swap_csv']}', HEADER=TRUE)").df()
        swap_stats["total"] = len(s)
        if len(s):
            swap_cu_q = s["cu_consumed"].describe(percentiles=[0.5, 0.9, 0.99]).to_dict()
            swap_fee_q = s["fee"].describe(percentiles=[0.5, 0.9, 0.99]).to_dict()

    # Dex swap
    dex_stats = {"total": 0, "by_project": {}}
    if paths["orca_raydium_csv"].exists():
        d = con.execute(f"SELECT * FROM read_csv_auto('{paths['orca_raydium_csv']}', HEADER=TRUE)").df()
        dex_stats["total"] = len(d)
        if "project" in d.columns:
            dex_stats["by_project"] = d.groupby("project").size().to_dict()

    # Binance
    bn_stats = {"rows": 0, "coverage_pct": None}
    if paths["binance_csv"].exists():
        b = con.execute(f"SELECT * FROM read_csv_auto('{paths['binance_csv']}', HEADER=TRUE)").df()
        bn_stats["rows"] = len(b)
        bn_stats["coverage_pct"] = bn_stats["rows"] / 86400

    # Analysis (健壮统计：用 median/p90，过滤 dust)
    analysis_stats = {}
    if paths["analysis_csv"].exists():
        a = con.execute(f"SELECT * FROM read_csv_auto('{paths['analysis_csv']}', HEADER=TRUE)").df()
        analysis_stats["total"] = len(a)
        analysis_stats["match_cex"] = int(a["cex_mid"].notna().sum())
        analysis_stats["n_dust"] = int(a.get("is_dust", pd.Series([False]*len(a))).sum()) if "is_dust" in a.columns else 0
        non_dust = a[~a.get("is_dust", pd.Series([False]*len(a))).astype(bool)] if "is_dust" in a.columns else a
        grp = non_dust.dropna(subset=["quote_deviation"]).groupby("amm_type")
        analysis_stats["by_amm"] = {
            name: {
                "n": int(len(g)),
                "med_dev": float(g["quote_deviation"].median()),
                "p90_dev": float(g["quote_deviation"].quantile(0.90)),
            }
            for name, g in grp
        }
        # HumidiFi delta_t 统计
        if "delta_t_sec" in a.columns:
            hmd = a[a["amm_type"] == "humidifi"].dropna(subset=["delta_t_sec"])
            if len(hmd):
                analysis_stats["hmd_delta_t"] = {
                    "n": int(len(hmd)),
                    "median_sec": float(hmd["delta_t_sec"].median()),
                    "p90_sec": float(hmd["delta_t_sec"].quantile(0.90)),
                    "max_sec": float(hmd["delta_t_sec"].max()),
                }

    # Write report
    lines = []
    lines.append(f"# {args.date} · Week 3 · 实验 0 数据准备报告")
    lines.append("")
    lines.append(
        f"**时间窗口（UTC）**：{datetime.fromtimestamp(start_ts, tz=timezone.utc)} ~ {datetime.fromtimestamp(end_ts, tz=timezone.utc)}"
    )
    lines.append("")
    lines.append("## 1. 抓取规模")
    lines.append("")
    lines.append("| 项 | 数量 |")
    lines.append("| --- | --- |")
    lines.append(f"| HumidiFi raw tx（含 revert） | {_fmt_int(total_raw)} |")
    lines.append(f"| HumidiFi raw missing | {_fmt_int(missing_raw)} |")
    lines.append(f"| Parsed oracle_update | {_fmt_int(oracle_stats['total'])} |")
    lines.append(f"| └ success | {_fmt_int(oracle_stats['success'])} |")
    lines.append(f"| └ revert | {_fmt_int(oracle_stats['revert'])} |")
    lines.append(f"| Parsed HumidiFi swap（SOL/USDC） | {_fmt_int(swap_stats['total'])} |")
    lines.append(f"| Orca+Raydium SOL/USDC trades | {_fmt_int(dex_stats['total'])} |")
    for p, n in (dex_stats.get("by_project") or {}).items():
        lines.append(f"| └ {p} | {_fmt_int(n)} |")
    lines.append(f"| Binance 1s klines | {_fmt_int(bn_stats['rows'])}（覆盖 {_fmt_pct(bn_stats['coverage_pct'])}） |")
    if total_raw:
        lines.append(
            f"| Revert ratio | {_fmt_pct(oracle_stats['revert'] / max(total_raw, 1))}（基于 raw tx） |"
        )
        parse_rate = (oracle_stats["total"] + swap_stats["total"]) / max(total_raw - missing_raw, 1)
        lines.append(f"| Parse 覆盖率（oracle+swap）/raw_ok | {_fmt_pct(parse_rate)} |")
    lines.append("")

    lines.append("## 2. HumidiFi 链上行为画像")
    lines.append("")
    if isinstance(state_counts, pd.DataFrame) and len(state_counts):
        lines.append("**Top 10 state_account（按 oracle update 次数）**")
        lines.append("")
        lines.append("| state_account | count |")
        lines.append("| --- | --- |")
        for _, row in state_counts.iterrows():
            lines.append(f"| `{row['state_account']}` | {_fmt_int(row['count'])} |")
        lines.append("")
    if cu_q:
        lines.append(
            f"- Oracle CU 消耗：mean={cu_q.get('mean'):.1f}  p50={cu_q.get('50%'):.0f}  p90={cu_q.get('90%'):.0f}  p99={cu_q.get('99%'):.0f}  max={cu_q.get('max'):.0f}"
        )
    if tip_q:
        lines.append(
            f"- Oracle Jito tip（lamports）：mean={tip_q.get('mean'):.0f}  p50={tip_q.get('50%'):.0f}  p90={tip_q.get('90%'):.0f}  p99={tip_q.get('99%'):.0f}  max={tip_q.get('max'):.0f}"
        )
    if swap_cu_q:
        lines.append(
            f"- Swap CU 消耗：mean={swap_cu_q.get('mean'):.0f}  p50={swap_cu_q.get('50%'):.0f}  p90={swap_cu_q.get('90%'):.0f}  p99={swap_cu_q.get('99%'):.0f}  max={swap_cu_q.get('max'):.0f}"
        )
    if swap_fee_q:
        lines.append(
            f"- Swap fee（lamports）：mean={swap_fee_q.get('mean'):.0f}  p50={swap_fee_q.get('50%'):.0f}  p90={swap_fee_q.get('90%'):.0f}  p99={swap_fee_q.get('99%'):.0f}  max={swap_fee_q.get('max'):.0f}"
        )
    lines.append("")

    lines.append("## 3. 分析宽表（analysis_table）")
    lines.append("")
    if analysis_stats:
        lines.append(f"- 总行数：{_fmt_int(analysis_stats['total'])}")
        cex_match = analysis_stats.get("match_cex")
        if cex_match is not None and analysis_stats["total"]:
            lines.append(
                f"- Binance 同秒 cex_mid 匹配率：{_fmt_pct(cex_match / analysis_stats['total'])}"
            )
        if analysis_stats.get("n_dust"):
            lines.append(f"- HumidiFi dust 路径（base_amount_sol < 0.001 SOL，aggregator 多跳内部腿）：{_fmt_int(analysis_stats['n_dust'])} → `price_usdc_per_sol` 置 NULL 不计入偏离统计")
        by_amm = analysis_stats.get("by_amm") or {}
        if by_amm:
            lines.append("")
            lines.append("| amm_type | n (非 dust) | median_quote_deviation | p90_quote_deviation |")
            lines.append("| --- | --- | --- | --- |")
            for k, v in by_amm.items():
                lines.append(
                    f"| {k} | {_fmt_int(v['n'])} | {v['med_dev']:.6%} | {v['p90_dev']:.6%} |"
                )
        dt = analysis_stats.get("hmd_delta_t") or {}
        if dt:
            lines.append("")
            lines.append(f"- HumidiFi `delta_t_sec`（swap vs 最近一次同池 oracle_update）：n={_fmt_int(dt['n'])}  median={dt['median_sec']:.1f}s  p90={dt['p90_sec']:.1f}s  max={dt['max_sec']:.1f}s")
    else:
        lines.append("（analysis_table 尚未生成）")
    lines.append("")

    lines.append("## 4. 数据质量说明")
    lines.append("")
    lines.append("### 4.1 混合抓取管线（Dune + Helius）")
    lines.append("- **HumidiFi swap（细节）**：用 Dune `dex_solana.trades` 拿到 SOL/USDC 成交签名清单，再以 Helius `getTransaction` 逐条拉原始数据，解析出 `state_account` / `side` / `base_amount_sol` / `quote_mint` / `price_usdc_per_sol` / `cu_consumed` / `fee`。Dune 本身不含这些链上细节，但提供了完整的成交签名列表，避免了对 HumidiFi 程序 PDA 做全量翻页。")
    lines.append("- **HumidiFi oracle_update（方案 B）**：按池子 PDA（`--pool-accounts`）调用 `getSignaturesForAddress` 翻页，对该池 1h 内所有相关 tx 一律拉细节；本 pilot 覆盖 top 1 热门池 `8sKQHfjNhvmA...`，共 59,653 条 tx，内含 28,146 次 oracle_update（7.82/s 更新节奏）。")
    lines.append("- **Orca/Whirlpool/Raydium（对照组）**：直接用 Dune 的 `dex_solana.trades` 已解码宽表（含 `amount_usd`、`token_bought/sold_*`），不走 Helius RPC。")
    lines.append("- **Binance 1s K 线**：HTTP `/api/v3/klines` 按日拉取，close 价当 `cex_mid`。")
    lines.append("")
    lines.append("### 4.2 已知限制")
    lines.append("- Solana `block_time` 仅秒级，Binance 1s 分桶 ≥ 秒级对齐为理论上限；`delta_t_sec` 几乎都落在 0~1s，说明 HumidiFi swap 与 oracle_update 在同一秒或相邻秒发生，已达数据精度极限。")
    lines.append("- **多跳 aggregator 的 dust 腿**：HumidiFi 在 Jupiter 等聚合器中做一个小池子（通常只是多跳路径的一腿），解析出的 `base_amount_sol` 可能 < 1e-3 SOL，这类 `price` 不具参考价值；统计时按 `is_dust=TRUE` 标记并从偏离率均值/分位计算中剔除。")
    lines.append("- **方案 B 只覆盖 top 1 池**：其它 ~50 个活跃池的 oracle_update 只有 77s 窗口样本（约 315 条），所以 `delta_t_sec` 主要来自 `8sKQHf...` 池的 swap。继续扩大可通过 `--pool-accounts` 逗号分隔多个池。")
    lines.append("- **Dune 摄取延迟**：`dex_solana.trades` 最新一天通常延迟 6–24h 后齐，pilot 选 2026-04-15 即为此；最新日用 Dune 跑会 0 行。")
    lines.append("- **revert 的 swap 目前丢弃**：`tx.err ≠ None` 的 swap 未解析；如果需要统计 revert 分布，可扩展 parser。")
    lines.append("")
    lines.append("### 4.3 落地产物")
    lines.append("- 原始数据：`data/raw/humidifi_txs_20260415_swap.jsonl.gz`（Dune 路径 15,002 tx）、`humidifi_txs_20260415_oracle.jsonl.gz`（池 PDA 路径 59,653 tx）、`humidifi_txs_20260415.jsonl.gz`（历史 77s 样本 4,976 tx）、`orca_raydium_sol_usdc_20260415*.csv`、`binance_sol_usdc_20260415.csv`。")
    lines.append("- 解析 CSV：`humidifi_oracle_updates_20260415.csv`、`humidifi_swaps_20260415.csv`。")
    lines.append("- 分析宽表：`analysis_table_20260415.csv`（含 `cex_mid`、`quote_deviation`、`is_dust`、`delta_t_sec`）。")
    lines.append("")

    paths["report_md"].write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {paths['report_md']}")


if __name__ == "__main__":
    main()
