"""用 DuckDB 把四份数据合并成分析宽表。

输入：
  data/processed/humidifi_oracle_updates_YYYYMMDD.csv
  data/processed/humidifi_swaps_YYYYMMDD.csv
  data/raw/orca_raydium_sol_usdc_YYYYMMDD.csv
  data/raw/binance_sol_usdc_YYYYMMDD.csv

输出：data/processed/analysis_table_YYYYMMDD.csv

合并规则：
  - 每笔 swap 按 floor(block_time) 同秒 join Binance close（cex_mid）。
  - HumidiFi swap 再 asof join 最近一次 *状态相同* 的 oracle update，取 delta_t_ms。
  - quote_deviation = abs(price - cex_mid) / cex_mid。
"""
from __future__ import annotations

import argparse

import duckdb

from .config import data_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--start-ts", type=int, default=None, help="UTC 秒，起始时间窗（可选，过滤所有表）")
    ap.add_argument("--end-ts", type=int, default=None, help="UTC 秒，结束时间窗（可选）")
    args = ap.parse_args()
    paths = data_paths(args.date)
    window_pred = ""
    if args.start_ts is not None and args.end_ts is not None:
        window_pred = f" AND block_time >= {args.start_ts} AND block_time < {args.end_ts}"
        print(f"[build] window filter: [{args.start_ts}, {args.end_ts})")

    con = duckdb.connect()
    # Binance: close per 1s bucket
    con.execute(
        """
        CREATE TABLE binance AS
        SELECT CAST(open_time_ms / 1000 AS BIGINT) AS ts_sec,
               CAST(close AS DOUBLE) AS cex_mid
        FROM read_csv_auto(?, HEADER=TRUE)
        """,
        [str(paths["binance_csv"])],
    )

    con.execute(
        f"""
        CREATE TABLE hmd_oracle AS
        SELECT signature, slot, block_time, status, state_account
        FROM read_csv_auto(?, HEADER=TRUE)
        WHERE TRUE{window_pred}
        """,
        [str(paths["oracle_csv"])],
    )

    # 注意：dust 路径（base<1e-3 SOL）通常是 aggregator 多跳内部腿，价格不具参考意义 → 置 NULL
    con.execute(
        f"""
        CREATE TABLE hmd_swap AS
        SELECT signature, slot, block_time, state_account, side,
               base_amount_sol, quote_mint, quote_amount_usdc,
               CASE WHEN base_amount_sol >= 0.001 THEN price_usdc_per_sol ELSE NULL END AS price_usdc_per_sol,
               CASE WHEN base_amount_sol >= 0.001 THEN amount_usd ELSE NULL END AS amount_usd,
               (base_amount_sol < 0.001) AS is_dust,
               cu_consumed, fee
        FROM read_csv_auto(?, HEADER=TRUE)
        WHERE TRUE{window_pred}
        """,
        [str(paths["swap_csv"])],
    )

    has_dune = paths["orca_raydium_csv"].exists() and paths["orca_raydium_csv"].stat().st_size > 0
    if has_dune:
        con.execute(
            f"""
            CREATE TABLE dex_swap AS
            WITH src AS (
              SELECT CAST(epoch(CAST(block_time AS TIMESTAMP)) AS BIGINT) AS block_time,
                     CAST(block_slot AS BIGINT) AS slot,
                     tx_id AS signature,
                     project AS amm_type,
                     token_bought_symbol, token_sold_symbol,
                     CAST(token_bought_amount AS DOUBLE) AS token_bought_amount,
                     CAST(token_sold_amount   AS DOUBLE) AS token_sold_amount,
                     CAST(amount_usd          AS DOUBLE) AS amount_usd
              FROM read_csv_auto(?, HEADER=TRUE)
            )
            SELECT block_time, slot, signature, amm_type,
                   CASE WHEN token_bought_symbol IN ('SOL','WSOL') THEN 'buy_sol' ELSE 'sell_sol' END AS side,
                   CASE WHEN token_bought_symbol IN ('SOL','WSOL') THEN token_bought_amount ELSE token_sold_amount END AS base_amount_sol,
                   CASE WHEN token_bought_symbol IN ('SOL','WSOL') THEN token_sold_amount   ELSE token_bought_amount END AS quote_amount_usdc,
                   amount_usd
            FROM src
            WHERE amount_usd IS NOT NULL
              AND amm_type IN ('orca','whirlpool','raydium')  -- humidifi 走 Helius 解析版，避免双计
              {window_pred}
            """,
            [str(paths["orca_raydium_csv"])],
        )
    else:
        con.execute("CREATE TABLE dex_swap AS SELECT NULL::BIGINT AS block_time, NULL::BIGINT AS slot, NULL::VARCHAR AS signature, NULL::VARCHAR AS amm_type, NULL::VARCHAR AS side, NULL::DOUBLE AS base_amount_sol, NULL::DOUBLE AS quote_amount_usdc, NULL::DOUBLE AS amount_usd WHERE 1=0")
    print(f"[build] has_dune={has_dune} dex_swap_rows={con.execute('SELECT count(*) FROM dex_swap').fetchone()[0]}")

    con.execute(
        """
        CREATE TABLE swaps_all AS
        SELECT
          'humidifi' AS amm_type,
          signature, slot, block_time, state_account,
          side, base_amount_sol, quote_mint, quote_amount_usdc, price_usdc_per_sol, amount_usd,
          is_dust, cu_consumed, fee
        FROM hmd_swap
        UNION ALL BY NAME
        SELECT
          amm_type,
          signature,
          slot,
          block_time,
          NULL::VARCHAR AS state_account,
          side,
          base_amount_sol,
          'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v' AS quote_mint,
          quote_amount_usdc,
          CASE WHEN base_amount_sol > 0 THEN quote_amount_usdc / base_amount_sol ELSE NULL END AS price_usdc_per_sol,
          amount_usd,
          FALSE AS is_dust,
          NULL::BIGINT AS cu_consumed,
          NULL::BIGINT AS fee
        FROM dex_swap
        """
    )

    # Attach cex_mid + delta_t_ms
    con.execute(
        """
        CREATE TABLE analysis AS
        WITH swap_enriched AS (
          SELECT s.*,
                 b.cex_mid,
                 CASE WHEN b.cex_mid IS NOT NULL AND s.price_usdc_per_sol IS NOT NULL
                      THEN abs(s.price_usdc_per_sol - b.cex_mid) / b.cex_mid END AS quote_deviation
          FROM swaps_all s
          LEFT JOIN binance b ON b.ts_sec = s.block_time
        ),
        hmd_with_prev AS (
          SELECT s.*,
                 (
                   SELECT max(o.block_time)
                   FROM hmd_oracle o
                   WHERE o.state_account = s.state_account
                     AND o.block_time <= s.block_time
                 ) AS prev_oracle_time,
                 (
                   SELECT max(o.block_time)
                   FROM hmd_oracle o
                   WHERE o.state_account = s.state_account
                     AND o.block_time <= s.block_time
                     AND o.status = 'success'
                 ) AS prev_oracle_success_time,
                 (
                   SELECT max(o.slot)
                   FROM hmd_oracle o
                   WHERE o.state_account = s.state_account
                     AND o.slot <= s.slot
                 ) AS prev_oracle_slot,
                 (
                   SELECT max(o.slot)
                   FROM hmd_oracle o
                   WHERE o.state_account = s.state_account
                     AND o.slot <= s.slot
                     AND o.status = 'success'
                 ) AS prev_oracle_success_slot
          FROM swap_enriched s
          WHERE s.amm_type = 'humidifi'
        )
        SELECT s.amm_type, s.signature, s.slot, s.block_time, s.side,
               s.base_amount_sol, s.quote_mint, s.quote_amount_usdc, s.price_usdc_per_sol,
               s.amount_usd, s.cex_mid, s.quote_deviation, s.is_dust,
               s.cu_consumed, s.fee,
               NULL::BIGINT AS prev_oracle_time,
               NULL::BIGINT AS prev_oracle_success_time,
               NULL::BIGINT AS delta_t_sec,
               NULL::BIGINT AS prev_oracle_slot,
               NULL::BIGINT AS prev_oracle_success_slot,
               NULL::BIGINT AS delta_t_slot
        FROM swap_enriched s
        WHERE s.amm_type <> 'humidifi'
        UNION ALL
        SELECT amm_type, signature, slot, block_time, side,
               base_amount_sol, quote_mint, quote_amount_usdc, price_usdc_per_sol,
               amount_usd, cex_mid, quote_deviation, is_dust,
               cu_consumed, fee,
               prev_oracle_time,
               prev_oracle_success_time,
               (block_time - prev_oracle_success_time) AS delta_t_sec,
               prev_oracle_slot,
               prev_oracle_success_slot,
               (slot - prev_oracle_success_slot) AS delta_t_slot
        FROM hmd_with_prev
        """
    )

    con.execute("COPY analysis TO ? (HEADER, DELIMITER ',')", [str(paths["analysis_csv"])])
    row_count = con.execute("SELECT count(*) FROM analysis").fetchone()[0]
    grp = con.execute(
        """
        SELECT amm_type,
               count(*) AS n,
               sum(CASE WHEN is_dust THEN 1 ELSE 0 END) AS n_dust,
               median(quote_deviation) FILTER (WHERE NOT is_dust) AS median_dev,
               quantile_cont(quote_deviation, 0.9) FILTER (WHERE NOT is_dust) AS p90_dev
        FROM analysis
        GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    print(f"[build] analysis rows: {row_count}")
    print("  amm_type            n      n_dust  median_dev      p90_dev")
    for row in grp:
        print(f"  {row[0]:<15} {row[1]:>8} {row[2]:>8}  {row[3]:<14} {row[4]}")
    print(f"[build] wrote {paths['analysis_csv']}")


if __name__ == "__main__":
    main()
