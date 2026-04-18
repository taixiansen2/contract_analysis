# Week 3 · HumidiFi 微观结构分析

**研究问题**：Solana 的微观结构特性（低 CU、400ms slot、无条件重推报价）如何支撑 Prop AMM（以 HumidiFi 为例）给出优于 CEX / 传统 AMM 的报价？

整个 Week 3 分成一个**数据准备实验（实验 0）+ 五个分析实验（1-5）**。本目录包含数据抓取 / 解析 / 对齐管线、每个实验的专用分析脚本、产出报告、以及所有数据产物。

---

## 当前进度（2026-04-18）

（实验 0 / 1 / 3 / 4 / 5 已完成，实验 2 暂缓待重抓 logMessages。）


| 实验    | 主题           | 状态       | 样本窗口                                 | 核心产出                                                                                                                                          |
| ----- | ------------ | -------- | ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **0** | 数据准备 / 宽表    | ✅ 完成     | 2026-04-15 00:00-01:00 UTC（1h pilot） | `data/processed/analysis_table_20260415.csv`；`[reports/20260415-Week3实验0数据准备报告.md](reports/20260415-Week3实验0数据准备报告.md)`                       |
| **1** | 改价频率 vs 报价质量 | ✅ 完成     | 同上                                   | `reports/tables/exp1_*.csv` + `reports/figures/exp1_*.png`；`[reports/20260415-Week3实验1-改价频率与报价质量.md](reports/20260415-Week3实验1-改价频率与报价质量.md)` |
| **2** | 改价成本与 CU 效率  | ⏸ **暂缓** | -                                    | 已定位 `_sum_tip` 只扫了 inner 指令导致 tip 全 0 的 bug；要做单指令级 CU 必须在 `slim_tx` 里保留 `logMessages` 后重抓 oracle，预估约 2-3h，待恢复时继续。详见本 README "TODO" 一节。        |
| **3** | 改价响应速度与价格跟踪  | ✅ 完成     | 同上                                   | `reports/tables/exp3_*.csv` + `reports/figures/exp3_*.png`；`[reports/20260415-Week3实验3-响应速度与价格跟踪.md](reports/20260415-Week3实验3-响应速度与价格跟踪.md)` |
| **4** | 量价关系与动态定价    | ✅ 完成     | 同上                                   | `reports/tables/exp4_*.csv` + `reports/figures/exp4_*.png`；`[reports/20260415-Week3实验4-量价关系.md](reports/20260415-Week3实验4-量价关系.md)`           |
| **5** | Revert 与逆向选择 | ✅ 完成     | 同上                                   | `reports/tables/exp5_*.csv` + `reports/figures/exp5_*.png`；`[reports/20260415-Week3实验5-revert与逆向选择.md](reports/20260415-Week3实验5-revert与逆向选择.md)` |


关键发现预览：

- 实验 1：HumidiFi swap 的 Δt（距上次成功 oracle）中位数仅数十毫秒，报价偏差随 Δt 线性上升；Whirlpool / Raydium 因没有主动改价，其 Δt 依赖被动成交密度。
- 实验 3：对 ≥10 bps 的 Binance 突变，HumidiFi oracle update 响应延迟 median = 0 s（≤ 1s 时间分辨率），即"报价无条件重推、不等套利者交易"。
- 实验 4：HumidiFi 是三者中唯一 `dev_bps ~ log10(amount_usd)` 斜率为负的 AMM（−2.26 bps/decade，p=1e-15）；在 $1k–$10k 档 median 仅 0.78 bps，比 Whirlpool 低约 30%、比 Raydium 低约 55%，与"动态定价、越大单越便宜"相符；Whirlpool/Raydium 斜率为正，符合恒定乘积 AMM 理论预测。
- 实验 5：在本 1h 低波动样本中，"前序 oracle revert → swap 更毒"假设**未得支持**：A/B 两组 `dev_bps`、毒性率、taker profit 的 Welch t 检验 p 值均在 0.17–0.70 之间；B 组相较 A 组的增量 informed-rent 仅 ~$2.6 / 1h。推测原因：多数 revert 属于协议内部状态校验而非被抢跑，且陈旧窗口通常只有 1–2 slot。需扩到 7 天并挑高波动小时复测。

---

## 目录结构

```
week3/
├── src/
│   ├── config.py              # HELIUS/DUNE env、HumidiFi program id、Jito tip accounts、data_paths()
│   ├── fetch_humidifi.py      # Helius RPC 异步抓取（getSignaturesForAddress + getTransaction）
│   ├── parse_humidifi.py      # 解析 raw tx → oracle_update / swap CSV
│   ├── fetch_dune_trades.py   # Dune dex_solana.trades：Orca / Whirlpool / Raydium / (option) HumidiFi
│   ├── fetch_binance.py       # Binance /api/v3/klines interval=1s
│   ├── build_table.py         # DuckDB 秒级 / slot 级对齐合并成分析宽表
│   ├── quality_report.py      # 实验 0 数据质量报告生成
│   ├── exp1_quote_vs_dt.py    # 实验 1：分桶统计 + OLS + 散点/柱图
│   ├── exp3_response_speed.py # 实验 3：事件检测 + 响应/收敛延迟 + 箱线/案例图
│   ├── exp4_amount_vs_dev.py  # 实验 4：分档统计 + log 回归 + 量价拟合图
│   └── exp5_revert_toxicity.py # 实验 5：A/B 分组 + Welch t 检验 + 毒性率/损失估算
├── data/
│   ├── raw/                   # *.jsonl.gz / *.csv / *.jsonl / *.log（抓取原件；大文件已被 .gitignore 排除）
│   └── processed/             # *.csv（清洗 / 对齐后的表）
├── reports/
│   ├── 20260415-Week3实验0数据准备报告.md
│   ├── 20260415-Week3实验1-改价频率与报价质量.md
│   ├── 20260415-Week3实验3-响应速度与价格跟踪.md
│   ├── 20260415-Week3实验4-量价关系.md
│   ├── 20260415-Week3实验5-revert与逆向选择.md
│   ├── figures/               # 各实验产出的 PNG（exp1/exp3/exp4/exp5_*.png）
│   └── tables/                # 各实验产出的 CSV（exp1/exp3/exp4/exp5_*.csv）
├── .env.example               # 填 HELIUS_RPC_URL / DUNE_API_KEY
├── requirements.txt
└── README.md (本文件)
```

---

## 环境与依赖

```bash
cd week3
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 在 .env 填入 HELIUS_RPC_URL 和 DUNE_API_KEY
```

`.venv/` 与 `data/raw/*.jsonl.gz` / `*.jsonl` / `_*.log` 不进 git。CSV 保留在仓库，便于直接复现报告。

---

## 一天试点：完整端到端

目前 pilot 选的是 2026-04-15 00:00-01:00 UTC 的 1h 窗口。如需跑整天，直接跑下面的序列即可（`date` 形如 `2026-04-15`，覆盖整 UTC 日）：

```bash
source .venv/bin/activate

# 1. Orca/Whirlpool/Raydium + 可选 HumidiFi tx_id 源（Dune）
python -m week3.src.fetch_dune_trades --date 2026-04-15 \
    --include-humidifi \
    --write-humidifi-sigs data/raw/humidifi_dune_sigs_20260415.txt

# 2. HumidiFi Helius 抓取
#    (a) 按 Dune tx_id 拉 swap 细节（推荐：Dune 覆盖更全）
python -m week3.src.fetch_humidifi --date 2026-04-15 \
    --from-sigs-file data/raw/humidifi_dune_sigs_20260415.txt \
    --out-suffix _swap
#    (b) 按 top pool PDA 拉 oracle update（方案 B）
python -m week3.src.fetch_humidifi --date 2026-04-15 \
    --pool-accounts 8sKQHfjNhvmAw94PhfvfMcytmqW6jmxvwieYyzXCCPu \
    --skip-sigs \
    --out-suffix _oracle

# 3. 解析（读两个 raw，自动去重）
python -m week3.src.parse_humidifi --date 2026-04-15 \
    --raw data/raw/humidifi_txs_20260415_oracle.jsonl.gz \
    --raw data/raw/humidifi_txs_20260415_swap.jsonl.gz

# 4. Binance 逐秒 K 线
python -m week3.src.fetch_binance --date 2026-04-15

# 5. 合并到分析宽表
python -m week3.src.build_table --date 2026-04-15

# 6. 数据质量报告
python -m week3.src.quality_report --date 2026-04-15

# 7. 实验 1 / 3 / 4 / 5
python -m week3.src.exp1_quote_vs_dt --date 2026-04-15
python -m week3.src.exp3_response_speed --date 2026-04-15
python -m week3.src.exp4_amount_vs_dev --date 2026-04-15
python -m week3.src.exp5_revert_toxicity --date 2026-04-15
```

---

## 如何扩到 7 天

目标：覆盖 2026-04-11 … 2026-04-17（或任一连续 7 天，建议至少含 1 个高波动日让实验 3 有意义的突变事件）。

### 1. 量级预估


| 项                                                     | 1 天量级                | 7 天量级     | 瓶颈                                                                      |
| ----------------------------------------------------- | -------------------- | --------- | ----------------------------------------------------------------------- |
| HumidiFi oracle update                                | ~1.4M 笔 / top 1 pool | ~10M      | Helius RPC；`fetch_humidifi.py` 单调用 ~8 rps ⇒ 全天约 **2-3h**，7 天 **14-20h** |
| HumidiFi swap (走 Dune tx_id)                          | ~0.4M 笔              | ~3M       | 同上；7 天 **8-12h**                                                        |
| Dune trades (Orca/Whirlpool/Raydium + HumidiFi tx_id) | 1 次查询                | 7 次查询     | Dune 免费档 query credit 有限；每次查询最多返回 250k 行                                |
| Binance 1s K 线                                        | 86,400 行             | 604,800 行 | 公开 API，分钟级限流，单日几分钟                                                      |


### 2. 建议流程（逐日循环 + 多日合并）

```bash
# 每日一次 pilot（并行友好：Binance / Dune / Helius 可同时跑）
for D in 2026-04-11 2026-04-12 2026-04-13 2026-04-14 2026-04-15 2026-04-16 2026-04-17; do
  python -m week3.src.fetch_dune_trades --date "$D" \
      --include-humidifi \
      --write-humidifi-sigs "data/raw/humidifi_dune_sigs_${D//-/}.txt"

  python -m week3.src.fetch_humidifi --date "$D" \
      --from-sigs-file "data/raw/humidifi_dune_sigs_${D//-/}.txt" \
      --out-suffix _swap

  python -m week3.src.fetch_humidifi --date "$D" \
      --pool-accounts 8sKQHfjNhvmAw94PhfvfMcytmqW6jmxvwieYyzXCCPu \
      --skip-sigs --out-suffix _oracle

  python -m week3.src.parse_humidifi --date "$D" \
      --raw "data/raw/humidifi_txs_${D//-/}_oracle.jsonl.gz" \
      --raw "data/raw/humidifi_txs_${D//-/}_swap.jsonl.gz"

  python -m week3.src.fetch_binance --date "$D"
  python -m week3.src.build_table --date "$D"
  python -m week3.src.quality_report --date "$D"
done
```

跑完后每天各有一套 `analysis_table_YYYYMMDD.csv`、`humidifi_oracle_updates_YYYYMMDD.csv`、`humidifi_swaps_YYYYMMDD.csv`。

### 3. 多日合并做实验 1 / 3

现阶段实验脚本只消费单日 `--date`。扩到 7 天后，把 7 份 CSV `pd.concat` 即可：

```python
import pandas as pd, glob
df = pd.concat(
    (pd.read_csv(p) for p in sorted(glob.glob("data/processed/analysis_table_202604*.csv"))),
    ignore_index=True,
)
```

各实验脚本中的 `load_and_prepare(date)` / `paths_for(date)` 可以改成接受一个 "日期列表" 或直接拼好 CSV 后保存成 `analysis_table_weekly.csv`，再传 `--date weekly`。

### 4. 扩到 7 天要盯的坑

- **Helius RPC 限流 & 并发**：Week 3 pilot 已经遇到过 Job A / Job B 并行导致 `RuntimeError: unreachable`，7 天内要**严格串行**（一天抓完再抓下一天），并把 `fetch_humidifi.py` 的 `MAX_CONCURRENCY` 留在 8 甚至降到 6。
- **Dune ingestion 延迟**：最近 15 分钟内的数据可能还没落到 `dex_solana.trades`。跑当天的 UTC 日要等到 UTC 次日凌晨以后。
- **Pool PDA 覆盖**：方案 B 目前只抓 top 1 pool (`8sKQHfjNhvmAw94PhfvfMcytmqW6jmxvwieYyzXCCPu`) 的 oracle。扩到多池需在 `--pool-accounts` 里用逗号拼接，抓取量线性上升。
- **磁盘**：1 天 oracle raw jsonl.gz ≈ 50-70 MB，7 天 ≈ 350-500 MB；processed CSV 一天 ≈ 20 MB，7 天 ≈ 140 MB。
- **参数 window**：`build_table.py` 支持 `--start-ts / --end-ts` 做窗口切片；默认是整个 UTC 日。
- **高波动日的挑选**：推荐在整合 Binance 数据后，用 `build_table` 生成的宽表找到 `max 5s |Δprice|/price` 最大的那天做实验 3 的主 case。

---

## TODO（下次恢复时优先）

1. **实验 2（改价成本）**：
  - 修 `parse_humidifi.py` 的 `_sum_tip`：同时扫 **outer + inner** 指令（之前只扫了 inner 导致 tip 全 0）
  - 在 `swap_csv` 中新增 `tip_amount` 列
  - `fetch_humidifi.py` 的 `slim_tx` 保留 `meta.logMessages`
  - 按现有 oracle sig 列表重抓一次（只 oracle，约 2-3h），在 `parse_humidifi.py` 中解析 `Program 9H6tua... consumed X of Y compute units` 得到 **单指令级 CU (`cu_program`)**
  - 新建 `week3/src/exp2_cost_analysis.py`：CU 分布 / 单次成本 / 24h 预算 / Tip-per-CU / ETH 反事实 / SolFi-TesseraV 引用对比
2. **扩窗到 7 天**：整合 7 天数据后重跑实验 1 / 3 / 4 / 5。
  - 实验 4 的 `>10k` 档（HumidiFi n=3）届时应能补到 ~20 条，做一次稳健性复跑。
  - 实验 5 B 组样本届时应能 ≥ 2–3k，且覆盖至少 1 个高波动小时；这次 p ∈ [0.17, 0.70] 的"未拒绝 H0"应该能真正检验。
  - 并考虑引入多 pool PDA 覆盖。
3. **实验 5 深化**（可选）：按 `oracle_updates.err_code` 细分 revert 根因，区分"协议状态校验 revert" vs "被抢跑 revert"，仅后者才期望表现出 stale-quote / toxic-flow 效应。

---

## 参考链接

- 实验卡：`[../2026-04 Week 3 实验卡.md](../2026-04%20Week%203%20实验卡.md)`
- HumidiFi 程序：`9H6tua7jkLhdm3w8BvgpTn5LZNU7g4ZynDmCiNN3q6Rp`
- HumidiFi top 1 pool：`8sKQHfjNhvmAw94PhfvfMcytmqW6jmxvwieYyzXCCPu`
- Binance SOL/USDC K 线：`https://api.binance.com/api/v3/klines?symbol=SOLUSDC&interval=1s`

