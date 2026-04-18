"""Week 3 · 实验 5：改价失败（revert）与逆向选择。

研究问题：当 HumidiFi 最近一次 oracle update 是 revert（pool 的价格没更新上），
这条"陈旧报价"下发生的 swap 是否更可能是有毒（informed）订单？

分组口径（per state_account 精确计数）：
  - 对每笔 HumidiFi swap，在它发生的 slot 之前，沿着同一 state_account 的
    oracle_updates 倒推：从最近一次 SUCCESS 到本 swap 之间，累计了多少次 revert。
  - A = consec_reverts == 0（最近一次 oracle 是成功）
  - B = consec_reverts >= 1（最近一次 oracle 是 revert，自上次成功以来累计 >=1 次）
  - B2 = consec_reverts >= 2（连续多次失败的子集）
  - 无法匹配到前序 oracle 的 swap（slot < 最早 oracle）归为 C，不参与 A/B。

说明：此口径比实验卡里 "prev_oracle_time != prev_oracle_success_time" 更严格 —
  因为 block_time 是 1s 粒度，HumidiFi 经常在同一秒里连续 revert+success，按 time
  比较会把"刚刚 revert 但紧接着又 success"的 case 也算到 A 里；用 slot 级（400ms）
  per-pool 计数能精确捕捉 "最近一次 oracle 事件是 revert" 的情形。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .config import data_paths

HORIZONS = [5, 30, 60]  # seconds


def paths_for(date: str) -> dict[str, Path]:
    p = data_paths(date)
    return {
        "analysis": p["analysis_csv"],
        "oracle": p["oracle_csv"],
        "swap": p["swap_csv"],
        "binance": p["binance_csv"],
    }


def load_humidifi_swaps_with_groups(analysis_csv: Path, swap_csv: Path,
                                     oracle_csv: Path, min_usd: float) -> pd.DataFrame:
    """Load HumidiFi swaps, attach state_account, compute per-pool consec_reverts."""
    an = pd.read_csv(analysis_csv)
    hmd = an[an["amm_type"] == "humidifi"].copy()
    sw = pd.read_csv(swap_csv, usecols=["signature", "state_account"])
    hmd = hmd.merge(sw, on="signature", how="left")

    hmd = hmd[(~hmd["is_dust"].astype(bool)) & (hmd["amount_usd"] >= min_usd)].copy()

    ora = pd.read_csv(oracle_csv, usecols=["state_account", "slot", "status"])
    ora = ora.sort_values(["state_account", "slot"], kind="stable").reset_index(drop=True)

    hmd["consec_reverts"] = np.nan
    hmd["slots_since_success"] = np.nan

    for pool, g_swaps in hmd.groupby("state_account", sort=False):
        g_or = ora[ora["state_account"] == pool]
        if len(g_or) == 0:
            continue
        slots = g_or["slot"].to_numpy()
        is_succ = (g_or["status"] == "success").astype(int).to_numpy()
        cs_succ = np.cumsum(is_succ)

        swap_slots = g_swaps["slot"].astype(float).to_numpy()
        idxs = np.searchsorted(slots, swap_slots, side="right") - 1
        consec = np.full(len(swap_slots), np.nan)
        since = np.full(len(swap_slots), np.nan)
        for j, idx in enumerate(idxs):
            if idx < 0:
                continue
            succ_cum_here = cs_succ[idx]
            if succ_cum_here == 0:
                consec[j] = idx + 1  # all reverts so far in this pool window
                since[j] = float(swap_slots[j] - slots[0])
                continue
            last_succ_idx = int(np.searchsorted(cs_succ, succ_cum_here, side="left"))
            consec[j] = idx - last_succ_idx
            since[j] = float(swap_slots[j] - slots[last_succ_idx])
        hmd.loc[g_swaps.index, "consec_reverts"] = consec
        hmd.loc[g_swaps.index, "slots_since_success"] = since

    def label(c: float) -> str:
        if pd.isna(c):
            return "C"
        if c == 0:
            return "A"
        return "B"

    hmd["group"] = hmd["consec_reverts"].apply(label)
    hmd["is_B2"] = hmd["consec_reverts"] >= 2
    hmd["dev_bps"] = hmd["quote_deviation"] * 1e4
    return hmd


def add_future_mid(df: pd.DataFrame, binance_csv: Path,
                   horizons: list[int] = HORIZONS) -> pd.DataFrame:
    bn = pd.read_csv(binance_csv, usecols=["open_time_ms", "close"])
    bn["ts_s"] = (bn["open_time_ms"].astype("int64") // 1000).astype("int64")
    bn = bn[["ts_s", "close"]].sort_values("ts_s").reset_index(drop=True)
    ts_arr = bn["ts_s"].to_numpy()
    close_arr = bn["close"].to_numpy()

    out = df.copy()
    bt = out["block_time"].astype("int64").to_numpy()
    p_s = out["price_usdc_per_sol"].to_numpy()
    side = out["side"].to_numpy()

    for h in horizons:
        target_ts = bt + h
        idx = np.searchsorted(ts_arr, target_ts)
        in_range = (idx < len(ts_arr)) & (ts_arr[np.clip(idx, 0, len(ts_arr) - 1)] == target_ts)
        mid = np.full(len(target_ts), np.nan)
        mid[in_range] = close_arr[idx[in_range]]

        profit_bps = np.full(len(mid), np.nan)
        buy_mask = (side == "buy_sol") & ~np.isnan(mid) & (p_s > 0)
        sell_mask = (side == "sell_sol") & ~np.isnan(mid) & (p_s > 0)
        profit_bps[buy_mask] = (mid[buy_mask] - p_s[buy_mask]) / p_s[buy_mask] * 1e4
        profit_bps[sell_mask] = (p_s[sell_mask] - mid[sell_mask]) / p_s[sell_mask] * 1e4

        out[f"mid_t{h}"] = mid
        out[f"taker_profit_bps_{h}"] = profit_bps
    return out


def group_summary(df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    rows = []
    groups = [
        ("A (consec=0)", df[df["group"] == "A"]),
        ("B (consec>=1)", df[df["group"] == "B"]),
        ("B2 (consec>=2)", df[df["is_B2"]]),
        ("C (no prior oracle)", df[df["group"] == "C"]),
        ("ALL (A+B+B2 union)", df[df["group"].isin(["A", "B"])]),
    ]
    for name, g in groups:
        row = {
            "group": name,
            "n": len(g),
            "mean_amount_usd": float(g["amount_usd"].mean()) if len(g) else None,
            "sum_amount_usd": float(g["amount_usd"].sum()) if len(g) else None,
            "mean_dev_bps": float(g["dev_bps"].mean()) if len(g) else None,
            "median_dev_bps": float(g["dev_bps"].median()) if len(g) else None,
        }
        for h in HORIZONS:
            col = f"taker_profit_bps_{h}"
            vals = g[col].dropna()
            if len(vals) == 0:
                row[f"mean_profit_bps_t{h}"] = None
                row[f"toxic_rate_t{h}"] = None
                row[f"n_with_future_t{h}"] = 0
            else:
                row[f"mean_profit_bps_t{h}"] = float(vals.mean())
                row[f"toxic_rate_t{h}"] = float((vals >= threshold).mean())
                row[f"n_with_future_t{h}"] = int(len(vals))
        rows.append(row)
    return pd.DataFrame(rows)


def welch_ttests(df: pd.DataFrame, main_horizon: int = 30) -> pd.DataFrame:
    A = df[df["group"] == "A"]
    B = df[df["group"] == "B"]
    B2 = df[df["is_B2"]]
    rows = []

    def tt(name: str, col: str, g1: pd.DataFrame, g2: pd.DataFrame, g1_label: str, g2_label: str):
        v1 = g1[col].dropna().to_numpy()
        v2 = g2[col].dropna().to_numpy()
        if len(v1) < 3 or len(v2) < 3:
            rows.append({"test": name, "col": col, "g1": g1_label, "n1": len(v1),
                         "g2": g2_label, "n2": len(v2), "t": None, "p": None,
                         "g1_mean": None, "g2_mean": None, "diff": None})
            return
        res = stats.ttest_ind(v1, v2, equal_var=False, nan_policy="omit")
        rows.append({
            "test": name,
            "col": col,
            "g1": g1_label, "n1": len(v1),
            "g2": g2_label, "n2": len(v2),
            "g1_mean": float(np.mean(v1)),
            "g2_mean": float(np.mean(v2)),
            "diff": float(np.mean(v2) - np.mean(v1)),
            "t": float(res.statistic),
            "p": float(res.pvalue),
        })

    for col in ["dev_bps", f"taker_profit_bps_{main_horizon}"]:
        tt(f"{col}: A vs B", col, A, B, "A", "B")
        tt(f"{col}: A vs B2", col, A, B2, "A", "B2")
    return pd.DataFrame(rows)


def loss_estimate(df: pd.DataFrame, main_horizon: int = 30) -> pd.DataFrame:
    rows = []
    for name, g in [("A", df[df["group"] == "A"]),
                    ("B", df[df["group"] == "B"]),
                    ("B2", df[df["is_B2"]])]:
        if len(g) == 0:
            rows.append({"group": name, "n": 0, "sum_amount_usd": 0,
                         "naive_loss_usd": None, "robust_loss_usd": None,
                         "robust_loss_rate_bps": None})
            continue
        total_usd = float(g["amount_usd"].sum())
        naive = float(g["dev_bps"].mean() / 1e4 * total_usd)
        col = f"taker_profit_bps_{main_horizon}"
        prof = g[col].fillna(0.0).clip(lower=0).to_numpy() / 1e4
        amt = g["amount_usd"].to_numpy()
        robust = float(np.sum(prof * amt))
        robust_rate_bps = (robust / total_usd * 1e4) if total_usd > 0 else None
        rows.append({
            "group": name,
            "n": len(g),
            "sum_amount_usd": total_usd,
            "naive_loss_usd": naive,
            "robust_loss_usd": robust,
            "robust_loss_rate_bps": robust_rate_bps,
        })
    df_rows = pd.DataFrame(rows)

    A_rate = df_rows.loc[df_rows["group"] == "A", "robust_loss_rate_bps"].iloc[0] if "A" in df_rows["group"].values else None
    incr_rows = []
    for name in ["B", "B2"]:
        r = df_rows[df_rows["group"] == name]
        if r.empty or A_rate is None or r["robust_loss_rate_bps"].iloc[0] is None:
            incr_rows.append({"group": name, "incremental_loss_usd": None})
            continue
        rate_diff_bps = r["robust_loss_rate_bps"].iloc[0] - A_rate
        incr = rate_diff_bps / 1e4 * r["sum_amount_usd"].iloc[0]
        incr_rows.append({"group": name, "incremental_loss_usd": float(incr),
                          "rate_excess_over_A_bps": float(rate_diff_bps)})
    incr_df = pd.DataFrame(incr_rows)
    return df_rows.merge(incr_df, on="group", how="left")


def draw_dev_box(df: pd.DataFrame, out_path: Path) -> None:
    groups = [
        ("A\n(consec=0)", df[df["group"] == "A"]["dev_bps"].dropna().to_numpy()),
        ("B\n(consec>=1)", df[df["group"] == "B"]["dev_bps"].dropna().to_numpy()),
        ("B2\n(consec>=2)", df[df["is_B2"]]["dev_bps"].dropna().to_numpy()),
    ]
    clipped = []
    for _, arr in groups:
        if len(arr) == 0:
            clipped.append(arr)
            continue
        cap = np.nanpercentile(arr, 99)
        clipped.append(np.clip(arr, 0, max(cap, 1e-9)))

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(
        clipped, tick_labels=[g[0] for g in groups],
        patch_artist=True, showfliers=False,
    )
    colors = ["#2B6CB0", "#C53030", "#9B2C2C"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    for med in bp["medians"]:
        med.set_color("#1A202C")
    for i, (_, arr) in enumerate(groups, start=1):
        if len(arr):
            ax.text(i, ax.get_ylim()[1] * 0.95,
                    f"n={len(arr)}\nmean={np.mean(arr):.2f}\nmed={np.median(arr):.2f}",
                    ha="center", va="top", fontsize=9)

    ax.set_ylabel("quote_deviation (bps, fliers hidden, clipped @ p99)")
    ax.set_title("Experiment 5: HumidiFi quote_deviation by prev-oracle state "
                 "(2026-04-15 00:00-01:00 UTC)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def draw_toxic_bar(summary: pd.DataFrame, out_path: Path, threshold: float = 2.0) -> None:
    groups_keep = ["A (consec=0)", "B (consec>=1)", "B2 (consec>=2)"]
    s = summary[summary["group"].isin(groups_keep)].set_index("group").reindex(groups_keep)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(HORIZONS))
    width = 0.25
    colors = {"A (consec=0)": "#2B6CB0", "B (consec>=1)": "#C53030", "B2 (consec>=2)": "#9B2C2C"}
    for i, grp in enumerate(groups_keep):
        rates = [s.loc[grp, f"toxic_rate_t{h}"] for h in HORIZONS]
        rates = [0 if r is None or pd.isna(r) else r * 100 for r in rates]
        ns = [s.loc[grp, f"n_with_future_t{h}"] for h in HORIZONS]
        bars = ax.bar(x + (i - 1) * width, rates, width, label=f"{grp}  n0={int(s.loc[grp,'n'])}",
                      color=colors[grp])
        for b, r, n in zip(bars, rates, ns):
            if n > 0:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{r:.1f}%\nn={int(n)}", ha="center", va="bottom",
                        fontsize=7, color=colors[grp])

    ax.set_xticks(x)
    ax.set_xticklabels([f"t+{h}s" for h in HORIZONS])
    ax.set_ylabel(f"toxic trade rate (%) | threshold: taker_profit_bps >= {threshold}")
    ax.set_xlabel("future horizon")
    ax.set_title("Experiment 5: toxic trade rate by prev-oracle state x horizon")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def draw_profit_hist(df: pd.DataFrame, out_path: Path, main_horizon: int = 30,
                     threshold: float = 2.0) -> None:
    col = f"taker_profit_bps_{main_horizon}"
    A = df[df["group"] == "A"][col].dropna().to_numpy()
    B = df[df["group"] == "B"][col].dropna().to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    if len(A) or len(B):
        lo = np.nanpercentile(np.concatenate([A, B]), 1)
        hi = np.nanpercentile(np.concatenate([A, B]), 99)
        edges = np.linspace(lo, hi, 61)
    else:
        edges = np.linspace(-10, 10, 61)
    if len(A):
        ax.hist(np.clip(A, edges[0], edges[-1]), bins=edges, alpha=0.55,
                color="#2B6CB0", label=f"A consec=0 (n={len(A)}, mean={A.mean():.2f})",
                density=True)
    if len(B):
        ax.hist(np.clip(B, edges[0], edges[-1]), bins=edges, alpha=0.55,
                color="#C53030", label=f"B consec>=1 (n={len(B)}, mean={B.mean():.2f})",
                density=True)
    ax.axvline(threshold, color="#1A202C", linestyle="--", linewidth=1,
               label=f"toxic threshold = +{threshold} bps")
    ax.axvline(-threshold, color="#1A202C", linestyle=":", linewidth=1)
    ax.set_xlabel(f"taker_profit_bps at t+{main_horizon}s (clipped to p1-p99)")
    ax.set_ylabel("density")
    ax.set_title(f"Experiment 5: taker_profit distribution at t+{main_horizon}s, A vs B")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="e.g. 20260415")
    ap.add_argument("--min-usd", type=float, default=1.0)
    ap.add_argument("--toxic-threshold-bps", type=float, default=2.0)
    ap.add_argument("--main-horizon", type=int, default=30, choices=HORIZONS)
    args = ap.parse_args()

    p = paths_for(args.date)
    print("[exp5] loading...")
    df = load_humidifi_swaps_with_groups(p["analysis"], p["swap"], p["oracle"],
                                          min_usd=args.min_usd)
    print(f"  total filtered HumidiFi swaps: {len(df)}")
    print(f"  group counts: {df['group'].value_counts().to_dict()}  B2={int(df['is_B2'].sum())}")

    df = add_future_mid(df, p["binance"], horizons=HORIZONS)
    for h in HORIZONS:
        n_ok = df[f"taker_profit_bps_{h}"].notna().sum()
        print(f"  future mid at t+{h}s: {n_ok}/{len(df)} in Binance range")

    reports = data_paths(args.date)["report_md"].parent
    tables_dir = reports / "tables"
    figures_dir = reports / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = group_summary(df, threshold=args.toxic_threshold_bps)
    summary.to_csv(tables_dir / "exp5_group_summary.csv", index=False)
    print(f"[exp5] wrote {tables_dir / 'exp5_group_summary.csv'}")

    ttests = welch_ttests(df, main_horizon=args.main_horizon)
    ttests.to_csv(tables_dir / "exp5_ttest.csv", index=False)
    print(f"[exp5] wrote {tables_dir / 'exp5_ttest.csv'}")

    losses = loss_estimate(df, main_horizon=args.main_horizon)
    losses.to_csv(tables_dir / "exp5_loss_estimate.csv", index=False)
    print(f"[exp5] wrote {tables_dir / 'exp5_loss_estimate.csv'}")

    draw_dev_box(df, figures_dir / "exp5_dev_box.png")
    print(f"[exp5] wrote {figures_dir / 'exp5_dev_box.png'}")

    draw_toxic_bar(summary, figures_dir / "exp5_toxic_bar.png", threshold=args.toxic_threshold_bps)
    print(f"[exp5] wrote {figures_dir / 'exp5_toxic_bar.png'}")

    draw_profit_hist(df, figures_dir / "exp5_profit_hist.png",
                     main_horizon=args.main_horizon, threshold=args.toxic_threshold_bps)
    print(f"[exp5] wrote {figures_dir / 'exp5_profit_hist.png'}")

    print("\n=== group summary ===")
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(summary.to_string(index=False))
    print("\n=== Welch t-tests (main horizon t+{}s) ===".format(args.main_horizon))
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(ttests.to_string(index=False))
    print("\n=== loss estimate ===")
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(losses.to_string(index=False))


if __name__ == "__main__":
    main()
