"""Week 3 · 实验 1：改价频率与报价质量。

读 analysis_table_YYYYMMDD.csv，对每笔 SOL/USDC swap 计算 Δt（slot / ms），
按桶 [0,400) / [400,800) / [800,1200) / >=1200 ms 做分桶统计 + OLS 回归，
出两张图（散点+阶梯、分桶均值柱）和结果 CSV。

Δt 定义：
  - HumidiFi：swap.slot - prev_oracle_success_slot（来自 build_table）
  - Orca/Whirlpool/Raydium：同 project 内按 slot 升序，前一笔 swap 的 slot 差（proxy）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import data_paths

SLOT_MS = 400  # Solana 理论 slot = 400ms

# 桶定义（ms 区间，右开）
BUCKETS = [
    ("B0 [0,400)", 0, 400),
    ("B1 [400,800)", 400, 800),
    ("B2 [800,1200)", 800, 1200),
    ("B3 >=1200", 1200, 10**9),
]

AMM_ORDER = ["humidifi", "whirlpool", "raydium", "orca"]
AMM_LABELS = {
    "humidifi": "HumidiFi (dt = swap - prev oracle, same pool)",
    "whirlpool": "Whirlpool (dt = prev swap in same project, proxy)",
    "orca": "Orca (dt = prev swap in same project, proxy)",
    "raydium": "Raydium (dt = prev swap in same project, proxy)",
}


def bucket_of(ms: float) -> str | None:
    if pd.isna(ms):
        return None
    for name, lo, hi in BUCKETS:
        if lo <= ms < hi:
            return name
    return None


def load_and_prepare(date: str) -> pd.DataFrame:
    """加载分析表，生成统一的 delta_t_ms / bucket 字段。"""
    paths = data_paths(date)
    df = pd.read_csv(paths["analysis_csv"])

    # 对照组：按 amm_type 内按 slot 升序填 delta_t_slot
    mask_ctrl = df["amm_type"].isin(["orca", "whirlpool", "raydium"])
    ctrl = df[mask_ctrl].copy()
    ctrl = ctrl.sort_values(["amm_type", "slot"], kind="stable")
    ctrl["delta_t_slot"] = ctrl.groupby("amm_type")["slot"].diff()
    df.loc[ctrl.index, "delta_t_slot"] = ctrl["delta_t_slot"]

    df["delta_t_ms"] = df["delta_t_slot"] * SLOT_MS

    # 有效样本：有 price / 有偏差 / 有 slot / 有 Δt；HumidiFi 额外要求非 dust
    mask_base = (
        df["price_usdc_per_sol"].notna()
        & df["quote_deviation"].notna()
        & df["slot"].notna()
        & df["delta_t_slot"].notna()
    )
    mask_hmd_dust = ~((df["amm_type"] == "humidifi") & df["is_dust"].astype(bool))
    df["valid"] = mask_base & mask_hmd_dust

    df["bucket"] = df["delta_t_ms"].apply(bucket_of)
    df["dev_bps"] = df["quote_deviation"] * 1e4
    return df


def bucket_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = df[df["valid"]].copy()
    for amm in AMM_ORDER:
        g_amm = valid[valid["amm_type"] == amm]
        for name, _lo, _hi in BUCKETS:
            g = g_amm[g_amm["bucket"] == name]
            n = len(g)
            if n == 0:
                rows.append({
                    "amm_type": amm, "bucket": name, "n": 0,
                    "mean_dev_bps": None, "median_dev_bps": None, "p90_dev_bps": None,
                    "mean_amount_usd": None,
                })
                continue
            rows.append({
                "amm_type": amm,
                "bucket": name,
                "n": n,
                "mean_dev_bps": float(g["dev_bps"].mean()),
                "median_dev_bps": float(g["dev_bps"].median()),
                "p90_dev_bps": float(g["dev_bps"].quantile(0.9)),
                "mean_amount_usd": float(g["amount_usd"].mean()) if g["amount_usd"].notna().any() else None,
            })
    return pd.DataFrame(rows)


def ols_for(df: pd.DataFrame, y_col: str = "quote_deviation") -> dict:
    """OLS: y ~ delta_t_ms + const，返回 coef/p/r2/n。"""
    n = len(df)
    if n < 5:
        return {"n": n, "coef": None, "std_err": None, "t": None, "p": None, "r2": None,
                "intercept": None, "intercept_p": None}
    X = sm.add_constant(df["delta_t_ms"].astype(float).values)
    y = df[y_col].astype(float).values
    model = sm.OLS(y, X, missing="drop").fit()
    return {
        "n": int(model.nobs),
        "intercept": float(model.params[0]),
        "intercept_p": float(model.pvalues[0]),
        "coef": float(model.params[1]),
        "std_err": float(model.bse[1]),
        "t": float(model.tvalues[1]),
        "p": float(model.pvalues[1]),
        "r2": float(model.rsquared),
    }


def ols_log(df: pd.DataFrame) -> dict:
    """敏感性：log10(dev) ~ log10(dt_ms + 1)。"""
    n = len(df)
    if n < 5:
        return {"n": n, "coef": None, "p": None, "r2": None}
    g = df[(df["quote_deviation"] > 0) & (df["delta_t_ms"].notna())].copy()
    if len(g) < 5:
        return {"n": len(g), "coef": None, "p": None, "r2": None}
    x = np.log10(g["delta_t_ms"].astype(float).values + 1.0)
    y = np.log10(g["quote_deviation"].astype(float).values)
    X = sm.add_constant(x)
    m = sm.OLS(y, X).fit()
    return {"n": int(m.nobs), "coef": float(m.params[1]), "p": float(m.pvalues[1]),
            "r2": float(m.rsquared), "intercept": float(m.params[0])}


def ols_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = df[df["valid"]]
    for amm in AMM_ORDER:
        g = valid[valid["amm_type"] == amm]
        r_lin = ols_for(g)
        r_log = ols_log(g)
        rows.append({
            "amm_type": amm,
            "n": r_lin["n"],
            "linear_coef_bps_per_ms": (r_lin["coef"] * 1e4) if r_lin["coef"] is not None else None,
            "linear_std_err_bps_per_ms": (r_lin["std_err"] * 1e4) if r_lin["std_err"] is not None else None,
            "linear_t": r_lin["t"],
            "linear_p": r_lin["p"],
            "linear_intercept_bps": (r_lin["intercept"] * 1e4) if r_lin["intercept"] is not None else None,
            "linear_r2": r_lin["r2"],
            "loglog_slope": r_log.get("coef"),
            "loglog_p": r_log.get("p"),
            "loglog_r2": r_log.get("r2"),
        })
    return pd.DataFrame(rows)


def draw_scatter_by_amm(df: pd.DataFrame, bucket_df: pd.DataFrame, out_path: Path) -> None:
    valid = df[df["valid"]]
    # 只画有数据的 amm（orca 在 Dune 归到 whirlpool，通常为空）
    present = [a for a in AMM_ORDER if (valid["amm_type"] == a).any()]
    n = len(present)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows), sharey=False)
    axes = np.atleast_1d(axes).flatten()
    for i, amm in enumerate(present):
        ax = axes[i]
        g = valid[valid["amm_type"] == amm]
        # 散点：dev_bps 在 [0, 99 percentile] 范围内更清晰；log 不适合因为有 dev=0
        dev_cap = np.nanpercentile(g["dev_bps"], 99)
        ax.scatter(
            g["delta_t_ms"].values + np.random.uniform(-40, 40, size=len(g)),  # jitter
            np.clip(g["dev_bps"].values, 0, dev_cap),
            s=6, alpha=0.25, color="#2B6CB0", label=f"n={len(g)}",
        )
        # 分桶均值阶梯
        b = bucket_df[bucket_df["amm_type"] == amm].set_index("bucket").reindex(
            [bn for bn, _, _ in BUCKETS]
        )
        centers = [(lo + min(hi, 2000)) / 2 for _, lo, hi in BUCKETS]
        means = b["mean_dev_bps"].values
        ax.step(centers, means, where="mid", color="#C53030", linewidth=2, label="bucket mean")
        ax.scatter(centers, means, color="#C53030", zorder=3, s=40)
        for xc, m, n in zip(centers, means, b["n"].values):
            if m is not None and not pd.isna(m):
                ax.annotate(f"{m:.1f}bps\nn={int(n)}", (xc, m), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8, color="#C53030")

        ax.set_title(AMM_LABELS[amm], fontsize=10)
        ax.set_xlabel("dt (ms)")
        ax.set_ylabel("quote_deviation (bps)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlim(-100, 2600)
    # 未使用的子图清掉
    for j in range(len(present), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Experiment 1: dt vs quote_deviation (2026-04-15 00:00-01:00 UTC)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def draw_bucket_bar(bucket_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    bucket_names = [b[0] for b in BUCKETS]
    x = np.arange(len(bucket_names))
    width = 0.2
    colors = {"humidifi": "#C53030", "whirlpool": "#2B6CB0", "orca": "#38A169", "raydium": "#D69E2E"}
    for i, amm in enumerate(AMM_ORDER):
        g = bucket_df[bucket_df["amm_type"] == amm].set_index("bucket").reindex(bucket_names)
        means = g["mean_dev_bps"].fillna(0).values
        ns = g["n"].fillna(0).astype(int).values
        bars = ax.bar(x + (i - 1.5) * width, means, width, label=amm, color=colors[amm])
        for b, n, m in zip(bars, ns, means):
            if n > 0:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                        f"n={n}", ha="center", va="bottom", fontsize=7, color=colors[amm])

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_names)
    ax.set_ylabel("mean quote_deviation (bps)")
    ax.set_title("Experiment 1: bucket mean comparison (dt bucket x AMM)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    args = ap.parse_args()
    paths = data_paths(args.date)

    df = load_and_prepare(args.date)

    # 确保报告目录存在
    reports = paths["report_md"].parent
    tables_dir = reports / "tables"
    figures_dir = reports / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    bstats = bucket_stats(df)
    bstats.to_csv(tables_dir / "exp1_bucket_stats.csv", index=False)
    print(f"[exp1] wrote {tables_dir / 'exp1_bucket_stats.csv'}")

    ols_tbl = ols_summary_table(df)
    ols_tbl.to_csv(tables_dir / "exp1_ols_summary.csv", index=False)
    print(f"[exp1] wrote {tables_dir / 'exp1_ols_summary.csv'}")

    draw_scatter_by_amm(df, bstats, figures_dir / "exp1_scatter_by_amm.png")
    print(f"[exp1] wrote {figures_dir / 'exp1_scatter_by_amm.png'}")

    draw_bucket_bar(bstats, figures_dir / "exp1_bucket_mean_bar.png")
    print(f"[exp1] wrote {figures_dir / 'exp1_bucket_mean_bar.png'}")

    # 控制台摘要
    print("\n=== bucket stats ===")
    print(bstats.to_string(index=False))
    print("\n=== OLS summary ===")
    print(ols_tbl.to_string(index=False))


if __name__ == "__main__":
    main()
