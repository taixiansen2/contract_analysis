"""Week 3 · 实验 4：量价关系与动态定价。

读 analysis_table_YYYYMMDD.csv，对 HumidiFi / Whirlpool / Raydium 的 SOL/USDC
swap 按成交量（amount_usd）做 4 档统计 + log 回归，比较"大单滑点"特性。

口径：
  - 滑点 ≡ quote_deviation（vs Binance 1s CEX mid），单位 bps。
  - 统一过滤 amount_usd < 1 USD（去聚合器 dust/中转）；HumidiFi 额外剔 is_dust。
  - 分档：<100 / 100-1k / 1k-10k / >10k USD。>10k 档样本稀，只做描述性。
  - OLS：每个 AMM 独立 `dev_bps ~ log10(amount_usd)`；附 log-log 敏感性。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import data_paths

BUCKETS: list[tuple[str, float, float]] = [
    ("<100", 0.0, 100.0),
    ("100-1k", 100.0, 1_000.0),
    ("1k-10k", 1_000.0, 10_000.0),
    (">10k", 10_000.0, 1e18),
]

AMM_ORDER = ["humidifi", "whirlpool", "raydium", "orca"]
AMM_LABELS = {
    "humidifi": "HumidiFi",
    "whirlpool": "Whirlpool",
    "raydium": "Raydium",
    "orca": "Orca",
}
AMM_COLORS = {
    "humidifi": "#C53030",
    "whirlpool": "#2B6CB0",
    "raydium": "#D69E2E",
    "orca": "#38A169",
}


def bucket_of(usd: float) -> str | None:
    if pd.isna(usd):
        return None
    for name, lo, hi in BUCKETS:
        if lo <= usd < hi:
            return name
    return None


def load_and_prepare(date: str, min_usd: float = 1.0) -> pd.DataFrame:
    paths = data_paths(date)
    df = pd.read_csv(paths["analysis_csv"])

    mask_base = (
        df["amount_usd"].notna()
        & df["price_usdc_per_sol"].notna()
        & df["quote_deviation"].notna()
        & (df["amount_usd"] >= min_usd)
    )
    mask_hmd_dust = ~((df["amm_type"] == "humidifi") & df["is_dust"].astype(bool))
    df["valid"] = mask_base & mask_hmd_dust

    df["bucket"] = df["amount_usd"].apply(bucket_of)
    df["dev_bps"] = df["quote_deviation"] * 1e4
    return df


def bucket_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = df[df["valid"]]
    for amm in AMM_ORDER:
        g_amm = valid[valid["amm_type"] == amm]
        for name, _lo, _hi in BUCKETS:
            g = g_amm[g_amm["bucket"] == name]
            n = len(g)
            if n == 0:
                rows.append({
                    "amm_type": amm, "bucket": name, "n": 0,
                    "mean_dev_bps": None, "median_dev_bps": None, "p90_dev_bps": None,
                    "mean_amount_usd": None, "median_amount_usd": None,
                })
                continue
            rows.append({
                "amm_type": amm,
                "bucket": name,
                "n": n,
                "mean_dev_bps": float(g["dev_bps"].mean()),
                "median_dev_bps": float(g["dev_bps"].median()),
                "p90_dev_bps": float(g["dev_bps"].quantile(0.9)),
                "mean_amount_usd": float(g["amount_usd"].mean()),
                "median_amount_usd": float(g["amount_usd"].median()),
            })
    return pd.DataFrame(rows)


def _fit_linear(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 5:
        return {"n": len(x), "coef": None, "std_err": None, "t": None, "p": None,
                "r2": None, "intercept": None, "intercept_p": None}
    X = sm.add_constant(x)
    m = sm.OLS(y, X).fit()
    return {
        "n": int(m.nobs),
        "intercept": float(m.params[0]),
        "intercept_p": float(m.pvalues[0]),
        "coef": float(m.params[1]),
        "std_err": float(m.bse[1]),
        "t": float(m.tvalues[1]),
        "p": float(m.pvalues[1]),
        "r2": float(m.rsquared),
    }


def ols_log_amount(df: pd.DataFrame) -> pd.DataFrame:
    """每 AMM 拟合 dev_bps ~ log10(amount_usd)，附 log-log 敏感性。"""
    rows = []
    valid = df[df["valid"]]
    for amm in AMM_ORDER:
        g = valid[valid["amm_type"] == amm].copy()
        n = len(g)
        if n < 5:
            rows.append({
                "amm_type": amm, "n": n,
                "slope_bps_per_decade": None, "slope_std_err": None,
                "slope_t": None, "slope_p": None,
                "intercept_bps": None, "r2": None,
                "loglog_slope": None, "loglog_p": None, "loglog_r2": None, "loglog_n": 0,
            })
            continue
        x = np.log10(g["amount_usd"].astype(float).values)
        y = g["dev_bps"].astype(float).values
        r = _fit_linear(x, y)

        g_pos = g[g["dev_bps"] > 0]
        if len(g_pos) >= 5:
            xl = np.log10(g_pos["amount_usd"].astype(float).values)
            yl = np.log10(g_pos["dev_bps"].astype(float).values)
            rl = _fit_linear(xl, yl)
        else:
            rl = {"n": len(g_pos), "coef": None, "p": None, "r2": None}

        rows.append({
            "amm_type": amm,
            "n": r["n"],
            "slope_bps_per_decade": r["coef"],
            "slope_std_err": r["std_err"],
            "slope_t": r["t"],
            "slope_p": r["p"],
            "intercept_bps": r["intercept"],
            "r2": r["r2"],
            "loglog_slope": rl.get("coef"),
            "loglog_p": rl.get("p"),
            "loglog_r2": rl.get("r2"),
            "loglog_n": rl.get("n"),
        })
    return pd.DataFrame(rows)


def draw_bucket_bar_grouped(bucket_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    bucket_names = [b[0] for b in BUCKETS]
    x = np.arange(len(bucket_names))
    present = [a for a in AMM_ORDER if (bucket_df[bucket_df["amm_type"] == a]["n"].sum() > 0)]
    k = len(present)
    width = 0.8 / max(k, 1)
    for i, amm in enumerate(present):
        g = bucket_df[bucket_df["amm_type"] == amm].set_index("bucket").reindex(bucket_names)
        meds = g["median_dev_bps"].fillna(0).values
        ns = g["n"].fillna(0).astype(int).values
        offset = (i - (k - 1) / 2) * width
        bars = ax.bar(x + offset, meds, width, label=AMM_LABELS[amm], color=AMM_COLORS[amm])
        for b, n, m in zip(bars, ns, meds):
            if n > 0:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{m:.2f}\nn={n}", ha="center", va="bottom",
                        fontsize=7, color=AMM_COLORS[amm])
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_names)
    ax.set_xlabel("amount_usd bucket")
    ax.set_ylabel("median quote_deviation (bps)")
    ax.set_title("Experiment 4: median quote_deviation by amount bucket x AMM"
                 " (2026-04-15 00:00-01:00 UTC, amount_usd >= $1)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def draw_scatter_fit(df: pd.DataFrame, bucket_df: pd.DataFrame, out_path: Path) -> None:
    valid = df[df["valid"]]
    present = [a for a in AMM_ORDER if (valid["amm_type"] == a).any()]
    n = len(present)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows), sharey=False)
    axes = np.atleast_1d(axes).flatten()

    for i, amm in enumerate(present):
        ax = axes[i]
        g = valid[valid["amm_type"] == amm].copy()
        g = g[g["amount_usd"] > 0]
        if len(g) == 0:
            ax.axis("off")
            continue

        dev_cap = np.nanpercentile(g["dev_bps"], 99) if len(g) else 10
        ax.scatter(
            g["amount_usd"].values,
            np.clip(g["dev_bps"].values, 0, dev_cap),
            s=6, alpha=0.25, color=AMM_COLORS[amm], label=f"n={len(g)}",
        )

        x = np.log10(g["amount_usd"].astype(float).values)
        y = g["dev_bps"].astype(float).values
        if len(g) >= 5:
            X = sm.add_constant(x)
            m = sm.OLS(y, X).fit()
            xs_line = np.linspace(x.min(), x.max(), 100)
            ys_line = m.params[0] + m.params[1] * xs_line
            ax.plot(10 ** xs_line, ys_line, color="#2D3748", linewidth=2,
                    label=f"slope={m.params[1]:.2f} bps/dec, r2={m.rsquared:.3f}")

        b = bucket_df[bucket_df["amm_type"] == amm].set_index("bucket").reindex(
            [bn for bn, _, _ in BUCKETS]
        )
        for (name, lo, hi), med, nb in zip(BUCKETS, b["median_dev_bps"].values, b["n"].fillna(0).astype(int).values):
            if nb == 0 or med is None or pd.isna(med):
                continue
            lo_plot = max(lo, 1)
            hi_plot = min(hi, 10 ** x.max() if len(x) else hi)
            ax.hlines(med, lo_plot, hi_plot, colors="#C53030", linewidth=2)
            ax.text((lo_plot * hi_plot) ** 0.5, med, f"{med:.1f}\nn={nb}",
                    ha="center", va="bottom", fontsize=8, color="#C53030")

        ax.set_xscale("log")
        ax.set_xlabel("amount_usd (USD, log)")
        ax.set_ylabel("quote_deviation (bps, clipped @ p99)")
        ax.set_title(AMM_LABELS[amm], fontsize=11)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    for j in range(len(present), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Experiment 4: amount vs quote_deviation (log-x), with OLS fit and bucket median",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def draw_bucket_box(df: pd.DataFrame, out_path: Path) -> None:
    valid = df[df["valid"]]
    present = [a for a in AMM_ORDER if (valid["amm_type"] == a).any()]
    bucket_names = [b[0] for b in BUCKETS]

    fig, ax = plt.subplots(figsize=(12, 6))
    positions = []
    data = []
    colors = []
    labels = []
    k = len(present)
    width = 0.8 / max(k, 1)

    for bi, bname in enumerate(bucket_names):
        for ai, amm in enumerate(present):
            g = valid[(valid["amm_type"] == amm) & (valid["bucket"] == bname)]
            vals = g["dev_bps"].values
            if len(vals) == 0:
                continue
            cap = np.nanpercentile(vals, 99) if len(vals) >= 5 else np.nanmax(vals)
            vals = np.clip(vals, 0, max(cap, 1e-9))
            positions.append(bi + (ai - (k - 1) / 2) * width)
            data.append(vals)
            colors.append(AMM_COLORS[amm])
            labels.append(f"{amm[:3]}")

    if data:
        bp = ax.boxplot(
            data, positions=positions, widths=width * 0.9,
            patch_artist=True, showfliers=False,
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
        for med in bp["medians"]:
            med.set_color("#1A202C")

    ax.set_xticks(np.arange(len(bucket_names)))
    ax.set_xticklabels(bucket_names)
    ax.set_xlabel("amount_usd bucket")
    ax.set_ylabel("quote_deviation (bps, fliers hidden, clipped @ p99)")
    ax.set_title("Experiment 4: quote_deviation distribution per bucket x AMM")
    ax.grid(True, axis="y", alpha=0.3)

    handles = [plt.Rectangle((0, 0), 1, 1, color=AMM_COLORS[a], alpha=0.5) for a in present]
    ax.legend(handles, [AMM_LABELS[a] for a in present], loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="e.g. 20260415")
    ap.add_argument("--min-usd", type=float, default=1.0,
                    help="drop swaps with amount_usd < this threshold (default 1.0)")
    args = ap.parse_args()
    paths = data_paths(args.date)

    df = load_and_prepare(args.date, min_usd=args.min_usd)

    reports = paths["report_md"].parent
    tables_dir = reports / "tables"
    figures_dir = reports / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    bstats = bucket_stats(df)
    bstats.to_csv(tables_dir / "exp4_bucket_stats.csv", index=False)
    print(f"[exp4] wrote {tables_dir / 'exp4_bucket_stats.csv'}")

    ols_tbl = ols_log_amount(df)
    ols_tbl.to_csv(tables_dir / "exp4_ols_summary.csv", index=False)
    print(f"[exp4] wrote {tables_dir / 'exp4_ols_summary.csv'}")

    draw_bucket_bar_grouped(bstats, figures_dir / "exp4_bucket_bar.png")
    print(f"[exp4] wrote {figures_dir / 'exp4_bucket_bar.png'}")

    draw_scatter_fit(df, bstats, figures_dir / "exp4_scatter_fit.png")
    print(f"[exp4] wrote {figures_dir / 'exp4_scatter_fit.png'}")

    draw_bucket_box(df, figures_dir / "exp4_bucket_box.png")
    print(f"[exp4] wrote {figures_dir / 'exp4_bucket_box.png'}")

    print("\n=== bucket stats ===")
    print(bstats.to_string(index=False))
    print("\n=== OLS summary (dev_bps ~ log10(amount_usd)) ===")
    print(ols_tbl.to_string(index=False))


if __name__ == "__main__":
    main()
