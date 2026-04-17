"""Week 3 · 实验 3：改价响应速度与价格跟踪。

基于 Binance SOL/USDC 1s K 线识别价格突变事件，量化以下两个指标：
  - 响应延迟 T1 - T0：
      · HumidiFi：第一次成功 oracle update 落地的 block_time
      · HumidiFi swap：第一笔成交的 block_time（作为 "报价已更新并被采用" 的侧证）
      · Whirlpool / Raydium：突变后第一笔 SOL/USDC swap 的 block_time（套利者代理）
  - 收敛时间 T2 - T0：
      · 任意 AMM 的首笔 swap 其成交价与同一秒 Binance close 的偏差 ≤ 5bps

数据分辨率说明：Binance K 线 + Solana blockTime 都是 1s，响应延迟实际上被量化到 ±1s
（≈2.5 slot），"slot 数" 维度仅作参考展示。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import WEEK3_DIR

# ---------- 常量 ----------

SLOT_MS = 400
EVENT_WIN_S = 5
EVENT_COOLDOWN_S = 5
CONVERGE_BPS = 5
MAX_TRACK_SEC = 30
THRESHOLDS = [0.0010, 0.0005, 0.0015]  # 主/敏感性
MAIN_THRESHOLD = 0.0010

AMM_ORDER = ["humidifi", "whirlpool", "raydium"]
AMM_COLORS = {
    "humidifi": "#C53030",
    "whirlpool": "#2B6CB0",
    "raydium": "#D69E2E",
}

# 1h 窗口
WINDOW_START = datetime(2026, 4, 15, 0, 0, 0, tzinfo=timezone.utc)
WINDOW_HOURS = 1


# ---------- 路径 ----------

def paths_for(date_str: str) -> dict[str, Path]:
    d = date_str.replace("-", "")
    raw = WEEK3_DIR / "data" / "raw"
    proc = WEEK3_DIR / "data" / "processed"
    reports = WEEK3_DIR / "reports"
    figs = reports / "figures"
    tabs = reports / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    return {
        "binance": raw / f"binance_sol_usdc_{d}.csv",
        "oracle": proc / f"humidifi_oracle_updates_{d}.csv",
        "hmd_swap": proc / f"humidifi_swaps_{d}.csv",
        "ctrl_swap": raw / f"orca_raydium_sol_usdc_{d}_h0-1.csv",
        "events_csv": tabs / "exp3_events.csv",
        "summary_csv": tabs / "exp3_delay_summary.csv",
        "fig_cases": figs / "exp3_event_cases.png",
        "fig_box": figs / "exp3_response_boxplot.png",
        "fig_slot_hist": figs / "exp3_slot_histogram.png",
    }


# ---------- 加载 ----------

def load_binance_1h(path: Path) -> pd.DataFrame:
    """返回 ts_s (int64) → close (float)；只保留 1h 窗口内的秒。"""
    df = pd.read_csv(path, usecols=["open_time_ms", "close"])
    df["ts_s"] = (df["open_time_ms"].astype("int64") // 1000).astype("int64")
    df["close"] = df["close"].astype(float)
    start_s = int(WINDOW_START.timestamp())
    end_s = start_s + WINDOW_HOURS * 3600
    df = df[(df["ts_s"] >= start_s) & (df["ts_s"] < end_s)].copy()
    df = df.sort_values("ts_s").reset_index(drop=True)
    return df[["ts_s", "close"]]


def load_oracle_updates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["slot", "block_time", "status"])
    df["slot"] = df["slot"].astype("int64")
    df["block_time"] = df["block_time"].astype("int64")
    df = df.sort_values("block_time", kind="stable").reset_index(drop=True)
    return df


def load_humidifi_swaps(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=[
            "slot", "block_time", "price_usdc_per_sol",
            "base_amount_sol", "amount_usd", "quote_mint",
        ],
    )
    df = df[df["price_usdc_per_sol"].notna()].copy()
    # 过滤 USDC 以外的 quote（price 只在 USDC 时有意义）
    df = df[df["quote_mint"] == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"]
    # 滤 dust（与实验 1 口径一致）
    df = df[df["base_amount_sol"].astype(float) >= 0.001]
    df["slot"] = df["slot"].astype("int64")
    df["block_time"] = df["block_time"].astype("int64")
    df["price_usdc_per_sol"] = df["price_usdc_per_sol"].astype(float)
    df = df.sort_values("block_time", kind="stable").reset_index(drop=True)
    return df[["slot", "block_time", "price_usdc_per_sol", "amount_usd"]]


def load_ctrl_swaps(path: Path) -> pd.DataFrame:
    """Dune Orca/Whirlpool/Raydium 交易，重构成统一 schema。"""
    df = pd.read_csv(path)
    # block_time: "2026-04-15 00:00:00.000 UTC" → int64 秒
    # pandas 2.x 对带 UTC 后缀的字符串返回 datetime64[us, UTC]，单位不固定，
    # 统一先转到 numpy datetime64[s] 再转 int64 最安全。
    arr = pd.to_datetime(df["block_time"], utc=True).to_numpy()
    df["block_time"] = arr.astype("datetime64[s]").astype("int64")
    df["slot"] = df["block_slot"].astype("int64")
    df["amm_type"] = df["project"].astype(str)

    SOL = {"SOL", "WSOL"}
    bought_is_sol = df["token_bought_symbol"].isin(SOL)
    base_amount = np.where(
        bought_is_sol,
        df["token_bought_amount"].astype(float),
        df["token_sold_amount"].astype(float),
    )
    quote_amount = np.where(
        bought_is_sol,
        df["token_sold_amount"].astype(float),
        df["token_bought_amount"].astype(float),
    )
    df["base_amount_sol"] = base_amount
    df["price_usdc_per_sol"] = np.where(
        base_amount > 0, quote_amount / base_amount, np.nan
    )
    df["amount_usd"] = df["amount_usd"].astype(float)
    # 只保留 whirlpool / raydium（orca V1 在该窗口下 0 条）
    df = df[df["amm_type"].isin(["whirlpool", "raydium"])].copy()
    df = df[df["price_usdc_per_sol"].notna() & (df["base_amount_sol"] > 0)]
    df = df.sort_values(["amm_type", "block_time"], kind="stable").reset_index(drop=True)
    return df[["amm_type", "slot", "block_time", "price_usdc_per_sol", "amount_usd"]]


# ---------- 事件检测 ----------

@dataclass
class Event:
    t0_s: int
    magnitude: float
    direction: str  # 'up' / 'down'
    pre_price: float
    peak_price: float
    trough_price: float
    threshold: float


def detect_events(binance: pd.DataFrame, threshold: float) -> list[Event]:
    """5s 滑窗内 max/min - 1 ≥ threshold 即为事件，冷却 EVENT_COOLDOWN_S。"""
    t = binance["ts_s"].to_numpy()
    p = binance["close"].to_numpy()
    n = len(p)
    events: list[Event] = []
    last_end = -1
    for i in range(n):
        if t[i] < last_end:
            continue
        # 找满足 t[j] - t[i] <= EVENT_WIN_S 的窗口
        j = i
        while j + 1 < n and t[j + 1] - t[i] <= EVENT_WIN_S:
            j += 1
        if j == i:
            continue
        window = p[i:j + 1]
        pmin = float(window.min())
        pmax = float(window.max())
        if pmin <= 0:
            continue
        mag = pmax / pmin - 1.0
        if mag >= threshold:
            # direction: 若最大值出现在最小值之后 → up；反之 down
            idx_min = int(window.argmin())
            idx_max = int(window.argmax())
            direction = "up" if idx_max > idx_min else "down"
            events.append(Event(
                t0_s=int(t[i]),
                magnitude=mag,
                direction=direction,
                pre_price=float(p[i]),
                peak_price=pmax,
                trough_price=pmin,
                threshold=threshold,
            ))
            last_end = int(t[i]) + EVENT_COOLDOWN_S
    return events


# ---------- 单事件测量 ----------

def _binance_price_at(binance: pd.DataFrame, ts_s: int) -> float | None:
    """取 ts_s 的 close；若无精确匹配，取 ≤ ts_s 的最近一条。"""
    arr_t = binance["ts_s"].to_numpy()
    arr_p = binance["close"].to_numpy()
    idx = np.searchsorted(arr_t, ts_s, side="right") - 1
    if idx < 0:
        return None
    return float(arr_p[idx])


def _first_after(series_t: np.ndarray, t0: int):
    """返回 series_t 中第一个 >= t0 的索引（不存在返回 -1）。"""
    idx = np.searchsorted(series_t, t0, side="left")
    if idx >= len(series_t):
        return -1
    return int(idx)


def measure_event(
    ev: Event,
    oracle_t: np.ndarray,
    hmd_t: np.ndarray,
    hmd_p: np.ndarray,
    ctrl_by_amm: dict[str, tuple[np.ndarray, np.ndarray]],
    binance: pd.DataFrame,
) -> list[dict]:
    """对单个事件，返回每个 AMM 一行的度量结果。"""
    rows: list[dict] = []
    t0 = ev.t0_s
    t_cutoff = t0 + MAX_TRACK_SEC

    # ---- HumidiFi oracle ----
    idx_o = _first_after(oracle_t, t0)
    if idx_o >= 0 and oracle_t[idx_o] <= t_cutoff:
        T1_oracle = int(oracle_t[idx_o])
    else:
        T1_oracle = None

    # ---- HumidiFi swap: T1 与 T2（收敛至 5bps） ----
    def first_converged(swap_t: np.ndarray, swap_p: np.ndarray) -> tuple[int | None, int | None]:
        idx = _first_after(swap_t, t0)
        if idx < 0 or swap_t[idx] > t_cutoff:
            return None, None
        T1 = int(swap_t[idx])
        T2 = None
        for k in range(idx, len(swap_t)):
            ts = int(swap_t[k])
            if ts > t_cutoff:
                break
            cex = _binance_price_at(binance, ts)
            if cex is None or cex <= 0:
                continue
            dev = abs(swap_p[k] / cex - 1.0)
            if dev <= CONVERGE_BPS * 1e-4:
                T2 = ts
                break
        return T1, T2

    # HumidiFi
    T1_hmd, T2_hmd = first_converged(hmd_t, hmd_p)
    rows.append({
        "threshold": ev.threshold,
        "t0_s": t0,
        "t0_iso": datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(),
        "magnitude": ev.magnitude,
        "direction": ev.direction,
        "pre_price": ev.pre_price,
        "peak_price": ev.peak_price,
        "trough_price": ev.trough_price,
        "amm": "humidifi",
        "T1_oracle_s": T1_oracle,
        "T1_swap_s": T1_hmd,
        "T2_swap_s": T2_hmd,
        "response_delay_oracle_s": (T1_oracle - t0) if T1_oracle is not None else None,
        "response_delay_swap_s": (T1_hmd - t0) if T1_hmd is not None else None,
        "convergence_delay_s": (T2_hmd - t0) if T2_hmd is not None else None,
        "converged": T2_hmd is not None,
    })

    # 对照组
    for amm, (c_t, c_p) in ctrl_by_amm.items():
        T1_c, T2_c = first_converged(c_t, c_p)
        rows.append({
            "threshold": ev.threshold,
            "t0_s": t0,
            "t0_iso": datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(),
            "magnitude": ev.magnitude,
            "direction": ev.direction,
            "pre_price": ev.pre_price,
            "peak_price": ev.peak_price,
            "trough_price": ev.trough_price,
            "amm": amm,
            "T1_oracle_s": None,
            "T1_swap_s": T1_c,
            "T2_swap_s": T2_c,
            "response_delay_oracle_s": None,
            "response_delay_swap_s": (T1_c - t0) if T1_c is not None else None,
            "convergence_delay_s": (T2_c - t0) if T2_c is not None else None,
            "converged": T2_c is not None,
        })

    return rows


# ---------- 汇总 ----------

def summarize(events_df: pd.DataFrame) -> pd.DataFrame:
    """按 threshold × amm × metric 汇总 n/median/mean/p90/max。"""
    rows = []
    metrics = {
        "response_delay_oracle_s": "response_oracle",
        "response_delay_swap_s": "response_swap",
        "convergence_delay_s": "convergence",
    }
    for th in sorted(events_df["threshold"].unique()):
        for amm in AMM_ORDER:
            g = events_df[(events_df["threshold"] == th) & (events_df["amm"] == amm)]
            if g.empty:
                continue
            for col, metric_name in metrics.items():
                if col not in g.columns:
                    continue
                vals = g[col].dropna().astype(float)
                rows.append({
                    "threshold": th,
                    "amm": amm,
                    "metric": metric_name,
                    "n_events_total": int(len(g)),
                    "n_with_value": int(len(vals)),
                    "median_s": float(vals.median()) if len(vals) else None,
                    "mean_s": float(vals.mean()) if len(vals) else None,
                    "p90_s": float(vals.quantile(0.9)) if len(vals) else None,
                    "max_s": float(vals.max()) if len(vals) else None,
                    "median_slot": float(vals.median() / (SLOT_MS / 1000.0)) if len(vals) else None,
                })
    return pd.DataFrame(rows)


# ---------- 绘图 ----------

def draw_event_cases(
    events: list[Event],
    binance: pd.DataFrame,
    hmd_t: np.ndarray, hmd_p: np.ndarray,
    oracle_t: np.ndarray,
    ctrl_by_amm: dict[str, tuple[np.ndarray, np.ndarray]],
    events_df: pd.DataFrame,
    out_path: Path,
) -> None:
    if not events:
        return
    n = len(events)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, 4.2 * rows), sharey=False)
    axes = np.atleast_1d(axes).flatten()

    for i, ev in enumerate(events):
        ax = axes[i]
        t0 = ev.t0_s
        lo, hi = t0 - 5, t0 + MAX_TRACK_SEC
        bx = binance[(binance["ts_s"] >= lo) & (binance["ts_s"] <= hi)]
        ax.plot(bx["ts_s"] - t0, bx["close"], color="#4A5568", linewidth=1.4, label="Binance close")

        def _scatter(t_arr: np.ndarray, p_arr: np.ndarray, color: str, marker: str, label: str):
            m = (t_arr >= lo) & (t_arr <= hi)
            if m.any():
                ax.scatter(
                    t_arr[m] - t0, p_arr[m],
                    color=color, marker=marker, s=24, alpha=0.75, label=label,
                )

        _scatter(hmd_t, hmd_p, AMM_COLORS["humidifi"], "o", "HumidiFi swap")
        for amm, (c_t, c_p) in ctrl_by_amm.items():
            _scatter(c_t, c_p, AMM_COLORS[amm], "^" if amm == "whirlpool" else "s", amm.capitalize())

        # oracle update 时间标记（底部条形）
        om = (oracle_t >= lo) & (oracle_t <= hi)
        if om.any():
            ymin, ymax = ax.get_ylim()
            ybar = ymin + (ymax - ymin) * 0.02
            ax.scatter(
                oracle_t[om] - t0, np.full(int(om.sum()), ybar),
                marker="|", color=AMM_COLORS["humidifi"], s=40, alpha=0.5,
                label="HumidiFi oracle update",
            )

        # 垂直线：T0、T1_oracle、T1_swap、T2
        ax.axvline(0, color="black", linestyle=":", linewidth=1)
        row_hmd = events_df[
            (events_df["threshold"] == MAIN_THRESHOLD)
            & (events_df["t0_s"] == t0) & (events_df["amm"] == "humidifi")
        ]
        if not row_hmd.empty:
            r = row_hmd.iloc[0]
            if pd.notna(r["T1_oracle_s"]):
                ax.axvline(r["T1_oracle_s"] - t0, color=AMM_COLORS["humidifi"], linestyle="--",
                           linewidth=1, alpha=0.7, label="HMD oracle T1")
            if pd.notna(r["T2_swap_s"]):
                ax.axvline(r["T2_swap_s"] - t0, color=AMM_COLORS["humidifi"], linestyle="-.",
                           linewidth=1, alpha=0.7, label="HMD swap T2")

        ax.set_title(
            f"Event @ {datetime.fromtimestamp(t0, tz=timezone.utc).strftime('%H:%M:%S')}  "
            f"mag={ev.magnitude*1e4:.1f}bps  dir={ev.direction}",
            fontsize=10,
        )
        ax.set_xlabel("seconds since T0")
        ax.set_ylabel("price (USDC/SOL)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)

    for j in range(len(events), len(axes)):
        axes[j].axis("off")
    fig.suptitle(
        f"Experiment 3: CEX shock response (threshold {MAIN_THRESHOLD*100:.2f}%/5s, "
        f"{len(events)} events, 2026-04-15 00:00-01:00 UTC)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def draw_response_boxplot(events_df: pd.DataFrame, out_path: Path) -> None:
    df = events_df[events_df["threshold"] == MAIN_THRESHOLD]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    # (a) response_delay：HumidiFi 用 oracle T1；ctrl 用首笔 swap
    box_data_r = []
    labels_r = []
    for amm in AMM_ORDER:
        g = df[df["amm"] == amm]
        if amm == "humidifi":
            vals = g["response_delay_oracle_s"].dropna().astype(float).to_numpy()
            labels_r.append("HumidiFi oracle")
        else:
            vals = g["response_delay_swap_s"].dropna().astype(float).to_numpy()
            labels_r.append(amm.capitalize())
        box_data_r.append(vals if len(vals) else np.array([np.nan]))

    ax = axes[0]
    bp = ax.boxplot(box_data_r, labels=labels_r, showfliers=True, patch_artist=True)
    for patch, amm in zip(bp["boxes"], AMM_ORDER):
        patch.set_facecolor(AMM_COLORS[amm])
        patch.set_alpha(0.5)
    # 散点
    for i, v in enumerate(box_data_r):
        xs = np.random.uniform(-0.08, 0.08, size=len(v)) + (i + 1)
        ax.scatter(xs, v, s=12, alpha=0.6, color="black")
    ax.set_title(f"Response delay (T1 - T0), threshold {MAIN_THRESHOLD*100:.2f}%/5s")
    ax.set_ylabel("delay (s)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(SLOT_MS / 1000.0, color="gray", linestyle=":", linewidth=1, label="1 slot")
    ax.axhline(2 * SLOT_MS / 1000.0, color="gray", linestyle="--", linewidth=1, label="2 slot")
    ax.legend(loc="upper left", fontsize=8)

    # (b) convergence_delay：全用 swap
    box_data_c = []
    labels_c = []
    for amm in AMM_ORDER:
        g = df[df["amm"] == amm]
        vals = g["convergence_delay_s"].dropna().astype(float).to_numpy()
        box_data_c.append(vals if len(vals) else np.array([np.nan]))
        labels_c.append(amm.capitalize())
    ax = axes[1]
    bp = ax.boxplot(box_data_c, labels=labels_c, showfliers=True, patch_artist=True)
    for patch, amm in zip(bp["boxes"], AMM_ORDER):
        patch.set_facecolor(AMM_COLORS[amm])
        patch.set_alpha(0.5)
    for i, v in enumerate(box_data_c):
        xs = np.random.uniform(-0.08, 0.08, size=len(v)) + (i + 1)
        ax.scatter(xs, v, s=12, alpha=0.6, color="black")
    ax.set_title("Convergence delay (T2 - T0, swap price within 5 bps of CEX)")
    ax.set_ylabel("delay (s)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Experiment 3: response & convergence delay by AMM", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def draw_slot_histogram(events_df: pd.DataFrame, out_path: Path) -> None:
    df = events_df[(events_df["threshold"] == MAIN_THRESHOLD) & (events_df["amm"] == "humidifi")]
    vals_s = df["response_delay_oracle_s"].dropna().astype(float).to_numpy()
    if len(vals_s) == 0:
        return
    vals_slot = vals_s * 1000.0 / SLOT_MS
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0, max(int(vals_slot.max()) + 2, 8))
    ax.hist(vals_slot, bins=bins, color=AMM_COLORS["humidifi"], alpha=0.75, edgecolor="white")
    for s in (1, 2):
        ax.axvline(s, color="gray", linestyle="--", linewidth=1)
        ax.text(s + 0.05, ax.get_ylim()[1] * 0.95, f"{s} slot", fontsize=9, color="gray")
    ax.set_title(
        f"HumidiFi oracle response delay, in slots (n={len(vals_slot)}, "
        f"threshold {MAIN_THRESHOLD*100:.2f}%/5s)",
    )
    ax.set_xlabel("slots (delay_s / 0.4)")
    ax.set_ylabel("# events")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------- 入口 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-04-15")
    args = ap.parse_args()
    p = paths_for(args.date)

    print(f"[load] binance: {p['binance']}")
    binance = load_binance_1h(p["binance"])
    print(f"       rows in 1h window: {len(binance)}")

    print(f"[load] oracle: {p['oracle']}")
    oracle = load_oracle_updates(p["oracle"])
    oracle_ok = oracle[oracle["status"] == "success"].reset_index(drop=True)
    print(f"       oracle total={len(oracle)}  success={len(oracle_ok)}")

    print(f"[load] humidifi swaps: {p['hmd_swap']}")
    hmd = load_humidifi_swaps(p["hmd_swap"])
    print(f"       rows (USDC, non-dust): {len(hmd)}")

    print(f"[load] ctrl swaps: {p['ctrl_swap']}")
    ctrl = load_ctrl_swaps(p["ctrl_swap"])
    print(f"       whirlpool={int((ctrl['amm_type']=='whirlpool').sum())} "
          f"raydium={int((ctrl['amm_type']=='raydium').sum())}")

    ctrl_by_amm = {
        amm: (ctrl[ctrl["amm_type"] == amm]["block_time"].to_numpy().astype("int64"),
              ctrl[ctrl["amm_type"] == amm]["price_usdc_per_sol"].to_numpy().astype(float))
        for amm in ("whirlpool", "raydium")
    }
    hmd_t = hmd["block_time"].to_numpy().astype("int64")
    hmd_p = hmd["price_usdc_per_sol"].to_numpy().astype(float)
    oracle_t = oracle_ok["block_time"].to_numpy().astype("int64")

    # ---- 事件检测 + 测量（主 + 敏感性阈值都跑） ----
    all_rows: list[dict] = []
    events_by_threshold: dict[float, list[Event]] = {}
    for th in THRESHOLDS:
        evs = detect_events(binance, th)
        events_by_threshold[th] = evs
        print(f"[events] threshold={th*100:.2f}%/5s  count={len(evs)}")
        for ev in evs:
            rows = measure_event(ev, oracle_t, hmd_t, hmd_p, ctrl_by_amm, binance)
            all_rows.extend(rows)

    events_df = pd.DataFrame(all_rows)
    events_df.to_csv(p["events_csv"], index=False)
    print(f"[write] events → {p['events_csv']}  ({len(events_df)} rows)")

    summary_df = summarize(events_df)
    summary_df.to_csv(p["summary_csv"], index=False)
    print(f"[write] summary → {p['summary_csv']}  ({len(summary_df)} rows)")

    # ---- 绘图（主阈值） ----
    draw_event_cases(
        events_by_threshold[MAIN_THRESHOLD], binance,
        hmd_t, hmd_p, oracle_t, ctrl_by_amm, events_df, p["fig_cases"],
    )
    print(f"[write] fig → {p['fig_cases']}")

    draw_response_boxplot(events_df, p["fig_box"])
    print(f"[write] fig → {p['fig_box']}")

    draw_slot_histogram(events_df, p["fig_slot_hist"])
    print(f"[write] fig → {p['fig_slot_hist']}")

    # ---- 终端摘要 ----
    print("\n[summary] MAIN threshold response/convergence medians:")
    main_sum = summary_df[summary_df["threshold"] == MAIN_THRESHOLD]
    for _, r in main_sum.iterrows():
        print(f"  {r['amm']:<10s}  {r['metric']:<17s}  n={r['n_with_value']}/{r['n_events_total']}  "
              f"median={r['median_s']}s  p90={r['p90_s']}s  max={r['max_s']}s")


if __name__ == "__main__":
    main()
