"""解析 data/raw/humidifi_txs_YYYYMMDD.jsonl.gz，输出 oracle_update 和 swap 两张 CSV。

分类规则（基于 Week 2 逆向结论 + Week 3 pilot 观察）：
  - oracle_update：HumidiFi 作为外层指令，accounts 数 == 3，data 长度 == 65 字节。
    `accounts[0]` 为池状态 PDA（join key），`accounts[1]` 为 signer，`accounts[2]` = SysvarClock。
    前 1 字节 tag，随后 8 个 u64（little-endian）。
  - swap：HumidiFi 作为外层或内层 CPI 指令，accounts 数 != 3 或 data 长度 != 65。
    账户布局（实测）：
      [0]=trader   [1]=pool_state_pda   [2..]=vaults / user ATAs / sysvars / tip
    **join key = accounts[1]**，与 oracle_update 的 accounts[0] 对齐。
    通过 preTokenBalances / postTokenBalances 找出变动了 wSOL 的 owner，
    再挑同 owner 下 |delta| 最大的非 SOL mint 作为 quote（支持任意 quote token，
    不限 USDC）；若 quote == USDC 则额外填 `price_usdc_per_sol` 方便与 Binance 对齐。

Tip：扫描 meta.innerInstructions 中对 Jito tip 账户的 system::transfer，
    记到 oracle_update 的 tip_amount 字段（lamports）。
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import struct
from collections import defaultdict
from pathlib import Path

import base58
from tqdm import tqdm

from .config import (
    HUMIDIFI_PROGRAM,
    JITO_TIP_ACCOUNTS,
    ORACLE_UPDATE_ACCOUNT_COUNT,
    ORACLE_UPDATE_DATA_LEN,
    SOL_MINT,
    USDC_MINT,
    data_paths,
)


def _flatten_account_keys(tx: dict) -> list[str]:
    keys = tx.get("accountKeys") or []
    out = []
    for k in keys:
        out.append(k["pubkey"] if isinstance(k, dict) else k)
    loaded = tx.get("loadedAddresses") or {}
    for k in (loaded.get("writable") or []):
        out.append(k)
    for k in (loaded.get("readonly") or []):
        out.append(k)
    return out


def _decode_b58(data: str | None) -> bytes:
    if not data:
        return b""
    try:
        return base58.b58decode(data)
    except Exception:
        return b""


def _sum_tip(tx: dict) -> int:
    """扫 innerInstructions 中向 Jito tip 账户的 system::transfer，返回 lamports 总和。"""
    total = 0
    for inner in tx.get("innerInstructions") or []:
        for ix in inner.get("instructions") or []:
            parsed = ix.get("parsed")
            if not parsed:
                continue
            if ix.get("program") != "system":
                continue
            if parsed.get("type") != "transfer":
                continue
            info = parsed.get("info") or {}
            dst = info.get("destination")
            lam = info.get("lamports")
            if dst in JITO_TIP_ACCOUNTS and isinstance(lam, int):
                total += lam
    return total


def _parse_oracle_data(raw: bytes) -> dict:
    """1 字节 tag + 8 × u64 LE。"""
    out = {"tag": raw[0]}
    for i in range(8):
        out[f"p{i}"] = struct.unpack_from("<Q", raw, 1 + i * 8)[0]
    return out


def _find_humidifi_instructions(tx: dict):
    """Yield (ix_dict, location) where programId == HumidiFi.
    location 为 ('outer', idx) 或 ('inner', parent_idx, sub_idx).
    """
    for idx, ix in enumerate(tx.get("instructions") or []):
        if ix.get("programId") == HUMIDIFI_PROGRAM:
            yield ix, ("outer", idx)
    for inner in tx.get("innerInstructions") or []:
        parent = inner.get("index")
        for j, ix in enumerate(inner.get("instructions") or []):
            if ix.get("programId") == HUMIDIFI_PROGRAM:
                yield ix, ("inner", parent, j)


def _is_oracle_update(ix: dict) -> bool:
    accts = ix.get("accounts") or []
    if len(accts) != ORACLE_UPDATE_ACCOUNT_COUNT:
        return False
    raw = _decode_b58(ix.get("data"))
    return len(raw) == ORACLE_UPDATE_DATA_LEN


def _token_deltas_by_owner(tx: dict) -> dict[str, dict[str, int]]:
    """{owner: {mint: post_amount - pre_amount (int)}}."""
    pre = tx.get("preTokenBalances") or []
    post = tx.get("postTokenBalances") or []

    def _index(balances):
        out = {}
        for b in balances:
            key = (b.get("owner"), b.get("mint"), b.get("accountIndex"))
            amt = int((b.get("uiTokenAmount") or {}).get("amount", "0") or "0")
            out[key] = amt
        return out

    pre_i = _index(pre)
    post_i = _index(post)
    keys = set(pre_i) | set(post_i)
    deltas: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for owner, mint, _ in keys:
        if owner is None:
            continue
        delta = post_i.get((owner, mint, _), 0) - pre_i.get((owner, mint, _), 0)
        if delta:
            deltas[owner][mint] += delta
    return deltas


def parse_tx(tx: dict):
    """Return (oracle_rows, swap_rows)."""
    oracle_rows, swap_rows = [], []
    sig = tx.get("signature")
    slot = tx.get("slot")
    bt = tx.get("blockTime")
    err = tx.get("err")
    fee = tx.get("fee")
    cu = tx.get("computeUnitsConsumed")
    status = "success" if err is None else "revert"
    keys = _flatten_account_keys(tx)
    tip = _sum_tip(tx)

    oracle_found = False
    humidifi_ixs = list(_find_humidifi_instructions(tx))
    for ix, loc in humidifi_ixs:
        if _is_oracle_update(ix) and loc[0] == "outer":
            oracle_found = True
            accounts = ix.get("accounts") or []
            state_account = accounts[0] if accounts else None
            raw = _decode_b58(ix.get("data"))
            decoded = _parse_oracle_data(raw)
            oracle_rows.append(
                {
                    "signature": sig,
                    "slot": slot,
                    "block_time": bt,
                    "status": status,
                    "err_code": json.dumps(err) if err else "",
                    "fee": fee,
                    "cu_consumed": cu,
                    "tip_amount": tip,
                    "state_account": state_account,
                    "tag": decoded["tag"],
                    **{f"p{i}": decoded[f"p{i}"] for i in range(8)},
                }
            )
            break  # 一笔 tx 通常只有一个 oracle update
    if oracle_found:
        return oracle_rows, swap_rows

    # 不是 oracle update → 按 swap 处理（每条 HumidiFi CPI 调用各推一行）
    if err is not None:
        return oracle_rows, swap_rows  # 失败的 swap 先忽略
    if not humidifi_ixs:
        return oracle_rows, swap_rows

    deltas = _token_deltas_by_owner(tx)
    for ix, loc in humidifi_ixs:
        accts = ix.get("accounts") or []
        if len(accts) < 2:
            continue
        # accounts[1] = 池状态 PDA（与 oracle_update 的 accounts[0] 对齐）
        state_account = accts[1]
        # trader = accounts[0]，用户侧通常在 deltas 里直接可见（他们的 ATA owner == trader）
        trader = accts[0]

        # 优先用 trader 侧的 deltas；若 trader 没有 SOL 变动再尝试其他 owner
        candidate_owners = [trader] + [o for o in deltas.keys() if o != trader]
        owner = None
        d_sol = 0
        for cand in candidate_owners:
            d = deltas.get(cand, {})
            if d.get(SOL_MINT, 0) != 0:
                owner = cand
                d_sol = d[SOL_MINT]
                break
        if owner is None or d_sol == 0:
            continue  # 这条 CPI 不涉及 wSOL，跳过

        owner_delta = deltas[owner]
        # trader 视角：d_sol < 0 表示 trader 付出 SOL（sell_sol），> 0 表示收到 SOL（buy_sol）
        # 如果我们用的是池 vault 侧，符号相反；用 trader 作为默认视角
        if owner == trader:
            side = "sell_sol" if d_sol < 0 else "buy_sol"
        else:
            side = "sell_sol" if d_sol > 0 else "buy_sol"

        base_amount = abs(d_sol) / 1e9
        # 找 |delta| 最大的非 SOL mint 当 quote
        other = [(m, dl) for m, dl in owner_delta.items() if m != SOL_MINT and dl != 0]
        if other:
            other.sort(key=lambda x: -abs(x[1]))
            quote_mint, d_quote = other[0]
        else:
            quote_mint, d_quote = None, 0

        # 只有 quote==USDC 才填 price_usdc_per_sol（USDC decimals=6）
        if quote_mint == USDC_MINT and base_amount:
            quote_amount_usdc = abs(d_quote) / 1e6
            price_usdc_per_sol = quote_amount_usdc / base_amount
            amount_usd = quote_amount_usdc
        else:
            quote_amount_usdc = None
            price_usdc_per_sol = None
            amount_usd = None

        swap_rows.append(
            {
                "signature": sig,
                "slot": slot,
                "block_time": bt,
                "state_account": state_account,
                "trader": trader,
                "side": side,
                "base_amount_sol": base_amount,
                "quote_mint": quote_mint or "",
                "quote_amount_raw": abs(d_quote),
                "quote_amount_usdc": quote_amount_usdc,
                "price_usdc_per_sol": price_usdc_per_sol,
                "amount_usd": amount_usd,
                "cu_consumed": cu,
                "fee": fee,
                "ix_location": loc[0],
            }
        )
    return oracle_rows, swap_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--raw", action="append", default=None, help="额外 raw jsonl.gz 路径，可多次指定；默认只读 paths.humidifi_raw")
    args = ap.parse_args()
    paths = data_paths(args.date)
    raw_paths = [paths["humidifi_raw"]]
    if args.raw:
        from pathlib import Path as _P
        raw_paths = [_P(p) for p in args.raw]
    raw_paths = [p for p in raw_paths if p.exists()]
    if not raw_paths:
        raise SystemExit(f"no raw files: expected {paths['humidifi_raw']} or --raw ...")
    print(f"[parse] inputs: {[str(p) for p in raw_paths]}")

    oracle_fields = [
        "signature", "slot", "block_time", "status", "err_code", "fee",
        "cu_consumed", "tip_amount", "state_account", "tag",
        "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7",
    ]
    swap_fields = [
        "signature", "slot", "block_time", "state_account", "trader", "side",
        "base_amount_sol", "quote_mint", "quote_amount_raw", "quote_amount_usdc",
        "price_usdc_per_sol", "amount_usd", "cu_consumed", "fee", "ix_location",
    ]

    n_total = n_oracle = n_swap = 0
    state_counts: dict[str, int] = defaultdict(int)
    seen_sigs: set[str] = set()  # 跨多 raw 文件去重
    with (
        paths["oracle_csv"].open("w", newline="") as fo,
        paths["swap_csv"].open("w", newline="") as fs,
    ):
        w_o = csv.DictWriter(fo, fieldnames=oracle_fields); w_o.writeheader()
        w_s = csv.DictWriter(fs, fieldnames=swap_fields); w_s.writeheader()
        for raw_path in raw_paths:
            with gzip.open(raw_path, "rt") as fin:
                for line in tqdm(fin, desc=f"parse {raw_path.name}", unit="tx"):
                    n_total += 1
                    try:
                        tx = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if tx.get("missing"):
                        continue
                    sig = tx.get("signature")
                    if sig in seen_sigs:
                        continue
                    seen_sigs.add(sig)
                    o_rows, s_rows = parse_tx(tx)
                    for r in o_rows:
                        w_o.writerow(r)
                        n_oracle += 1
                        state_counts[r.get("state_account") or ""] += 1
                    for r in s_rows:
                        w_s.writerow(r)
                        n_swap += 1

    print(f"[parse] scanned={n_total} oracle_updates={n_oracle} swaps={n_swap}")
    print("[parse] top HumidiFi state accounts (by oracle update count):")
    for addr, c in sorted(state_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {addr}  {c}")
    print(f"[parse] wrote: {paths['oracle_csv']}  {paths['swap_csv']}")


if __name__ == "__main__":
    main()
