"""抓取 HumidiFi 某一 UTC 天的全部链上交易。

两阶段：
  1. getSignaturesForAddress 反向翻页拉全部签名到目标时间窗
  2. 按 100 一批用 JSON-RPC batch getTransaction 拉 jsonParsed 交易

落盘：data/raw/humidifi_txs_YYYYMMDD.jsonl.gz  （每行一个 tx）
失败签名：data/raw/humidifi_failed_YYYYMMDD.jsonl
签名索引：data/raw/humidifi_sigs_YYYYMMDD.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from tqdm import tqdm

from .config import (
    HELIUS_RPC_URL,
    HUMIDIFI_PROGRAM,
    data_paths,
    utc_day_bounds,
)

SIGS_PER_CALL = 1000
MAX_CONCURRENCY = 8  # 免费档 Helius 实测 ~10 RPS，留一点余量
MAX_RETRIES = 6

HEADERS = {
    "user-agent": "Mozilla/5.0 (Week3 pilot)",
    "accept": "application/json",
    "content-type": "application/json",
}


async def post_json(session: aiohttp.ClientSession, payload):
    """POST JSON-RPC with retry + backoff. 单条调用（batch 在 Helius 上被 403）。"""
    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(
                HELIUS_RPC_URL,
                json=payload,
                headers=HEADERS,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as r:
                if r.status == 429:
                    await asyncio.sleep(0.5 + 0.5 * attempt + random.random())
                    continue
                if r.status >= 500:
                    await asyncio.sleep(0.5 * (2**attempt) + random.random())
                    continue
                r.raise_for_status()
                return await r.json()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(0.5 * (2**attempt) + random.random())
    raise RuntimeError("unreachable")


async def _jump_before_sig(session: aiohttp.ClientSession, target_ts: int) -> str | None:
    """找一个 blockTime ≈ target_ts 附近的任意 sig，用作 getSignaturesForAddress 的 before= 起点，
    避免从最新一路翻回老日期。"""
    # 1) 当前 slot + blockTime
    d = await post_json(session, {"jsonrpc": "2.0", "id": 1, "method": "getSlot", "params": []})
    now_slot = d["result"]
    d = await post_json(session, {"jsonrpc": "2.0", "id": 2, "method": "getBlockTime", "params": [now_slot]})
    now_ts = d["result"]
    # 2) 线性外推 slot（Solana ~0.4s / slot），向前多跳一点点到 target_ts+120，保险
    slot_delta = int((now_ts - (target_ts + 120)) / 0.4)
    target_slot = max(1, now_slot - slot_delta)
    print(f"[jump] now_slot={now_slot} now_ts={now_ts} → target_ts={target_ts} est_slot={target_slot}")
    # 3) getBlock 找附近的 block（可能被 skip，逐步向前试）
    for offset in range(0, 500):
        slot = target_slot + offset
        payload = {
            "jsonrpc": "2.0", "id": 3, "method": "getBlock",
            "params": [slot, {"encoding": "json", "transactionDetails": "signatures", "maxSupportedTransactionVersion": 0, "rewards": False}],
        }
        try:
            d = await post_json(session, payload)
        except Exception:
            continue
        if isinstance(d, dict) and d.get("result"):
            res = d["result"]
            sigs = res.get("signatures") or []
            if sigs:
                print(f"[jump] landed on slot={slot} blockTime={res.get('blockTime')} first_sig={sigs[0][:12]}...")
                return sigs[0]
    return None


async def fetch_signatures(session: aiohttp.ClientSession, start_ts: int, end_ts: int, sigs_path: Path, jump: bool = False, address: str | None = None) -> list[dict]:
    """分页拉签名，直到 blockTime < start_ts 为止。

    address: 默认 HumidiFi 程序地址；也可传入某个池子 PDA（方案 B：按池翻页抓 oracle+swap）。
    """
    addr = address or HUMIDIFI_PROGRAM
    print(f"[sigs] fetching signatures for {addr} in [{start_ts}, {end_ts}) ...")
    all_sigs: list[dict] = []
    before: str | None = None
    if jump:
        before = await _jump_before_sig(session, end_ts)
        if before is None:
            print("[sigs][warn] jump 失败，回退到顺序翻页")
    calls = 0
    pbar = tqdm(desc=f"sigs[{addr[:6]}..]", unit="sig")
    while True:
        params = [addr, {"limit": SIGS_PER_CALL}]
        if before:
            params[1]["before"] = before
        payload = {"jsonrpc": "2.0", "id": calls, "method": "getSignaturesForAddress", "params": params}
        data = await post_json(session, payload)
        calls += 1
        res = data.get("result") or []
        if not res:
            break
        # 按时间窗过滤 + 判停
        stop = False
        for item in res:
            bt = item.get("blockTime")
            if bt is None:
                continue
            if bt >= end_ts:
                continue
            if bt < start_ts:
                stop = True
                break
            all_sigs.append(item)
        pbar.update(len(res))
        before = res[-1]["signature"]
        if stop or len(res) < SIGS_PER_CALL:
            break
    pbar.close()
    # 落盘签名索引
    with sigs_path.open("w") as f:
        for s in all_sigs:
            f.write(json.dumps(s, separators=(",", ":")) + "\n")
    print(f"[sigs] {len(all_sigs)} signatures in window; {calls} RPC calls; saved -> {sigs_path}")
    return all_sigs


def slim_tx(sig: str, tx: dict | None) -> dict:
    """只保留解析必需字段，节省磁盘。"""
    if tx is None:
        return {"signature": sig, "missing": True}
    meta = tx.get("meta") or {}
    message = (tx.get("transaction") or {}).get("message") or {}
    return {
        "signature": sig,
        "slot": tx.get("slot"),
        "blockTime": tx.get("blockTime"),
        "err": meta.get("err"),
        "fee": meta.get("fee"),
        "computeUnitsConsumed": meta.get("computeUnitsConsumed"),
        "preBalances": meta.get("preBalances"),
        "postBalances": meta.get("postBalances"),
        "preTokenBalances": meta.get("preTokenBalances"),
        "postTokenBalances": meta.get("postTokenBalances"),
        "loadedAddresses": meta.get("loadedAddresses"),
        "innerInstructions": meta.get("innerInstructions"),
        "accountKeys": message.get("accountKeys"),
        "instructions": message.get("instructions"),
    }


async def fetch_one_tx(session: aiohttp.ClientSession, sig: str, sem: asyncio.Semaphore) -> tuple[str, dict | None]:
    async with sem:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTransaction",
            "params": [
                sig,
                {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0, "commitment": "confirmed"},
            ],
        }
        try:
            data = await post_json(session, payload)
        except Exception:
            return sig, None
    return sig, (data.get("result") if isinstance(data, dict) else None)


async def fetch_all_transactions(session: aiohttp.ClientSession, sigs: list[str], out_path: Path, fail_path: Path, max_tx: int | None = None):
    if max_tx is not None and len(sigs) > max_tx:
        print(f"[tx] 限流: 只抓取前 {max_tx}/{len(sigs)} 笔（时间近 → 远）")
        sigs = sigs[:max_tx]
    total = len(sigs)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    print(f"[tx] {total} txs @ concurrency {MAX_CONCURRENCY} (single getTransaction)")

    failed = 0
    wrote = 0
    start = time.time()
    with gzip.open(out_path, "wt") as fout, fail_path.open("w") as fbad:
        pbar = tqdm(total=total, desc="tx", unit="tx")
        tasks = [asyncio.create_task(fetch_one_tx(session, s, sem)) for s in sigs]
        for coro in asyncio.as_completed(tasks):
            sig, tx = await coro
            if tx is None:
                failed += 1
                fbad.write(sig + "\n")
            else:
                fout.write(json.dumps(slim_tx(sig, tx), separators=(",", ":")) + "\n")
                wrote += 1
            pbar.update(1)
        pbar.close()
    elapsed = time.time() - start
    print(f"[tx] done: wrote={wrote} failed={failed} elapsed={elapsed:.1f}s ({wrote/max(elapsed,1):.1f} tx/s)")
    return wrote, failed, elapsed


async def main_async(args):
    paths = data_paths(args.date)
    start_ts, end_ts = utc_day_bounds(args.date)
    if args.recent_hours:
        end_ts = int(time.time())
        start_ts = end_ts - args.recent_hours * 3600
        print(f"[cfg] --recent-hours={args.recent_hours} → 时间窗 [{start_ts}, {end_ts}) = {datetime.fromtimestamp(start_ts, tz=timezone.utc)} ~ {datetime.fromtimestamp(end_ts, tz=timezone.utc)}")
    elif args.hours:
        end_ts = start_ts + args.hours * 3600
        print(f"[cfg] --hours={args.hours} → 时间窗 [{start_ts}, {end_ts}) = {datetime.fromtimestamp(start_ts, tz=timezone.utc)} ~ {datetime.fromtimestamp(end_ts, tz=timezone.utc)}")

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENCY * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        # 签名来源优先级：--from-sigs-file > --pool-accounts > --skip-sigs 复用 > 新拉
        if args.from_sigs_file:
            sig_strs = [l.strip() for l in Path(args.from_sigs_file).open() if l.strip()]
            print(f"[sigs] 来自文件 {args.from_sigs_file}: {len(sig_strs)} 条")
        elif args.pool_accounts:
            pool_list = [a.strip() for a in args.pool_accounts.split(",") if a.strip()]
            agg: dict[str, dict] = {}
            suffix = args.out_suffix or "_poolpda"
            for addr in pool_list:
                per_pool_path = paths["humidifi_sigs"].with_name(
                    f"humidifi_sigs_{args.date.replace('-', '')}{suffix}_{addr[:8]}.jsonl"
                )
                sigs_i = await fetch_signatures(session, start_ts, end_ts, per_pool_path, jump=args.jump, address=addr)
                for s in sigs_i:
                    agg[s["signature"]] = s
            sig_strs = list(agg.keys())
            print(f"[sigs] --pool-accounts 合并去重后: {len(sig_strs)} 条（{len(pool_list)} 个池）")
        elif args.skip_sigs and paths["humidifi_sigs"].exists():
            sigs = [json.loads(l) for l in paths["humidifi_sigs"].open()]
            sigs = [s for s in sigs if start_ts <= (s.get("blockTime") or 0) < end_ts]
            print(f"[sigs] 复用已存在: {len(sigs)} 条")
            sig_strs = [s["signature"] for s in sigs]
        else:
            sigs = await fetch_signatures(session, start_ts, end_ts, paths["humidifi_sigs"], jump=args.jump)
            sig_strs = [s["signature"] for s in sigs]

        if args.shuffle_sample:
            import random
            random.seed(args.shuffle_seed)
            if args.shuffle_sample < len(sig_strs):
                sig_strs = random.sample(sig_strs, args.shuffle_sample)
            print(f"[sigs] --shuffle-sample 后: {len(sig_strs)} 条")

        if args.sigs_only:
            print("[exit] --sigs-only")
            return

        out_raw = paths["humidifi_raw"]
        out_fail = paths["humidifi_failed"]
        if args.out_suffix:
            # humidifi_txs_YYYYMMDD.jsonl.gz → humidifi_txs_YYYYMMDD<suffix>.jsonl.gz
            out_raw = out_raw.with_name(f"humidifi_txs_{args.date.replace('-', '')}{args.out_suffix}.jsonl.gz")
            out_fail = out_fail.with_name(f"humidifi_failed_{args.date.replace('-', '')}{args.out_suffix}.jsonl")
        print(f"[out] raw={out_raw}  failed={out_fail}")
        await fetch_all_transactions(
            session,
            sig_strs,
            out_raw,
            out_fail,
            max_tx=args.max_tx,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="UTC date YYYY-MM-DD")
    ap.add_argument("--hours", type=int, default=None, help="覆盖窗口小时数 (默认 24)，从 date 00:00 UTC 起")
    ap.add_argument("--recent-hours", type=int, default=None, help="改为最近 N 小时（截至脚本启动时间），pilot 推荐用")
    ap.add_argument("--max-tx", type=int, default=None, help="对 tx 细节拉取限流（取最近的 N 笔）")
    ap.add_argument("--sigs-only", action="store_true", help="只拉签名，不拉 tx 细节")
    ap.add_argument("--skip-sigs", action="store_true", help="复用已有签名索引")
    ap.add_argument("--jump", action="store_true", help="用 getSlot+getBlock 跳到 end_ts 附近的 sig 作为 before= 起点（老日期大幅提速）")
    ap.add_argument("--from-sigs-file", default=None, help="跳过翻页，直接从给定文件读 tx_id（每行一个）去拉 tx 细节")
    ap.add_argument("--pool-accounts", default=None, help="逗号分隔的池子 PDA 地址列表；按每个池 getSignaturesForAddress 翻页，合并去重（方案 B：全覆盖 top 池）")
    ap.add_argument("--shuffle-sample", type=int, default=None, help="从签名池随机抽 N 条（配合 --skip-sigs 做 oracle 稀疏覆盖）")
    ap.add_argument("--shuffle-seed", type=int, default=42)
    ap.add_argument("--out-suffix", default="", help="输出 raw/failed 文件名后缀，避免与主任务覆盖")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
