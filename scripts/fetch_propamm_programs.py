#!/usr/bin/env python3

import base64
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


RPC_URL = None

TARGETS = [
    {
        "name": "HumidiFi",
        "program_id": "9H6tua7jkLhdm3w8BvgpTn5LZNU7g4ZynDmCiNN3q6Rp",
        "sources": [
            "https://orbmarkets.io/address/9H6tua7jkLhdm3w8BvgpTn5LZNU7g4ZynDmCiNN3q6Rp/history",
            "https://solscan.io/account/9H6tua7jkLhdm3w8BvgpTn5LZNU7g4ZynDmCiNN3q6Rp",
            "https://danielandsolana.substack.com/p/inside-humidifis-34-billion-mystery",
        ],
    },
    {
        "name": "GoonFi",
        "program_id": "goonERTdGsjnkZqWuVjs73BZ3Pb9qoCUdBUL17BnS5j",
        "sources": [
            "https://orbmarkets.io/address/goonERTdGsjnkZqWuVjs73BZ3Pb9qoCUdBUL17BnS5j/history",
            "https://solscan.io/account/goonERTdGsjnkZqWuVjs73BZ3Pb9qoCUdBUL17BnS5j",
            "https://solana.com/news/understanding-proprietary-amms",
        ],
    },
    {
        "name": "ZeroFi",
        "program_id": "ZERor4xhbUycZ6gb9ntrhqscUcZmAbQDjEAtCf4hbZY",
        "sources": [
            "https://orbmarkets.io/address/ZERor4xhbUycZ6gb9ntrhqscUcZmAbQDjEAtCf4hbZY/history",
            "https://solscan.io/account/ZERor4xhbUycZ6gb9ntrhqscUcZmAbQDjEAtCf4hbZY",
            "https://www.helius.dev/blog/solanas-proprietary-amm-revolution",
        ],
    },
]

STRING_FILTER = (
    "price|quote|swap|oracle|update|get|set|init|market|pool|error|account|"
    "instruction|router|state|dflow|jupiter|vault|amm|bid|ask"
)

ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
BASE58_MAP = {c: i for i, c in enumerate(ALPHABET)}


def load_rpc_url():
    env_path = Path("/home/ubuntu/blockchain/.env")
    for line in env_path.read_text().splitlines():
        if line.startswith("RPC_URL="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("RPC_URL not found in .env")


def rpc_call(method, params, retries=6):
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    data = json.dumps(payload).encode()
    last_error = None
    for attempt in range(retries):
        try:
            req = Request(RPC_URL, data=data, headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=45) as resp:
                result = json.load(resp)
            if "error" in result:
                raise RuntimeError(result["error"])
            return result["result"]
        except HTTPError as exc:
            last_error = exc
            if exc.code != 429 or attempt == retries - 1:
                raise
        except Exception as exc:
            last_error = exc
            if attempt == retries - 1:
                raise
        time.sleep(min(2 ** attempt, 12))
    raise RuntimeError(f"RPC failed: {last_error}")


def b58decode(value):
    number = 0
    for ch in value:
        number = number * 58 + BASE58_MAP[ch]
    out = bytearray()
    while number:
        number, rem = divmod(number, 256)
        out.append(rem)
    out = bytes(reversed(out))
    pad = 0
    for ch in value:
        if ch == "1":
            pad += 1
        else:
            break
    return (b"\x00" * pad) + out


def run_command(args):
    proc = subprocess.run(args, check=False, capture_output=True, text=True)
    return {
        "command": " ".join(args),
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def sample_instruction_accounts(tx, account_indexes):
    keys = list(tx["transaction"]["message"]["accountKeys"])
    loaded = tx["meta"].get("loadedAddresses", {})
    keys.extend(loaded.get("writable", []))
    keys.extend(loaded.get("readonly", []))
    return [keys[i] if i < len(keys) else f"<idx:{i}>" for i in account_indexes]


def cluster_instructions(program_id, signatures, per_program_limit=20):
    clusters = Counter()
    examples = {}
    processed = 0

    for item in signatures:
        if processed >= per_program_limit:
            break
        tx = rpc_call(
            "getTransaction",
            [
                item["signature"],
                {
                    "encoding": "json",
                    "commitment": "confirmed",
                    "maxSupportedTransactionVersion": 0,
                },
            ],
        )
        if not tx:
            continue
        processed += 1
        keys = tx["transaction"]["message"]["accountKeys"]
        for ix in tx["transaction"]["message"]["instructions"]:
            pid = keys[ix["programIdIndex"]]
            if pid != program_id:
                continue
            raw = b58decode(ix["data"]) if ix.get("data") else b""
            key = (len(ix.get("accounts", [])), len(raw))
            clusters[key] += 1
            if key not in examples:
                examples[key] = {
                    "signature": item["signature"],
                    "memo": item.get("memo"),
                    "error": item.get("err"),
                    "accounts": sample_instruction_accounts(tx, ix.get("accounts", [])),
                    "data_hex_prefix": raw[:24].hex(),
                    "data_len_mod_8": len(raw) % 8,
                }
        time.sleep(0.15)

    out = []
    for (account_count, data_len), count in clusters.most_common():
        item = {
            "sample_count": count,
            "account_count": account_count,
            "data_len": data_len,
            "layout_hint": infer_layout_hint(data_len),
        }
        item.update(examples[(account_count, data_len)])
        out.append(item)
    return out


def infer_layout_hint(data_len):
    if data_len == 0:
        return "no_data"
    if data_len % 8 == 0:
        return f"{data_len // 8}x_u64_or_i64_words"
    if data_len > 1 and (data_len - 1) % 8 == 0:
        return f"1_byte_tag_plus_{(data_len - 1) // 8}x_8byte_fields"
    if data_len > 8 and (data_len - 8) % 8 == 0:
        return f"8_byte_tag_plus_{(data_len - 8) // 8}x_8byte_fields"
    return "mixed_or_packed"


def filtered_strings(elf_path):
    strings_cmd = run_command(["strings", "-n", "8", str(elf_path)])
    if strings_cmd["exit_code"] != 0:
        return strings_cmd
    lines = []
    for line in strings_cmd["stdout"].splitlines():
        lower = line.lower()
        if any(
            token in lower
            for token in [
                "price",
                "quote",
                "swap",
                "oracle",
                "update",
                "init",
                "market",
                "pool",
                "error",
                "account",
                "instruction",
                "router",
                "jupiter",
                "dflow",
                "state",
                "contract/src/",
            ]
        ):
            lines.append(line)
    return {
        "command": strings_cmd["command"],
        "exit_code": 0,
        "matches": lines[:120],
    }


def analyze_target(base_dir, target):
    out_dir = base_dir / target["name"].lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    program_info = rpc_call(
        "getAccountInfo",
        [target["program_id"], {"encoding": "jsonParsed", "commitment": "confirmed"}],
    )["value"]
    program_data_address = program_info["data"]["parsed"]["info"]["programData"]

    program_data_info = rpc_call(
        "getAccountInfo",
        [program_data_address, {"encoding": "jsonParsed", "commitment": "confirmed"}],
    )["value"]
    program_data_raw_info = rpc_call(
        "getAccountInfo",
        [program_data_address, {"encoding": "base64", "commitment": "confirmed"}],
    )["value"]

    raw = base64.b64decode(program_data_raw_info["data"][0])
    elf_offset = raw.index(b"\x7fELF")
    elf = raw[elf_offset:]

    raw_path = out_dir / "programdata.bin"
    elf_path = out_dir / "program.so"
    raw_path.write_bytes(raw)
    elf_path.write_bytes(elf)

    dynsym = run_command(["readelf", "-s", str(elf_path)])
    elf_header = run_command(["readelf", "-h", str(elf_path)])
    file_info = run_command(["file", str(elf_path)])
    strings_info = filtered_strings(elf_path)

    signatures = rpc_call(
        "getSignaturesForAddress",
        [target["program_id"], {"limit": 25, "commitment": "confirmed"}],
    )
    clusters = cluster_instructions(target["program_id"], signatures, per_program_limit=18)

    metadata = {
        "name": target["name"],
        "program_id": target["program_id"],
        "sources": target["sources"],
        "program_account": {
            "owner": program_info["owner"],
            "executable": program_info["executable"],
            "lamports": program_info["lamports"],
            "space": program_info["space"],
        },
        "program_data_account": {
            "address": program_data_address,
            "authority": program_data_info["data"]["parsed"]["info"].get("authority"),
            "slot": program_data_info["data"]["parsed"]["info"].get("slot"),
            "lamports": program_data_info["lamports"],
            "space": program_data_info["space"],
        },
        "binary": {
            "raw_len": len(raw),
            "elf_offset": elf_offset,
            "elf_len": len(elf),
            "sha256": hashlib.sha256(elf).hexdigest(),
            "file": file_info,
            "elf_header": elf_header,
            "dynsym": dynsym,
        },
        "static_hints": strings_info,
        "instruction_clusters": clusters,
    }

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


def main():
    global RPC_URL
    RPC_URL = load_rpc_url()

    base_dir = Path("/home/ubuntu/blockchain/contracts")
    base_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for target in TARGETS:
        summary.append(analyze_target(base_dir, target))
        time.sleep(0.5)

    summary_path = base_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
