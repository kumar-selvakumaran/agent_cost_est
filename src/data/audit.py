"""Pass 1 metadata scan (spec §3.3) over raw AgentTrove Parquet shards.

Reads every row of a shard, keeps the metadata columns and derives cheap
per-row transcript statistics (message counts, hashes, flags) without
tokenizing. Output: data_cache/meta/shard_XXXXX.parquet, one row per dataset row.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import pyarrow.parquet as pq
import pandas as pd
import xxhash

META_COLS = [
    "agent", "model", "date", "task", "episode", "run_id", "trial_name",
    "model_provider", "original_source", "original_teacher", "result",
    "trace_source", "path", "instruction", "judgment", "split",
]
TASK_MARK = "Task Description:"
TERM_MARK = "Current terminal state:"
SUMM_RE = re.compile(
    r"(picking up work from a previous|previous (?:ai )?agent|other agent provided|"
    r"summary of (?:the )?(?:previous|prior) (?:work|agent|conversation)|"
    r"(?:conversation|context) (?:has been|was) summari[sz]ed)", re.I)
OBS_PREFIXES = ("Previous response had warnings", "Current terminal state", "New Terminal Output")
WARNING_PREFIX = "[SYSTEM WARNING]"
TC_RE = re.compile(r'"task_complete"\s*:\s*true', re.I)


def extract_task_text(first_user: str) -> str:
    """Task statement embedded in the terminus-2 first user message."""
    i = first_user.rfind(TASK_MARK)
    if i < 0:
        return first_user
    s = first_user[i + len(TASK_MARK):]
    j = s.rfind(TERM_MARK)
    if j >= 0:
        s = s[:j]
    return s.strip()


def h64(s: str) -> int:
    return xxhash.xxh64_intdigest(s.encode("utf-8", "surrogatepass"))


def conv_stats(conv) -> dict:
    roles = conv["role"] if isinstance(conv, dict) else [m.get("role") for m in conv]
    contents = conv["content"] if isinstance(conv, dict) else [m.get("content") for m in conv]
    contents = [c if isinstance(c, str) else "" for c in contents]
    n = len(roles)
    n_asst = sum(r in ("assistant", "gpt") for r in roles)
    n_user = sum(r in ("user", "human") for r in roles)
    n_sys = sum(r == "system" for r in roles)
    first_user = next((c for r, c in zip(roles, contents) if r in ("user", "human")), "")
    task_text = extract_task_text(first_user)
    i_task = first_user.rfind(TASK_MARK)
    harness = first_user[:i_task] if i_task >= 0 else ""
    hx = xxhash.xxh64()
    for r, c in zip(roles, contents):
        hx.update((r or "").encode()); hx.update(b"\x00")
        hx.update(c.encode("utf-8", "surrogatepass")); hx.update(b"\x01")
    think = sum(1 for r, c in zip(roles, contents) if r in ("assistant", "gpt") and ("<think>" in c or "</think>" in c))
    summ_first = None
    n_summ = 0
    seen_asst = False
    for r, c in zip(roles, contents):
        if r in ("assistant", "gpt"):
            seen_asst = True
            continue
        if r in ("user", "human") and seen_asst:
            head = c[:600]
            if head.startswith(WARNING_PREFIX) or head.startswith(OBS_PREFIXES):
                continue
            if SUMM_RE.search(head):
                n_summ += 1
                if summ_first is None:
                    summ_first = c[:80]
    last_asst = next((c for r, c in zip(reversed(roles), reversed(contents)) if r in ("assistant", "gpt")), "")
    lens = [len(c) for c in contents]
    return dict(
        n_msgs=n, n_asst=n_asst, n_user=n_user, n_sys=n_sys,
        first_role=roles[0] if n else None,
        total_chars=sum(lens), max_msg_chars=max(lens) if lens else 0,
        first_user_chars=len(first_user), first_user_head=first_user[:100], task_text_chars=len(task_text),
        h_conv=hx.intdigest(), h_first_user=h64(first_user), h_task_text=h64(task_text),
        h_harness=h64(harness), n_think=think, n_summ_msgs=n_summ, summ_first80=summ_first,
        last_task_complete=bool(TC_RE.search(last_asst[-3000:])) if last_asst else False,
        has_warning_ctx=any(isinstance(c, str) and c.startswith(WARNING_PREFIX + " Context is") for c in contents),
    )


def scan_shard(path: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    shard = int(re.search(r"train-(\d+)-of", path).group(1))
    out = os.path.join(out_dir, f"shard_{shard:05d}.parquet")
    if os.path.exists(out):
        return out
    pf = pq.ParquetFile(path)
    have = set(pf.schema_arrow.names)
    cols = [c for c in META_COLS if c in have]
    msg_col = "conversations" if "conversations" in have else "messages"
    frames = []
    offset = 0
    for rg in range(pf.metadata.num_row_groups):
        tb = pf.read_row_group(rg, columns=cols + [msg_col])
        meta = tb.select(cols).to_pandas()
        convs = tb.column(msg_col).to_pylist()
        stats = pd.DataFrame([conv_stats(c) if c else conv_stats([]) for c in convs])
        df = pd.concat([meta.reset_index(drop=True), stats], axis=1)
        df.insert(0, "shard", shard)
        df.insert(1, "row", range(offset, offset + len(df)))
        offset += len(df)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for c in META_COLS:
        if c not in df:
            df[c] = None
    df.to_parquet(out + ".tmp", index=False)
    os.replace(out + ".tmp", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/home/user/hf_raw")
    ap.add_argument("--out", default="data_cache/meta")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(a.raw, "train-*.parquet"))):
        print(scan_shard(p, a.out), flush=True)


if __name__ == "__main__":
    main()
