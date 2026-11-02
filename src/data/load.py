"""Load transcripts of selected runs, reconstruct cost, write runs.parquet / turns.parquet (spec §3.3, §5)."""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.cost.tokenizers import count_tokens
from src.data.parse import parse_run
from src.utils import load_config


def _tok(texts):
    return count_tokens(texts, num_threads=1)


def process_shard(args):
    path, rows, uids, out_dir, cfg_cost = args
    out_r = os.path.join(out_dir, f"runs_{os.path.basename(path)}")
    out_t = os.path.join(out_dir, f"turns_{os.path.basename(path)}")
    if os.path.exists(out_r) and os.path.exists(out_t):
        return out_r
    pf = pq.ParquetFile(path)
    col = "conversations" if "conversations" in pf.schema_arrow.names else "messages"
    # map global row -> (row group, offset)
    rg_starts = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)])
    order = np.argsort(rows)
    rows, uids = np.asarray(rows)[order], np.asarray(uids)[order]
    rg_of = np.searchsorted(rg_starts, rows, side="right") - 1
    runs, turns = [], []
    lam = dict(lambda_c=cfg_cost["lambda_c"], lambda_o=cfg_cost["lambda_o"], lambda_o_hi=cfg_cost["lambda_o_hi"])
    for rg in np.unique(rg_of):
        m = rg_of == rg
        tb = pf.read_row_group(int(rg), columns=[col])
        local = rows[m] - rg_starts[rg]
        convs = tb.column(col).take(local).to_pylist()
        for uid, conv in zip(uids[m], convs):
            try:
                run, tr = parse_run(conv, _tok, cfg_cost["c_msg"], lam)
                run["parse_ok"] = True
            except Exception as e:  # unparseable -> excluded later (T0 step 1)
                runs.append(dict(uid=uid, parse_ok=False, parse_error=repr(e)[:200]))
                continue
            run["uid"] = uid
            run.pop("msg_prefix_hashes", None)
            runs.append(run)
            t = pd.DataFrame(tr)
            t.insert(0, "uid", uid)
            turns.append(t)
    pd.DataFrame(runs).to_parquet(out_r, index=False)
    (pd.concat(turns, ignore_index=True) if turns else pd.DataFrame()).to_parquet(out_t, index=False)
    return out_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--workers", type=int, default=4)
    cfg = load_config(ap.parse_args().config)
    cache = cfg["data"]["cache_dir"]
    sel = pd.read_parquet(os.path.join(cache, "selection.parquet"))
    parts = os.path.join(cache, "parts")
    os.makedirs(parts, exist_ok=True)
    jobs = []
    for shard, g in sel.groupby("shard"):
        path = os.path.join(cfg["data"]["raw_dir"], f"train-{shard:05d}-of-00038.parquet")
        jobs.append((path, g.row.values, g.uid.values, parts, cfg["cost"]))
    jobs.sort(key=lambda j: -len(j[1]))
    with ProcessPoolExecutor(ap.parse_args().workers if False else 4) as ex:
        for o in ex.map(process_shard, jobs):
            print(o, flush=True)


if __name__ == "__main__":
    main()
