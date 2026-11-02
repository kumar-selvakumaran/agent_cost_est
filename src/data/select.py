"""Pass 2 (spec §3.3): exclusions on metadata, cell definition and stratified cell selection.

Writes:
  data_cache/meta_derived.parquet   all terminus-2 rows with derived ids + exclusion step
  data_cache/selection.parquet      selected runs (uid -> shard,row) with ids
  results/tables/t0_exclusion_flow.csv, results/tables/sample_size.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from src.data.ids import derive_ids
from src.utils import load_config, seed_for


def union_find_groups(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Connected components over a bipartite graph of keys a<->b; returns component id per row."""
    ka, ia = np.unique(a, return_inverse=True)
    kb, ib = np.unique(b, return_inverse=True)
    n = len(ka) + len(kb)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for x, y in zip(ia, ib + len(ka)):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    roots = np.array([find(x) for x in range(n)])
    return roots[ia]


def build_derived(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_parquet(os.path.join(cfg["data"]["cache_dir"], "meta_all.parquet"))
    flow = [("all dataset rows", len(d))]
    d = d[d.agent.eq(cfg["selection"]["agent"])]
    flow.append(("agent == terminus-2", len(d)))
    ts = d.trace_source.fillna("")
    d = d[~ts.str.startswith("summarization")]
    flow.append(("drop summarization sub-call rows (trace_source summarization-*)", len(d)))
    d = derive_ids(d)
    d = d[d.model_id.notna()]
    flow.append(("model_id resolvable (drop ambiguous `model`)", len(d)))
    # §3.4 exclusions that are decidable from metadata
    d = d[(d.n_msgs > 0) & (d.task_text_chars < d.first_user_chars)]
    flow.append(("1. transcript parseable: non-empty, first user message holds the task statement", len(d)))
    d = d[d.n_asst > 0]
    flow.append(("2. >= 1 assistant turn", len(d)))
    # 3. duplicates: row-per-episode snapshots (keep the longest per trial), then exact duplicate transcripts
    d = d.sort_values(["n_asst", "total_chars", "uid"], ascending=[False, False, True])
    d = d.drop_duplicates("trial_id")
    flow.append(("3a. one row per trial (longest episode snapshot kept)", len(d)))
    d = d.drop_duplicates("h_conv")
    flow.append(("3b. drop exact duplicate transcripts", len(d)))
    d = d[d.max_msg_chars <= cfg["exclusion"]["max_message_chars"]]
    flow.append(("4. no message > 2,000,000 chars", len(d)))
    # task identity = exact task statement; split group = component linking names and texts
    d["task_name"] = d["task_id"]
    d["task_id"] = "t" + d.h_task_text.astype(np.uint64).astype(str)
    d["split_group"] = union_find_groups(d.task_name.values, d.task_id.values)
    d["cell_id"] = d.task_id + "|" + d.model_id
    return d, pd.DataFrame(flow, columns=["step", "rows_remaining"])


def select_cells(d: pd.DataFrame, cfg) -> pd.DataFrame:
    sc = cfg["selection"]
    rng = np.random.default_rng(seed_for(cfg, "select"))
    nmax = sc["n_runs_max"]
    cap_src = max(int(sc["cap_per_source_frac"] * nmax), sc["cap_per_source_min"])
    cap_cell = sc["n_runs_cell_cap"]
    n = d.groupby("cell_id").size()
    elig = n[n >= sc["n_min_cell"]].index
    e = d[d.cell_id.isin(elig)]
    cells = e.groupby("cell_id").agg(task_id=("task_id", "first"), model_id=("model_id", "first"),
                                    source=("source", lambda s: s.mode().iloc[0]), n=("uid", "size"))
    cells["n_take"] = cells.n.clip(upper=cap_cell)
    # crossed tasks: >= 2 models among eligible cells; prioritise tasks with >= 2 models having >= 4 runs
    nm = cells.groupby("task_id").model_id.nunique()
    nm4 = cells[cells.n >= 4].groupby("task_id").model_id.nunique()
    crossed = nm[nm >= 2].index
    strong = nm4[nm4 >= 2].index
    tasks = list(crossed)
    rng.shuffle(tasks)
    tasks = sorted(tasks, key=lambda t: t not in set(strong))  # stable: strong first, random within
    chosen, total = [], 0
    crossed_quota = int(sc.get("crossed_max_share", 0.5) * nmax)
    by_task = cells.groupby("task_id")
    src_count: dict = {}
    for t in tasks:
        g = by_task.get_group(t)
        k = int(g.n_take.sum())
        if total + k > crossed_quota:
            continue
        chosen.extend(g.index)
        total += k
        for s, v in g.groupby("source").n_take.sum().items():
            src_count[s] = src_count.get(s, 0) + int(v)
    n_crossed_runs = total
    # stratified fill by source
    rest = cells.drop(index=chosen)
    rest = rest[~rest.task_id.isin(set(cells.loc[chosen].task_id))]
    order = rest.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
    for s, g in order.groupby("source", sort=True):
        for cid, k in g.n_take.items():
            if total >= nmax:
                break
            if src_count.get(s, 0) + k > cap_src:
                continue
            chosen.append(cid)
            src_count[s] = src_count.get(s, 0) + int(k)
            total += int(k)
    sel_cells = cells.loc[chosen]
    rows = []
    for cid, g in e[e.cell_id.isin(sel_cells.index)].groupby("cell_id"):
        if len(g) > cap_cell:
            g = g.sample(cap_cell, random_state=int(rng.integers(1 << 31)))
        rows.append(g)
    sel = pd.concat(rows)
    sel["crossed_task"] = sel.task_id.isin(set(crossed))
    print(f"selected runs={len(sel)} cells={sel.cell_id.nunique()} crossed-task runs={n_crossed_runs}")
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    cfg = load_config(ap.parse_args().config)
    d, flow = build_derived(cfg)
    cache = cfg["data"]["cache_dir"]
    os.makedirs("results/tables", exist_ok=True)
    d.to_parquet(os.path.join(cache, "meta_derived.parquet"), index=False)
    sel = select_cells(d, cfg)
    keep = ["uid", "shard", "row", "trial_id", "task_id", "task_name", "split_group", "model_id", "model",
            "original_teacher", "source", "cell_id", "crossed_task", "date", "result", "judgment", "split",
            "n_asst", "episode", "trace_source", "last_task_complete"]
    sel[keep].to_parquet(os.path.join(cache, "selection.parquet"), index=False)
    flow.loc[len(flow)] = ("selected (cells with >= 2 runs, stratified, <= 32 runs/cell)", len(sel))
    flow.to_csv("results/tables/t0_exclusion_flow.csv", index=False)
    ss = sel.groupby("source").agg(runs=("uid", "size"), cells=("cell_id", "nunique"), tasks=("task_id", "nunique"))
    ss.to_csv("results/tables/sample_size.csv")
    print(flow.to_string())
    print(ss.to_string())


if __name__ == "__main__":
    main()
