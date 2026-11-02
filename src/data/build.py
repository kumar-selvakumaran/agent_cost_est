"""Build runs.parquet, turns.parquet, splits.parquet from the loaded parts (spec §5.5, §8)."""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

from src.utils import load_config, seed_for

SOURCE_FAMILY = {
    "swesmith": "code_repair", "r2egym": "code_repair", "SWEGym": "code_repair", "Inferred Bugs": "code_repair",
    "defects 4j": "code_repair",
    "nl2bash": "shell", "bash textbook": "shell", "taskmaster": "shell", "staqc": "shell",
    "code_contests": "competitive_programming", "codeforces": "competitive_programming",
    "math": "math", "gsm8k": "math", "Nemotron Prism": "math", "puzzles": "math",
    "Qasper": "data_qa", "WikiTable Format Conversion": "data_qa", "stack_exchange": "data_qa",
    "StackExchange Overflow": "data_qa",
}


def success_label(result, judgment, split):
    r = None if result is None or (isinstance(result, float) and np.isnan(result)) else str(result)
    if r in ("1.0", "success"):
        return 1.0
    if r in ("0.0", "failure"):
        return 0.0
    if r is not None:
        return 0.0  # error result: agent did not finish
    if judgment == "Yes":
        return 1.0
    if split == "success":
        return 1.0
    if split == "failure":
        return 0.0
    return np.nan


def add_turn_derived(turns: pd.DataFrame) -> pd.DataFrame:
    """Cumulative tokens N_k and input size of the next call I_{k+1} (null after the last call)."""
    turns["n_tok"] = turns.I + turns.O
    turns["cum_tokens"] = turns.groupby("run_id").n_tok.cumsum()
    turns["I_next"] = turns.groupby("run_id").I.shift(-1).astype("Int64")
    return turns


def concat_parts(cache):
    parts = os.path.join(cache, "parts")
    runs = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(parts, "runs_*.parquet")))],
                     ignore_index=True)
    runs.to_parquet(os.path.join(cache, "runs_raw.parquet"), index=False)
    return runs


def build(cfg):
    cache = cfg["data"]["cache_dir"]
    runs_raw = pd.read_parquet(os.path.join(cache, "runs_raw.parquet"))
    sel = pd.read_parquet(os.path.join(cache, "selection.parquet"))
    flags = pd.read_parquet(os.path.join(cache, "summ_flags.parquet"))
    r = sel.merge(runs_raw, on="uid", how="inner")
    r = r[r.parse_ok & (r.K > 0)].copy()
    r = r.merge(flags, on="uid", how="left")
    r["summ_flag"] = r.summ_flag.fillna(False).astype(bool)
    r["y"] = np.log(r.Y)
    r["y_tok"] = np.log(r.Y_tok)
    r["y_hi"] = np.log(r.Y_hi)
    r["source_family"] = r.source.map(SOURCE_FAMILY).fillna("other")
    r["success"] = [success_label(a, b, c) for a, b, c in zip(r.result, r.judgment, r.split)]
    err = r.result.notna() & ~r.result.isin(["1.0", "0.0", "success", "failure", "timeout"])
    r["completed"] = (~err & r.final_task_complete).astype(float)
    r["date_ts"] = pd.to_datetime(r.date, errors="coerce", utc=True, format="mixed")
    # turn cap per source (E0 rule)
    caps = pd.read_csv("results/tables/e0_turn_caps.csv")
    capmap = dict(zip(caps.source, caps.K_cap))
    r["K_cap"] = r.source.map(capmap)
    r["at_cap"] = (r.K == r.K_cap).astype(int)
    r = r.reset_index(drop=True)
    r["run_id"] = np.arange(len(r))
    r.to_parquet(os.path.join(cache, "runs.parquet"), index=False)

    parts = os.path.join(cache, "parts")
    turns = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(parts, "turns_*.parquet")))],
                      ignore_index=True)
    turns = turns.merge(r[["uid", "run_id"]], on="uid", how="inner").drop(columns="uid")
    turns = turns.sort_values(["run_id", "k"]).reset_index(drop=True)
    turns = add_turn_derived(turns)
    turns.to_parquet(os.path.join(cache, "turns.parquet"), index=False)
    print("runs", len(r), "turns", len(turns))
    return r


def make_splits(cfg, r: pd.DataFrame) -> pd.DataFrame:
    seed = cfg["seed"]
    n_splits = cfg["cv"]["n_splits"]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(r), -1)
    for f, (_, te) in enumerate(sgkf.split(r, r.source, groups=r.split_group)):
        fold[te] = f
    sp = pd.DataFrame({"run_id": r.run_id, "uid": r.uid, "task_id": r.task_id, "split_group": r.split_group,
                       "fold": fold})
    # inner split of each training fold by group: 70 fit / 15 val / 15 cal
    for f in range(n_splits):
        role = np.where(fold == f, "test", "").astype(object)
        tr = np.flatnonzero(fold != f)
        groups = r.split_group.values[tr]
        ug = np.unique(groups)
        rng = np.random.default_rng(seed_for(cfg, f"inner_{f}"))
        perm = rng.permutation(ug)
        n_fit = int(round(cfg["cv"]["inner_fit"] * len(ug)))
        n_val = int(round(cfg["cv"]["inner_val"] * len(ug)))
        gfit, gval = set(perm[:n_fit]), set(perm[n_fit:n_fit + n_val])
        role[tr] = ["fit" if g in gfit else ("val" if g in gval else "cal") for g in groups]
        sp[f"role_{f}"] = role
    # warm split: cells with >= 6 runs, half history (extra run to H) / half target
    rng = np.random.default_rng(seed_for(cfg, "warm_halves"))
    cn = r.groupby("cell_id").run_id.transform("size")
    half = np.array([""] * len(r), dtype=object)
    for cid, idx in r[cn >= cfg["warm"]["min_cell_runs"]].groupby("cell_id").groups.items():
        idx = np.array(list(idx))
        p = rng.permutation(idx)
        nh = (len(p) + 1) // 2
        half[p[:nh]] = "H"
        half[p[nh:]] = "T"
    sp["warm_half"] = half
    return sp


def main():
    cfg = load_config()
    r = build(cfg)
    sp = make_splits(cfg, r)
    sp.to_parquet(os.path.join(cfg["data"]["cache_dir"], "splits.parquet"), index=False)
    print(sp.groupby("fold").size())


if __name__ == "__main__":
    main()
