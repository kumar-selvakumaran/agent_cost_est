"""Warm-task history features for setting S2 (spec §6.2). Features of a target-half (T) run are computed only
from history-half (H) runs, which are disjoint from every target run."""
from __future__ import annotations

import numpy as np
import pandas as pd

HIST_COLS = ["h_n", "h_mean", "h_sd", "h_q10", "h_q50", "h_q90", "h_K_mean", "x_mean", "x_sd", "x_n"]


def history_features(runs: pd.DataFrame, warm_half: np.ndarray) -> pd.DataFrame:
    """Returns a frame aligned with `runs` holding history features for T-half runs (NaN elsewhere)."""
    r = runs[["run_id", "task_id", "model_id", "cell_id", "y", "K"]].copy()
    r["half"] = warm_half
    H = r[r.half == "H"]
    hs = H.groupby("cell_id").agg(h_n=("y", "size"), h_mean=("y", "mean"), h_sd=("y", "std"),
                                  h_q10=("y", lambda s: s.quantile(0.1)), h_q50=("y", "median"),
                                  h_q90=("y", lambda s: s.quantile(0.9)), h_K_mean=("K", "mean"),
                                  task_id=("task_id", "first"), model_id=("model_id", "first"))
    # cross-model history: H runs of other models on the same task
    by_task = H.groupby(["task_id", "model_id"]).y.agg(["sum", "size"])
    by_task["sq"] = H.assign(y2=H.y ** 2).groupby(["task_id", "model_id"]).y2.sum()
    tt = by_task.groupby(level=0).sum()
    rows = []
    for cid, row in hs.iterrows():
        t, m = row.task_id, row.model_id
        if t in tt.index:
            tot = tt.loc[t]
            own = by_task.loc[(t, m)] if (t, m) in by_task.index else pd.Series({"sum": 0, "size": 0, "sq": 0})
            n = tot["size"] - own["size"]
            s = tot["sum"] - own["sum"]
            sq = tot["sq"] - own["sq"]
        else:
            n = s = sq = 0
        mean = s / n if n > 0 else np.nan
        sd = np.sqrt(max(sq / n - mean ** 2, 0) * n / (n - 1)) if n > 1 else np.nan
        rows.append((cid, mean, sd, n))
    x = pd.DataFrame(rows, columns=["cell_id", "x_mean", "x_sd", "x_n"]).set_index("cell_id")
    hs = hs.join(x)
    out = pd.DataFrame(np.nan, index=runs.index, columns=HIST_COLS)
    tmask = r.half.values == "T"
    joined = hs.loc[r.cell_id[tmask].values, HIST_COLS].values
    out.loc[tmask, HIST_COLS] = joined
    return out
