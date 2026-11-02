"""EXPLORATORY (not pre-registered): is the online model's error much smaller than the spread of cost itself, and
does it beat cheap lookup tables of stratified constants? Out-of-fold throughout (tables fit on training folds).

For each checkpoint k (0 = pre-run) on runs with K > k, point forecasts of total cost Y:
  spread      : no forecast — dispersion of Y itself (median |Y - median Y| / Y over the pooled training runs' median)
  table_sm    : median Y of the run's (source, model) cell in the training folds (P1-style lookup)
  table_ratio : C_k x median(Y / C_k) of the run's (source, model) in the training folds (runtime lookup, k >= 1)
  P5 / O2     : model medians (P5 pre-run; O2 per-k online model from E5)
Metrics: median |Yhat - Y| / Y (total), median |Yhat - Y| / (Y - C_k) (remaining), share within +-25% of Y.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.common import load_data


def metrics(yhat, Y, C):
    rel = np.abs(yhat - Y) / Y
    rem = np.abs(yhat - Y) / np.maximum(Y - C, 1.0)
    return dict(median_miss_total=np.median(rel), median_miss_remaining=np.median(rem), within_25=np.mean(rel <= 0.25),
                p90_miss_total=np.quantile(rel, 0.9))


def main():
    cfg, runs, splits = load_data()
    turns = pd.read_parquet("data_cache/turns.parquet", columns=["run_id", "k", "cum_cost"])
    e5 = pd.read_parquet("results/predictions/e5_online.parquet", columns=["run_id", "method", "k", "q50"])
    R = runs.set_index("run_id")
    R["fold"] = splits.set_index("run_id").fold
    rows = []
    for k in [0, 1, 2, 4, 8, 16, 32]:
        ids = R.index[R.K > k].values
        Y = R.loc[ids, "Y"].values
        if k > 0:
            C = turns[turns.k == k].set_index("run_id").cum_cost.loc[ids].values
        else:
            C = np.zeros(len(ids))
        fold = R.loc[ids, "fold"].values
        key = (R.loc[ids, "source"] + "|" + R.loc[ids, "model_id"]).values
        Cs = pd.Series(C, index=ids)
        f_const, f_sm, f_ratio = np.empty(len(ids)), np.empty(len(ids)), np.empty(len(ids))
        for f in range(cfg["cv"]["n_splits"]):
            tr, te = fold != f, fold == f
            gmed = np.median(Y[tr])
            f_const[te] = gmed
            sm = pd.Series(Y[tr]).groupby(key[tr]).median()
            f_sm[te] = pd.Series(key[te]).map(sm).fillna(gmed).values
            if k > 0:
                ratio = pd.Series(Y[tr] / C[tr]).groupby(key[tr]).median()
                gr = np.median(Y[tr] / C[tr])
                f_ratio[te] = C[te] * pd.Series(key[te]).map(ratio).fillna(gr).values
        preds = {"spread (global median)": f_const, "table: source x model median": f_sm}
        if k > 0:
            preds["table: cost-so-far x (source x model) ratio"] = f_ratio
        m = e5[(e5.k == k) & (e5.method == ("O2" if k > 0 else "P5"))].set_index("run_id").q50
        preds["model: " + ("O2 (online)" if k > 0 else "P5 (pre-run)")] = np.exp(m.loc[ids].values)
        sd = float(np.std(np.log(Y)))
        for name, yhat in preds.items():
            rows.append(dict(k=k, n_runs=len(ids), forecast=name, sd_logY=sd,
                             median_cost=float(np.median(Y)), spent_share_median=float(np.median(C / Y)),
                             **metrics(yhat, Y, C)))
    df = pd.DataFrame(rows)
    df.to_csv("results/tables/x1_lookup_baselines.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
