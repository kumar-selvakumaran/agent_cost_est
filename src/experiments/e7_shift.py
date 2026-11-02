"""E7 — distribution shift and adaptive conformal inference (spec §10 E7, §8.4)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.bootstrap import boot_task_metric, task_cluster_indices
from src.eval.metrics import QCOLS, interp_q
from src.experiments.common import load_data, plt, write_log
from src.experiments.e3_cold import fit_fold, load_pred
from src.predictors.aci import aci_run, CQRScores
from src.predictors.cqr import one_per_task, conformal_quantile
from src.utils import seed_for


def temporal_splits(cfg, runs, splits):
    cut = runs.date_ts.quantile(0.7)
    train = runs.date_ts <= cut
    test = ~train & ~runs.split_group.isin(set(runs.split_group[train]))
    role = np.full(len(runs), "drop", dtype=object)
    role[test.values] = "test"
    groups = runs.split_group[train].unique()
    rng = np.random.default_rng(seed_for(cfg, "e7_inner"))
    perm = rng.permutation(groups)
    nf, nv = int(round(0.7 * len(perm))), int(round(0.15 * len(perm)))
    gmap = {g: ("fit" if i < nf else ("val" if i < nf + nv else "cal")) for i, g in enumerate(perm)}
    role[train.values] = runs.split_group[train].map(gmap).values
    sp = splits.copy()
    sp["role_temporal"] = role
    return sp, cut, int(train.sum()), int(test.sum()), int((~train & runs.split_group.isin(set(runs.split_group[train]))).sum())


def main():
    cfg, runs, splits = load_data()
    best = pd.read_csv("results/tables/e3_best_method.csv").best_method.iloc[0]
    sp, cut, ntr, nte, nrem = temporal_splits(cfg, runs, splits)
    keep = sp.role_temporal != "drop"
    r2, s2 = runs[keep].reset_index(drop=True), sp[keep].reset_index(drop=True)
    method = best.split("(")[0] if best.startswith("CQR") else best
    fit_fold(cfg, r2, s2, 0, exp="e7", methods=[method if method != "P3-EVT" else "P3-EVT"],
             group_col="role_temporal", svd_tag="temporal", n_seeds=1)
    d = load_pred("e7", method, 0)
    q_cal, q_te = d["cal"].astype(float), d["test"].astype(float)
    rid_cal, rid_te = d["cal_run_id"].astype(np.int64), d["test_run_id"].astype(np.int64)
    R = runs.set_index("run_id")
    rng = np.random.default_rng(seed_for(cfg, "e7_cal"))
    meta_cal = R.loc[rid_cal].reset_index()
    pick = one_per_task(meta_cal, rng)
    q_cal, y_cal = q_cal[pick], R.loc[rid_cal[pick], "y"].values
    order = np.argsort(R.loc[rid_te, "date_ts"].values, kind="stable")
    q_te, rid_te = q_te[order], rid_te[order]
    y_te = R.loc[rid_te, "y"].values
    task = R.loc[rid_te, "task_id"].values
    a = cfg["aci"]["alpha"]
    # static CQR 90%
    s = np.maximum(interp_q(q_cal, a / 2) - y_cal, y_cal - interp_q(q_cal, 1 - a / 2))
    Q = conformal_quantile(s, 1 - a)
    lo_s, hi_s = interp_q(q_te, a / 2) - Q, interp_q(q_te, 1 - a / 2) + Q
    cov_s = ((y_te >= lo_s) & (y_te <= hi_s)).astype(float)
    # ACI
    lo_a, hi_a, path, err = aci_run(q_te, y_te, CQRScores(q_cal, y_cal), a, cfg["aci"]["gamma"], tuple(cfg["aci"]["clip"]))
    cov_a = 1 - err
    # raw (unconformalized) 90% interval
    cov_r = ((y_te >= interp_q(q_te, a / 2)) & (y_te <= interp_q(q_te, 1 - a / 2))).astype(float)
    B = cfg["bootstrap"]["B"]
    cw = task_cluster_indices(task, B, seed_for(cfg, "e7_boot"))
    rows = []
    for name, cov, w in [("raw " + method, cov_r, interp_q(q_te, 1 - a / 2) - interp_q(q_te, a / 2)),
                         ("static CQR", cov_s, hi_s - lo_s), ("ACI", cov_a, hi_a - lo_a)]:
        c = boot_task_metric(cov, task, B, 0, "run", cw)
        wd = boot_task_metric(w, task, B, 0, "run", cw)
        win = cfg["aci"]["window"]
        roll = pd.Series(cov).rolling(win).mean().dropna()
        rows.append(dict(method=name, n_test=len(y_te), coverage=c["est"], cov_lo=c["lo"], cov_hi=c["hi"],
                         width=wd["est"], width_lo=wd["lo"], width_hi=wd["hi"], rolling_cov_min=roll.min(),
                         rolling_cov_max=roll.max(), share_windows_below_85=float((roll < 0.85).mean())))
    tab = pd.DataFrame(rows)
    tab.to_csv("results/tables/e7_shift.csv", index=False)
    pd.DataFrame({"run_id": rid_te, "y": y_te, "static_lo": lo_s, "static_hi": hi_s, "aci_lo": lo_a, "aci_hi": hi_a,
                  "alpha_t": path}).to_parquet("results/predictions/e7_intervals.parquet", index=False)
    win = cfg["aci"]["window"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(y_te))
    for name, cov, c in [("static CQR", cov_s, "#0072B2"), ("ACI", cov_a, "#D55E00"), ("raw " + method, cov_r, "#999999")]:
        ax.plot(x, pd.Series(cov).rolling(win).mean(), color=c, label=name, lw=1.2)
    ax.axhline(0.9, color="k", ls="--", lw=1)
    ax.set_xlabel("test runs in date order")
    ax.set_ylabel(f"rolling coverage (window {win})")
    ax.set_title("Coverage of the 90% interval under temporal shift")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/fig_e7_rolling_coverage.png")
    plt.close(fig)
    te_src = R.loc[rid_te].source.value_counts().to_dict()
    pd.DataFrame([dict(cut_date=str(cut)[:10], n_train=ntr, n_test=nte, n_removed=nrem, method=method,
                       **{f"test_runs_{k}": v for k, v in te_src.items()})]).to_csv("results/tables/e7_design.csv", index=False)
    L = ["# E7 — Distribution shift and ACI\n",
         f"Temporal split (date coverage 100%): training = runs dated ≤ {cut} (70th percentile, n = {ntr:,}); test = "
         f"later runs whose task (CV group) never appears in training (n = {nte:,}; {nrem:,} later runs removed because "
         f"their task appears in training). The base forecaster is the best S1 method ({method}) refit on the "
         f"temporal training set (70/15/15 fit/val/cal by task group). Test-side sources: {te_src}. The shift is "
         "therefore largely a shift to unseen sources.\n",
         tab.round(4).to_markdown(index=False) + "\n"]
    write_log("e7", L)
    print("\n".join(L))


if __name__ == "__main__":
    main()
