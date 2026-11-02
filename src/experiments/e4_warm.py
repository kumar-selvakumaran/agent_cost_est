"""E4 — warm pre-run benchmark (S2; tests H6). Spec §10 E4, §8.2.

Cells with >= 6 runs; each cell's runs split into a history half H and target half T (splits.parquet).
Training: T runs of training-fold cells with features from their H; evaluation: T runs of test-fold cells.
Methods: P1 (S1 baseline, from E3), P2 / P3-EVT / P4 with and without history features (trained on the same
rows), and the cell oracle (leave-one-out empirical quantiles over all other runs of the cell, cells >= 10 runs).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.eval.bootstrap import task_cluster_indices, boot_task_metric, paired_diff, holm
from src.eval.metrics import QCOLS, TAUS, qcrps, central_cover
from src.experiments.common import load_data, plt, write_log, color
from src.experiments.e3_cold import load_pred
from src.experiments.folds import FoldData
from src.features.history import history_features, HIST_COLS
from src.predictors.base import get_hp
from src.predictors.p2_hetero import P2Hetero
from src.predictors.p3_quantile import P3Quantile, P3EVT
from src.predictors.p4_mdn import P4MDN
from src.utils import seed_for

PRED = "data_cache/preds"
METHODS = ["P2", "P3-EVT", "P4"]


def oracle_quantiles(runs: pd.DataFrame, ids: np.ndarray) -> np.ndarray:
    r = runs.set_index("run_id")
    by_cell = r.groupby("cell_id").y.apply(lambda s: s)
    out = np.empty((len(ids), 99))
    cell_y = {c: g for c, g in r.groupby("cell_id").y}
    for i, rid in enumerate(ids):
        g = cell_y[r.at[rid, "cell_id"]]
        out[i] = np.quantile(g.drop(rid).values, TAUS)
    return out


def main():
    cfg, runs, splits = load_data()
    hist = history_features(runs, splits.warm_half.values)
    T = splits.warm_half.values == "T"
    mc = cfg["mdn"]
    os.makedirs(PRED, exist_ok=True)
    for f in range(cfg["cv"]["n_splits"]):
        path = os.path.join(PRED, f"e4__f{f}.npz")
        if os.path.exists(path):
            continue
        out = {}
        for with_h in (True, False):
            fd = FoldData(cfg, runs, splits, f, extra=hist if with_h else None)
            fit, val, test = (fd.bundle_mask((fd.role == r) & T) for r in ["fit", "val", "test"])
            out["test_run_id"] = test.meta.run_id.values
            tag = "H" if with_h else "noH"
            p2 = P2Hetero(get_hp("P2_mean"), get_hp("P2_var"), seed=seed_for(cfg, "e4P2")).fit(fit, val)
            out[f"P2_{tag}"] = p2.predict_quantiles(test)
            p3 = P3Quantile(get_hp("P3"), seed=seed_for(cfg, "e4P3")).fit(fit, val)
            ev = P3EVT(p3, cfg["evt"]["threshold_q"], cfg["evt"]["min_exceedances"]).fit(fit, val)
            out[f"P3-EVT_{tag}"] = ev.predict_quantiles(test)
            out[f"P3-EVT_{tag}_fallback"] = np.array([ev.fallback])
            p4 = P4MDN(3, tuple(mc["hidden"]), mc["dropout"], mc["lr"], mc["weight_decay"], mc["batch"],
                       mc["max_epochs"], mc["patience"], seed=seed_for(cfg, "e4P4_s0")).fit(fit, val)
            out[f"P4_{tag}"] = p4.predict_quantiles(test)
            print(f"fold {f} {tag}: fit={len(fit)} val={len(val)} test={len(test)}", flush=True)
        np.savez_compressed(path, **out)

    # ---- evaluation
    B = cfg["bootstrap"]["B"]
    seed = seed_for(cfg, "e4_boot")
    q = {}
    ids_all = []
    for f in range(cfg["cv"]["n_splits"]):
        d = np.load(os.path.join(PRED, f"e4__f{f}.npz"))
        ids_all.append(d["test_run_id"])
        for k in d.files:
            if k.startswith(("P2_", "P3-EVT_", "P4_")) and not k.endswith("fallback"):
                q.setdefault(k, []).append(d[k])
        p1 = load_pred("e3", "P1", f)
        m = dict(zip(p1["test_run_id"].astype(np.int64), p1["test"]))
        q.setdefault("P1", []).append(np.stack([m[i] for i in d["test_run_id"]]))
    ids = np.concatenate(ids_all)
    q = {k: np.concatenate(v).astype(float) for k, v in q.items()}
    r = runs.set_index("run_id")
    y = r.loc[ids, "y"].values
    task = r.loc[ids, "task_id"].values
    ncell = r.groupby("cell_id").size()
    has_or = ncell.loc[r.loc[ids, "cell_id"]].values >= cfg["oracle"]["min_cell_runs"]
    q["oracle"] = np.full((len(ids), 99), np.nan)
    q["oracle"][has_or] = oracle_quantiles(runs, ids[has_or])
    names = ["P1"] + [f"{m}_{t}" for m in METHODS for t in ("noH", "H")] + ["oracle"]
    rows = []
    s_all = {n: qcrps(q[n], y) for n in names if n != "oracle"}
    s_all["oracle"] = np.where(has_or, qcrps(np.nan_to_num(q["oracle"]), y), np.nan)
    for pop, mask in [("all S2 test runs", np.ones(len(ids), bool)), ("cells with ≥ 10 runs (oracle defined)", has_or)]:
        cw = task_cluster_indices(task[mask], B, seed)
        p1 = boot_task_metric(s_all["P1"][mask], task[mask], B, seed, "task", cw)
        orc = boot_task_metric(s_all["oracle"][mask], task[mask], B, seed, "task", cw) if mask is has_or else None
        for n in names:
            if n == "oracle" and orc is None:
                continue
            s = s_all[n][mask]
            b = boot_task_metric(s, task[mask], B, seed, "task", cw)
            cov = boot_task_metric(central_cover(np.nan_to_num(q[n][mask]), y[mask], 0.9), task[mask], B, seed, "task", cw)
            row = dict(population=pop, method=n, n_runs=int(mask.sum()), qcrps=b["est"], lo=b["lo"], hi=b["hi"],
                       cov90=cov["est"])
            g = (p1["reps"] - b["reps"]) / p1["reps"]
            row.update(gain_vs_P1=(p1["est"] - b["est"]) / p1["est"], gain_lo=np.percentile(g, 2.5),
                       gain_hi=np.percentile(g, 97.5))
            if orc is not None:
                gc = (p1["reps"] - b["reps"]) / (p1["reps"] - orc["reps"])
                row.update(gap_closed=(p1["est"] - b["est"]) / (p1["est"] - orc["est"]),
                           gap_lo=np.percentile(gc, 2.5), gap_hi=np.percentile(gc, 97.5))
            if n != "P1":
                d = paired_diff(s, s_all["P1"][mask], task[mask], B, seed, cw)
                row.update(diff_vs_P1=d["est"], p_vs_P1=d["p"])
            rows.append(row)
    tab = pd.DataFrame(rows)
    m = tab.p_vs_P1.notna()
    tab.loc[m, "p_holm"] = holm(tab.loc[m, "p_vs_P1"].values)
    tab.to_csv("results/tables/e4_warm.csv", index=False)
    pd.DataFrame([dict(n_runs=len(ids), n_tasks=len(set(task)), n_cells=r.loc[ids, "cell_id"].nunique(),
                       n_runs_oracle=int(has_or.sum()))]).to_csv("results/tables/e4_design.csv", index=False)
    # paired history vs no history
    cw = task_cluster_indices(task, B, seed)
    pr = []
    for mth in METHODS:
        d = paired_diff(s_all[f"{mth}_H"], s_all[f"{mth}_noH"], task, B, seed, cw)
        pr.append(dict(method=mth, diff_H_minus_noH=d["est"], lo=d["lo"], hi=d["hi"], p=d["p"]))
    pr = pd.DataFrame(pr)
    pr["p_holm"] = holm(pr.p.values)
    pr.to_csv("results/tables/e4_history_paired.csv", index=False)
    # H6: relative gain of history (best warm method with history, vs P1, on S2 runs) minus relative gain of text
    # (best E3 method vs P1, on all S1 runs) -- bootstrap CI from independent resamples of the two experiments
    allp = tab[tab.population == "all S2 test runs"].set_index("method")
    best_h = allp.loc[[f"{m}_H" for m in METHODS], "qcrps"].idxmin()
    p1b = boot_task_metric(s_all["P1"], task, B, seed, "task", cw)
    hb = boot_task_metric(s_all[best_h], task, B, seed, "task", cw)
    g_hist = (p1b["reps"] - hb["reps"]) / p1b["reps"]
    e3g = pd.read_csv("results/tables/e3_gain_reps.csv")
    g_text = e3g.reps.values
    n = min(len(g_hist), len(g_text))
    diff = g_hist[:n] - g_text[:n]
    est = (p1b["est"] - hb["est"]) / p1b["est"] - e3g.est.iloc[0]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    verdict = "supported" if lo > 0 else ("not supported" if hi < 0 else "inconclusive")
    pd.DataFrame([dict(hypothesis="H6", verdict=verdict, best_history_method=best_h,
                       gain_history=(p1b["est"] - hb["est"]) / p1b["est"], gain_text=e3g.est.iloc[0],
                       diff=est, lo=lo, hi=hi)]).to_csv("results/tables/h6.csv", index=False)
    preds = []
    for n in names:
        df = pd.DataFrame(q[n], columns=QCOLS)
        df.insert(0, "run_id", ids)
        df.insert(1, "task_id", task)
        df.insert(2, "model_id", r.loc[ids, "model_id"].values)
        df.insert(3, "fold", splits.set_index("run_id").loc[ids, "fold"].values)
        df.insert(4, "method", n)
        df.insert(5, "k", 0)
        df["y"] = y
        preds.append(df)
    pd.concat(preds).to_parquet("results/predictions/e4_warm.parquet", index=False)
    L = ["# E4 — Warm pre-run benchmark\n",
         f"S2 test units: target-half runs of cells with ≥ 6 runs; n = {len(ids):,} runs, {len(set(task)):,} tasks. "
         "History features from the disjoint history half. `_H` = with history features, `_noH` = same predictor, "
         "same rows, without them. P1 = the S1 baseline from E3 restricted to these runs. Oracle = leave-one-out "
         "empirical quantiles of the cell (cells ≥ 10 runs).\n",
         tab.round(4).to_markdown(index=False) + "\n",
         "## History vs no history (paired, task-cluster bootstrap)\n", pr.round(4).to_markdown(index=False) + "\n",
         f"**H6 ({verdict})**: relative CRPS gain vs P1 from history ({best_h}) − relative gain from text (best E3 "
         f"method) = {est:.3f} ({lo:.3f}, {hi:.3f}).\n"]
    write_log("e4", L)
    sub = tab[tab.population.str.startswith("cells")]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    order = sub.sort_values("qcrps", ascending=False)
    ax.barh(order.method, order.qcrps, xerr=[order.qcrps - order.lo, order.hi - order.qcrps],
            color=[color(m.split("_")[0]) if m != "oracle" else "#000000" for m in order.method],
            alpha=[1.0 if m.endswith("_H") or m in ("P1", "oracle") else 0.5 for m in order.method][0])
    p1v = sub[sub.method == "P1"].qcrps.iloc[0]
    ov = sub[sub.method == "oracle"].qcrps.iloc[0]
    ax.axvline(p1v, color="k", ls="--", lw=1, label="P1")
    ax.axvline(ov, color="k", ls=":", lw=1, label="cell oracle")
    ax.set_xlabel("task-weighted QCRPS (log cost), cells with ≥ 10 runs")
    ax.set_title("Warm setting: gap between P1 and the cell oracle")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/fig_e4_gap_closed.png")
    plt.close(fig)
    print("\n".join(L))


if __name__ == "__main__":
    main()
