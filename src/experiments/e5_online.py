"""E5 — online benchmark (S3; tests H4). Spec §10 E5.

For each k in the grid and fold: O2 / O2-noregime (one quantile GBM per k) and O1 (one hazard model per k),
trained on fit-role runs with K > k, early-stopped on val-role runs with K > k, predicting test runs with K > k.
k = 0 is the best S1 predictor from E3 (its stored out-of-fold predictions).
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from src.eval.bootstrap import task_cluster_indices, boot_task_metric, paired_diff, holm
from src.eval.metrics import QCOLS, qcrps, central_cover
from src.experiments.common import load_data, plt, write_log, color, sig3
from src.experiments.e3_cold import load_pred
from src.experiments.folds import FoldData
from src.features.prefix import features_at
from src.predictors.base import get_hp
from src.predictors.o1_structural import O1Structural, online_state
from src.predictors.o2_prefix_gbm import O2Prefix
from src.utils import seed_for

PRED_DIR = "data_cache/preds"


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def fit_fold(cfg, runs, splits, turns, f, ks, methods=("O2", "O2-noregime", "O1")):
    fd = FoldData(cfg, runs, splits, f)
    K = runs.K.values
    hp3 = get_hp("P3")
    hph = get_hp("P5_haz")
    for k in ks:
        path = os.path.join(PRED_DIR, f"e5__k{k}__f{f}.npz")
        if os.path.exists(path):
            continue
        elig = K > k
        roles = {r: (fd.role == r) & elig for r in ["fit", "val", "test"]}
        b = {r: fd.bundle_mask(m) for r, m in roles.items()}
        ids = np.concatenate([b[r].meta.run_id.values for r in b])
        pf = features_at(turns[turns.run_id.isin(ids)], k)
        pfr = {r: pf[pf.run_id.isin(b[r].meta.run_id.values)] for r in b}
        out = {"test_run_id": b["test"].meta.run_id.values}
        for name in methods:
            if name.startswith("O2"):
                m = O2Prefix(hp3, seed=seed_for(cfg, f"{name}_k{k}"), noregime=name.endswith("noregime"))
                m.fit_k(k, b["fit"], b["val"], pfr["fit"], pfr["val"])
                out[name] = m.predict_k(k, b["test"], pfr["test"])
            else:
                m = O1Structural(hph, seed=seed_for(cfg, f"O1_k{k}"), M=cfg["p5"]["mc_draws"],
                                 max_pp=cfg["p5"]["max_person_period_rows"])
                m.fit_k(k, b["fit"], b["val"], pfr["fit"], pfr["val"])
                st = online_state(turns[turns.run_id.isin(b["test"].meta.run_id.values)], k)
                out[name] = m.predict_k(k, b["test"], pfr["test"], st, cfg["cost"]["lambda_c"], cfg["cost"]["lambda_o"])
            log(f"fold {f} k={k} {name} n_test={len(b['test'])}")
        np.savez_compressed(path, **{kk: np.asarray(v, dtype=np.float32 if kk != "test_run_id" else np.int64)
                                     for kk, v in out.items()})


def evaluate(cfg, runs, best_s1, ks):
    B = cfg["bootstrap"]["B"]
    seed = seed_for(cfg, "e5_boot")
    yv = runs.set_index("run_id").y
    rows, preds_out = [], []
    # k = 0 reference: best S1 out-of-fold predictions
    q0 = {}
    for f in range(cfg["cv"]["n_splits"]):
        d = load_pred("e3", best_s1, f)
        for rid, q in zip(d["test_run_id"].astype(np.int64), d["test"]):
            q0[rid] = q
    allk = {0: {"ref": q0}}
    for k in ks:
        allk[k] = {}
        for f in range(cfg["cv"]["n_splits"]):
            d = np.load(os.path.join(PRED_DIR, f"e5__k{k}__f{f}.npz"))
            for name in ["O1", "O2", "O2-noregime"]:
                if name in d:
                    allk[k].setdefault(name, {}).update(dict(zip(d["test_run_id"], d[name])))
    K = runs.set_index("run_id").K
    panel_ids = K[K >= cfg["online"]["panel_min_K"]].index.values
    for pop in ["eligible", "panel"]:
        for k in [0] + list(ks):
            ids = K[K > k].index.values if pop == "eligible" else panel_ids
            task = runs.set_index("run_id").loc[ids, "task_id"].values
            codes_W = task_cluster_indices(task, B, seed + k)
            y = yv.loc[ids].values
            names = ["ref"] if k == 0 else ["O1", "O2", "O2-noregime"]
            for name in names:
                q = np.stack([allk[k][name][i] for i in ids]).astype(float)
                s = qcrps(q, y)
                r = boot_task_metric(s, task, B, seed, "task", codes_W)
                cov = boot_task_metric(central_cover(q, y, 0.9), task, B, seed, "task", codes_W)
                # paired vs k=0 reference on the same runs
                q_ref = np.stack([q0[i] for i in ids]).astype(float)
                d = paired_diff(s, qcrps(q_ref, y), task, B, seed, codes_W)
                rr = r["reps"] / boot_task_metric(qcrps(q_ref, y), task, B, seed, "task", codes_W)["reps"]
                rows.append(dict(population=pop, k=k, method=(best_s1 + " (k=0)") if name == "ref" else name,
                                 n_runs=len(ids), n_tasks=len(set(task)), qcrps=r["est"], lo=r["lo"], hi=r["hi"],
                                 cov90=cov["est"], cov90_lo=cov["lo"], cov90_hi=cov["hi"],
                                 diff_vs_k0=d["est"], diff_lo=d["lo"], diff_hi=d["hi"], p_vs_k0=d["p"],
                                 ratio_vs_k0=r["est"] / boot_task_metric(qcrps(q_ref, y), task, B, seed, "task", codes_W)["est"],
                                 ratio_lo=np.percentile(rr, 2.5), ratio_hi=np.percentile(rr, 97.5)))
                if pop == "eligible":
                    df = pd.DataFrame(q, columns=QCOLS)
                    meta = runs.set_index("run_id").loc[ids]
                    df.insert(0, "run_id", ids)
                    df.insert(1, "task_id", meta.task_id.values)
                    df.insert(2, "model_id", meta.model_id.values)
                    df.insert(3, "fold", -1)
                    df.insert(4, "method", name if name != "ref" else best_s1)
                    df.insert(5, "k", k)
                    df["y"] = y
                    preds_out.append(df)
    tab = pd.DataFrame(rows)
    tab["p_holm"] = np.nan
    m = tab.k > 0
    tab.loc[m, "p_holm"] = holm(tab.loc[m, "p_vs_k0"].values)
    return tab, pd.concat(preds_out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--folds", default="0,1,2,3,4")
    a = ap.parse_args()
    cfg, runs, splits = load_data()
    ks = cfg["online"]["k_grid"]
    turns = pd.read_parquet(os.path.join(cfg["data"]["cache_dir"], "turns.parquet"))
    if a.stage in ("all", "fit"):
        for f in [int(x) for x in a.folds.split(",")]:
            fit_fold(cfg, runs, splits, turns, f, ks)
    if a.stage in ("all", "eval"):
        best = pd.read_csv("results/tables/e3_best_method.csv").best_method.iloc[0]
        tab, preds = evaluate(cfg, runs, best, ks)
        # fold ids for prediction file
        fold = splits.set_index("run_id").fold
        preds["fold"] = fold.loc[preds.run_id].values
        preds.to_parquet("results/predictions/e5_online.parquet", index=False)
        tab.to_csv("results/tables/e5_online.csv", index=False)
        L = ["# E5 — Online benchmark\n",
             f"k = 0 reference: best S1 method from E3 validation ({best}). O2 hyperparameters reuse P3's tuned "
             "configuration; O1's hazard models reuse P5's hazard configuration (see DEVIATIONS.md).\n"]
        for pop in ["eligible", "panel"]:
            t = tab[tab.population == pop]
            L.append(f"## Population: {'runs with K > k (eligible)' if pop == 'eligible' else 'fixed panel K ≥ 33'}\n")
            L.append(t[["k", "method", "n_runs", "qcrps", "lo", "hi", "ratio_vs_k0", "ratio_lo", "ratio_hi", "cov90",
                        "p_holm"]].round(4).to_markdown(index=False) + "\n")
        # H4: monotone improvement for O2 and k at which CRPS is 50% of pre-run CRPS (eligible population)
        o2 = tab[(tab.population == "eligible") & (tab.method == "O2")].sort_values("k")
        mono = bool(np.all(np.diff(o2.qcrps.values) < 0))
        half = o2[o2.ratio_vs_k0 <= 0.5].k
        k50 = int(half.iloc[0]) if len(half) else None
        o2p = tab[(tab.population == "panel") & (tab.method == "O2")].sort_values("k")
        mono_p = bool(np.all(np.diff(o2p.qcrps.values) < 0))
        r4 = o2[o2.k == 4].iloc[0]
        verdict = "supported" if (mono and mono_p) else ("inconclusive" if (mono or mono_p) else "not supported")
        pd.DataFrame([dict(hypothesis="H4", verdict=verdict, monotone_eligible=mono, monotone_panel=mono_p,
                           ratio_k4=r4.ratio_vs_k0, lo=r4.ratio_lo, hi=r4.ratio_hi, k_at_50=k50)]).to_csv(
            "results/tables/h4.csv", index=False)
        L.append(f"**H4 ({verdict})**: O2 QCRPS decreasing in k — eligible population: {mono}; fixed panel: {mono_p}. "
                 f"CRPS(k=4)/CRPS(k=0) = {r4.ratio_vs_k0:.3f} ({r4.ratio_lo:.3f}, {r4.ratio_hi:.3f}); first grid k with "
                 f"ratio ≤ 0.5: {k50}.\n")
        write_log("e5", L)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, pop in zip(axes[:2], ["eligible", "panel"]):
            t = tab[tab.population == pop]
            ref = t[t.k == 0].iloc[0]
            for name in ["O1", "O2", "O2-noregime"]:
                s = t[t.method == name].sort_values("k")
                xs = np.r_[0, s.k.values]
                ys = np.r_[ref.qcrps, s.qcrps.values]
                lo = np.r_[ref.lo, s.lo.values]
                hi = np.r_[ref.hi, s.hi.values]
                ax.plot(xs, ys, "o-", color=color(name), label=name)
                ax.fill_between(xs, lo, hi, color=color(name), alpha=0.2)
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xlabel("k (turns observed; 0 = pre-run best S1)")
            ax.set_ylabel("task-weighted QCRPS (log cost)")
            ax.set_title("Runs with K > k" if pop == "eligible" else "Fixed panel K ≥ 33")
            ax.legend()
        t = tab[tab.population == "eligible"]
        for name in ["O1", "O2", "O2-noregime"]:
            s = t[t.method == name].sort_values("k")
            axes[2].errorbar(s.k, s.cov90, yerr=[s.cov90 - s.cov90_lo, s.cov90_hi - s.cov90], fmt="o-",
                             color=color(name), label=name)
        axes[2].axhline(0.9, color="k", ls="--", lw=1)
        axes[2].set_xscale("log", base=2)
        axes[2].set_xlabel("k")
        axes[2].set_ylabel("coverage of central 90% interval")
        axes[2].set_title("Coverage vs k (eligible)")
        axes[2].legend()
        fig.tight_layout()
        fig.savefig("results/figures/fig_e5_crps_vs_k.png")
        plt.close(fig)
        print("\n".join(L))


if __name__ == "__main__":
    main()
