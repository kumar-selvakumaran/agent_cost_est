"""E3 stage 2 — pool out-of-fold predictions, select the best method on validation, apply CQR, evaluate."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.eval.bootstrap import task_cluster_indices, holm
from src.eval.evaluate import Evaluator, comparisons_table, pit_by_group
from src.eval.metrics import QCOLS, TAUS, qcrps, interp_q, ks_uniform
from src.experiments.common import load_data, plt, write_log, color, sig3
from src.experiments.e3_cold import assemble, load_pred, PRED_DIR
from src.predictors.cqr import CQR, one_per_task, LEVELS_2S, LEVELS_1S
from src.utils import seed_for

BASE_METHODS = ["P0", "P1", "P2", "P3", "P3-EVT", "P4", "P5", "P6"]
ABLATION = ["P4-K1", "P4-K2"]
SEEDS = {"P4": 3, "P4-K1": 3, "P4-K2": 3, "P5": 3}


def val_score(cfg, runs, method, seed=0):
    R = runs.set_index("run_id")
    vals = []
    for f in range(cfg["cv"]["n_splits"]):
        d = load_pred("e3", method, f, seed)
        rid = d["val_run_id"].astype(np.int64)
        s = qcrps(d["val"].astype(float), R.loc[rid, "y"].values)
        vals.append(pd.Series(s).groupby(R.loc[rid, "task_id"].values).mean().mean())
    return float(np.mean(vals))


def cqr_predictions(cfg, runs, method, mondrian=False):
    """Fold-wise CQR on one-run-per-task calibration sets; returns (test DataFrame grid-adjusted, interval dict)."""
    R = runs.set_index("run_id")
    parts, iv = [], []
    for f in range(cfg["cv"]["n_splits"]):
        d = load_pred("e3", method, f)
        qc, rc = d["cal"].astype(float), d["cal_run_id"].astype(np.int64)
        pick = one_per_task(R.loc[rc].reset_index(), np.random.default_rng(seed_for(cfg, f"cqr_f{f}")))
        qc, rc = qc[pick], rc[pick]
        c = CQR(mondrian_min=200).fit(qc, R.loc[rc, "y"].values, R.loc[rc, "model_id"].values)
        qt, rt = d["test"].astype(float), d["test_run_id"].astype(np.int64)
        models = R.loc[rt, "model_id"].values
        adj = c.adjust_grid(qt)
        df = pd.DataFrame(adj, columns=QCOLS)
        df.insert(0, "run_id", rt)
        df.insert(1, "fold", f)
        parts.append(df)
        rec = {"run_id": rt}
        for mond in (False, True):
            tag = "mond_" if mond else ""
            for L in LEVELS_2S:
                lo, hi = c.intervals(qt, L, models, mond)
                rec[f"{tag}lo{int(L*100)}"], rec[f"{tag}hi{int(L*100)}"] = lo, hi
            for t in LEVELS_1S:
                rec[f"{tag}up{int(t*100)}"] = c.upper(qt, t, models, mond)
        rec["n_cal"] = np.full(len(rt), len(qc))
        iv.append(pd.DataFrame(rec))
    df = pd.concat(parts, ignore_index=True)
    meta = R.loc[df.run_id, ["task_id", "model_id"]].reset_index(drop=True)
    df.insert(1, "task_id", meta.task_id.values)
    df.insert(2, "model_id", meta.model_id.values)
    df.insert(4, "method", f"CQR({method})")
    return df, pd.concat(iv, ignore_index=True)


def main():
    cfg, runs, splits = load_data()
    B = cfg["bootstrap"]["B"]
    R = runs.set_index("run_id")
    os.makedirs("results/predictions", exist_ok=True)
    # 1. validation-based selection of the best method (pre-specified: task-weighted val QCRPS, mean over folds)
    vs = pd.DataFrame([dict(method=m, val_qcrps=val_score(cfg, runs, m)) for m in BASE_METHODS + ABLATION])
    vs.to_csv("results/tables/e3_validation_scores.csv", index=False)
    cand = vs[vs.method.isin([m for m in BASE_METHODS if m not in ("P0", "P1")])]
    best = cand.sort_values("val_qcrps").method.iloc[0]
    pd.DataFrame([dict(best_method=best)]).to_csv("results/tables/e3_best_method.csv", index=False)

    # 2. assemble predictions (seed 0 = first pre-specified seed), add CQR wrappers
    preds = {m: assemble("e3", m, runs) for m in BASE_METHODS + ABLATION}
    cqr_targets = ["P3-EVT"] + ([best] if best != "P3-EVT" else [])
    intervals = {}
    for m in cqr_targets:
        df, iv = cqr_predictions(cfg, runs, m)
        name = "CQR(P3-EVT)" if m == "P3-EVT" else "CQR(best)"
        df["method"] = name
        preds[name] = df
        intervals[name] = iv
    if best == "P3-EVT":
        preds["CQR(best)"] = preds["CQR(P3-EVT)"].assign(method="CQR(best)")
        intervals["CQR(best)"] = intervals["CQR(P3-EVT)"]
    order = np.sort(preds["P1"].run_id.values)
    for m in preds:
        preds[m] = preds[m].set_index("run_id").loc[order].reset_index()
        preds[m]["k"] = 0
        preds[m]["y"] = R.loc[order, "y"].values
        cols = ["run_id", "task_id", "model_id", "fold", "method", "k"] + QCOLS + ["y"]
        preds[m][cols].to_parquet(f"results/predictions/e3_{m}.parquet", index=False)
    task = R.loc[order, "task_id"].values
    ev = Evaluator(runs, task, B, seed_for(cfg, "e3_boot"))
    main_methods = BASE_METHODS + ABLATION + ["CQR(P3-EVT)", "CQR(best)"]
    rows = [ev.summarize(m, preds[m]) for m in main_methods]
    tab = pd.DataFrame(rows)
    for m in main_methods:
        g = ev.rel_gain(m, "P1")
        tab.loc[tab.method == m, ["gain_vs_P1", "gain_lo", "gain_hi"]] = [g["gain"], g["gain_lo"], g["gain_hi"]]
    # seeds: mean +- SD of task-weighted QCRPS across the 3 seeds
    sd_rows = []
    for m, ns in SEEDS.items():
        vals = []
        for s in range(ns):
            p = assemble("e3", m, runs, seed=s).set_index("run_id").loc[order]
            v = qcrps(p[QCOLS].to_numpy(float), R.loc[order, "y"].values)
            vals.append(pd.Series(v).groupby(task).mean().mean())
        sd_rows.append(dict(method=m, n_seeds=ns, qcrps_mean=np.mean(vals), qcrps_sd=np.std(vals, ddof=1),
                            seeds=";".join(f"{v:.4f}" for v in vals)))
    sdt = pd.DataFrame(sd_rows)
    sdt.to_csv("results/tables/e3_seed_variation.csv", index=False)
    tab = tab.merge(sdt[["method", "qcrps_mean", "qcrps_sd"]].rename(columns={"qcrps_mean": "qcrps_seed_mean",
                                                                             "qcrps_sd": "qcrps_seed_sd"}),
                    on="method", how="left")
    tab.to_csv("results/tables/e3_main.csv", index=False)
    comp = comparisons_table(ev, main_methods, "P1")
    comp.to_csv("results/tables/e3_comparisons.csv", index=False)
    # H2
    g = ev.rel_gain(best, "P1")
    pd.DataFrame({"reps": g["reps"]}).assign(est=g["gain"]).to_csv("results/tables/e3_gain_reps.csv", index=False)
    row = comp[(comp.a == best) & (comp.b == "P1")].iloc[0]
    h2 = "supported" if (row["diff"] < 0 and row.p_holm < 0.05) else (
        "not supported" if row["diff"] >= 0 or row.hi > 0 and row.lo > 0 else "inconclusive")
    if row["diff"] < 0 and row.p_holm >= 0.05:
        h2 = "inconclusive"
    pd.DataFrame([dict(hypothesis="H2", verdict=h2, best_method=best, gain=g["gain"], lo=g["gain_lo"],
                       hi=g["gain_hi"], diff=row["diff"], p_holm=row.p_holm)]).to_csv("results/tables/h2.csv", index=False)
    # H3: PIT KS of non-Gaussian (P3-EVT, P4, P5) vs P2, paired task-cluster bootstrap of the KS difference
    codes, W = ev.codes_W
    p2 = ev.per_run["P2"][4]
    h3rows = []
    for m in ["P3-EVT", "P4", "P5"]:
        pm = ev.per_run[m][4]
        diffs = []
        for b in range(min(B, 400)):
            idx = np.repeat(np.arange(len(pm)), W[b][codes])
            diffs.append(ks_uniform(pm[idx]) - ks_uniform(p2[idx]))
        h3rows.append(dict(method=m, ks=ks_uniform(pm), ks_P2=ks_uniform(p2), diff=ks_uniform(pm) - ks_uniform(p2),
                           lo=np.percentile(diffs, 2.5), hi=np.percentile(diffs, 97.5)))
    h3t = pd.DataFrame(h3rows)
    sig_lower = (h3t.hi < 0).sum()
    h3 = "supported" if sig_lower == 3 else ("not supported" if sig_lower == 0 else "inconclusive")
    bestng = h3t.sort_values("ks").iloc[0]
    h3t.to_csv("results/tables/e3_h3_pit.csv", index=False)
    pd.DataFrame([dict(hypothesis="H3", verdict=h3, ks_P2=bestng.ks_P2, best_non_gaussian=bestng.method,
                       ks_best=bestng.ks, diff=bestng["diff"], lo=bestng.lo, hi=bestng.hi,
                       n_sig_lower=int(sig_lower))]).to_csv("results/tables/h3.csv", index=False)
    # PIT / coverage by model and family
    pg = pd.concat([pit_by_group(ev, m, c) for m in main_methods for c in ["model_id", "source_family"]])
    pg.to_csv("results/tables/e3_pit_by_group.csv", index=False)
    # CQR interval coverage (spec-level intervals; pooled and Mondrian)
    cr = []
    y = R.loc[order, "y"].values
    cw = task_cluster_indices(task, B, seed_for(cfg, "e3_boot"))
    from src.eval.bootstrap import boot_task_metric
    for name, iv in intervals.items():
        iv = iv.set_index("run_id").loc[order]
        for mond in ("", "mond_"):
            for L in LEVELS_2S:
                c = ((y >= iv[f"{mond}lo{int(L*100)}"].values) & (y <= iv[f"{mond}hi{int(L*100)}"].values)).astype(float)
                b = boot_task_metric(c, task, B, 0, "run", cw)
                w = iv[f"{mond}hi{int(L*100)}"].values - iv[f"{mond}lo{int(L*100)}"].values
                cr.append(dict(method=name, variant="mondrian" if mond else "pooled", interval=f"central {int(L*100)}%",
                               nominal=L, coverage=b["est"], lo=b["lo"], hi=b["hi"], mean_width=float(np.mean(w))))
            for t in LEVELS_1S:
                c = (y <= iv[f"{mond}up{int(t*100)}"].values).astype(float)
                b = boot_task_metric(c, task, B, 0, "run", cw)
                cr.append(dict(method=name, variant="mondrian" if mond else "pooled", interval=f"upper {int(t*100)}%",
                               nominal=t, coverage=b["est"], lo=b["lo"], hi=b["hi"], mean_width=np.nan))
    crt = pd.DataFrame(cr)
    crt.to_csv("results/tables/e3_cqr_coverage.csv", index=False)
    # P4 mixture weights (K = 3, seed 0)
    wrows = []
    for f in range(cfg["cv"]["n_splits"]):
        d = load_pred("e3", "P4", f)
        w, mu = d["test_w"].astype(float), d["test_mu"].astype(float)
        dom = w.argmax(1)
        hi_comp = mu.argmax(1)
        wrows.append(dict(fold=f, mean_max_weight=w.max(1).mean(), share_dominant_high_mean=(dom == hi_comp).mean(),
                          mean_weight_high_mean=w[np.arange(len(w)), hi_comp].mean()))
    pd.DataFrame(wrows).to_csv("results/tables/e3_p4_mixture.csv", index=False)
    # per-model QCRPS (visible-cost threat: report per model)
    pm = []
    for m in main_methods:
        s = ev.per_run[m][0].qcrps.values
        for mod, idx in pd.Series(np.arange(len(order))).groupby(R.loc[order, "model_id"].values).groups.items():
            idx = np.asarray(list(idx))
            pm.append(dict(method=m, model_id=mod, n=len(idx),
                           qcrps_task_wt=pd.Series(s[idx]).groupby(task[idx]).mean().mean()))
    pd.DataFrame(pm).to_csv("results/tables/e3_by_model.csv", index=False)
    figures(ev, main_methods, tab)
    # log
    show = tab.set_index("method")
    L = ["# E3 — Cold pre-run benchmark (S1)\n",
         f"Pooled out-of-fold predictions over 5 task-grouped folds: {len(order):,} runs, {len(set(task)):,} tasks. "
         f"Best method on validation (task-weighted val QCRPS, mean over folds): **{best}**. Stochastic predictors: "
         "main rows use the first pre-specified seed; mean ± SD over 3 seeds in `e3_seed_variation.csv`.\n",
         "## Validation scores\n", vs.round(4).to_markdown(index=False) + "\n",
         "## Main table\n",
         show[["qcrps", "qcrps_lo", "qcrps_hi", "qcrps_run", "gain_vs_P1", "gain_lo", "gain_hi", "pin50", "pin90",
               "pin99", "spearman", "pearson", "abs_err50", "width80", "cov80", "cov90", "cov95", "pit_ks", "dq_p_90",
               "w1_cells", "es_ratio"]].round(4).to_markdown() + "\n",
         "## Comparisons (paired task-cluster bootstrap, Holm-corrected)\n", comp.round(4).to_markdown(index=False) + "\n",
         "## Seed variation\n", sdt.round(4).to_markdown(index=False) + "\n",
         "## CQR coverage\n", crt.round(4).to_markdown(index=False) + "\n",
         f"**H2 ({h2})**: relative CRPS gain of {best} vs P1 = {g['gain']:.3f} ({g['gain_lo']:.3f}, {g['gain_hi']:.3f}); "
         f"Holm p = {row.p_holm:.3g}.\n",
         f"**H3 ({h3})**: PIT KS P2 = {bestng.ks_P2:.3f}; " + "; ".join(
             f"{r.method} = {r.ks:.3f} (diff {getattr(r, 'diff'):.3f} [{r.lo:.3f}, {r.hi:.3f}])" for r in h3t.itertuples()) + "\n"]
    write_log("e3", L)
    print("\n".join(L))


def figures(ev, methods, tab):
    n = len(methods)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.4 * rows), sharey=True)
    for ax, m in zip(axes.flat, methods):
        p = ev.per_run[m][4]
        ax.hist(p, bins=10, range=(0, 1), density=True, color=color(m), edgecolor="white")
        ax.axhline(1, color="k", ls="--", lw=1)
        ax.set_title(f"{m} (KS {tab.set_index('method').loc[m, 'pit_ks']:.3f})", fontsize=8)
        ax.set_xlabel("PIT")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("PIT histograms (uniform = calibrated)")
    fig.tight_layout()
    fig.savefig("results/figures/fig_e3_pit_grid.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98])
    for m in methods:
        _, q, y, _, _ = ev.per_run[m]
        emp = [np.mean((y >= interp_q(q, (1 - L) / 2)) & (y <= interp_q(q, 1 - (1 - L) / 2))) for L in levels]
        ax.plot(levels, emp, "o-", ms=3, color=color(m), label=m)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("nominal central-interval coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Reliability of central intervals")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig("results/figures/fig_e3_reliability.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
