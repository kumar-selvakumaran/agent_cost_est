"""E3 sensitivities (primary metric only): secondary targets, summarization-flagged runs excluded,
visible-cost (hidden-reasoning models excluded), leave-source-family-out CV, feature ablations of the best method."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.bootstrap import task_cluster_indices, boot_task_metric, paired_diff
from src.eval.metrics import qcrps
from src.experiments.common import load_data, write_log
from src.experiments.e3_cold import fit_fold, load_pred, assemble
from src.utils import seed_for

HIDDEN = {"gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-codex"}


def pooled(exp, method, runs, target, n_folds=5, seed=0):
    df = assemble(exp, method, runs, n_folds, seed).set_index("run_id").sort_index()
    from src.eval.metrics import QCOLS
    q = df[QCOLS].to_numpy(float)
    y = runs.set_index("run_id").loc[df.index, target].values
    return df.index.values, qcrps(q, y)


def tw(cfg, s, task, key):
    cw = task_cluster_indices(task, cfg["bootstrap"]["B"], seed_for(cfg, key))
    return boot_task_metric(s, task, cfg["bootstrap"]["B"], 0, "task", cw), cw


def main():
    cfg, runs, splits = load_data()
    best = pd.read_csv("results/tables/e3_best_method.csv").best_method.iloc[0]
    R = runs.set_index("run_id")
    rows = []

    def add(analysis, method, ids, s, extra=None):
        task = R.loc[ids, "task_id"].values
        b, _ = tw(cfg, s, task, "sens_" + analysis)
        row = dict(analysis=analysis, method=method, n_runs=len(ids), qcrps=b["est"], lo=b["lo"], hi=b["hi"])
        row.update(extra or {})
        rows.append(row)
        return b

    # main-target reference
    ids, s1 = pooled("e3", "P1", runs, "y")
    _, sb = pooled("e3", best, runs, "y")
    task = R.loc[ids, "task_id"].values
    for name, mask in [("primary target (reference)", np.ones(len(ids), bool)),
                       ("summarization-flagged runs excluded", ~R.loc[ids, "summ_flag"].values.astype(bool)),
                       ("hidden-reasoning models excluded (visible = billed)", ~R.loc[ids, "model_id"].isin(HIDDEN).values)]:
        for m, s in [("P1", s1), (best, sb)]:
            add(name, m, ids[mask], s[mask])
        d = paired_diff(sb[mask], s1[mask], task[mask], cfg["bootstrap"]["B"], 0)
        rows[-1].update(diff_vs_P1=d["est"], diff_lo=d["lo"], diff_hi=d["hi"],
                        rel_gain=-d["est"] / rows[-2]["qcrps"])
    # secondary targets: retrain P1 and best
    for tgt, exp in [("y_tok", "e3tok"), ("y_hi", "e3hi")]:
        for f in range(cfg["cv"]["n_splits"]):
            fit_fold(cfg, runs, splits, f, exp=exp, methods=["P1", best], target=tgt, n_seeds=1, pred_roles=("test",))
        i1, t1 = pooled(exp, "P1", runs, tgt)
        _, tb = pooled(exp, best, runs, tgt)
        add(f"target {tgt}", "P1", i1, t1)
        add(f"target {tgt}", best, i1, tb)
        d = paired_diff(tb, t1, R.loc[i1, "task_id"].values, cfg["bootstrap"]["B"], 0)
        rows[-1].update(diff_vs_P1=d["est"], diff_lo=d["lo"], diff_hi=d["hi"], rel_gain=-d["est"] / rows[-2]["qcrps"])
    # leave-source-family-out
    fams = sorted(runs.source_family.unique())
    for fam in fams:
        test = (runs.source_family == fam).values
        tr_groups = runs.split_group[~test].unique()
        rng = np.random.default_rng(seed_for(cfg, "loso_" + fam))
        perm = rng.permutation(tr_groups)
        nf, nv = int(round(0.7 * len(perm))), int(round(0.15 * len(perm)))
        gmap = {g: ("fit" if i < nf else ("val" if i < nf + nv else "cal")) for i, g in enumerate(perm)}
        role = np.where(test, "test", runs.split_group.map(gmap).values)
        sp = splits.copy()
        sp["role_loso"] = role
        tag = "loso_" + fam.replace(" ", "_")
        fit_fold(cfg, runs, sp, 0, exp=tag, methods=["P1", best], group_col="role_loso", svd_tag=tag, n_seeds=1, pred_roles=("test",))
        for m in ["P1", best]:
            d = load_pred(tag, m, 0)
            rid = d["test_run_id"].astype(np.int64)
            s = qcrps(d["test"].astype(float), R.loc[rid, "y"].values)
            ref = pd.Series(s1 if m == "P1" else sb, index=ids).loc[rid].values
            b = add(f"leave-family-out: {fam}", m, rid, s)
            rr = add(f"grouped-task CV on same runs: {fam}", m, rid, ref)
            rows[-1]["degradation_loso_minus_cv"] = b["est"] - rr["est"]
    # feature ablations of the best method
    for fs, label in [("a", "(a) model + source only"), ("b", "(b) + numeric text stats"), ("c", "(c) + keywords")]:
        exp = "e3abl" + fs
        for f in range(cfg["cv"]["n_splits"]):
            fit_fold(cfg, runs, splits, f, exp=exp, methods=[best], feature_set=fs, n_seeds=1, pred_roles=("test",))
        i_, s_ = pooled(exp, best, runs, "y")
        add("ablation " + label, best, i_, s_)
    add("ablation (d) + TF-IDF/SVD (full)", best, ids, sb)
    tab = pd.DataFrame(rows)
    tab.to_csv("results/tables/e3_sensitivity.csv", index=False)
    write_log("e3_sensitivity", ["# E3 sensitivities (primary metric: task-weighted QCRPS)\n",
                                 f"Best method: {best}. Secondary targets, leave-family-out and ablations refit the "
                                 "method (single seed); exclusion analyses re-evaluate the main predictions on the subset.\n",
                                 tab.round(4).to_markdown(index=False)])
    print(tab.round(4).to_string())


if __name__ == "__main__":
    main()
