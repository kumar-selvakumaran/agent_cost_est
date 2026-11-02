"""E8 — budget-policy replay (spec §10 E8).

Per-run cap = exp(predicted upper quantile q_tau). Replaying each test run's turns: if the cumulative cost
exceeds the cap, the run is killed at the end of that turn (cost = cumulative cost through that turn).
Methods: the best S1 method with one-sided CQR adjustment, and P1 (its raw quantiles).
Success: the spec label (`result`/`judgment`/`split`) is degenerate in this sample (all labelled runs are
errors/failures), so the outcome is the exploratory `completed` proxy; the spec label is reported too.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.eval.bootstrap import task_cluster_indices
from src.eval.metrics import interp_q
from src.experiments.common import load_data, plt, write_log, color
from src.experiments.e3_cold import load_pred
from src.predictors.cqr import one_per_task, conformal_quantile
from src.utils import seed_for

TAUS_TABLE = [0.8, 0.9, 0.95, 0.99]
TAUS_CURVE = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99]


def caps_for(cfg, runs, method, cqr: bool, taus):
    """Returns run_id array and dict tau -> log cap, pooled over folds (CQR calibrated per fold)."""
    R = runs.set_index("run_id")
    ids, caps = [], {t: [] for t in taus}
    for f in range(cfg["cv"]["n_splits"]):
        d = load_pred("e3", method, f)
        q, rid = d["test"].astype(float), d["test_run_id"].astype(np.int64)
        ids.append(rid)
        if cqr:
            qc, rc = d["cal"].astype(float), d["cal_run_id"].astype(np.int64)
            pick = one_per_task(R.loc[rc].reset_index(), np.random.default_rng(seed_for(cfg, f"cqr_f{f}")))
            qc, yc = qc[pick], R.loc[rc[pick], "y"].values
        for t in taus:
            base = interp_q(q, t)
            if cqr:
                base = base + conformal_quantile(yc - interp_q(qc, t), t)
            caps[t].append(base)
    return np.concatenate(ids), {t: np.concatenate(v) for t, v in caps.items()}


def replay(turns_g, ids, logcap):
    """Return realized (capped) cost and killed flag per run."""
    cost = np.empty(len(ids))
    killed = np.zeros(len(ids), bool)
    cap = np.exp(logcap)
    for i, rid in enumerate(ids):
        cc = turns_g[rid]
        over = np.flatnonzero(cc > cap[i])
        if len(over) and over[0] < len(cc) - 1:
            cost[i] = cc[over[0]]
            killed[i] = True
        else:
            cost[i] = cc[-1]
    return cost, killed


def outcomes(cost, killed, Y, succ, w=None):
    w = np.ones(len(cost)) if w is None else w
    s = np.nan_to_num(succ) * w
    has = ~np.isnan(succ)
    n_succ = s.sum()
    out = dict(cost_saved=1 - (w * cost).sum() / (w * Y).sum(), runs_killed=(w * killed).sum() / w.sum(),
               succ_lost=(s * killed).sum() / n_succ if n_succ > 0 else np.nan,
               cost_per_success=(w * cost).sum() / (s * ~killed).sum() if (s * ~killed).sum() > 0 else np.nan,
               cost_per_success_nocap=(w * Y).sum() / n_succ if n_succ > 0 else np.nan)
    return out


def main():
    cfg, runs, splits = load_data()
    best = pd.read_csv("results/tables/e3_best_method.csv").best_method.iloc[0]
    base_method = best.replace("CQR(", "").rstrip(")")
    turns = pd.read_parquet(os.path.join(cfg["data"]["cache_dir"], "turns.parquet"), columns=["run_id", "cum_cost"])
    tg = {rid: g.values for rid, g in turns.groupby("run_id").cum_cost}
    R = runs.set_index("run_id")
    B = cfg["bootstrap"]["B"]
    rows, curves = [], []
    methods = [("P1", "P1", False), (f"CQR({base_method})", base_method, True)]
    for label, m, cqr in methods:
        ids, caps = caps_for(cfg, runs, m, cqr, TAUS_CURVE)
        Y = R.loc[ids, "Y"].values
        task = R.loc[ids, "task_id"].values
        codes, W = task_cluster_indices(task, B, seed_for(cfg, "e8_boot"))
        for succ_name in ["completed", "success"]:
            succ = R.loc[ids, succ_name].values.astype(float)
            for t in TAUS_CURVE:
                cost, killed = replay(tg, ids, caps[t])
                o = outcomes(cost, killed, Y, succ)
                row = dict(method=label, success_label=succ_name, tau=t, **o)
                if t in TAUS_TABLE:
                    bs = {k: [] for k in o}
                    for b in range(B):
                        w = W[b][codes].astype(float)
                        ob = outcomes(cost, killed, Y, succ, w)
                        for k in ob:
                            bs[k].append(ob[k])
                    for k in o:
                        row[k + "_lo"], row[k + "_hi"] = np.nanpercentile(bs[k], [2.5, 97.5])
                rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv("results/tables/e8_replay.csv", index=False)
    main_t = tab[(tab.success_label == "completed") & tab.tau.isin(TAUS_TABLE)]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for label, _, _ in methods:
        s = tab[(tab.method == label) & (tab.success_label == "completed")].sort_values("tau")
        ax.plot(100 * s.succ_lost, 100 * s.cost_saved, "o-", color=color("P1" if label == "P1" else "CQR(best)"),
                label=label)
        for _, r in s[s.tau.isin(TAUS_TABLE)].iterrows():
            ax.annotate(f"τ={r.tau}", (100 * r.succ_lost, 100 * r.cost_saved), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
    ax.set_xlabel("completed runs killed (%)  [exploratory success proxy]")
    ax.set_ylabel("total cost saved (%)")
    ax.set_title("Budget-cap replay: savings vs successes lost")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/fig_e8_frontier.png")
    plt.close(fig)
    # savings at matched success loss (interpolated on the curves) for the headline
    def at_loss(label, loss):
        s = tab[(tab.method == label) & (tab.success_label == "completed")].sort_values("succ_lost")
        return float(np.interp(loss, s.succ_lost, s.cost_saved))
    ref_loss = float(main_t[(main_t.method == "P1") & (main_t.tau == 0.9)].succ_lost.iloc[0])
    matched = pd.DataFrame([dict(success_loss=ref_loss, P1=at_loss("P1", ref_loss),
                                 best=at_loss(methods[1][0], ref_loss), best_method=methods[1][0])])
    matched.to_csv("results/tables/e8_matched.csv", index=False)
    L = ["# E8 — Budget-policy replay\n",
         "Outcome label: the spec success label (from `result`/`judgment`/`split`) covers few runs and those are almost "
         "all failures (error results), so 'successes lost' under it is undefined or 0. The table uses the "
         "exploratory `completed` proxy (no error result and final `task_complete: true`). CIs: task-cluster bootstrap.\n",
         main_t[["method", "tau", "cost_saved", "cost_saved_lo", "cost_saved_hi", "runs_killed", "succ_lost",
                 "succ_lost_lo", "succ_lost_hi", "cost_per_success", "cost_per_success_nocap"]].round(4).to_markdown(index=False) + "\n",
         f"At the success loss of P1 with τ = 0.9 ({ref_loss:.2%}), interpolated cost savings: P1 "
         f"{matched.P1.iloc[0]:.2%}, {methods[1][0]} {matched.best.iloc[0]:.2%}.\n"]
    write_log("e8", L)
    print("\n".join(L))


if __name__ == "__main__":
    main()
