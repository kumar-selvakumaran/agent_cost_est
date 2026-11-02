"""E1 — stylized facts of agent cost (spec §10 E1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from src.eval.bootstrap import task_cluster_indices
from src.experiments.common import load_data, plt, sig3, ci, write_log, OKABE
from src.utils import seed_for


def hill(x: np.ndarray, frac=0.05) -> float:
    x = np.sort(x)[::-1]
    k = max(int(np.floor(frac * len(x))), 5)
    if len(x) <= k + 1:
        return np.nan
    return 1.0 / np.mean(np.log(x[:k] / x[k]))


def main():
    cfg, runs, _ = load_data()
    B = cfg["bootstrap"]["B"]
    L = ["# E1 — Stylized facts\n"]
    models = runs.model_id.value_counts()
    models = models[models >= 50].index.tolist()

    # distribution per model / family
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    order = runs[runs.model_id.isin(models)].groupby("model_id").y.median().sort_values().index
    axes[0].boxplot([runs.y[runs.model_id == m] for m in order], vert=False, tick_labels=order, showfliers=False,
                    whis=(1, 99))
    axes[0].set_xlabel("log Y (log reference units)")
    axes[0].set_title("Cost per model (whiskers: 1st–99th pct)")
    fo = runs.groupby("source_family").y.median().sort_values().index
    axes[1].boxplot([runs.y[runs.source_family == f] for f in fo], vert=False, tick_labels=fo, showfliers=False,
                    whis=(1, 99))
    axes[1].set_xlabel("log Y (log reference units)")
    axes[1].set_title("Cost per source family")
    fig.tight_layout()
    fig.savefig("results/figures/fig_e1_cost_by_model.png")
    plt.close(fig)
    desc = runs.groupby("model_id").y.describe(percentiles=[0.1, 0.5, 0.9]).round(3)
    desc.to_csv("results/tables/e1_logY_by_model.csv")
    runs.groupby("source_family").y.describe(percentiles=[0.1, 0.5, 0.9]).round(3).to_csv(
        "results/tables/e1_logY_by_family.csv")
    L.append("## Distribution of log Y\n")
    L.append(desc.to_markdown() + "\n")

    # within-cell spread
    cn = runs.groupby("cell_id").y.transform("size")
    c10 = runs[cn >= 10].groupby("cell_id").agg(maxmin=("Y", lambda s: s.max() / s.min()),
                                                p90p10=("Y", lambda s: s.quantile(0.9) / s.quantile(0.1)),
                                                sd_logY=("y", "std"), n=("y", "size"),
                                                model_id=("model_id", "first"))
    rng = np.random.default_rng(seed_for(cfg, "e1_cells"))
    rows = []
    for col in ["maxmin", "p90p10", "sd_logY"]:
        v = c10[col].values
        for stat, f in [("median", np.median), ("p90", lambda a: np.quantile(a, 0.9))]:
            bs = [f(v[rng.integers(0, len(v), len(v))]) for _ in range(B)]
            rows.append(dict(metric=col, stat=stat, value=f(v), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    sp = pd.DataFrame(rows)
    share30 = float((c10.maxmin >= 30).mean())
    sp.loc[len(sp)] = dict(metric="maxmin", stat="share_cells_ge_30x", value=share30, lo=np.nan, hi=np.nan)
    sp.loc[len(sp)] = dict(metric="maxmin", stat="max", value=float(c10.maxmin.max()), lo=np.nan, hi=np.nan)
    sp.to_csv("results/tables/e1_within_cell_spread.csv", index=False)
    L.append(f"## Within-cell spread (cells with ≥ 10 runs, n = {len(c10)})\n")
    L.append(sp.round(3).to_markdown(index=False) + "\n")
    L.append(f"Bai et al. report up to 30× between runs of the same task. Here the largest max/min ratio in a cell "
             f"is {sig3(c10.maxmin.max())}× and {share30:.1%} of cells (≥ 10 runs) exceed 30×.\n")

    # tails: Hill per model with task-cluster bootstrap
    rows = []
    for m in models:
        g = runs[runs.model_id == m]
        if len(g) < 200:
            continue
        a = hill(g.Y.values)
        codes, W = task_cluster_indices(g.task_id.values, B, seed_for(cfg, "hill" + m))
        bs = []
        for b in range(B):
            idx = np.repeat(np.arange(len(g)), W[b][codes])
            bs.append(hill(g.Y.values[idx]))
        rows.append(dict(model_id=m, n=len(g), hill_alpha=a, lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5)))
    hl = pd.DataFrame(rows)
    hl.to_csv("results/tables/e1_hill.csv", index=False)
    L.append("## Tail index (Hill estimator on Y, top 5% per model; α < 2 = infinite variance, α < 4 = heavy)\n")
    L.append(hl.round(3).to_markdown(index=False) + "\n")

    # bimodality: cells >= 20 runs
    c20 = runs[cn >= 20]
    pref2 = []
    for cid, g in c20.groupby("cell_id"):
        x = g.y.values.reshape(-1, 1)
        b1 = GaussianMixture(1, random_state=0).fit(x).bic(x)
        b2 = GaussianMixture(2, random_state=0, n_init=3).fit(x).bic(x)
        pref2.append(b2 < b1)
    pref2 = np.array(pref2)
    bs = [pref2[rng.integers(0, len(pref2), len(pref2))].mean() for _ in range(B)] if len(pref2) else [np.nan]
    bim = pd.DataFrame([dict(n_cells=len(pref2), share_bic_prefers_2=pref2.mean() if len(pref2) else np.nan,
                             lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5))])
    bim.to_csv("results/tables/e1_bimodality.csv", index=False)
    L.append("## Bimodality (cells with ≥ 20 runs; BIC 2- vs 1-component Gaussian mixture on log Y)\n")
    L.append(bim.round(3).to_markdown(index=False) + "\n")

    # scaling: log Y on log K per model, slope CI by task-cluster bootstrap
    rows = []
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, m in enumerate(models):
        g = runs[runs.model_id == m]
        x, y = np.log(g.K.values), g.y.values
        if np.var(x) == 0 or len(g) < 50:
            continue
        slope, icpt = np.polyfit(x, y, 1)
        codes, W = task_cluster_indices(g.task_id.values, B, seed_for(cfg, "scale" + m))
        T = W.shape[1]
        sx = np.bincount(codes, x, T); sy = np.bincount(codes, y, T); n = np.bincount(codes, minlength=T).astype(float)
        sxx = np.bincount(codes, x * x, T); sxy = np.bincount(codes, x * y, T)
        Wn, Wx, Wy, Wxx, Wxy = W @ n, W @ sx, W @ sy, W @ sxx, W @ sxy
        bsl = (Wxy - Wx * Wy / Wn) / (Wxx - Wx ** 2 / Wn)
        rows.append(dict(model_id=m, n=len(g), slope=slope, lo=np.nanpercentile(bsl, 2.5), hi=np.nanpercentile(bsl, 97.5),
                         r2=np.corrcoef(x, y)[0, 1] ** 2))
        c = OKABE[i % len(OKABE)]
        sub = g.sample(min(len(g), 1500), random_state=0)
        ax.scatter(np.log(sub.K), sub.y, s=2, alpha=0.25, color=c)
        xs = np.linspace(x.min(), x.max(), 10)
        ax.plot(xs, icpt + slope * xs, color=c, lw=1.5, label=f"{m} (slope {slope:.2f})")
    ax.set_xlabel("log K (number of turns)")
    ax.set_ylabel("log Y (log reference units)")
    ax.legend(fontsize=6, ncol=2)
    ax.set_title("Cost vs run length")
    fig.tight_layout()
    fig.savefig("results/figures/fig_e1_scaling.png")
    plt.close(fig)
    sc = pd.DataFrame(rows)
    sc.to_csv("results/tables/e1_scaling.csv", index=False)
    L.append("## Scaling of log Y with log K (hypothesis from context growth: slope in [1, 2])\n")
    L.append(sc.round(3).to_markdown(index=False) + "\n")

    # cost vs success
    rows = []
    for lab in ["success", "completed"]:
        g = runs[runs[lab].notna()]
        for src, h in g.groupby("source"):
            if h[lab].nunique() < 2 or len(h) < 30:
                continue
            rows.append(dict(label=lab, source=src, n=len(h), share_label1=h[lab].mean(),
                             median_logY_1=h.y[h[lab] == 1].median(), median_logY_0=h.y[h[lab] == 0].median(),
                             cost_share_on_0=h.Y[h[lab] == 0].sum() / h.Y.sum()))
    cs = pd.DataFrame(rows)
    cs.to_csv("results/tables/e1_cost_vs_success.csv", index=False)
    lab_cov = runs.success.notna().mean()
    tot = {}
    for lab in ["success", "completed"]:
        g = runs[runs[lab].notna()]
        tot[lab] = (len(g), g.Y[g[lab] == 0].sum() / g.Y.sum(), (g[lab] == 0).mean())
    L.append("## Cost vs success\n")
    L.append(f"Spec success label available for {lab_cov:.1%} of runs (from `result`/`judgment`/`split`); the "
             f"exploratory `completed` proxy (no error and final `task_complete: true`) is available for all runs. "
             f"Share of total cost spent on label-0 runs: success label {tot['success'][1]:.1%} "
             f"(failures are {tot['success'][2]:.1%} of {tot['success'][0]:,} labelled runs); completed proxy "
             f"{tot['completed'][1]:.1%} (not completed: {tot['completed'][2]:.1%} of runs).\n")
    L.append(cs.round(3).to_markdown(index=False) + "\n")
    # sources where every labelled run succeeds (selection-on-outcome flag, §11.6)
    allsucc = runs[runs.success.notna()].groupby("source").success.agg(["size", "mean"])
    flag = allsucc[allsucc["mean"] == 1.0]
    L.append("Sources where every labelled run succeeds (possible outcome filtering): "
             + (", ".join(f"{s} (n={int(r['size'])})" for s, r in flag.iterrows()) if len(flag) else "none") + "\n")
    write_log("e1", L)
    print("\n".join(L))


if __name__ == "__main__":
    main()
