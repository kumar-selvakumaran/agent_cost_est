"""E2 — variance decomposition (tests H1), spec §10 E2."""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from src.eval.decomposition import fit_level_mixed, fit_spread_mixed
from src.experiments.common import load_data, plt, write_log, sig3
from src.features.prerun import base_numeric, svd_features, gbm_frame
from src.predictors.base import lgb_train, DEFAULT_HP
from src.utils import seed_for


def resample_tasks(df: pd.DataFrame, rng) -> pd.DataFrame:
    tasks = df.task_id.unique()
    pick = rng.choice(tasks, size=len(tasks), replace=True)
    g = df.groupby("task_id").indices
    parts = []
    for i, t in enumerate(pick):
        sub = df.iloc[g[t]].copy()
        sub["task_id"] = f"{t}#{i}"
        if "cell_id" in sub:
            sub["cell_id"] = sub["cell_id"] + f"#{i}"
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def _boot_level(df, seed):
    try:
        return fit_level_mixed(resample_tasks(df, np.random.default_rng(seed)))
    except Exception:
        return None


def _boot_spread(cells, seed):
    try:
        return fit_spread_mixed(resample_tasks(cells, np.random.default_rng(seed)))
    except Exception:
        return None


def main():
    cfg, runs, splits = load_data()
    Bm = cfg["bootstrap"]["B_mixed"]
    base = seed_for(cfg, "e2")
    L = ["# E2 — Variance decomposition\n"]
    cn = runs.groupby("cell_id").y.transform("size")
    c4 = runs[cn >= 4]
    mc = c4.groupby("model_id").cell_id.nunique()
    keep_models = mc[mc >= 20].index
    dropped = sorted(set(c4.model_id) - set(keep_models))
    c4 = c4[c4.model_id.isin(keep_models)]
    nm = c4.groupby("task_id").model_id.nunique()
    X = c4[c4.task_id.isin(nm[nm >= 2].index)].copy()
    L.append(f"Crossed design: tasks with ≥ 2 models each having ≥ 4 runs. Models with < 20 such cells are dropped "
             f"for identifiability ({', '.join(dropped) or 'none'}). n tasks = {X.task_id.nunique():,}, cells = "
             f"{X.cell_id.nunique():,}, runs = {len(X):,}; models: {X.model_id.value_counts().to_dict()}.\n")

    # 1. level decomposition
    lv = fit_level_mixed(X)
    boots = Parallel(n_jobs=4, verbose=5)(delayed(_boot_level)(X, base + b) for b in range(Bm))
    boots = [b for b in boots if b is not None]
    rows = []
    for k in ["model", "task", "cell", "run"]:
        v = [b[k] for b in boots]
        rows.append(dict(component=k, share=lv[k], lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5)))
    lvl = pd.DataFrame(rows)
    lvl.to_csv("results/tables/e2_level_shares.csv", index=False)
    L.append(f"## Level decomposition of Var(log Y) (MixedLM REML, {len(boots)} task-cluster bootstrap refits)\n")
    L.append(lvl.round(3).to_markdown(index=False) + "\n")
    L.append(f"Total variance of log Y in this subset: {lv['total']:.3f}; converged: {lv['converged']}.\n")

    # 2. spread decomposition
    cells = X.groupby("cell_id").agg(task_id=("task_id", "first"), model_id=("model_id", "first"),
                                     s=("y", "std"), n=("y", "size")).reset_index()
    n_zero = int((cells.s < 1e-6).sum())
    cells["log_s"] = np.log(cells.s.clip(lower=1e-3))
    sp = fit_spread_mixed(cells)
    sboots = Parallel(n_jobs=4, verbose=5)(delayed(_boot_spread)(cells, base + 10_000 + b) for b in range(Bm))
    sboots = [b for b in sboots if b is not None]
    rows = []
    for k in ["model", "task", "residual"]:
        v = [b[k] for b in sboots]
        rows.append(dict(component=k, share_corrected=sp[k], lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5),
                         share_raw=sp[k + "_raw"]))
    diff = np.array([b["task"] - b["model"] for b in sboots])
    rows.append(dict(component="task - model", share_corrected=sp["task"] - sp["model"],
                     lo=np.percentile(diff, 2.5), hi=np.percentile(diff, 97.5), share_raw=sp["task_raw"] - sp["model_raw"]))
    spr = pd.DataFrame(rows)
    spr.to_csv("results/tables/e2_spread_shares.csv", index=False)
    h1 = (sp["task"] > sp["model"]) and np.percentile(diff, 2.5) > 0
    h1_verdict = "supported" if h1 else ("not supported" if np.percentile(diff, 97.5) < 0 else "inconclusive")
    pd.DataFrame([dict(hypothesis="H1", verdict=h1_verdict, stat="task share - model share (spread, noise-corrected)",
                       est=sp["task"] - sp["model"], lo=np.percentile(diff, 2.5), hi=np.percentile(diff, 97.5))]
                 ).to_csv("results/tables/h1.csv", index=False)
    L.append(f"## Spread decomposition of log SD within cell (cells ≥ 4 runs in crossed tasks; n cells = {len(cells):,}; "
             f"{n_zero} cells with SD = 0 clipped at 1e-3)\n")
    L.append(f"Sampling-noise variance of log s (mean of 1/(2(n−1))): {sp['noise']:.4f}, subtracted from the residual "
             f"variance {sp['v_res']:.4f}.\n")
    L.append(spr.round(3).to_markdown(index=False) + "\n")
    L.append(f"**H1 ({h1_verdict})**: task share − model share = {sp['task'] - sp['model']:.3f} "
             f"(95% CI {np.percentile(diff, 2.5):.3f}, {np.percentile(diff, 97.5):.3f}).\n")

    # 3. spread transfer between models
    rng = np.random.default_rng(base + 1)
    allc = runs[cn >= 4]
    cs = allc.groupby("cell_id").agg(task_id=("task_id", "first"), model_id=("model_id", "first"),
                                     s=("y", "std")).reset_index()
    cs["log_s"] = np.log(cs.s.clip(lower=1e-3))
    # split-half reliability per model
    half = []
    for cid, g in allc.groupby("cell_id"):
        y = g.y.values[rng.permutation(len(g))]
        h = len(y) // 2
        if h >= 2:
            half.append((cid, g.model_id.iloc[0], np.log(max(np.std(y[:h], ddof=1), 1e-3)),
                         np.log(max(np.std(y[h:2 * h], ddof=1), 1e-3))))
    half = pd.DataFrame(half, columns=["cell_id", "model_id", "a", "b"])
    rel = {}
    for m, g in half.groupby("model_id"):
        if len(g) >= 30:
            rhh = stats.spearmanr(g.a, g.b).statistic
            rel[m] = 2 * rhh / (1 + rhh)
    piv = cs.pivot_table(index="task_id", columns="model_id", values="log_s")
    models = [m for m in piv.columns if m in rel]
    rows = []
    M = pd.DataFrame(np.nan, index=models, columns=models)
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            both = piv[[a, b]].dropna()
            if len(both) < 50:
                continue
            r_obs = stats.spearmanr(both[a], both[b]).statistic
            den = np.sqrt(rel[a] * rel[b]) if rel[a] > 0 and rel[b] > 0 else np.nan
            r_dis = r_obs / den if den == den else np.nan
            bs = []
            for _ in range(cfg["bootstrap"]["B"]):
                idx = rng.integers(0, len(both), len(both))
                bs.append(stats.spearmanr(both[a].values[idx], both[b].values[idx]).statistic / den)
            rows.append(dict(model_a=a, model_b=b, shared_tasks=len(both), r_obs=r_obs, rel_a=rel[a], rel_b=rel[b],
                             r_disattenuated=r_dis, lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5)))
            M.loc[a, b] = M.loc[b, a] = r_dis
    for m in models:
        M.loc[m, m] = 1.0
    tr = pd.DataFrame(rows)
    tr.to_csv("results/tables/e2_spread_transfer.csv", index=False)
    L.append("## Spread transfer (Spearman of log s across shared tasks, disattenuated by split-half reliability)\n")
    L.append("Split-half reliabilities (Spearman–Brown): " + ", ".join(f"{k} {v:.3f}" for k, v in rel.items()) + "\n")
    L.append((tr.round(3).to_markdown(index=False) if len(tr) else "No model pair has ≥ 50 shared tasks.") + "\n")
    fig, ax = plt.subplots(figsize=(4.8, 4))
    Mv = M.loc[[m for m in models if M.loc[m].notna().sum() > 1], [m for m in models if M.loc[m].notna().sum() > 1]]
    im = ax.imshow(Mv.values.astype(float), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(Mv)), Mv.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(Mv)), Mv.index)
    for i in range(len(Mv)):
        for j in range(len(Mv)):
            v = Mv.values[i, j]
            if v == v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="disattenuated Spearman r of log within-cell SD")
    ax.set_title("Spread transfer between models")
    fig.tight_layout()
    fig.savefig("results/figures/fig_e2_spread_transfer.png")
    plt.close(fig)

    # 4. predictability of spread from text (grouped CV, held-out tasks)
    num = base_numeric(runs, cfg)
    cell_rows = runs[cn >= 4].groupby("cell_id").head(1).index.values
    cs2 = runs.loc[cell_rows, ["cell_id", "run_id"]].merge(cs[["cell_id", "log_s"]], on="cell_id")
    fold = splits.set_index("run_id").loc[cs2.run_id, "fold"].values
    preds = np.full(len(cs2), np.nan)
    for f in range(cfg["cv"]["n_splits"]):
        trm = splits.fold.values != f
        Z = svd_features(runs, trm, cfg, f"fold{f}")
        Xall = gbm_frame(runs, num, Z, trm)
        Xc = Xall.loc[cs2.run_id.values].reset_index(drop=True)
        te = fold == f
        trc = np.flatnonzero(~te)
        rng2 = np.random.default_rng(base + f)
        va = rng2.random(len(trc)) < 0.15
        mdl = lgb_train(Xc.iloc[trc[~va]], cs2.log_s.values[trc[~va]], Xc.iloc[trc[va]], cs2.log_s.values[trc[va]],
                        DEFAULT_HP, seed=base)
        preds[te] = mdl.predict(Xc[te])
    y = cs2.log_s.values
    r2 = 1 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)
    # bootstrap over cells
    bs = []
    for _ in range(cfg["bootstrap"]["B"]):
        idx = rng.integers(0, len(y), len(y))
        bs.append(1 - np.sum((y[idx] - preds[idx]) ** 2) / np.sum((y[idx] - y[idx].mean()) ** 2))
    r2t = pd.DataFrame([dict(n_cells=len(y), r2_heldout=r2, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                             spearman=stats.spearmanr(y, preds).statistic)])
    r2t.to_csv("results/tables/e2_spread_from_text.csv", index=False)
    L.append("## Predictability of the within-cell spread from pre-run features (held-out tasks, 5-fold grouped CV)\n")
    L.append(r2t.round(3).to_markdown(index=False) + "\n")
    write_log("e2", L)
    print("\n".join(L))


if __name__ == "__main__":
    main()
