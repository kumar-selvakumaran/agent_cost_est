"""E9 — trajectory-integrated evaluation of online cost forecasts (extension §3–§7).

Stage `fit` (per fold): forecasts at every decision point d_k (k = 0..K-1) of every test run for
  Static (best S1 = P5 out-of-fold, never updated), O2 and O2-noregime (pooled, §3.2), O1 (per-k base models at grid
  k from E5's stored forecasts, pooled hazard elsewhere, §3.3). At d_0 every method uses the Static forecast.
Stage `eval`: metrics (§4), inference (§4.6), figures (§7.2), tables (§7.3), log.
"""
from __future__ import annotations

import argparse
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.experiments.common import load_data
from src.experiments.e3_cold import load_pred
from src.experiments.folds import FoldData
from src.eval.trajectory import decision_points
from src.features.prefix import prefix_features
from src.predictors.base import get_hp, lgb_train
from src.predictors.o1_structural import o1_quantiles_at_k
from src.predictors.o2_pooled import O2Pooled, cap_k, design, sample_rows, KMAX
from src.predictors.p5_survival import person_period
from src.utils import seed_for

PRED = "data_cache/preds"
GRID = [1, 2, 4, 8, 16, 32]


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def turn_state(turns: pd.DataFrame) -> pd.DataFrame:
    """Observable state after k calls, one row per (run, k >= 1): only turns <= k."""
    t = turns.sort_values(["run_id", "k"])
    g = t.groupby("run_id", sort=False)
    return pd.DataFrame({"run_id": t.run_id.values, "k": t.k.values, "I_k": t.I.values.astype(float),
                         "O_k": t.O.values.astype(float), "obs_k": t.obs.values.astype(float),
                         "I_1": g.I.transform("first").values.astype(float),
                         "O_mean": (g.O.cumsum() / t.k).values, "C_k": t.cum_cost.values})


def fit_fold(cfg, runs, splits, turns, pf_all, f, per_run=2, pooled_haz_k_per_run=2):
    path = os.path.join(PRED, f"e9__f{f}.npz")
    if os.path.exists(path):
        return
    rng = np.random.default_rng(seed_for(cfg, f"e9_f{f}"))
    fd = FoldData(cfg, runs, splits, f)
    Kser = runs.set_index("run_id").K
    pos = pd.Series(np.arange(len(runs)), index=runs.run_id.values)
    role = pd.Series(fd.role, index=runs.run_id.values)
    yv = runs.set_index("run_id").y
    kcap_s = runs.set_index("run_id").K_cap.astype(float)
    pf_all = cap_k(pf_all)
    out = {}
    # ---- test decision points k >= 1
    te_ids = runs.run_id[fd.role == "test"].values
    pf_te = pf_all[pf_all.run_id.isin(te_ids) & (pf_all.k <= pf_all.run_id.map(Kser) - 1)].reset_index(drop=True)
    out["dp_run_id"], out["dp_k"] = pf_te.run_id.values, pf_te.k.values
    # ---- O2 / O2-noregime pooled
    tr = {r: sample_rows(pf_all[pf_all.run_id.isin(runs.run_id[fd.role == r])], Kser, per_run, rng) for r in ["fit", "val"]}
    hp = get_hp("P3")
    for name, nr in [("O2", False), ("O2-noregime", True)]:
        ck = os.path.join(PRED, f"e9__f{f}__{name}.npy")
        if os.path.exists(ck):
            out[name] = np.load(ck)
            continue
        Xtr = design(fd.X, pos.loc[tr["fit"].run_id].values, tr["fit"], nr)
        Xva = design(fd.X, pos.loc[tr["val"].run_id].values, tr["val"], nr)
        m = O2Pooled(hp, seed=seed_for(cfg, "e9_" + name), noregime=nr).fit(
            Xtr, tr["fit"].w.values, yv.loc[tr["fit"].run_id].values, Xva, tr["val"].w.values,
            yv.loc[tr["val"].run_id].values)
        out[name] = m.predict(design(fd.X, pos.loc[pf_te.run_id].values, pf_te, nr))
        np.save(ck, out[name])
        log(f"fold {f} {name}: train rows {len(Xtr)}, test decision points {len(pf_te)}")
    out["n_capped_k"] = np.array([int((pf_te.k > KMAX).sum())])
    # ---- O1: base per-k forecasts at grid k (E5), pooled hazard elsewhere
    o1 = np.full((len(pf_te), 99), np.nan, dtype=np.float32)
    src = np.zeros(len(pf_te), dtype=np.int8)  # 1 = per-k base model, 2 = pooled
    for k in GRID:
        d = np.load(os.path.join(PRED, f"e5__k{k}__f{f}.npz"))
        mp = dict(zip(d["test_run_id"], range(len(d["test_run_id"]))))
        sel = np.flatnonzero(pf_te.k.values == k)
        o1[sel] = d["O1"][[mp[r] for r in pf_te.run_id.values[sel]]]
        src[sel] = 1
    # pooled hazard model: up to `pooled_haz_k_per_run` decision points per training run, person-period rows j > k
    hh = get_hp("P5_haz")
    pp = {}
    for r, cap_rows in [("fit", cfg["p5"]["max_person_period_rows"]), ("val", 500_000)]:
        rows = sample_rows(pf_all[pf_all.run_id.isin(runs.run_id[fd.role == r])], Kser, pooled_haz_k_per_run, rng)
        Xr = design(fd.X, pos.loc[rows.run_id].values, rows)
        Kr = Kser.loc[rows.run_id].values
        Xs, evs = [], []
        # build person-period rows per distinct k (start_after = k), capped in total
        budget = cap_rows
        for k, idx in rows.groupby("k").indices.items():
            Xpp, ev, _ = person_period(Xr.iloc[idx].reset_index(drop=True), Kr[idx],
                                       kcap_s.loc[rows.run_id.values[idx]].values, budget, rng, start_after=int(k))
            Xs.append(Xpp)
            evs.append(ev)
            budget -= len(ev)
            if budget <= 0:
                break
        pp[r] = (pd.concat(Xs, ignore_index=True), np.concatenate(evs))
    haz = lgb_train(pp["fit"][0], pp["fit"][1], pp["val"][0], pp["val"][1], hh, objective="binary",
                    seed=seed_for(cfg, "e9_O1_pooled"))
    log(f"fold {f} O1 pooled hazard: {len(pp['fit'][1])} person-period rows")
    jmax = int(np.ceil(1.5 * runs.K[fd.role == "fit"].max()))
    st = turn_state(turns[turns.run_id.isin(te_ids)]).set_index(["run_id", "k"])
    need = np.flatnonzero(src == 0)
    Xte = design(fd.X, pos.loc[pf_te.run_id].values, pf_te)
    M = cfg["p5"]["mc_draws"]
    for k, posn in pd.Series(need).groupby(pf_te.k.values[need]).indices.items():
        idx = need[np.asarray(posn)]
        rid = pf_te.run_id.values[idx]
        s = st.loc[list(zip(rid, np.full(len(rid), k)))]
        state = SimpleNamespace(**{c: s[c].values for c in ["I_k", "O_k", "obs_k", "I_1", "O_mean", "C_k"]})
        o1[idx] = o1_quantiles_at_k(haz, int(k), Xte.iloc[idx], kcap_s.loc[rid].values, state, jmax, M,
                                    np.random.default_rng(seed_for(cfg, f"e9_o1_{f}_{k}")),
                                    cfg["cost"]["lambda_c"], cfg["cost"]["lambda_o"])
        src[idx] = 2
    out["O1"], out["O1_src"] = o1, src
    log(f"fold {f} O1 done: per-k {int((src == 1).sum())}, pooled {int((src == 2).sum())}")
    np.savez_compressed(path, **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--folds", default="0,1,2,3,4")
    a = ap.parse_args()
    cfg, runs, splits = load_data()
    turns = pd.read_parquet(os.path.join(cfg["data"]["cache_dir"], "turns.parquet"))
    if a.stage in ("all", "fit"):
        pf_all = prefix_features(turns)
        for f in [int(x) for x in a.folds.split(",")]:
            fit_fold(cfg, runs, splits, turns, pf_all, f)
    if a.stage in ("all", "eval"):
        from src.experiments.e9_eval import evaluate
        evaluate(cfg, runs, splits, turns)


if __name__ == "__main__":
    main()
