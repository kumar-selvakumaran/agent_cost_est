"""E3 — cold pre-run benchmark (S1; tests H2, H3). Spec §10 E3.

Stage 1 (`fit`): per fold, fit every predictor on the fold's fit/val roles and store quantile predictions
for the val, cal and test roles in data_cache/preds/.
Stage 2 (`eval`): pool out-of-fold test predictions, apply CQR, compute the full evaluation suite.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from src.eval.metrics import QCOLS, qcrps
from src.experiments.common import load_data
from src.experiments.folds import FoldData
from src.experiments.tuning import tune_p2, tune_p3, tune_p5
from src.predictors.p0_global import P0Global
from src.predictors.p1_cell import P1SourceModel
from src.predictors.p2_hetero import P2Hetero
from src.predictors.p3_quantile import P3Quantile, P3EVT
from src.predictors.p4_mdn import P4MDN
from src.predictors.p5_survival import P5Structural
from src.predictors.p6_linear import P6Linear
from src.utils import seed_for

PRED_DIR = "data_cache/preds"
SEEDED = {"P4": 3, "P4-K1": 3, "P4-K2": 3, "P5": 3}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def save_pred(exp, method, fold, seed, roles: dict):
    os.makedirs(PRED_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(PRED_DIR, f"{exp}__{method}__f{fold}__s{seed}.npz"),
                        **{k: v.astype(np.float32) for k, v in roles.items()})


def pred_path(exp, method, fold, seed):
    return os.path.join(PRED_DIR, f"{exp}__{method}__f{fold}__s{seed}.npz")


def have(exp, method, fold, seed=0):
    return os.path.exists(pred_path(exp, method, fold, seed))


def load_pred(exp, method, fold, seed=0):
    return dict(np.load(pred_path(exp, method, fold, seed)))


def predict_roles(model, bundles: dict) -> dict:
    out = {}
    for r, b in bundles.items():
        out[r] = model.predict_quantiles(b)
        out[r + "_run_id"] = b.meta.run_id.values.astype(np.float32)
    return out


def fit_fold(cfg, runs, splits, f, exp="e3", methods=None, feature_set="full", target="y", group_col="fold",
             svd_tag=None, n_seeds=None, extra=None):
    fd = FoldData(cfg, runs, splits, f, feature_set, target, group_col, svd_tag, extra)
    fit, val, cal, test = (fd.bundle(r) for r in ["fit", "val", "cal", "test"])
    fitval = fd.bundle(["fit", "val"])
    B = dict(val=val, cal=cal, test=test)
    n_cfg = cfg["hyperparams"]["n_configs"]
    methods = methods or ["P0", "P1", "P2", "P3", "P3-EVT", "P4", "P4-K1", "P4-K2", "P5", "P6"]
    seeds_for = lambda m: (n_seeds if n_seeds is not None else SEEDED.get(m, 1))
    log(f"[{exp}] fold {f}: fit={len(fit)} val={len(val)} cal={len(cal)} test={len(test)}")
    if "P0" in methods and not have(exp, "P0", f):
        save_pred(exp, "P0", f, 0, predict_roles(P0Global().fit(fitval), B))
    if "P1" in methods and not have(exp, "P1", f):
        save_pred(exp, "P1", f, 0, predict_roles(P1SourceModel(cfg["p1_shrinkage_n0"]).fit(fitval), B))
    if "P2" in methods and not have(exp, "P2", f):
        hm, hv = tune_p2(cfg, fit, val, n_cfg) if exp == "e3" or True else (None, None)
        m = P2Hetero(hm, hv, seed=seed_for(cfg, "P2")).fit(fit, val)
        save_pred(exp, "P2", f, 0, predict_roles(m, B))
        log(f"P2 c={m.c:.2f}")
    if ("P3" in methods or "P3-EVT" in methods) and not (have(exp, "P3", f) and have(exp, "P3-EVT", f)):
        hp = tune_p3(cfg, fit, val, cfg["hyperparams"].get("n_configs_p3", n_cfg))
        p3 = P3Quantile(hp, seed=seed_for(cfg, "P3")).fit(fit, val)
        save_pred(exp, "P3", f, 0, predict_roles(p3, B))
        evt = P3EVT(p3, cfg["evt"]["threshold_q"], cfg["evt"]["min_exceedances"]).fit(fit, val)
        save_pred(exp, "P3-EVT", f, 0, predict_roles(evt, B))
        with open(os.path.join(PRED_DIR, f"{exp}__evt_params.csv"), "a") as fh:
            fh.write(f"{f},{evt.fallback},{evt.n_exc},{getattr(evt, 'xi', np.nan)},{getattr(evt, 'beta', np.nan)},"
                     f"{evt.u},{evt.zeta}\n")
        log(f"P3-EVT fallback={evt.fallback} n_exc={evt.n_exc} xi={getattr(evt, 'xi', None)}")
    mc = cfg["mdn"]
    for name, K in [("P4", 3), ("P4-K1", 1), ("P4-K2", 2)]:
        if name not in methods:
            continue
        for s in range(seeds_for(name)):
            if have(exp, name, f, s):
                continue
            m = P4MDN(K, tuple(mc["hidden"]), mc["dropout"], mc["lr"], mc["weight_decay"], mc["batch"],
                      mc["max_epochs"], mc["patience"], seed=seed_for(cfg, f"{name}_s{s}")).fit(fit, val)
            roles = predict_roles(m, B)
            if K == 3:
                w, mu, sd = m.mixture(test)
                roles["test_w"], roles["test_mu"], roles["test_sd"] = w, mu, sd
            save_pred(exp, name, f, s, roles)
            log(f"{name} seed {s} epochs={m.epochs} val_nll={m.val_nll:.4f}")
    if "P5" in methods:
        hh, hc = tune_p5(cfg, fit, val, n_cfg)
        for s in range(seeds_for("P5")):
            if have(exp, "P5", f, s):
                continue
            m = P5Structural(hh, hc, seed=seed_for(cfg, f"P5_s{s}"), M=cfg["p5"]["mc_draws"],
                             max_pp=cfg["p5"]["max_person_period_rows"]).fit(fit, val)
            save_pred(exp, "P5", f, s, predict_roles(m, B))
            log(f"P5 seed {s} done (jmax={m.jmax})")
    if "P6" in methods and not have(exp, "P6", f):
        m = P6Linear(tuple(cfg["p6"]["ridge_alphas"])).fit(fit, val)
        save_pred(exp, "P6", f, 0, predict_roles(m, B))
        log(f"P6 alpha_mean={m.alpha_mean} alpha_var={m.alpha_var} c={m.c:.2f}")


def assemble(exp, method, runs, n_folds=5, seed=0, role="test") -> pd.DataFrame:
    """Pool one method's predictions for a role across folds -> DataFrame (run_id, fold, q01..q99, y)."""
    parts = []
    for f in range(n_folds):
        d = load_pred(exp, method, f, seed)
        rid = d[role + "_run_id"].astype(np.int64)
        df = pd.DataFrame(d[role].astype(float), columns=QCOLS)
        df.insert(0, "run_id", rid)
        df.insert(1, "fold", f)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    meta = runs.set_index("run_id").loc[df.run_id, ["task_id", "model_id"]].reset_index(drop=True)
    df.insert(1, "task_id", meta.task_id.values)
    df.insert(2, "model_id", meta.model_id.values)
    df.insert(4, "method", method)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--methods", default=None)
    a = ap.parse_args()
    cfg, runs, splits = load_data()
    methods = a.methods.split(",") if a.methods else None
    for f in [int(x) for x in a.folds.split(",")]:
        fit_fold(cfg, runs, splits, f, methods=methods)


if __name__ == "__main__":
    main()
