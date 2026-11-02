"""Hyperparameter search on the validation split of fold 1 (index 0), then frozen (spec §7.13)."""
from __future__ import annotations

import time

import numpy as np

from src.eval.metrics import TAUS, qcrps
from src.predictors.base import sample_configs, save_hp, get_hp, lgb_train
from src.predictors.p2_hetero import LOGCHI2_BIAS, choose_c
from src.predictors.p3_quantile import LEVELS
from src.predictors.p5_survival import person_period
from src.utils import seed_for


def _log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def tune_p2(cfg, fit, val, n_configs):
    if get_hp("P2_mean") and get_hp("P2_var"):
        return get_hp("P2_mean"), get_hp("P2_var")
    space = cfg["hyperparams"]["search_space"]
    confs = sample_configs(space, n_configs, seed_for(cfg, "hp_p2_mean"))
    best = None
    for c in confs:
        m = lgb_train(fit.X, fit.y, val.X, val.y, c, seed=0)
        mse = float(np.mean((val.y - m.predict(val.X)) ** 2))
        if best is None or mse < best[0]:
            best = (mse, c, m)
    save_hp("P2_mean", best[1], best[0], len(confs))
    _log(f"P2_mean best val MSE {best[0]:.4f}")
    mean_hp, mean_m = best[1], best[2]
    # variance component: residuals from a 5-fold OOF mean on fit (as in P2.fit)
    from sklearn.model_selection import GroupKFold
    oof = np.zeros(len(fit.y))
    n_r = max(mean_m.best_iteration, 10)
    for tr, te in GroupKFold(5).split(fit.X, fit.y, groups=fit.meta.split_group.values):
        oof[te] = lgb_train(fit.X.iloc[tr], fit.y[tr], None, None, mean_hp, num_rounds=n_r).predict(fit.X.iloc[te])
    r2 = np.log((fit.y - oof) ** 2 + 1e-6)
    mu_v = mean_m.predict(val.X)
    r2v = np.log((val.y - mu_v) ** 2 + 1e-6)
    confs = sample_configs(space, n_configs, seed_for(cfg, "hp_p2_var"))
    best = None
    for c in confs:
        v = lgb_train(fit.X, r2, val.X, r2v, c, seed=0)
        sig = np.sqrt(np.exp(v.predict(val.X) + LOGCHI2_BIAS))
        cc, s = choose_c(mu_v, sig, val.y)
        if best is None or s < best[0]:
            best = (s, c)
    save_hp("P2_var", best[1], best[0], len(confs))
    _log(f"P2_var best val CRPS {best[0]:.4f}")
    return mean_hp, best[1]


def tune_p3(cfg, fit, val, n_configs, name="P3"):
    if get_hp(name):
        return get_hp(name)
    space = cfg["hyperparams"]["search_space"]
    confs = sample_configs(space, n_configs, seed_for(cfg, "hp_" + name))
    best = None
    for c in confs:
        losses = []
        for a in LEVELS:
            m = lgb_train(fit.X, fit.y, val.X, val.y, c, objective="quantile", alpha=float(a), seed=0)
            u = val.y - m.predict(val.X)
            losses.append(np.mean(u * (a - (u < 0))))
        s = float(np.mean(losses))
        if best is None or s < best[0]:
            best = (s, c)
        _log(f"{name} config {c} pinball {s:.4f}")
    save_hp(name, best[1], best[0], len(confs))
    return best[1]


def tune_p5(cfg, fit, val, n_configs):
    if get_hp("P5_haz") and get_hp("P5_cost"):
        return get_hp("P5_haz"), get_hp("P5_cost")
    space = cfg["hyperparams"]["search_space"]
    rng = np.random.default_rng(seed_for(cfg, "hp_p5"))
    Xpp, ev, _ = person_period(fit.X, fit.meta.K.values, fit.meta.K_cap.values, cfg["p5"]["max_person_period_rows"], rng)
    Xppv, evv, _ = person_period(val.X, val.meta.K.values, val.meta.K_cap.values, 500_000, rng)
    best = None
    for c in sample_configs(space, n_configs, seed_for(cfg, "hp_p5_haz")):
        m = lgb_train(Xpp, ev, Xppv, evv, c, objective="binary", seed=0)
        p = np.clip(m.predict(Xppv), 1e-7, 1 - 1e-7)
        ll = float(-np.mean(evv * np.log(p) + (1 - evv) * np.log(1 - p)))
        if best is None or ll < best[0]:
            best = (ll, c)
    save_hp("P5_haz", best[1], best[0], n_configs)
    _log(f"P5_haz best logloss {best[0]:.4f}")
    Xk, Xkv = fit.X.copy(), val.X.copy()
    for X, K in ((Xk, fit.meta.K.values), (Xkv, val.meta.K.values)):
        X["K"] = K
        X["log_K"] = np.log(K)
    bestc = None
    for c in sample_configs(space, n_configs, seed_for(cfg, "hp_p5_cost")):
        m = lgb_train(Xk, fit.y, Xkv, val.y, c, seed=0)
        mse = float(np.mean((val.y - m.predict(Xkv)) ** 2))
        if bestc is None or mse < bestc[0]:
            bestc = (mse, c)
    save_hp("P5_cost", bestc[1], bestc[0], n_configs)
    _log(f"P5_cost best MSE {bestc[0]:.4f}")
    return best[1], bestc[1]
