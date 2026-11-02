"""Common predictor interface and LightGBM helpers (spec §7.1, §7.13)."""
from __future__ import annotations

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.eval.metrics import TAUS, monotone, qcrps

HP_PATH = "results/tables/hyperparams.json"


class Bundle:
    """Feature bundle for one subset of runs."""

    def __init__(self, X: pd.DataFrame, y: np.ndarray, meta: pd.DataFrame, Xd: np.ndarray | None = None):
        self.X, self.y, self.meta, self.Xd = X, np.asarray(y, dtype=float), meta, Xd

    def __len__(self):
        return len(self.y)

    def subset(self, mask):
        mask = np.asarray(mask)
        return Bundle(self.X[mask].reset_index(drop=True) if self.X is not None else None, self.y[mask],
                      self.meta[mask].reset_index(drop=True), None if self.Xd is None else self.Xd[mask])


class Predictor:
    name = "base"

    def fit(self, train: Bundle, val: Bundle):
        raise NotImplementedError

    def predict_quantiles(self, b: Bundle) -> np.ndarray:
        raise NotImplementedError


def empirical_q(y: np.ndarray) -> np.ndarray:
    return np.quantile(y, TAUS, method="linear")  # Hyndman-Fan type 7


BASE_PARAMS = dict(verbose=-1, n_jobs=4, bagging_freq=1, max_bin=255, seed=0)
DEFAULT_HP = dict(num_leaves=31, learning_rate=0.05, min_child_samples=50, feature_fraction=0.8,
                  bagging_fraction=0.9, lambda_l2=1)


def lgb_train(X, y, Xv, yv, hp: dict, objective="regression", alpha=None, seed=0, max_rounds=5000,
              patience=100, num_rounds=None, feval_metric=None):
    params = dict(BASE_PARAMS)
    params.update(hp)
    params["objective"] = objective
    params["seed"] = seed
    if alpha is not None:
        params["alpha"] = alpha
    if objective == "binary":
        params["metric"] = "binary_logloss"
    elif objective == "quantile":
        params["metric"] = "quantile"
    else:
        params["metric"] = "l2"
    dtr = lgb.Dataset(X, y, free_raw_data=False)
    if num_rounds is not None:
        return lgb.train(params, dtr, num_boost_round=int(num_rounds))
    dv = lgb.Dataset(Xv, yv, reference=dtr, free_raw_data=False)
    return lgb.train(params, dtr, num_boost_round=max_rounds, valid_sets=[dv],
                     callbacks=[lgb.early_stopping(patience, verbose=False)])


def sample_configs(space: dict, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = [dict(DEFAULT_HP)]
    while len(out) < n:
        c = {k: v[int(rng.integers(len(v)))] for k, v in space.items()}
        if c not in out:
            out.append(c)
    return out


def load_hp() -> dict:
    if os.path.exists(HP_PATH):
        with open(HP_PATH) as f:
            return json.load(f)
    return {}


def save_hp(component: str, hp: dict, score: float, n_tried: int):
    allhp = load_hp()
    allhp[component] = dict(hp=hp, val_score=score, n_configs=n_tried)
    os.makedirs(os.path.dirname(HP_PATH), exist_ok=True)
    with open(HP_PATH, "w") as f:
        json.dump(allhp, f, indent=1)
    rows = [dict(component=k, val_score=v["val_score"], n_configs=v["n_configs"], **v["hp"]) for k, v in allhp.items()]
    pd.DataFrame(rows).to_csv("results/tables/hyperparams.csv", index=False)


def get_hp(component: str) -> dict | None:
    v = load_hp().get(component)
    return None if v is None else v["hp"]


def finalize(q: np.ndarray) -> np.ndarray:
    return monotone(q)


def crps_mean(q, y):
    return float(qcrps(q, y).mean())
