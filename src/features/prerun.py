"""Pre-run feature matrices for settings S1/S2 (spec §6.1). All learned transforms fit on training folds only."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.features.text import TextSVD, text_stats
from src.utils import seed_for

CAT_COLS = ["model_id", "source", "source_family"]
NUM_BASE = ["log_instr_tokens", "log_sys_tokens"]


def base_numeric(runs: pd.DataFrame, cfg) -> pd.DataFrame:
    cache = os.path.join(cfg["data"]["cache_dir"], "text_stats.parquet")
    if os.path.exists(cache):
        ts = pd.read_parquet(cache)
        if len(ts) == len(runs):
            return ts
    ts = text_stats(runs.task_text, cfg["features"]["keywords"])
    ts.insert(0, "log_instr_tokens", np.log1p(runs.instr_tokens.values))
    ts.insert(1, "log_sys_tokens", np.log1p(runs.sys_tokens.values))
    ts = ts.reset_index(drop=True)
    ts.to_parquet(cache, index=False)
    return ts


def svd_features(runs: pd.DataFrame, train_mask: np.ndarray, cfg, tag: str) -> np.ndarray:
    """SVD-128 of TF-IDF, fit on runs[train_mask] texts; cached per tag."""
    path = os.path.join(cfg["data"]["cache_dir"], f"svd_{tag}.npy")
    if os.path.exists(path):
        Z = np.load(path)
        if len(Z) == len(runs):
            return Z
    fc = cfg["features"]
    m = TextSVD(fc["tfidf_min_df"], fc["tfidf_max_features"], fc["svd_dim"], seed_for(cfg, "svd_" + tag))
    m.fit(runs.task_text[train_mask])
    Z = m.transform(runs.task_text)
    np.save(path, Z)
    return Z


def gbm_frame(runs: pd.DataFrame, num: pd.DataFrame, Z: np.ndarray | None, train_mask: np.ndarray,
              feature_set: str = "full") -> pd.DataFrame:
    """LightGBM frame with native categoricals. feature_set in {'a','b','c','full'} (E3 ablation)."""
    X = pd.DataFrame(index=runs.index)
    for c in CAT_COLS:
        cats = pd.Index(sorted(runs.loc[train_mask, c].unique()))
        X[c] = pd.Categorical(runs[c], categories=cats)
    if feature_set in ("b", "c", "full"):
        cols = [c for c in num.columns if not c.startswith("kw_")]
        for c in cols:
            X[c] = num[c].values
    if feature_set in ("c", "full"):
        for c in num.columns:
            if c.startswith("kw_"):
                X[c] = num[c].values
    if feature_set == "full" and Z is not None:
        S = pd.DataFrame(Z, index=X.index, columns=[f"svd{j}" for j in range(Z.shape[1])])
        X = pd.concat([X, S], axis=1)
    return X


def dense_frame(runs: pd.DataFrame, num: pd.DataFrame, Z: np.ndarray, train_mask: np.ndarray,
                extra: pd.DataFrame | None = None) -> np.ndarray:
    """Standardized numeric + one-hot top-20 models / top-30 sources + SVD (for P4, P6)."""
    tm = runs.model_id[train_mask].value_counts().index[:20]
    ts = runs.source[train_mask].value_counts().index[:30]
    oh_m = np.stack([(runs.model_id == m).values for m in tm], 1).astype(np.float32)
    oh_s = np.stack([(runs.source == s).values for s in ts], 1).astype(np.float32)
    numv = num.values.astype(np.float32)
    if extra is not None:
        numv = np.c_[numv, extra.values.astype(np.float32)]
    mu = np.nanmean(numv[train_mask], 0)
    sd = np.nanstd(numv[train_mask], 0) + 1e-6
    numv = np.nan_to_num((numv - mu) / sd)
    zs = Z.astype(np.float32)
    zmu, zsd = zs[train_mask].mean(0), zs[train_mask].std(0) + 1e-6
    return np.c_[numv, oh_m, oh_s, (zs - zmu) / zsd].astype(np.float32)
