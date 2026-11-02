"""Variance decomposition helpers (spec §10 E2)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM, VCSpec


def _model_dummies(models: pd.Series) -> tuple[np.ndarray, list]:
    levels = sorted(models.unique())
    X = np.column_stack([np.ones(len(models))] + [(models == m).values.astype(float) for m in levels[1:]])
    return X, levels


def fit_level_mixed(df: pd.DataFrame) -> dict:
    """y ~ model (fixed) + (1 | task) + (1 | cell nested in task) + run noise, REML.

    The cell variance component is passed as per-task indicator matrices (VCSpec) instead of a global
    patsy dummy matrix; the model is identical."""
    df = df.sort_values(["task_id", "cell_id"]).reset_index(drop=True)
    X, _ = _model_dummies(df.model_id)
    groups = df.task_id.values
    names, colnames, mats = [["cell"]], [[]], [[]]
    for t, idx in df.groupby("task_id", sort=False).indices.items():
        cells = np.asarray(df.cell_id.to_numpy(dtype=object))[idx]
        u = pd.unique(cells)
        mats[0].append((cells[:, None] == u[None, :]).astype(float))
        colnames[0].append([str(c) for c in u])
    vcs = VCSpec(names[0], colnames, mats)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = MixedLM(df.y.values, X, groups=groups, exog_re=np.ones((len(df), 1)), exog_vc=vcs)
        r = md.fit(reml=True, method=["lbfgs"])
    v_model = float(np.var(X @ r.fe_params))
    v_task = float(np.asarray(r.cov_re)[0, 0])
    v_cell = float(r.vcomp[0])
    v_run = float(r.scale)
    tot = v_model + v_task + v_cell + v_run
    return dict(model=v_model / tot, task=v_task / tot, cell=v_cell / tot, run=v_run / tot,
                v_model=v_model, v_task=v_task, v_cell=v_cell, v_run=v_run, total=tot, converged=bool(r.converged))


def fit_spread_mixed(cells: pd.DataFrame) -> dict:
    """log s_tm = nu + beta'_m + a'_t + e ; noise-corrected shares."""
    X, _ = _model_dummies(cells.model_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = MixedLM(cells.log_s.values, X, groups=cells.task_id.values, exog_re=np.ones((len(cells), 1)))
        r = md.fit(reml=True, method=["lbfgs"])
    v_model = float(np.var(X @ r.fe_params))
    v_task = float(np.asarray(r.cov_re)[0, 0])
    v_res = float(r.scale)
    noise = float(np.mean(1.0 / (2.0 * (cells.n.values - 1))))
    v_res_c = max(v_res - noise, 0.0)
    tot_c = v_model + v_task + v_res_c
    tot_raw = v_model + v_task + v_res
    return dict(model=v_model / tot_c, task=v_task / tot_c, residual=v_res_c / tot_c,
                model_raw=v_model / tot_raw, task_raw=v_task / tot_raw, residual_raw=v_res / tot_raw,
                v_model=v_model, v_task=v_task, v_res=v_res, noise=noise, converged=bool(r.converged))
