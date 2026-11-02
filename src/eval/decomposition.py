"""Variance decomposition helpers (spec §10 E2)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def fit_level_mixed(df: pd.DataFrame) -> dict:
    """y ~ C(model) + (1 | task) + (1 | cell within task), REML. Returns variance shares."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = smf.mixedlm("y ~ C(model_id)", df, groups=df["task_id"], re_formula="1",
                         vc_formula={"cell": "0 + C(cell_id)"})
        r = md.fit(reml=True, method=["lbfgs"])
    fe = r.fe_params
    X = md.exog
    v_model = float(np.var(X @ fe.values))
    v_task = float(r.cov_re.iloc[0, 0])
    v_cell = float(r.vcomp[0])
    v_run = float(r.scale)
    tot = v_model + v_task + v_cell + v_run
    return dict(model=v_model / tot, task=v_task / tot, cell=v_cell / tot, run=v_run / tot,
                v_model=v_model, v_task=v_task, v_cell=v_cell, v_run=v_run, total=tot, converged=bool(r.converged))


def fit_spread_mixed(cells: pd.DataFrame) -> dict:
    """log s_tm = nu + beta'_m + a'_t + e ; noise-corrected shares."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = smf.mixedlm("log_s ~ C(model_id)", cells, groups=cells["task_id"], re_formula="1")
        r = md.fit(reml=True, method=["lbfgs"])
    v_model = float(np.var(md.exog @ r.fe_params.values))
    v_task = float(r.cov_re.iloc[0, 0])
    v_res = float(r.scale)
    noise = float(np.mean(1.0 / (2.0 * (cells.n.values - 1))))
    v_res_c = max(v_res - noise, 0.0)
    tot_c = v_model + v_task + v_res_c
    tot_raw = v_model + v_task + v_res
    return dict(model=v_model / tot_c, task=v_task / tot_c, residual=v_res_c / tot_c,
                model_raw=v_model / tot_raw, task_raw=v_task / tot_raw, residual_raw=v_res / tot_raw,
                v_model=v_model, v_task=v_task, v_res=v_res, noise=noise, converged=bool(r.converged))
