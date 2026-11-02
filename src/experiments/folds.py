"""Per-fold feature bundles shared by E3/E4/E5/E7."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.prerun import base_numeric, svd_features, gbm_frame, dense_frame
from src.predictors.base import Bundle

META_COLS = ["run_id", "task_id", "model_id", "source", "source_family", "split_group", "cell_id", "K", "K_cap",
             "instr_tokens", "summ_flag", "y", "y_tok", "y_hi", "date_ts", "success", "completed"]


class FoldData:
    def __init__(self, cfg, runs: pd.DataFrame, splits: pd.DataFrame, fold: int, feature_set="full",
                 target="y", group_col="fold", svd_tag=None, extra: pd.DataFrame | None = None):
        assert np.array_equal(runs.run_id.values, splits.run_id.values)
        self.cfg, self.runs, self.fold = cfg, runs, fold
        if group_col == "fold":
            role = splits[f"role_{fold}"].values
        else:
            role = splits[group_col].values
        self.role = role
        train_mask = role != "test"
        num = base_numeric(runs, cfg)
        Z = svd_features(runs, train_mask, cfg, svd_tag or f"fold{fold}")
        X = gbm_frame(runs, num, Z, train_mask, feature_set)
        if extra is not None:
            X = pd.concat([X, extra.reset_index(drop=True)], axis=1)
        Xd = dense_frame(runs, num, Z, train_mask, extra)
        self.X, self.Xd = X, Xd
        self.y = runs[target].values
        self.meta = runs[[c for c in META_COLS if c in runs]].reset_index(drop=True)

    def bundle(self, roles) -> Bundle:
        roles = [roles] if isinstance(roles, str) else roles
        m = np.isin(self.role, roles)
        return Bundle(self.X[m].reset_index(drop=True), self.y[m], self.meta[m].reset_index(drop=True), self.Xd[m])


def _bundle_mask(self, m) -> Bundle:
    m = np.asarray(m)
    return Bundle(self.X[m].reset_index(drop=True), self.y[m], self.meta[m].reset_index(drop=True), self.Xd[m])


FoldData.bundle_mask = _bundle_mask
