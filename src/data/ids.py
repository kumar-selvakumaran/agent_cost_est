"""Identifier derivation (spec §4.1)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

DATE_SUFFIX = re.compile(r"-20\d\d-\d\d-\d\d$")
COPY_SUFFIX = re.compile(r"_copy\d+$")

# raw `model` string (after lowercasing / prefix stripping) -> canonical model id.
# Generic or non-model strings map to None and are resolved with the teacher (see resolve_model).
MODEL_ALIASES = {
    "glm-4.6-awq": "glm-4.6",
    "glm-4.6": "glm-4.6",
    "glm5-awq": "glm-5",
    "minimax-m2-awq": "minimax-m2",
    "qwen3-coder-480b-a35b-instruct-awq": "qwen3-coder-480b",
    "code_contests-qwen3-coder-480b-traces": "qwen3-coder-480b",
    "qwen3-coder-30b-a3b-instruct": "qwen3-coder-30b",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gpt-5": "gpt-5",
    "gpt-5-codex": "gpt-5-codex",
    "gpt-5-mini": "gpt-5-mini",
    "gpt-5-nano": "gpt-5-nano",
    "kimi-k2-thinking": "kimi-k2-thinking",
    "gpt-oss-120b": "gpt-oss-120b",
    "glm": None,                  # generic: family only, resolved by teacher
    "taskmaster2-64ep": None,     # a dataset name, not a model: ambiguous
}
# teacher label -> canonical id (used only to refine a generic family string)
TEACHER_CANON = {"GLM-4.6": "glm-4.6", "GLM-4.7": "glm-4.7", "GLM-5.0": "glm-5"}
PREFIXES = ("hosted_vllm/", "together_ai/", "quanttrio/", "zai-org/", "openai/", "moonshotai/",
            "qwen/", "dcagent/")


def strip_model(raw: str) -> str:
    s = raw.strip().lower()
    changed = True
    while changed:
        changed = False
        for p in PREFIXES:
            if s.startswith(p):
                s = s[len(p):]
                changed = True
    return DATE_SUFFIX.sub("", s)


def resolve_model(model, teacher) -> tuple[str | None, str]:
    """Return (model_id, rule). model_id None -> drop row (ambiguous / missing)."""
    if isinstance(model, str) and model:
        s = strip_model(model)
        if s.isdigit():
            return f"ckpt-{s}", "checkpoint-id"
        if s in MODEL_ALIASES:
            canon = MODEL_ALIASES[s]
            if canon is not None:
                return canon, "model"
            if s == "glm" and teacher in TEACHER_CANON:
                return TEACHER_CANON[teacher], "glm+teacher"
            return None, "ambiguous-model"
        return s, "model-unmapped"
    if isinstance(teacher, str) and teacher:
        return teacher.lower().replace(" ", "-"), "teacher"
    return None, "missing"


def derive_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pairs = df[["model", "original_teacher"]].drop_duplicates()
    res = {(m, t): resolve_model(m, t) for m, t in pairs.itertuples(index=False)}
    mt = list(zip(df.model, df.original_teacher))
    df["model_id"] = [res[k][0] for k in mt]
    df["model_rule"] = [res[k][1] for k in mt]
    tn_prefix = df.trial_name.str.split("__").str[0]
    df["task_raw"] = df.task.where(df.task.notna(), tn_prefix)
    df["task_rule"] = np.where(df.task.notna(), "task", np.where(tn_prefix.notna(), "trial_name", "hash"))
    df["task_id"] = df.task_raw.str.replace(COPY_SUFFIX, "", regex=True)
    miss = df.task_id.isna()
    df.loc[miss, "task_id"] = "h" + df.loc[miss, "h_task_text"].astype(str)
    df["trial_id"] = df.trial_name.where(df.trial_name.notna(),
                                         df.run_id.where(df.run_id.notna(), df.shard.astype(str) + ":" + df.row.astype(str)))
    df["source"] = df.original_source.fillna("unknown")
    df["uid"] = df.shard.astype(str) + ":" + df.row.astype(str)
    return df
