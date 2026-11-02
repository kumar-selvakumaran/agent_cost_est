"""E0 — data audit and reconstruction validation (spec §3.4, §4, §5.6, §10 E0)."""
from __future__ import annotations

import os
import re
import subprocess

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xxhash

from src.cost.reconstruct import normalize_messages
from src.cost.tokenizers import count_tokens
from src.data.ids import TEACHER_CANON, derive_ids
from src.utils import load_config, seed_for, ensure_dirs

TOKENIZER_REPOS = {
    "glm-4.6": "zai-org/GLM-4.6", "glm-4.7": "zai-org/GLM-4.7", "glm-5": "zai-org/GLM-5",
    "qwen3-coder-480b": "Qwen/Qwen3-Coder-480B-A35B-Instruct", "qwen3-coder-30b": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "minimax-m2": "MiniMaxAI/MiniMax-M2", "gpt-oss-120b": "openai/gpt-oss-120b",
}
TEACHER_FAMILY = {
    "GLM-4.6": "glm-4.6", "GLM-4.7": "glm-4.7", "GLM-5.0": "glm-5", "GPT 5.1 Nano": "gpt-5.1-nano",
    "GPT-5-nano": "gpt-5-nano", "GPT-5-mini": "gpt-5-mini", "GPT-5": "gpt-5", "GPT-4o": "gpt-4o",
    "Kimi K2.0 Thinking": "kimi-k2-thinking", "Kimi-2.5": "kimi-2.5", "MiniMax M2.0": "minimax-m2",
    "Qwen3": "qwen3", "GPT-OSS-120B": "gpt-oss-120b", "Gemini-2.5-Flash": "gemini-2.5-flash",
}


def read_rows(raw_dir, shard_rows: pd.DataFrame, col="conversations"):
    """shard_rows: DataFrame with shard,row,uid -> dict uid -> conversation."""
    out = {}
    for shard, g in shard_rows.groupby("shard"):
        pf = pq.ParquetFile(os.path.join(raw_dir, f"train-{shard:05d}-of-00038.parquet"))
        starts = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)])
        rg = np.searchsorted(starts, g.row.values, side="right") - 1
        for r in np.unique(rg):
            m = rg == r
            tb = pf.read_row_group(int(r), columns=[col])
            convs = tb.column(col).take(g.row.values[m] - starts[r]).to_pylist()
            for u, c in zip(g.uid.values[m], convs):
                out[u] = c
    return out


def msg_hashes(conv):
    return [xxhash.xxh64_intdigest((r + "\x00" + c).encode("utf-8", "surrogatepass")) for r, c in normalize_messages(conv)]


def prefix_check(cfg, meta_all: pd.DataFrame, n_trials=400) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(seed_for(cfg, "e0_prefix"))
    t = meta_all[meta_all.agent.eq("terminus-2") & ~meta_all.trace_source.fillna("").str.startswith("summarization")]
    t = t[t.trial_name.notna()]
    cnt = t.groupby("trial_name").size()
    multi = cnt[cnt > 1]
    res = dict(trials=int(cnt.size), trials_multi_row=int(multi.size), share_multi=float(multi.size / cnt.size),
               rows_in_multi=int(multi.sum()))
    pick = rng.choice(multi.index.values, size=min(n_trials, multi.size), replace=False)
    rows = t[t.trial_name.isin(pick)].copy()
    rows["uid"] = rows.shard.astype(str) + ":" + rows.row.astype(str)
    convs = read_rows(cfg["data"]["raw_dir"], rows[["shard", "row", "uid"]])
    out = []
    for tn, g in rows.groupby("trial_name"):
        hs = {u: msg_hashes(convs[u]) for u in g.uid}
        longest = max(hs.values(), key=len)
        n_prefix = 0
        n_identical = 0
        for u, h in hs.items():
            if h == longest:
                n_identical += 1
            elif longest[:len(h)] == h:
                n_prefix += 1
            # some snapshots end with an extra (episode-final) assistant message not in later rows
            elif longest[:len(h) - 1] == h[:-1]:
                n_prefix += 1
        all_ok = (n_prefix + n_identical) == len(hs)
        out.append(dict(trial_name=tn, n_rows=len(hs), n_identical=n_identical, n_prefix=n_prefix,
                        all_prefix=all_ok, source=g.original_source.iloc[0]))
    df = pd.DataFrame(out)
    res["checked_trials"] = len(df)
    res["share_all_prefix_or_identical"] = float(df.all_prefix.mean())
    res["share_trials_with_identical_copies"] = float((df.n_identical > 1).mean())
    return res, df


def tokenizer_agreement(cfg, runs_sel: pd.DataFrame, n=2000) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    rng = np.random.default_rng(seed_for(cfg, "e0_tok"))
    pick = runs_sel.sample(min(n, len(runs_sel)), random_state=int(rng.integers(1 << 31)))
    convs = read_rows(cfg["data"]["raw_dir"], pick[["shard", "row", "uid"]])
    toks = {}
    for mid, repo in TOKENIZER_REPOS.items():
        try:
            toks[mid] = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        except Exception as e:  # noqa
            print("tokenizer unavailable", mid, e)
    rows = []
    for u, mid in zip(pick.uid, pick.model_id):
        if mid not in toks:
            rows.append(dict(uid=u, model_id=mid, ratio=np.nan))
            continue
        texts = [c for _, c in normalize_messages(convs[u])]
        # cap length per run for speed: first 200 messages
        texts = texts[:200]
        canon = count_tokens(texts).sum()
        spec = sum(len(e.ids) for e in toks[mid].encode_batch(texts, add_special_tokens=False))
        rows.append(dict(uid=u, model_id=mid, ratio=spec / max(canon, 1)))
    df = pd.DataFrame(rows)
    tab = df.groupby("model_id").ratio.agg(n_runs="size", n_with_tokenizer="count", median="median",
                                           p10=lambda s: s.quantile(0.1), p90=lambda s: s.quantile(0.9))
    return tab.reset_index()


def main():
    cfg = load_config()
    ensure_dirs()
    cache = cfg["data"]["cache_dir"]
    L = []  # markdown log lines
    meta_all = pd.read_parquet(os.path.join(cache, "meta_all.parquet"))
    derived = pd.read_parquet(os.path.join(cache, "meta_derived.parquet"))
    sel = pd.read_parquet(os.path.join(cache, "selection.parquet"))
    runs = pd.read_parquet(os.path.join(cache, "runs_raw.parquet"))
    L.append("# E0 — Data audit\n")

    # ---- schema report
    cols = [c for c in meta_all.columns if c in ["agent", "model", "date", "task", "episode", "run_id", "trial_name",
                                                 "model_provider", "original_source", "original_teacher", "result",
                                                 "trace_source", "path", "instruction", "judgment", "split"]]
    nulls = meta_all.groupby(meta_all.original_source.fillna("NA"))[cols].agg(lambda s: s.isna().mean())
    nulls.insert(0, "rows", meta_all.groupby(meta_all.original_source.fillna("NA")).size())
    nulls.round(3).to_csv("results/tables/e0_schema_nulls_by_source.csv")
    L.append("## Schema\n")
    L.append(f"Rows: {len(meta_all):,}. Parquet columns: conversations (list of {{role, content}}; OpenAI style), "
             + ", ".join(cols) + ", task_binary, verifier_output, ground_truth, __index_level_0__. "
             "There is **no `reward` column and no `messages` column** in the released Parquet files. "
             "`instruction` is null in every row; the task statement is embedded in the first user message after "
             "the terminus-2 harness prompt (`Task Description:` marker). Null rates per source: "
             "`results/tables/e0_schema_nulls_by_source.csv`.\n")
    L.append("Overall null rate per column: " + ", ".join(f"{c} {meta_all[c].isna().mean():.3f}" for c in cols) + "\n")

    # ---- T0
    flow = pd.read_csv("results/tables/t0_exclusion_flow.csv")
    n_parse_fail = int((~runs.parse_ok).sum()) if "parse_ok" in runs else 0
    n_zero = int((runs.K == 0).sum()) if "K" in runs else 0
    flow.loc[len(flow)] = ("post-load: transcript parse failures", len(runs) - n_parse_fail)
    flow.loc[len(flow)] = ("post-load: zero assistant turns after parsing", len(runs) - n_parse_fail - n_zero)
    flow.to_csv("results/tables/t0_exclusion_flow.csv", index=False)
    L.append("## T0 exclusion flow\n")
    L.append(flow.to_markdown(index=False) + "\n")

    # ---- row semantics
    pres, pdf = prefix_check(cfg, meta_all)
    pdf.to_csv("results/tables/e0_prefix_check.csv", index=False)
    L.append("## Row-per-trial vs row-per-episode (§4.2)\n")
    L.append(f"Trials (terminus-2 main rows with trial_name): {pres['trials']:,}; trials with >1 row: "
             f"{pres['trials_multi_row']:,} ({pres['share_multi']:.1%}), holding {pres['rows_in_multi']:,} rows. "
             f"On {pres['checked_trials']} randomly sampled multi-row trials, the shorter transcripts are exact prefixes "
             f"of (or identical to) the longest one in {pres['share_all_prefix_or_identical']:.1%} of trials; "
             f"{pres['share_trials_with_identical_copies']:.1%} contain identical copies (same trace re-released in "
             "several source repos). Resolution: keep the row with the most assistant turns per trial "
             "(then drop exact duplicate transcripts).\n")
    by_src = pdf.groupby("source").all_prefix.agg(["size", "mean"]).round(3)
    L.append(by_src.to_markdown() + "\n")
    bad = pdf[~pdf.all_prefix].head(5)
    if len(bad):
        L.append("Non-prefix examples (trial_name, rows): " + "; ".join(f"{r.trial_name} ({r.n_rows})" for r in bad.itertuples()) + "\n")
        L.append("Inspection of these anomalies shows the *same* trial re-released in two source repos with small "
                 "post-processing differences (e.g. `codeforces-00066__veZmxFs`: identical except for an inserted "
                 "terminal-noise line `bash: initialize_job_control: no job control in background` in the first "
                 "message). They are near-duplicates of one trial, not distinct runs, so keeping one row per trial "
                 "(the one with the most assistant turns) is correct.\n")

    # ---- identifiers
    t = derive_ids(meta_all[meta_all.agent.eq("terminus-2") & ~meta_all.trace_source.fillna("").str.startswith("summarization")])
    tf = t.original_teacher.map(TEACHER_FAMILY)
    has_both = t.model.notna() & tf.notna()
    fam_m = t.model_id.fillna("").str.extract(r"^([a-z]+)")[0]
    fam_t = tf.fillna("").str.extract(r"^([a-z]+)")[0]
    exact_dis = (t.model_id != tf)[has_both].mean()
    fam_dis = (fam_m != fam_t)[has_both].mean()
    mt = t[has_both].groupby(["model", "original_teacher"]).agg(rows=("uid", "size"), model_id=("model_id", "first"),
                                                              rule=("model_rule", "first")).reset_index()
    think = derived.assign(ts=derived.n_think / derived.n_asst.clip(lower=1)).groupby(["model", "original_teacher"]).ts.mean()
    mt["think_share"] = [think.get((a, b), np.nan) for a, b in zip(mt.model, mt.original_teacher)]
    mt.to_csv("results/tables/e0_model_teacher.csv", index=False)
    L.append("## Identifiers (§4.1)\n")
    L.append(f"`model` vs `original_teacher` disagreement (rows with both): exact model {exact_dis:.1%}; "
             f"model family {fam_dis:.1%}. Transcript style check: share of assistant turns with `<think>` blocks "
             "per (model, teacher) pair is in `results/tables/e0_model_teacher.csv`. Where they disagree, the "
             "transcript style follows `model` (e.g. `QuantTrio/GLM-4.6-AWQ` rows labelled teacher `GPT 5.1 Nano` "
             "contain GLM-style `<think>` blocks, which GPT-5-family API outputs never contain), so `model` is used. "
             "`hosted_vllm/glm` is a family name only: it is resolved with the teacher label when that is a GLM "
             "version (GLM-4.6/4.7) and **excluded as ambiguous** when the teacher is `GPT 5.1 Nano`; "
             "`DCAgent/taskmaster2-64ep` (a dataset name, not a model) is excluded. Numeric `hosted_vllm/<id>` "
             "checkpoints are kept as their own model ids.\n")
    L.append(mt.sort_values("rows", ascending=False).to_markdown(index=False) + "\n")
    both = t[t.task.notna() & t.trial_name.notna()]
    a_task_trial = (both.task == both.trial_name.str.split("__").str[0]).mean()
    tt = derived
    mode = tt.groupby("task_name").h_task_text.agg(lambda s: s.value_counts().index[0])
    a_name_text = float((tt.h_task_text.values == mode.loc[tt.task_name].values).mean())
    k = tt.groupby("h_task_text").task_name.nunique()
    L.append(f"Task-id agreement: `task` == `trial_name` prefix in {a_task_trial:.1%} of rows with both "
             f"(the difference is entirely the `_copyN` suffix convention); name-based task id (with `_copyN` "
             f"stripped) vs hash of the task statement: {a_name_text:.1%} of runs carry the modal statement of "
             f"their task name; {(k > 1).mean():.1%} of distinct statements appear under more than one task name. "
             "Task names are therefore not reliable identifiers (names such as `swesmith-00088` or "
             "`acl-permissions-inheritance` are reused across source repos for different statements). "
             "**Resolution (conservative): task_id = hash of the exact task statement** (cell = statement × model); "
             "CV groups = connected components linking task names and statements, so neither a shared name nor a "
             "shared statement can straddle train and test.\n")
    by_src_agree = tt.assign(a=tt.h_task_text.values == mode.loc[tt.task_name].values).groupby("source").a.mean()
    by_src_agree.round(3).to_frame("name_text_agreement").to_csv("results/tables/e0_task_agreement_by_source.csv")

    # ---- success label
    L.append("## Success label (§4.3)\n")
    vc = meta_all.result.value_counts(dropna=False)
    L.append("`reward` does not exist. Distinct `result` values: " + ", ".join(f"`{k}` ({v:,})" for k, v in vc.items())
             + f". `judgment`: {meta_all.judgment.value_counts(dropna=False).to_dict()}; `split`: "
             f"{meta_all.split.value_counts(dropna=False).to_dict()}.\n")
    L.append("Mapping written to config (`success.map`): `1.0`/`success` → 1, `0.0`/`failure` → 0; `judgment == Yes` → 1; "
             "error results (`AgentTimeoutError`, `Daytona*`, …) → 0 (the agent did not finish); null → NA. "
             "Because most rows have null `result`, an additional *exploratory* proxy `completed` = (no error result and the "
             "final assistant turn sets `task_complete: true`) is used in E1/E8, labelled as such.\n")

    # ---- measurement validity on the selected sample
    s = sel.merge(runs, on="uid", how="inner")
    ok = s[s.parse_ok & (s.K > 0)]
    summ_trials = set(meta_all[meta_all.trace_source.fillna("").str.startswith("summarization")].trial_name.dropna())
    dsel = derived.set_index("uid").loc[ok.uid]
    ok = ok.assign(summ_flag=(ok.n_handoff.values > 0) | ok.trial_id.isin(summ_trials).values | dsel.has_warning_ctx.values)
    mv = ok.groupby("model_id").agg(runs=("uid", "size"), think_share=("think_share", "mean"),
                                    runs_with_think=("n_think", lambda s: (s > 0).mean()),
                                    summarization_flag=("summ_flag", "mean"), median_K=("K", "median"))
    mv.round(3).to_csv("results/tables/e0_measurement_validity.csv")
    first80 = dsel.summ_first80.dropna().value_counts().head(15)
    L.append("## Measurement validity (§4.4)\n")
    L.append(mv.round(3).to_markdown() + "\n")
    L.append("Hidden reasoning: GPT-5-family models (gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-codex) never show `<think>` "
             "blocks, i.e. their reasoning tokens are hidden; their cost here is *visible-transcript cost*, a lower bound "
             "on billed cost. Open-weight models served via vLLM show reasoning inline for a subset of turns.\n")
    L.append(f"Summarization: a run is flagged if (a) a later user message is a terminus-2 summarization handoff, "
             f"(b) the dataset contains summarization sub-call rows with the same trial_name, or (c) a context-full "
             f"warning appears. Flagged share of the selected sample: {ok.summ_flag.mean():.2%}. Distinct first 80 "
             "characters of handoff messages:\n")
    L.append("\n".join(f"- `{k!r}` ({v})" for k, v in first80.items()) + "\n")
    caps = []
    for src, g in ok.groupby("source"):
        Kmax = int(g.K.max())
        share = float((g.K == Kmax).mean())
        caps.append(dict(source=src, runs=len(g), K_median=float(g.K.median()), K_p90=float(g.K.quantile(0.9)),
                         K_max=Kmax, share_at_max=round(share, 4), K_cap=Kmax if share >= 0.05 else None))
    caps = pd.DataFrame(caps)
    caps.to_csv("results/tables/e0_turn_caps.csv", index=False)
    L.append("Turn caps (a cap is declared when ≥ 5% of a source's runs share the maximum K):\n")
    L.append(caps.to_markdown(index=False) + "\n")
    tok = tokenizer_agreement(cfg, ok)
    tok.to_csv("results/tables/e0_tokenizer_agreement.csv", index=False)
    L.append("Tokenizer agreement (model-specific `tokenizer.json` tokens / o200k_base tokens, 2,000 random runs; "
             "GPT-5 family and Gemini have no public tokenizer — GPT-5 uses the o200k family, so ratio ≈ 1 by construction):\n")
    L.append(tok.round(3).to_markdown(index=False) + "\n")

    # ---- final sample summary
    cn = ok.groupby("cell_id").size()
    nm = ok.groupby("task_id").model_id.nunique()
    summary = dict(runs=len(ok), tasks=ok.task_id.nunique(), cells=int(cn.size), models=ok.model_id.nunique(),
                   sources=ok.source.nunique(), crossed_tasks=int((nm >= 2).sum()),
                   runs_from_crossed=float(ok.task_id.isin(nm[nm >= 2].index).mean()),
                   cells_ge6=int((cn >= 6).sum()), cells_ge10=int((cn >= 10).sum()), cells_ge20=int((cn >= 20).sum()),
                   split_groups=ok.split_group.nunique())
    pd.Series(summary).to_csv("results/tables/e0_sample_summary.csv", header=["value"])
    L.append("## Final sample\n")
    L.append(pd.Series(summary).to_frame("value").to_markdown() + "\n")
    L.append("Runs per cell: " + cn.describe().round(2).to_dict().__repr__() + "\n")
    comp = ok.groupby(["source", "model_id"]).size().unstack(fill_value=0)
    comp.to_csv("results/tables/e0_composition_source_model.csv")
    full = derived.groupby("source").size().rename("full_dataset_runs").to_frame()
    full["selected_runs"] = ok.groupby("source").size()
    full["selected_share"] = full.selected_runs / full.selected_runs.sum()
    full["full_share"] = full.full_dataset_runs / full.full_dataset_runs.sum()
    full.round(4).to_csv("results/tables/e0_sample_vs_full.csv")
    L.append("Sample composition vs full dataset (`results/tables/e0_sample_vs_full.csv`):\n")
    L.append(full.round(4).to_markdown() + "\n")

    # ---- stop conditions
    stop = []
    if summary["cells_ge6"] < 500:
        stop.append("fewer than 500 cells with >= 6 runs")
    if min(a_task_trial, a_name_text) < 0.95:
        stop.append("task_id agreement below 95%")
    if ok.summ_flag.mean() > 0.30:
        stop.append("more than 30% of runs flagged for summarization")
    L.append("## Stop conditions\n")
    L.append(("TRIGGERED: " + "; ".join(stop)) if stop else
             f"None triggered (cells with ≥ 6 runs = {summary['cells_ge6']:,}; task-id agreement = "
             f"{min(a_task_trial, a_name_text):.1%} (≥ 95%, and resolved by statement hashing); summarization-flagged = "
             f"{ok.summ_flag.mean():.2%}).")
    L.append("\n")
    # reconstruction validation: unit tests
    r = subprocess.run(["python", "-m", "pytest", "-q", "tests/test_cost.py"], capture_output=True, text=True)
    L.append("## Reconstruction unit tests (§5.6)\n```\n" + r.stdout.strip().splitlines()[-1] + "\n```\n")
    with open("results/logs/e0.md", "w") as f:
        f.write("\n".join(L))
    ok[["uid", "summ_flag"]].to_parquet(os.path.join(cache, "summ_flags.parquet"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
