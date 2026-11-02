# Deviations from the specification

Every departure from `agenttrove_cost_benchmark_spec.md` (v1.0), and every choice the spec left open, with the reason.
The rule applied under ambiguity: choose the most conservative option (least leakage, least flattering), record it here.

## Compute and scale
1. **N_RUNS_MAX = 100,000 (spec: 300,000).** The machine has 4 CPU cores and 15 GB RAM (spec target: 16+ cores,
   64 GB). The per-source cap scales with it: max(5% × 100k, 2,000) = 5,000 runs. Selected: 98,501 runs.
2. **Raw data access.** The 38 Parquet shards were downloaded once (19.5 GB) and scanned locally; Pass 1 reads all
   rows (metadata + cheap transcript statistics such as message counts and hashes), Pass 2 loads transcripts only for
   selected rows, as specified.

## Data semantics (E0)
3. **No `reward` column exists** in the released Parquet files (nor `messages`; transcripts are in `conversations`,
   OpenAI style). The success label follows §4.3 using `result`, `judgment`, `split`: `1.0`/`success`/`judgment=Yes`
   → 1, `0.0`/`failure` → 0, any error result (`AgentTimeoutError`, `Daytona*`, …) → 0, else NA. In the selected
   sample this label exists for 12% of runs and those are essentially all errors/failures, so it is degenerate for E1
   and E8. An **exploratory** proxy `completed` (no error result and the final assistant turn sets
   `task_complete: true`) is reported alongside and labelled as exploratory everywhere it is used.
4. **`instruction` is null in every row.** The task statement is the text after the `Task Description:` marker in the
   first user message (terminus-2 harness prompt before it). This is the "instruction text" for features (§6.1); the
   harness prompt portion is the "system prompt" length (there are no `system` messages).
5. **Summarization sub-call rows** (`trace_source` = `summarization-*`, 46,132 rows) are separate model calls of the
   summarizer, not trials; they are excluded from runs. Their cost is not attributed to the parent trial (they cannot
   be linked reliably); parent trials that have such rows are flagged as summarized (§4.4).
6. **Model identity.** `model` disagrees with `original_teacher` on 45% of rows; transcript style follows `model`, so
   `model` is used (spec rule). `hosted_vllm/glm` is a family name only: resolved with the teacher label when that is
   a GLM version (GLM-4.6/4.7); rows with teacher `GPT 5.1 Nano` are **excluded as ambiguous** (22,862 rows), as is
   `DCAgent/taskmaster2-64ep` (a dataset name). Quantized/served variants are merged with their base model
   (`QuantTrio/GLM-4.6-AWQ`, `hosted_vllm/QuantTrio/GLM-4.6-AWQ`, `together_ai/zai-org/GLM-4.6` → `glm-4.6`). Numeric
   `hosted_vllm/<id>` checkpoints are kept as separate model ids (none was selected: their cells have < 2 runs).
7. **Task identity = hash of the exact task statement**, not the `task` name. Names are reused across source repos
   for different statements (name ↔ statement agreement 95.2%; 14.4% of statements appear under several names) and
   `_copyN` suffixes mark repeated rollouts of the same statement. The cell is (statement, model). CV groups are the
   connected components of the bipartite graph linking task names and statements, so neither a shared name nor a
   shared statement can straddle train/test (more conservative than either identifier alone).
8. **Row-per-episode duplicates.** 16% of trials have several rows (episode snapshots and re-releases of the same
   trial in several repos). One row per trial is kept (most assistant turns), then exact duplicate transcripts are
   dropped. ~11% of multi-row trials are not exact prefixes; inspection shows the same trial re-released with small
   text differences (e.g. an inserted terminal-noise line), i.e. near-duplicates, so keeping one row is correct.
9. **Exclusion order.** The metadata-decidable exclusions of §3.4 (empty/unparseable, zero assistant turns, duplicate
   trials, >2M-character messages) are applied *before* cell selection, so that duplicates cannot inflate a cell's run
   count; parse failures after loading are checked again (none occurred). T0 records both.
10. **Crossed-task quota.** Cells of tasks run by ≥ 2 models are selected first, prioritising tasks where ≥ 2 models have
    ≥ 4 runs (E2 needs them), capped at 50% of N_RUNS_MAX; crossed cells are exempt from the per-source cap (they
    are concentrated in two sources). Result: 51% of runs come from crossed tasks (target ≥ 40%).
11. **Summarization flag** = a later user message is a summarization hand-off ("Here are the answers the other agent
    provided…", "picking up work from a previous agent…"), or the trial has summarization sub-call rows, or a
    context-full warning appears. Generic "summary"/"summarize" matches in terminal output were false positives and are
    not used.
12. **Tokenizer agreement** uses each model's public `tokenizer.json` loaded with the `tokenizers` library (no remote
    code execution). Kimi (custom tokenizer code), GPT-5 family and Gemini have no loadable public tokenizer and are
    reported as NA.
13. **Turn caps**: declared per source when ≥ 5% of its runs share the maximum K (spec rule); 6 sources have one.

## Features and predictors
14. **E2 identifiability.** Models with < 20 cells of ≥ 4 runs in crossed tasks (gpt-5, gpt-5-mini, kimi-k2-thinking)
    are dropped from the E2 mixed models (the fixed effect would be estimated from a handful of cells). The cell
    variance component is passed to statsmodels as per-task indicator matrices (`VCSpec`), which is the same model as
    `vc_formula={"cell": "0 + C(cell_id)"}` but ~10× faster; estimates agree to 3 decimals.
15. **P3 levels.** The spec text says 14 levels but lists 15 (0.01, 0.025, 0.05, 0.10, …, 0.90, 0.95, 0.975, 0.99);
    all 15 are used. One hyperparameter configuration is shared by all levels (selected by mean pinball over levels).
16. **P2 tuning criteria.** The mean component is tuned by validation MSE; the variance component by validation CRPS
    of the resulting P2 forecast (after choosing c). P6 uses the same split (ridge α of the mean by MSE, of the
    variance by CRPS).
17. **P0/P1** are fit on the fit + validation roles of each training fold (they have no tuning); the calibration role
    is reserved for CQR for every method.
18. **Stochastic predictors (P4 variants, P5).** Run with 3 seeds; the main E3 table reports the first pre-specified
    seed with its bootstrap CI, and `e3_seed_variation.csv` reports mean ± SD across seeds (never the best seed).
19. **P5 hazards are evaluated on a j-grid** (every j ≤ 64, then ~4% geometric steps up to j_max) and interpolated in
    log j; the cost-given-K curve likewise. This is a numerical approximation for speed (LightGBM is piecewise
    constant in j, so the error is small).
20. **CQR full-distribution version.** To score CQR with QCRPS, every symmetric quantile pair (τ, 1−τ) of the grid is
    conformalized at level 1−2τ (49 pairs; the median is unchanged). The spec's level-specific intervals
    (50/80/90/95%, one-sided 90/95/99%) are computed exactly as specified and reported in `e3_cqr_coverage.csv`.
21. **O2 / O1 hyperparameters.** O2 reuses P3's tuned configuration and O1's hazard models reuse P5's hazard
    configuration (no separate 20-config search per k) to stay within compute; this does not use test data.
22. **O1 context growth.** ĝ = (I_k + O_k + obs_k − I_1)/k, the mean per-turn growth of the history observed up to and
    including the k-th observation (defined at k = 1; minimum 1 token), as the spec's "mean per-turn context growth
    over turns ≤ k".

## Extension compatibility (agenttrove_trajectory_metric_extension.md)
23. `turns.parquet` additionally stores `n_tok = I_k + O_k`, cumulative tokens `cum_tokens` (N_k) and `I_next`
    (I_{k+1}; null after the last call). Additive only; no base quantity changed.
24. Prediction files carry a `k` column (0 for pre-run forecasts) in addition to the spec's columns.

## Experiments
25. **Verdict rules (fixed before results were seen, in code):** H2 supported if the best-on-validation method beats P1
    with Holm p < 0.05; H3 supported only if all three non-Gaussian methods (P3-EVT, P4, P5) have significantly lower
    PIT KS than P2, not supported if none does, inconclusive otherwise; H4 supported if O2's QCRPS decreases at every
    grid k in both populations (eligible and fixed panel), inconclusive if in only one; H6 compares the relative gain of
    the best history model (E4 test runs) with that of the best E3 method (all runs) via independent bootstrap draws.
26. **E3 best method** = lowest task-weighted validation QCRPS averaged over folds among P2–P6 (P5). CQR(best) =
    CQR(P5).
27. **E4 P1** is the E3 P1 forecast restricted to the S2 target runs (P1 has no history variant). P4 in E4 uses one
    seed. The oracle is computed on cells with ≥ 10 runs; gap-closed is reported on that subset.
28. **E5 eligible population** at k uses all runs with K > k; the k = 0 reference is the E3 out-of-fold P5 forecast
    on exactly the same runs (ratios are paired). The fixed panel is K ≥ 33 (spec: K > 32).
29. **E7** uses the best S1 method (P5) refit once on the temporal training set (single seed), with a 70/15/15
    fit/val/cal split by task group inside it.
30. **E8 kill rule.** A run is killed at the end of the first turn whose cumulative cost exceeds the cap, unless that
    turn is the run's last (then it finished anyway). Caps for P1 use its raw upper quantile; for the best method the
    one-sided CQR upper quantile (calibrated per fold, one run per task). τ = 0.8 was added to CQR's one-sided levels.
    Outcome = exploratory `completed` proxy (spec label degenerate, item 3).
31. **E1 bootstrap CIs** for within-cell spread and bimodality resample cells (each cell belongs to one task in the
    statistic's population, so this is the cluster bootstrap); Hill and scaling CIs resample tasks.
32. **E6 skipped**: `llm_predictor.enabled` is false by default and no API key is configured (spec A.8); H5 = skipped.
33. **Optional `h_last1`/`h_last5` history variant** (§6.2) not run: E0 did not establish that `date` orders runs
    within a cell (dates are per job batch for many sources).
34. **E3 sensitivities refit the best method with one seed and score only the test role** (no val/cal predictions,
    which these analyses do not use) to fit the compute budget. Leave-family-out uses 6 source families (the spec's
    "other" family collects sources without a manual family).

## Extension E9 (trajectory-integrated evaluation)
Base-spec deviations that affect the extension, and how E9 adapts (read before E9 per extension §A.3):
- Target, cost weights, k grid and folds are unchanged from the base spec; the best S1 method is P5 (E3 validation).
- Base O1/O2 models were not persisted (only their forecasts), so E9 refits what it needs; base E5 per-k O1
  forecasts are reused at grid k (exact same forecasts), the pooled O1 hazard elsewhere.
E9-specific choices:
35. **Static is scored uncorrected; all other methods are clipped at log C_k.** Extension §2.3 asks to clip every
    method before scoring, but §5 defines Static ("never updated") and Static-truncated (clipped) as distinct and
    sanity check §A.7.1 requires Static's ITS to equal its pre-run QCRPS, which only holds unclipped. Static's
    infeasible-quantile share is still reported as a diagnostic. (Conflict internal to the extension; base unaffected.)
36. **At d_0 every method uses the Static (P5) forecast** (O1/O2 define no k = 0 behaviour of their own).
37. **O2-pooled training rows are subsampled**: at most 2 decision points per training run (uniform over its
    k = 1..min(K−1, 64)), each run keeping total weight 1 spread over its sampled rows. With all rows (~600k per fold)
    × 15 quantile levels × 2 variants × 5 folds the fit did not finish within the compute budget (first attempt with
    4 rows/run took > 50 min for one variant of one fold).
38. **O1 pooled hazard** is trained on person-period rows (run, k, j > k) for up to 2 sampled decision points per
    training run (k ≤ 64, k as a feature via pf_k capped at 64), capped at 2M rows; its hazards are evaluated on every
    j in (k, k+16] then ~8% geometric steps (interpolated in log j). The per-k base models are used at k ∈ {1, 2, 4,
    8, 16, 32}; `O1_src` records which model produced each forecast (diagnostics table).
39. **E9 sensitivity (b) for Y_tok / Y_hi** is computed for Static and Static-truncated only (from the E3 sensitivity
    refits of P5 on those targets); refitting the online models for two extra targets was beyond the compute budget.
