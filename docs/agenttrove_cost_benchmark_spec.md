# AgentTrove Token-Cost Prediction Benchmark — Implementation Specification

Version 1.0 · 2026-11-01 · Author: Kumar (with Claude)

This document is the complete, self-contained specification for an experiment that benchmarks methods for **predicting the token cost of an LLM agent run**, using the trajectories in the AgentTrove dataset. It is written to be handed to a Claude Code session that will implement every component, run every experiment, and produce a results report. Where this document says MUST, the implementation must follow it exactly. Where it says SHOULD, deviations are allowed but must be logged in `results/DEVIATIONS.md` with a reason. Nothing here should require guessing; if the implementer finds an ambiguity, the rule is: choose the most conservative option (the one least likely to leak information from test to train, or to flatter a method), record the choice in `DEVIATIONS.md`, and continue.

**How the document is organized.** Part A (immediately below) is your brief: the task, why it matters, the exact steps, how to communicate, what the results should look like, and when you are done. Sections 0–15 are the technical specification that Part A refers to. Appendix A shows the expected shape of the outputs. Read the whole document before writing any code.

---

## Part A — Brief for the implementing agent

### A.1 Your task
You are a research engineer. Build, run and report a rigorous benchmark of methods that forecast the **probability distribution** of an LLM agent run's token cost, using the public AgentTrove dataset of agent trajectories. Concretely:
1. Obtain a stratified sample of AgentTrove trajectories (§3).
2. Verify what the data actually contains and resolve its ambiguities (§4, experiment E0), then **stop and report** to the user (checkpoint 1, §A.4).
3. Reconstruct per-turn token usage and the cost target from the transcripts (§5).
4. Implement the feature sets (§6), the predictors (§7), the splits (§8) and the evaluation suite (§9).
5. Run experiments E1–E8 (§10), in the order of §13.
6. Write `results/REPORT.md` exactly as specified in §12, with every number traceable to a file in `results/`.

You work autonomously between checkpoints. You do not need permission for routine engineering choices that this document leaves open; log them in `DEVIATIONS.md`.

### A.2 Why this matters (the intent)
People running LLM agents cannot currently say, before or during a run, what it will cost, or how sure they are. Published work predicts a single number and finds it predicts poorly (correlations up to about 0.39), while runs of the same task can vary by up to 30× in tokens. The working hypothesis of this project is that **the right object to predict is a distribution, not a number**, and that methods from quantitative finance and actuarial science — built for exactly this kind of heavy-tailed, heteroscedastic quantity — can produce well-calibrated cost distributions with cheap models.

The study has to answer four questions, in this priority order:
1. **How much of cost variability is predictable at all?** Decompose it into task, model, task × model, and irreducible run-to-run noise (E2). The project author's prior is that variability is mostly a property of the *task* (hypothesis H1); test it honestly.
2. **Which methods give calibrated, sharp cost distributions before a run starts?** (E3, E4)
3. **How quickly does uncertainty shrink as a run unfolds?** (E5)
4. **Are such forecasts useful?** Do they support budgets that save money without killing successful runs, and do they stay calibrated when conditions shift? (E7, E8)

A negative or null result is a valid, valuable outcome. If cost turns out to be mostly unpredictable noise, say so clearly and quantify it. Never tune, filter or re-define things after looking at test results to make a method look better.

### A.3 Step-by-step procedure
Follow these steps in order. Each step lists its exit criterion; do not start a step until the previous one's criterion is met.

| Step | What to do | Exit criterion |
|---|---|---|
| 1. Setup | Create the repository layout (§2.2), `config.yaml` with every constant from §14, `requirements.txt`; install packages. | `pytest` runs (even with zero tests). |
| 2. Core tests | Write unit tests for cost reconstruction (§5.6) and metrics (§13 item 1). | All tests pass. |
| 3. Data access | Check network access to huggingface.co. If unreachable, stop and ask the user for local Parquet shard paths. | Metadata of all rows can be read. |
| 4. Metadata scan and selection | Pass 1 and Pass 2 of §3.3. | `data_cache/selection.parquet` written; sample-size table logged. |
| 5. E0 audit | Everything in §4, §3.4 and E0 (§10). | `results/logs/e0.md` written. **Checkpoint 1** (§A.4). |
| 6. Build caches | `runs.parquet`, `turns.parquet`, `splits.parquet`; leakage tests (§11 item 7). | Tests pass; row counts match E0. |
| 7. E1, E2 | Stylized facts and variance decomposition. | Logs, tables and figures written. |
| 8. Pipeline on baselines | Implement P0 and P1 and the full evaluation of §9 end to end; sanity-check numbers (§A.6). | E3 tables exist for P0 and P1, and pass the sanity checks. |
| 9. Remaining predictors | P2–P6, P3-EVT, P4 variants, CQR; run E3 in full. | E3 complete. **Checkpoint 2.** |
| 10. E4, E5, E7, E8 | Warm, online, shift, replay. | All logs written. |
| 11. E6 | Only if `llm_predictor.enabled: true` and an API key is present (default **false**; see §A.8). | Done or skipped with a logged reason. |
| 12. Report | `REPORT.md` per §12; consistency check (§A.6). | Definition of done (§A.7) met. |

### A.4 Checkpoints and communication
- **Checkpoint 1 (after E0).** Send the user a short summary: sample size (runs, tasks, cells, models, sources), how the data-semantics questions of §4 were resolved, measurement-validity findings, and whether any E0 stop condition was triggered. If a stop condition was triggered, wait for the user's decision. Otherwise continue without waiting.
- **Checkpoint 2 (after E3).** Send the main E3 table and a 3-sentence interpretation. Continue without waiting.
- **Blockers.** If something makes a part of the plan impossible (missing column, no multi-run cells, out of memory), say what it is, what you will do instead, and log it. Do not silently drop an experiment.
- **Progress.** Keep a task list of the steps in §A.3 and update it as you go. Avoid narrating routine work.

### A.5 What the results should look like
- Every experiment produces: CSV tables in `results/tables/`, PNG figures in `results/figures/`, a markdown log in `results/logs/eN.md`, and (for predictors) out-of-fold predictions in `results/predictions/` as Parquet with columns `run_id, task_id, model_id, fold, method, q01 … q99, y`.
- Every reported number has a 95% confidence interval from the task-cluster bootstrap (§9.4), unless it is a count.
- Numbers in tables are rounded to 3 significant figures; CRPS values to 3 decimals.
- Appendix A gives the exact shape of the main tables and the hypothesis table. Expect values roughly in these ranges (sanity, not targets): log Y on the order of 8–16 (thousands to millions of reference units); within-cell SD of log Y between about 0.2 and 1.5; QCRPS of P0 on the order of 0.3–1.0.
- The report must be readable by someone who has not seen this document: define CRPS, PIT and coverage briefly in the report itself.

### A.6 Sanity checks to run before trusting any result
1. QCRPS of P1 is no worse than QCRPS of P0 on the pooled test set, beyond the bootstrap CI of their difference (P1 has strictly more information). If it is worse, investigate for a bug before continuing.
2. The cell oracle (E4) has the lowest CRPS in E4. If any method beats it materially, suspect leakage.
3. Coverage of P3's raw 90% interval is within ±10 points of 90%; CQR's 90% coverage on test tasks is within ±2 points of 90%.
4. PIT histograms of a simulated, correctly specified forecast are flat (unit test).
5. Online CRPS at k = 1 is not worse than the k = 0 reference by more than its CI.
6. Every number in `REPORT.md` matches the corresponding CSV (write a script that checks this).
7. No `task_id` overlaps between training and test in any fold (test).

### A.7 Definition of done
- All of E0–E5, E7, E8 completed (E6 completed or skipped with a reason).
- `results/REPORT.md` follows §12 and contains the hypothesis table with a verdict for H1–H6.
- All sanity checks in §A.6 pass, or their failure is explained in the report.
- `results/DEVIATIONS.md` lists every departure from this spec.
- The run is reproducible: `python -m src.run_all --config config.yaml` regenerates every table and figure from the cached data.
- Final message to the user: the headline findings (5–8 sentences), the hypothesis verdicts, and the paths to `REPORT.md` and `DEVIATIONS.md`.

### A.8 Decisions already made (do not re-ask)
- **Cost units:** reference units (§5.3), not dollars.
- **Tokenizer:** `o200k_base` canonical tokenizer for everything (§5.1).
- **E6 (LLM predictor):** off by default. Turn it on only if the user says so and an API key is configured.
- **Scale:** at most 300,000 runs (§3.3); reduce if compute requires and log it.
- **Primary metric:** task-weighted mean QCRPS on log cost (§9.1).

### A.9 Things you must not do
- Do not use test-fold data in any fitting, tuning, feature transformation, recalibration or conformal calibration.
- Do not drop runs or sources after seeing test results.
- Do not report a method's best seed or best fold; report the pre-specified aggregates.
- Do not invent numbers, fill gaps with estimates, or describe figures you did not generate.
- Do not change hypotheses H1–H6; label any additional analysis as exploratory.

### A.10 Glossary
- **Run / trajectory / trial:** one execution of the agent on one task by one model.
- **Turn (call):** one assistant message, i.e. one model call. K = number of turns in a run.
- **Cell:** a (task, model) pair. Multi-run cells give the true per-task cost distribution.
- **Y:** reference cost of a run (§5.3); **y = log Y** is what every predictor forecasts.
- **Quantile forecast:** the 99 values q01…q99 a predictor outputs for y.
- **QCRPS:** quantile-approximated continuous ranked probability score (§9.1); lower is better.
- **PIT:** probability integral transform; uniform when the forecast is calibrated (§9.3).
- **Coverage:** share of outcomes inside a predicted interval.
- **Cold / warm / online:** prediction before a run without the task's history / with it / after observing the first k turns.

---

## 0. Executive summary

**Question.** How much of an agent run's token cost is predictable — before the run starts and while it runs — and which forecasting methods produce *calibrated predictive distributions* of that cost?

**Core idea.** Treat run cost like a financial risk quantity: a positive, heavy-tailed scalar whose spread depends on context. Forecast its full distribution (99 quantiles), and evaluate with the tools quantitative finance uses for density forecasts: proper scoring rules (CRPS), coverage backtests, PIT calibration, and decision-value replays. Exploit the one advantage agents have over finance: **many runs of the same task**, which reveal the true per-task cost distribution and let us measure the predictability ceiling directly.

**What gets produced.**
1. A cost-reconstruction pipeline that turns AgentTrove transcripts into per-turn token accounting.
2. Descriptive "stylized facts" of agent cost (tails, bimodality, scaling with run length).
3. A variance decomposition answering: *is cost variability a property of the task, the model, their interaction, or irreducible run noise?*
4. A benchmark of 9 predictors across 3 prediction settings (cold pre-run, warm pre-run, online mid-run), plus conformal wrappers.
5. Calibration backtests, a distribution-shift experiment, and a budget-policy replay.
6. `results/REPORT.md` with every table and figure defined in §12.

---

## 1. Background and rationale (read before implementing)

### 1.1 Why distributions, not point estimates
Prior agent-cost work (Bai et al. 2026, arXiv:2604.22750; TokenCast, arXiv:2609.35760) evaluates point predictions by correlation or absolute error. Bai et al. report runs of the same task differing by up to 30× in tokens. When identical inputs produce outcomes spread over an order of magnitude, a point estimate is the wrong object: the right object is a predictive distribution whose width is honest. A calibrated forecaster that cannot predict an outcome should output a wide distribution; one that can should output a narrow one. Proper scoring rules (CRPS) reward exactly this: calibration *and* sharpness (Gneiting & Raftery 2007).

### 1.2 Why finance methods
Each predictor in §7 is grounded in a mature method from financial econometrics or actuarial science that solves a structurally similar problem:

| Agent-cost property | Finance/actuarial analog | Method used here |
|---|---|---|
| Spread varies by task | Heteroscedasticity (volatility) | P2 heteroscedastic model (GARCH-style variance equation) |
| Budgets are quantiles | Value at Risk | P3 direct quantile regression (CAViaR-style, cross-sectional) |
| Heavy right tail | Extreme losses | P3-EVT generalized Pareto tail (McNeil & Frey 2000) |
| Converge-or-loop bimodality | Regime switching; PD × LGD | P4 mixture density model |
| Cost ≈ f(number of turns) | Duration / hazard models (Engle & Russell 1998) | P5 two-stage structural survival model |
| Per-task history | Realized-volatility history (HAR-RV, Corsi 2009) | Warm-task features in setting S2 |
| Intraday information | Nowcasting, adaptive execution | Online predictors O1, O2 |
| Interval guarantees under shift | Conformal prediction | CQR (Romano et al. 2019) and ACI (Gibbs & Candès 2021) |
| Forecast backtesting | VaR backtests (Kupiec 1995; Christoffersen 1998; DQ test, Engle & Manganelli 2004) | §9.3 |

### 1.3 Pre-registered hypotheses
State these in the report and give a verdict for each with its evidence. Do not add hypotheses after seeing results without labeling them exploratory.

- **H1 (task-driven spread).** Across cells, the between-task variance component of the per-cell cost spread (log SD) exceeds the between-model component. Tested in E2.
- **H2 (text carries signal).** In the cold pre-run setting, the best feature-based predictor achieves a significantly lower task-weighted mean CRPS than P1 (source × model marginal). Tested in E3.
- **H3 (shape matters).** Predictors that do not assume a Gaussian shape on log cost (P3-EVT, P4, P5) achieve better calibration (lower PIT KS distance) than P2. Tested in E3.
- **H4 (online gain).** CRPS of total cost improves monotonically with the number of observed turns k. Report the curve and the k at which CRPS is 50% of the pre-run CRPS. Tested in E5.
- **H5 (capacity is not the bottleneck).** A small LLM predictor is not significantly worse than a large LLM predictor, and neither beats the best feature-based predictor. Tested in E6 (optional module).
- **H6 (history beats text).** In the warm setting, history features reduce CRPS more than text features do in the cold setting (relative to P1). Tested in E4.

---

## 2. Environment and repository layout

### 2.1 Software
Python 3.11. Required packages (pin versions in `requirements.txt` after install):
`pandas, pyarrow, numpy, scipy, scikit-learn, lightgbm, statsmodels, tiktoken, torch (CPU is fine), matplotlib, tqdm, datasets, huggingface_hub, joblib, pytest`.
Optional: `transformers` (model-specific tokenizers, §4.4), `anthropic` (E6).

### 2.2 Layout
```
agenttrove-cost/
  README.md
  requirements.txt
  config.yaml               # every constant in this spec lives here
  src/
    run_all.py              # runs every step after data caching, in §13 order
    data/  audit.py  select.py  load.py  parse.py
    cost/  reconstruct.py  tokenizers.py
    features/  prerun.py  history.py  prefix.py  text.py
    predictors/  base.py  p0_global.py  p1_cell.py  p2_hetero.py  p3_quantile.py
                 p4_mdn.py  p5_survival.py  p6_linear.py  p7_llm.py  cqr.py  aci.py
                 o1_structural.py  o2_prefix_gbm.py
    eval/  metrics.py  backtests.py  bootstrap.py  decomposition.py  replay.py
    experiments/  e0_audit.py ... e8_replay.py
    report/  make_report.py
  tests/
  data_cache/               # parquet subsets, git-ignored
  results/
    tables/  figures/  logs/  predictions/  REPORT.md  DEVIATIONS.md
```

### 2.3 Determinism
Global seed `SEED = 20261101`. Every stochastic component receives a seed derived as `SEED + hash(component_name) % 10_000`. Stochastic predictors (P4, P5 Monte Carlo) are run with 3 seeds; report mean ± SD across seeds. All splits are computed once in E0 and saved to `data_cache/splits.parquet`; every experiment reads them from there.

### 2.4 Compute budget
Target: completes on one machine with 16+ CPU cores and 64 GB RAM in under 24 hours, excluding E6. If the selected sample (§3.3) is too large for this, reduce `N_RUNS_MAX` and log it.

---

## 3. Data acquisition and selection

### 3.1 Source
`open-thoughts/AgentTrove` on Hugging Face (Apache-2.0), about 1.7M rows, Parquet. Access via `datasets.load_dataset(..., streaming=True)` or by downloading Parquet shards with `huggingface_hub`. If the network cannot reach Hugging Face, stop and ask the user for local shard paths (`config.yaml: local_parquet_glob`).

### 3.2 Known schema (verify in E0; do not trust)
The dataset card shows columns including: `conversations` (list of messages), `agent`, `model`, `model_provider`, `date`, `task`, `episode`, `run_id`, `trial_name`, `original_source`, `original_teacher`, `result`, `reward`, `judgment`, `verifier_output`, `instruction`, `trace_source`, `path`, `split`. Many columns are null for some sources. The card's prose also mentions a `messages` column with `role`/`content`. Messages may be ShareGPT-style (`from`/`value`, roles `system`/`human`/`gpt`) or OpenAI-style (`role`/`content`). The parser MUST handle both.

The harness is **terminus-2** (Harbor framework). Its assistant turns are JSON objects with keys such as `analysis`, `plan`, `commands` (list of `{keystrokes, duration}`), `task_complete`. Observations come back as user messages containing terminal output.

### 3.3 Metadata scan and cell selection (two-pass)
Reading every transcript is expensive. Do two passes.

**Pass 1 (metadata only).** Read all rows but only metadata columns (everything except `conversations`/`messages`). Build a table with one row per dataset row. Derive:
- `task_id` (§4.1), `model_id` (§4.1), `trial_id` (§4.1), `source` = `original_source`.
- Counts of distinct `trial_id` per `(task_id, model_id)` — the **cell** — and per task.

**Pass 2 (selection).** Select *cells*, never individual runs, so that every selected cell keeps all its runs (required for the multi-run analysis):
1. Keep rows with `agent == "terminus-2"` (log the share dropped).
2. Define a cell as eligible if it has `n_runs ≥ N_MIN_CELL = 2`.
3. Stratify by `source`. For each source, sample eligible cells uniformly at random until the source contributes at most `CAP_PER_SOURCE = max(5% of N_RUNS_MAX, 2000 runs)`.
4. Within a cell, if `n_runs > N_RUNS_CELL_CAP = 32`, sample 32 runs uniformly.
5. Stop when total selected runs reach `N_RUNS_MAX = 300,000`.
6. Additionally retain **all** cells from tasks that appear with ≥ 2 models (crossed tasks), up to the cap, because E2 needs them. Fill this quota first (target ≥ 40% of runs from crossed tasks if they exist).

Then load transcripts only for the selected rows. Save as `data_cache/runs.parquet`.

### 3.4 Exclusion rules (apply in this order; log counts at each step)
1. Transcript unparseable or empty.
2. Zero assistant turns.
3. Duplicate trial (§4.2) — keep one.
4. Any message content longer than 2,000,000 characters (corrupt).
Record an exclusion flow table (Table T0).

---

## 4. Data semantics (E0 must resolve these before anything else)

### 4.1 Identifiers
- `model_id`: use `model` if non-null; else `original_teacher`; else drop the row. **E0 MUST check** whether `model` and `original_teacher` disagree on the same row, and whether the transcript's style is consistent with `model` (the dataset preview shows a row with `model = gpt-5-nano` and `original_teacher = GPT-5-mini`). Report the disagreement rate. If `model` is ambiguous for a source, exclude that source and log it.
- Normalize `model_id` strings: lowercase, strip date suffixes (e.g. `-2025-08-07`), map aliases to one canonical name per model in `config.yaml: model_aliases`.
- `task_id`: use `task` if non-null; else `trial_name.split("__")[0]`; else `hash(instruction)`. **E0 MUST verify** that these three agree where all are available (report agreement rate). A task is the unit of generalization; getting it wrong leaks information.
- `trial_id`: `trial_name` if non-null; else `run_id`; else the row index.

### 4.2 Row-per-trial vs row-per-episode
The `episode` column (e.g. `episode-7`) suggests some sources may store one row per *turn snapshot* rather than one row per trial. E0 MUST:
1. Group by `trial_id` and count rows.
2. For groups with > 1 row, test whether the shorter transcripts are exact prefixes of the longest one (compare message lists).
3. If yes, keep only the longest row per trial. If transcripts within a trial are *not* prefixes of each other, keep the row with the most assistant turns and log the anomaly with examples.
Report the share of trials affected.

### 4.3 Success label
Define `success ∈ {0, 1, NA}`: `reward ≥ 0.5 → 1`, `reward < 0.5 → 0`; if `reward` is null, map `result` and `judgment` values (E0 lists their distinct values and the mapping is written into `config.yaml`); otherwise NA. Success is used in E1 and E8 only, never as a prediction feature in pre-run settings.

### 4.4 Measurement validity checks
E0 MUST also report, per model:
- **Hidden reasoning.** The share of assistant messages containing `<think>` or similar reasoning blocks. Reasoning models served via APIs that hide reasoning (e.g. GPT-5 family) will have unobserved output tokens. Our cost is therefore *visible-transcript cost*, a lower bound on billed cost for those models. This does not invalidate predictability analysis of visible cost, but the report must say so.
- **Context summarization.** terminus-2 can summarize history when context grows large. Flag runs where any user message matches the summarization handoff pattern (search for case-insensitive `summary` or `summarize` in a user message that follows a long gap in turn structure; also list the distinct first 80 characters of such messages so the pattern can be confirmed by eye). Flagged runs violate the linear-history assumption in §5.2. Report the share; run E3 sensitivity with flagged runs excluded.
- **Turn caps.** For each source, the distribution of the number of assistant turns K. If ≥ 5% of runs share the maximum K, treat that as a turn cap `K_cap(source)` (used in P5).
- **Tokenizer agreement.** On 2,000 random runs, compare canonical token counts (§5.1) with model-specific tokenizers where available (open models via `transformers`). Report the ratio distribution per model.

---

## 5. Cost reconstruction (the target variable)

### 5.1 Canonical tokenization
All primary analyses use one **canonical tokenizer**: `tiktoken` encoding `o200k_base`, applied to every message's text content. Rationale: we are studying the predictability of agent behavior; using one tokenizer removes tokenizer differences between models, which are a constant per-model multiplier and are not what we want to predict. Model-specific tokenizers are a sensitivity analysis only (§4.4).

Let `tok(m)` = number of `o200k_base` tokens of message m's content, plus a per-message overhead `C_MSG = 4` tokens (role and delimiters). Assistant messages that are JSON are tokenized as their raw string.

### 5.2 Turn accounting
Order the transcript's messages as `m_1, …, m_M`. Let the assistant messages be at indices `a_1 < a_2 < … < a_K`; K is the number of model calls (turns). Under the linear-history assumption (the full history is re-sent at every call):

- Input tokens of call k: `I_k = Σ_{i < a_k} tok(m_i)`
- Output tokens of call k: `O_k = tok(m_{a_k})`
- Cached input of call k (perfect prefix caching): `Cch_k = I_{k-1} + O_{k-1}` for k ≥ 2, and `Cch_1 = 0`. Clip so `Cch_k ≤ I_k`.
- Uncached input: `U_k = I_k − Cch_k`.

### 5.3 Target variables
- **Primary target:** reference cost in *reference units*
  `Y = Σ_k ( U_k + λ_c · Cch_k + λ_o · O_k )` with `λ_c = 0.1`, `λ_o = 4`.
  Rationale: these ratios reflect typical frontier-API pricing (cached input ≈ 10% of input; output ≈ 4× input), and using reference units avoids stale dollar prices. The model is the same for every run, so Y differs from dollars only by a per-model constant that the predictors learn.
- **Secondary targets (sensitivity, E3 only):** `Y_tok = Σ_k (I_k + O_k)` (no caching, all tokens equal), and `Y_hi = Σ_k (U_k + 0.1·Cch_k + 8·O_k)`.
- **Run length:** K.
- All models predict `y = log Y` (natural log). Distributions are reported back on the Y scale by exponentiating quantiles (quantiles are invariant to monotone transforms).

### 5.4 Censoring note (important)
Runs stopped by a turn cap or timeout have *realized* costs that are the true cost of running that harness with that cap. The primary target is realized cost, so **there is no censoring problem for the target.** Censoring only matters for extrapolating to an uncapped harness, which is out of scope. P5 models the cap explicitly as part of the data-generating process.

### 5.5 Per-turn table
Save `data_cache/turns.parquet` with one row per (run, k): `I_k, O_k, Cch_k, U_k`, cumulative cost through k, observation length (tokens of the user message after the assistant turn), and the parsed prefix features of §6.3. This table feeds the online experiments and the replay.

### 5.6 Unit tests (must pass before any experiment)
- A synthetic 3-turn transcript with known token counts reproduces hand-computed `I_k, O_k, Cch_k, U_k, Y`.
- `Σ_k U_k + Σ_k Cch_k = Σ_k I_k` exactly, and `U_k ≥ 0` for every k.
- ShareGPT and OpenAI message formats yield identical Y for the same content.

---

## 6. Features

All feature transformers that learn anything (TF-IDF vocabulary, SVD, category encodings, standardization) MUST be fit on the training fold only.

### 6.1 Pre-run features (settings S1 and S2)
Available before the run starts:
- `model_id` (categorical; LightGBM native categorical).
- `source` (categorical) and `source_family` (manual grouping in `config.yaml`: code repair, shell, competitive programming, math, data/QA, other).
- Instruction text = the first user message (or `instruction` column if present; E0 decides which is the true task statement). Numeric stats: character length, canonical token length, line count, count of fenced code blocks, count of file-path-like strings (regex `[\w./-]+\.\w{1,5}`), count of URLs, count of numbers.
- System prompt canonical token length.
- Keyword indicators (binary): `test`, `fix`, `bug`, `implement`, `refactor`, `install`, `compile`, `error`, `performance`, `multiple files`.
- Text representation: TF-IDF (word 1–2 grams, `min_df = 5`, `max_features = 200,000`, sublinear tf) on the instruction, reduced by TruncatedSVD to 128 dimensions. Fit on the training fold only.
- Optional (ablation only): a sentence-embedding model if available offline; never required.

### 6.2 Warm-task history features (setting S2 only)
For a target run r of cell (t, m), with a **history set** H(t, m) of other runs of the same cell that is disjoint from all target runs (§8.2):
- `h_n`: number of history runs.
- `h_mean, h_sd, h_q10, h_q50, h_q90`: of log Y over H(t, m).
- `h_K_mean`: mean K over H(t, m).
- Cross-model history on the same task: `x_mean`, `x_sd` of log Y over runs of task t by *other* models in the history pool, and `x_n`.
- These are the HAR-RV analog (Corsi 2009): realized cost history as a predictor of future cost. Because runs within a cell have no reliable time order, use set statistics rather than lags over 1, 5, 25 runs; if E0 finds `date` reliable within cells, add `h_last1`, `h_last5` (means of most recent 1 and 5 history runs by date) as an extra variant.

### 6.3 Prefix features (setting S3, after observing turns 1..k)
Computed from turns ≤ k only:
- `k`, `log k`, cumulative cost through k (`C_≤k`, log), current context length `I_k + O_k` (log).
- Mean and last value of `O_j` and of observation length.
- Context growth rate: least-squares slope of `I_j` on j over j ≤ k.
- Loop/stuck indicators (the regime signals): share of assistant turns ≤ k whose `commands` keystrokes exactly repeat a previous turn's; count of observations containing any of `Traceback`, `Error`, `error:`, `command not found`, `No such file`, `FAILED`, `Permission denied`; longest streak of consecutive error-containing observations; count of `task_complete: true` attempts so far; share of turns with empty `commands`; count of JSON-parse failures in assistant turns.
- All pre-run features of §6.1.

---

## 7. Predictors

### 7.1 Common interface
Every predictor implements:
```
fit(train_df, val_df) -> self
predict_quantiles(df) -> array of shape (n, 99)   # quantiles of y = log Y at τ_j = j/100, j = 1..99
```
Quantiles MUST be non-decreasing in τ; enforce by sorting each row (monotone rearrangement, Chernozhukov et al. 2010). This single representation lets every metric in §9 be computed identically for every method.

### 7.2 P0 — Global marginal (per model)
Empirical quantiles (Hyndman–Fan type 7) of y over training runs of the same `model_id`. Unseen model → pooled training quantiles.
*Why:* the "know nothing about the task" floor. Any method that cannot beat P0 has learned nothing.

### 7.3 P1 — Source × model marginal with shrinkage
For cell (source s, model m) with `n` training runs: `q = w·q_{s,m} + (1−w)·q_m`, with `w = n / (n + 30)`, quantile-by-quantile.
*Why:* the strongest "no text" baseline. H2 is defined relative to P1; it isolates what the instruction text adds beyond knowing the benchmark and the model. This is "historical simulation" in VaR terms.

### 7.4 P2 — Heteroscedastic log-normal (two-stage gradient boosting)
1. Mean: LightGBM regression (L2) of y on features → `μ(x)`. Use out-of-fold predictions (5-fold grouped by task, inside the training fold) to get residuals `r = y − μ̂_oof(x)`.
2. Variance: LightGBM regression of `log(r² + 1e-6)` on features → `g(x)`. Then `σ̂²(x) = exp(g(x) + 1.2704)` (bias correction: `E[log χ²₁] = −1.2704`).
3. Recalibrate: a single scalar `c` multiplying σ̂, chosen on validation to minimize mean CRPS (grid `c ∈ [0.5, 2.0]`, step 0.01).
4. Quantiles: `μ(x) + c·σ̂(x)·Φ⁻¹(τ_j)`.
*Why:* the GARCH idea transplanted to the cross-section: model the variance as a function of information, not as a constant. It tests whether "the spread is predictable" (H1's practical consequence) with the simplest possible distributional assumption.

### 7.5 P3 — Direct quantile gradient boosting, with EVT tail (P3 and P3-EVT)
**P3.** Fit separate LightGBM models with the quantile (pinball) objective at `τ ∈ {0.01, 0.025, 0.05, 0.10, 0.20, …, 0.90, 0.95, 0.975, 0.99}` (14 levels). Sort per row, then interpolate to the 99-level grid with monotone PCHIP in τ.
*Why:* CAViaR's insight (Engle & Manganelli 2004) — model the quantile directly, with no distributional assumption. Here it is cross-sectional (no autoregression across time), since runs are not a time series.

**P3-EVT.** Replace the upper tail of P3 with a generalized Pareto tail fitted by peaks-over-threshold:
1. On the validation split, standardize: `z = (y − q̂_{0.5}(x)) / max(q̂_{0.9}(x) − q̂_{0.5}(x), 0.05)`.
2. Threshold `u` = empirical 0.90 quantile of z; exceedance probability `ζ = P(z > u)`.
3. Fit GPD shape ξ and scale β by MLE on `z − u` for `z > u` (`scipy.stats.genpareto.fit` with `floc = 0`). Require ≥ 200 exceedances; otherwise fall back to P3 and log it.
4. For `τ_j ≥ 1 − ζ`: `q_z(τ) = u + (β/ξ)·[((1 − τ)/ζ)^(−ξ) − 1]` (use the `ξ → 0` limit `u − β·log((1 − τ)/ζ)` if `|ξ| < 1e-4`). Map back: `q_y(τ) = q̂_{0.5}(x) + scale(x)·q_z(τ)`. Re-sort.
*Why:* quantile regression is unstable in the far tail where data are scarce; McNeil & Frey's two-step (standardize, then fit a Pareto tail) is the standard fix in risk management and gives defensible p99 estimates.

### 7.6 P4 — Mixture density network (regimes)
A multilayer perceptron mapping standardized dense features (numeric + one-hot of top-20 models and top-30 sources + SVD-128) to a K = 3 component Gaussian mixture on y: mixing weights via softmax, means unconstrained, standard deviations via softplus plus a floor of 0.05.
- Architecture: 2 hidden layers of 128 units, ReLU, dropout 0.1.
- Loss: negative log-likelihood. Adam, lr 1e-3, weight decay 1e-4, batch 512, at most 100 epochs, early stopping on validation NLL with patience 10.
- Quantiles: invert the mixture CDF per row by bisection (tolerance 1e-4).
- Report the learned mixing weights' distribution and the fraction of test runs whose dominant component is the high-mean one.
*Why:* if runs either converge or loop, cost is multimodal and any unimodal model is miscalibrated. This is Hamilton-style regime switching in the cross-section, and the credit-risk PD × LGD decomposition (probability of entering the bad regime × cost given it). Also run K = 1 (which is a neural P2) and K = 2 as an ablation.

### 7.7 P5 — Two-stage structural model (hazard on K, then cost given K)
Cost is close to a deterministic function of the number of turns K (context grows every turn), so model K's distribution, then the cost given K.
1. **Discrete-time hazard for K.** Build person-period rows (run, j) for j = 1..K. Label `event = 1` at `j = K` if the run stopped on its own; at a capped run (`K = K_cap(source)`), the stop at `j = K` is a forced stop: label `event = 1` but add the binary feature `at_cap = (j == K_cap)`. Features: pre-run features, `j`, `log j`. Fit a LightGBM binary classifier → hazard `h(j | x)`. Downsample person-period rows to at most 2M uniformly by run (keep all rows of a sampled run).
2. **Cost given K.** LightGBM regression of y on `(log K, K, pre-run features)` → `ŷ(K, x)`. Store out-of-fold residuals `e`, binned by K decile.
3. **Simulation.** For each test run, draw `M = 2000` samples: draw K from `P(K = j) = h(j) Π_{i<j}(1 − h(i))` (for sources without a cap, beyond `j_max = 1.5 × max training K` assume constant hazard equal to the mean of the last 5 hazards; truncate at `5 × j_max`); draw e from the residual bin matching K's decile; `y = ŷ(K, x) + e`. Quantiles = empirical quantiles of the samples.
*Why:* this is the duration-model approach (Engle & Russell's ACD; hazard models in credit risk). It puts the uncertainty where it lives (how long the run lasts) and handles turn caps explicitly. It is also the natural precursor to the online model O1.

### 7.8 P6 — Linear heteroscedastic baseline
Same as P2 but with ridge regressions (α chosen on validation from {0.1, 1, 10, 100}) on standardized dense features instead of LightGBM.
*Why:* an interpretable, low-capacity reference. If P6 ≈ P2, nonlinearity does not matter; this also feeds the capacity question (H5).

### 7.9 P7 — LLM self-report predictor (optional; E6 only)
Runs only if an LLM API key is available (`config.yaml: llm_predictor.enabled`).
- Prompt (fixed, saved in `config.yaml`): describes the terminus-2 harness (it sends the full conversation every turn), gives the agent model name, the source benchmark name, and the instruction (truncated to 6,000 canonical tokens), and asks for a JSON object `{"p10": int, "p50": int, "p90": int}` estimating **total tokens processed** (`Y_tok`) for one run. Temperature 0. One call per test *task-model cell* (not per run).
- Convert to a distribution on log Y_tok as a split normal: median `log p50`, left SD `(log p50 − log p10)/1.2816`, right SD `(log p90 − log p50)/1.2816` (floor 0.05).
- Map to the primary target with a per-model linear calibration `y = a_m + b_m · log Y_tok_pred` fitted on 300 validation cells (this gives the LLM a fair chance: it is scored after the same kind of calibration other methods get). Report both raw and calibrated results.
- Two predictor models: a small one and a large one (e.g. `claude-haiku-5-5` and `claude-sonnet-5-5`; any small/large pair is fine — record which).
*Why:* reproduces Bai et al.'s self-prediction idea in a fair setting and directly tests whether a less capable model suffices (H5).

### 7.10 CQR — Conformalized quantile regression wrapper
Applied to P3-EVT and to the best-performing predictor from E3 validation.
- Calibration set: a held-out set of training-fold tasks (§8.1), using **one randomly chosen run per task** so that calibration points are exchangeable at the task level.
- For each nominal level `1 − α ∈ {0.5, 0.8, 0.9, 0.95}`: scores `E_i = max(q̂_{α/2}(x_i) − y_i, y_i − q̂_{1−α/2}(x_i))`; `Q̂` = the `⌈(1 − α)(n + 1)⌉`-th smallest score; interval `[q̂_{α/2} − Q̂, q̂_{1−α/2} + Q̂]`.
- One-sided budget version for upper quantiles `τ ∈ {0.9, 0.95, 0.99}`: scores `y_i − q̂_τ(x_i)`, adjust by their `⌈τ(n + 1)⌉`-th order statistic.
- Mondrian variant: separate calibration per `model_id` (only for models with ≥ 200 calibration tasks; others use the pooled set).
*Why:* turns any predictor into intervals with finite-sample marginal coverage guarantees under task-level exchangeability. Essential for using forecasts as budgets.

### 7.11 Online predictors (setting S3)
Both predict the distribution of **total** log cost y given turns 1..k, so CRPS is comparable with pre-run predictors.

**O1 — Structural online.** Known: `C_≤k`. Unknown: remaining turns and their cost.
1. Conditional K distribution: P5's hazard model, conditioned on `K > k`, with prefix features added to the hazard model's inputs (refit a version on person-period rows j > k using prefix features at k; for simplicity fit one hazard model per k ∈ the evaluation grid).
2. Future turn costs computed mechanically: for future turn j > k, `I_j ≈ (I_k + O_k) + (j − k − 1)·ĝ`, where ĝ is the mean per-turn context growth over turns ≤ k (minimum 1 token); `O_j ≈` mean of `O_i` for i ≤ k; cached share per §5.2. Remaining cost is a deterministic function of K given these, so sample K (M = 2000) and add `C_≤k`.
*Why:* the TokenCast insight (carried context makes remaining cost compound) in its simplest explicit form; uncertainty lives only in K.

**O2 — Prefix quantile GBM.** P3 trained on rows (run, k) with prefix features (§6.3) and target y, one model per k in the grid `{1, 2, 4, 8, 16, 32}`. Include runs only when `K > k`.
**O2-noregime.** O2 without the loop/stuck indicators (ablation for the regime signal).
*Why:* the nowcasting approach — high-frequency information updates a low-frequency forecast. The ablation measures how much of the gain comes from detecting the stuck regime.

### 7.12 Oracles (upper bounds, not competitors)
- **Cell oracle (S2 ceiling):** for a test run in cell c, the empirical quantiles of y over all *other* runs of c (leave-one-out). Defined only for cells with ≥ 10 runs. It is the best achievable by any pre-run predictor that knows task and model but not run-specific randomness.
- **Within-cell noise floor:** the CRPS of the cell oracle measures irreducible run-to-run randomness given task and model.

### 7.13 Hyperparameters
For each LightGBM component, search 20 random configurations on the **validation split of fold 1 only**, then freeze for all folds: `num_leaves ∈ {15, 31, 63, 127}`, `learning_rate ∈ {0.02, 0.05, 0.1}`, `min_child_samples ∈ {20, 50, 100, 200}`, `feature_fraction ∈ {0.6, 0.8, 1.0}`, `bagging_fraction ∈ {0.7, 0.9, 1.0}`, `lambda_l2 ∈ {0, 1, 10}`, `n_estimators` by early stopping (patience 100, max 5000). Selection criterion: mean validation CRPS (for P3: mean pinball across its 14 levels). Record chosen configurations in `results/tables/hyperparams.csv`.

---

## 8. Splits and prediction settings

### 8.1 Grouped cross-validation (used by S1 and S3)
`StratifiedGroupKFold(n_splits = 5, shuffle = True, random_state = SEED)`, groups = `task_id`, stratified by `source`. No task appears in both train and test of a fold. Inside each training fold, split tasks (again grouped) into: 70% fit, 15% validation (early stopping, hyperparameters, P2's c, P7 calibration), 15% calibration (CQR only). Every test run of every fold receives a prediction from each method; metrics are computed on the pooled out-of-fold predictions.

### 8.2 Warm-task split (S2)
Only cells with ≥ 6 runs. Fix one 5-fold grouped split as in §8.1, but for test tasks, randomly split each cell's runs into a **history half** H and a **target half** T (if odd, the extra run goes to H). Training data: training-fold tasks, using the same half-split construction on *their* cells, so the model learns how history maps to targets. Features come from H; evaluation is on T only. Cross-model features `x_*` use only H runs of other models on the same task.

### 8.3 Leave-source-out (stress test, E3 secondary)
Hold out each `source_family` in turn (groups = source family); report CRPS degradation relative to grouped-task CV.

### 8.4 Temporal split (E7)
If `date` coverage ≥ 80%: training = runs before the 70th percentile date; test = after. Tasks appearing on both sides are removed from the test side.

### 8.5 Settings summary

| Setting | Information available | Test units | Predictors |
|---|---|---|---|
| S1 cold pre-run | Instruction, model, source | All runs, grouped-task CV | P0–P7, CQR |
| S2 warm pre-run | S1 + history of the same task | Target-half runs in cells with ≥ 6 runs | P1, P2, P3-EVT, P4 with history features; cell oracle |
| S3 online | S1 + turns 1..k | Runs with K > k, grouped-task CV | O1, O2, O2-noregime; best S1 predictor as the k = 0 reference |

---

## 9. Evaluation

### 9.1 Primary metric: quantile-approximated CRPS on log cost
For a prediction `q_1 ≤ … ≤ q_99` and outcome y:
```
QCRPS(q, y) = (2/99) · Σ_{j=1}^{99} ρ_{τ_j}(y − q_j),   ρ_τ(u) = u·(τ − 1{u < 0}),   τ_j = j/100
```
This approximates `CRPS = 2∫₀¹ ρ_τ(y − F⁻¹(τ)) dτ` and is identical for every predictor. Units: log cost. Lower is better.

**Aggregation.** Primary: **task-weighted mean** — average QCRPS within each task, then average across tasks (so heavily sampled tasks do not dominate). Secondary: run-weighted mean. Report both.

### 9.2 Secondary metrics
- **Pinball loss** at τ = 0.5, 0.9, 0.99.
- **Point accuracy (for comparison with the literature):** Pearson and Spearman correlation between `q_50` and y; MAE of `q_50` on log scale.
- **Sharpness:** mean width `q_90 − q_10` (log units).
- **Distribution distance on multi-run cells:** for test cells with ≥ 10 runs (all runs share the same prediction in S1 since features are identical), the 1-Wasserstein distance on log scale between predicted and empirical cell quantiles: `W1 = (1/99) Σ_j |q_j − q̂^emp_j|`, empirical quantiles type 7. This compares the forecast with the *whole* true distribution rather than one draw — the advantage finance lacks.
- **Expected shortfall:** `ES_0.95` of y = mean of `q_j` for `τ_j ≥ 0.95`. Report the realized average of y above the predicted `q_95` vs the predicted ES (ratio; 1 = calibrated tail mean).

### 9.3 Calibration backtests
- **PIT.** For each test run, `PIT = F̂(y)` by linear interpolation of τ between neighboring quantiles; if `y < q_1`, draw `PIT ~ U(0, 0.01)`; if `y > q_99`, draw `PIT ~ U(0.99, 1)` (randomized PIT, seeded). Report: 10-bin histogram (figure), Kolmogorov–Smirnov distance from U(0, 1), per model and per source family.
- **Coverage.** Empirical coverage of central intervals at 50/80/90/95% and of one-sided upper quantiles at 90/95/99%. Report 95% CIs by cluster bootstrap over tasks (§9.4).
- **Kupiec unconditional coverage test** (secondary; it assumes independent hits, which runs of the same task violate — the cluster-bootstrap CI is primary). For hits `h_i = 1{y_i > q_τ,i}`, `p = 1 − τ`, `x = Σ h_i`, `n` runs:
  `LR = −2 [ (n − x) ln(1 − p) + x ln p − (n − x) ln(1 − x/n) − x ln(x/n) ] ~ χ²₁`.
- **Dynamic-quantile (DQ) style conditional test, cross-sectional.** Regress `h_i − p` by OLS on `Z_i = [1, q_τ,i, model dummies (top 10), source-family dummies, log instruction length]`; test all coefficients = 0 with a Wald test using task-clustered standard errors; report the χ² statistic and p-value per method for τ = 0.9 and 0.95. A significant result means exceedances are predictable from information available at forecast time — the forecast is miscalibrated *somewhere* even if overall coverage is right (the cross-sectional analog of Christoffersen's independence test).

### 9.4 Statistical inference
- **Cluster bootstrap over tasks** (B = 1000, seeded): resample tasks with replacement, carry all their test runs. Use for CIs of every metric and for paired differences between methods.
- **Method comparisons:** for each method vs P1 and vs the best method, the bootstrap 95% CI and two-sided p-value of the difference in task-weighted mean QCRPS. Apply Holm–Bonferroni correction across all comparisons within an experiment.
- Report effect sizes as relative CRPS reduction vs P1: `(CRPS_P1 − CRPS_method) / CRPS_P1`.

---

## 10. Experiments

Each experiment is a script `src/experiments/eN_*.py` that reads cached data, writes tables to `results/tables/`, figures to `results/figures/`, and a short markdown summary to `results/logs/eN.md`.

### E0 — Data audit and reconstruction validation
All checks in §4 and §5.6. Outputs: T0 exclusion flow; schema report (columns, null rates per source); row-semantics resolution; identifier agreement rates; success-label mapping; hidden-reasoning and summarization shares per model; turn caps per source; tokenizer-agreement table; final sample summary (runs, tasks, cells, models, sources, crossed tasks, runs-per-cell distribution). **Stop and report to the user if** any of these hold: fewer than 500 cells with ≥ 6 runs; `task_id` agreement below 95%; more than 30% of runs flagged for summarization. These would undermine the design.

### E1 — Stylized facts
- Distribution of log Y per model and per source family (figure: ridgeline or box plots).
- Within-cell spread: per cell with ≥ 10 runs, `max Y / min Y` and the ratio of the 90th to 10th percentile; compare with Bai et al.'s up-to-30× figure.
- Tail heaviness: Hill estimator of the tail index of Y using the top 5% per model (report with bootstrap CI).
- Bimodality: per cell with ≥ 20 runs, fit 1- and 2-component Gaussian mixtures to log Y; report the share where BIC prefers 2 components.
- Scaling with run length: per model, regress log Y on log K; report slope with CI. Hypothesis from the context-growth structure: slope between 1 and 2.
- Cost vs success: median log Y by success, per source; share of total cost spent on failed runs.

### E2 — Variance decomposition (tests H1)
Use crossed tasks: tasks with ≥ 2 models each having ≥ 4 runs.
1. **Level decomposition.** Linear mixed model `y_{tmr} = μ + β_m + a_t + c_{tm} + ε_{tmr}`, model as a fixed effect, task `a_t ~ N(0, σ²_task)`, cell interaction `c_{tm} ~ N(0, σ²_cell)`, run noise `ε ~ N(0, σ²_run)`. Fit with `statsmodels MixedLM` (groups = task, variance component for cell nested in task), REML. Report shares of total variance: `Var(Xβ̂)` (model), `σ²_task`, `σ²_cell`, `σ²_run`. 95% CIs from 200 task-cluster bootstrap refits.
2. **Spread decomposition.** For each cell with ≥ 4 runs: `s_tm = SD(y)` within the cell. Fit `log s_tm = ν + β'_m + a'_t + e_tm` (groups = task). Correct for sampling noise: the sampling variance of `log s` from n normal draws is approximately `1/(2(n − 1))`; subtract the mean of this quantity from the residual variance and report the corrected share. **H1 is supported** if the corrected task share exceeds the model share and the bootstrap CI of their difference excludes 0.
3. **Spread transfer.** For each pair of models with ≥ 50 shared tasks: Spearman correlation of `log s_tm` across shared tasks, disattenuated for measurement noise using split-half reliability (split each cell's runs randomly in half, compute s on each half, take the correlation r_hh across cells, apply Spearman–Brown `r_rel = 2r_hh/(1 + r_hh)`; disattenuated `r = r_obs / √(r_rel,1 · r_rel,2)`). High transfer means a task's "volatility" measured with one model predicts it for another.
4. **Predictability of spread from text.** On held-out tasks (grouped CV), R² of LightGBM predicting `log s_tm` from pre-run features. This tells whether the task's spread, if it is a task property, is visible in the task description.

### E3 — Cold pre-run benchmark (S1; tests H2, H3)
All predictors P0–P6 (P7 in E6), P3-EVT, P4 with K ∈ {1, 2, 3}, CQR on P3-EVT and on the best method. Metrics: everything in §9, with CIs and comparisons. Sensitivities (primary metric only): secondary targets `Y_tok` and `Y_hi`; runs flagged for summarization excluded; leave-source-family-out CV; feature ablations for the best method — (a) model + source only, (b) + numeric text stats, (c) + keywords, (d) + TF-IDF/SVD.

### E4 — Warm pre-run benchmark (S2; tests H6)
Predictors P1, P2, P3-EVT, P4 with history features, the same predictors without history features (on the same S2 test runs, for a paired comparison), and the cell oracle. Report the gap closed: `(CRPS_P1 − CRPS_method) / (CRPS_P1 − CRPS_oracle)`. H6 is supported if history reduces CRPS (vs P1) by more than text does in E3, comparing relative reductions with bootstrap CIs.

### E5 — Online benchmark (S3; tests H4)
For k ∈ {0, 1, 2, 4, 8, 16, 32} (k = 0 is the best S1 predictor), predictors O1, O2, O2-noregime. Because the population with K > k shrinks as k grows, report each k on its own eligible runs **and** on a fixed panel of runs with K > 32 (so the curve is not confounded by survivorship). Figure: QCRPS vs k with CIs, both populations. Also report coverage of the 90% interval vs k.

### E6 — Predictor capacity (tests H5; optional, needs API access)
On 1,000 test cells sampled from fold 1's test tasks (stratified by source family): P7-small, P7-large (raw and calibrated), the best feature-based method, and P6. Paired comparisons by cell bootstrap. Log API cost.

### E7 — Distribution shift and adaptive conformal inference
Using the temporal split (§8.4), or leave-one-model-out if dates are unreliable:
- Static CQR 90% interval (calibrated on the pre-shift calibration set).
- ACI: process test runs in date order; at each step use level `α_t` (start α = 0.1), compute the interval from the CQR scores at level `1 − α_t`, observe `err_t = 1{y_t outside}`, update `α_{t+1} = α_t + γ(α − err_t)` with `γ = 0.005`; clip `α_t` to [0.001, 0.5].
- Report overall coverage, and rolling coverage over windows of 500 runs (figure), for static vs ACI. Also report interval width.

### E8 — Budget-policy replay (decision value)
For the best S1 method with CQR, and for P1: set a per-run cap at the predicted upper quantile `q_τ` for `τ ∈ {0.8, 0.9, 0.95, 0.99}`. Replay each test run's turns from `turns.parquet`: if cumulative cost exceeds the cap, the run is killed at the end of that turn. Outcomes: total cost saved (%), share of runs killed, share of **successful** runs killed (successes lost), and cost per success. Figure: cost saved vs successes lost (the efficient frontier, Almgren–Chriss style), one curve per method. A good forecaster dominates P1 (more savings at the same success loss).

---

## 11. Robustness and threats to validity (the report must discuss each)

1. **Visible vs billed cost.** Hidden reasoning tokens are unobserved; results describe visible-transcript cost. Report per-model results and a sensitivity excluding models flagged in §4.4.
2. **Linear-history assumption.** Violated by summarization; sensitivity in E3.
3. **Canonical tokenizer.** Per-model multiplicative bias is absorbed by model features; report the tokenizer-agreement table.
4. **Pricing ratios.** Results shown for three cost definitions.
5. **Selection.** Cell sampling is stratified by source; report sample composition vs the full dataset's metadata counts.
6. **Training-data provenance.** AgentTrove was assembled as training data; it may be filtered (e.g. toward successful or shorter traces). If E0 finds sources where every run succeeds, flag them; selection on outcome truncates the cost distribution.
7. **Leakage.** Every learned transformation fit on training folds only; task-grouped splits; history features from disjoint runs. Add a test asserting no `task_id` overlap between train and test in every fold.
8. **Multiple comparisons.** Holm correction within each experiment; hypotheses H1–H6 pre-registered.

---

## 12. Report specification (`results/REPORT.md`)

Sections, in order:
1. **Summary** — 5–8 sentences answering the core question, with the headline numbers (variance shares, best method's relative CRPS gain over P1, online CRPS reduction at k = 4, budget-replay savings at fixed success loss).
2. **Hypotheses** — table: H1–H6, verdict (supported / not supported / inconclusive), key statistic with CI.
3. **Data** — T0 exclusion flow; sample composition; measurement-validity findings from E0.
4. **Stylized facts** (E1) — figures and the tail/bimodality/scaling table.
5. **Variance decomposition** (E2) — level and spread share tables with CIs; spread-transfer matrix (heatmap); R² of spread from text.
6. **Cold pre-run results** (E3) — main table: rows = methods; columns = task-weighted QCRPS (CI), run-weighted QCRPS, relative gain vs P1, pinball@0.5/0.9/0.99, Spearman, MAE, sharpness, coverage 80/90/95, PIT KS, DQ p-value (τ = 0.9). Best value per column in bold. PIT histogram grid figure. Sensitivity and ablation tables.
7. **Warm pre-run results** (E4) — table including the oracle and the gap-closed column.
8. **Online results** (E5) — CRPS-vs-k figure (two populations), coverage-vs-k.
9. **Capacity** (E6) — table, if run.
10. **Shift** (E7) — rolling-coverage figure, summary table.
11. **Decision value** (E8) — frontier figure and table.
12. **Threats to validity** — §11 items with what was found.
13. **Reproducibility** — config hash, package versions, seeds, run times, `DEVIATIONS.md` summary.

Figures: PNG, 150 dpi, labeled axes with units, colorblind-safe palette, consistent method colors across all figures.

---

## 13. Implementation order and checkpoints

1. Scaffold the repository, `config.yaml`, unit tests for §5 and §9.1 (QCRPS of a Gaussian forecast vs the closed-form Gaussian CRPS within 1%; quantile monotonicity; PIT uniformity on simulated calibrated forecasts; Kupiec LR on a known example).
2. E0. **Checkpoint:** summarize E0 findings to the user, including any stop condition in E0, before continuing.
3. Build `runs.parquet`, `turns.parquet`, `splits.parquet`.
4. E1, E2.
5. Predictors P0, P1 → full evaluation pipeline end to end on them (debug the metrics on the simplest methods first).
6. P2–P6, CQR → E3.
7. E4, E5, E7, E8.
8. E6 if enabled.
9. Report.

At each experiment's end, write its log file and a one-paragraph summary. If any experiment fails, record the failure in `DEVIATIONS.md` and continue with the next; do not silently drop it from the report.

---

## 14. Constants (`config.yaml` defaults)

| Name | Value |
|---|---|
| SEED | 20261101 |
| N_RUNS_MAX | 300,000 |
| N_RUNS_CELL_CAP | 32 |
| N_MIN_CELL | 2 |
| C_MSG | 4 |
| λ_c, λ_o (primary) | 0.1, 4 |
| Quantile grid | τ_j = j/100, j = 1..99 |
| P3 levels | 0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99 |
| P1 shrinkage n0 | 30 |
| EVT threshold, min exceedances | 0.90 quantile, 200 |
| MDN | K = 3, 2 × 128, dropout 0.1, lr 1e-3, wd 1e-4, batch 512, patience 10 |
| P5 Monte Carlo draws | 2000 |
| Online k grid | 1, 2, 4, 8, 16, 32 |
| CV | 5-fold StratifiedGroupKFold by task, stratified by source |
| Inner split | 70 / 15 / 15 (fit / validation / calibration) by task |
| Bootstrap | B = 1000 (metrics), 200 (mixed-model refits) |
| ACI | α = 0.1, γ = 0.005, window 500 |
| Hyperparameter search | 20 configs, fold-1 validation |

---

## 15. References
- Bai et al. (2026). How Do AI Agents Spend Your Money? arXiv:2604.22750.
- Ouyang et al. (2026). TokenCast: Forecasting Token Consumption During LLM Agent Execution. arXiv:2609.35760.
- Corsi (2009). A Simple Approximate Long-Memory Model of Realized Volatility. J. Financial Econometrics 7(2).
- Engle & Manganelli (2004). CAViaR. J. Business & Economic Statistics 22(4).
- McNeil & Frey (2000). Estimation of tail-related risk measures for heteroscedastic financial time series. J. Empirical Finance 7.
- Engle & Russell (1998). Autoregressive Conditional Duration. Econometrica 66(5).
- Hamilton (1989). A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle. Econometrica 57(2).
- Diebold, Gunther & Tay (1998). Evaluating Density Forecasts. International Economic Review 39(4).
- Christoffersen (1998). Evaluating Interval Forecasts. International Economic Review 39(4).
- Kupiec (1995). Techniques for Verifying the Accuracy of Risk Measurement Models. J. Derivatives 3(2).
- Gneiting & Raftery (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. JASA 102(477).
- Romano, Patterson & Candès (2019). Conformalized Quantile Regression. NeurIPS.
- Gibbs & Candès (2021). Adaptive Conformal Inference Under Distribution Shift. NeurIPS.
- Chernozhukov, Fernández-Val & Galichon (2010). Quantile and Probability Curves Without Crossing. Econometrica 78(3).
- Almgren & Chriss (2000). Optimal Execution of Portfolio Transactions. J. Risk 3.

---

## Appendix A — Expected shape of the outputs

Values below are placeholders (`…`) showing format only. Do not copy them; fill every cell from your own results.

### A1. Hypothesis table (REPORT.md §2)

| Hypothesis | Verdict | Key statistic (95% CI) | Experiment |
|---|---|---|---|
| H1 task-driven spread | supported / not supported / inconclusive | task share − model share of log-spread variance = … (…, …) | E2 |
| H2 text carries signal | … | relative CRPS gain of best method vs P1 = …% (…, …) | E3 |
| H3 shape matters | … | PIT KS: P2 = …, best non-Gaussian = … ; difference (…, …) | E3 |
| H4 online gain | … | CRPS at k = 4 / CRPS at k = 0 = … (…, …); k at 50% = … | E5 |
| H5 capacity not the bottleneck | … or skipped | CRPS small − large = … (…, …) | E6 |
| H6 history beats text | … | relative gain (history) − relative gain (text) = … (…, …) | E3, E4 |

### A2. Variance decomposition (E2)

| Component | Level share of Var(log Y) | Spread share of Var(log SD), noise-corrected |
|---|---|---|
| Model (fixed effects) | … (…, …) | … (…, …) |
| Task | … (…, …) | … (…, …) |
| Task × model | … (…, …) | — |
| Run noise / residual | … (…, …) | … (…, …) |
| n tasks / cells / runs | … / … / … | … / … |

### A3. Main cold pre-run table (E3)

| Method | QCRPS task-wt (CI) | QCRPS run-wt | Gain vs P1 | Pinball .5 / .9 / .99 | Spearman | Cov 80 / 90 / 95 | Width 10–90 | PIT KS | DQ p (τ=.9) |
|---|---|---|---|---|---|---|---|---|---|
| P0 global | … | … | … | … / … / … | … | … / … / … | … | … | … |
| P1 source × model | … | … | 0 | … | … | … | … | … | … |
| P2 hetero log-normal | … | … | … | … | … | … | … | … | … |
| P3 quantile GBM | … | … | … | … | … | … | … | … | … |
| P3-EVT | … | … | … | … | … | … | … | … | … |
| P4 MDN (K = 3) | … | … | … | … | … | … | … | … | … |
| P5 structural | … | … | … | … | … | … | … | … | … |
| P6 linear | … | … | … | … | … | … | … | … | … |
| CQR(P3-EVT) | … | … | … | … | … | … | … | … | … |

### A4. Online table (E5), both populations

| k | Eligible runs | O1 QCRPS | O2 QCRPS | O2-noregime QCRPS | Panel (K > 32) O2 QCRPS | O2 cov 90 |
|---|---|---|---|---|---|---|
| 0 (best S1) | … | … | … | … | … | … |
| 1 | … | … | … | … | … | … |
| … | … | … | … | … | … | … |
| 32 | … | … | … | … | … | … |

### A5. Budget replay (E8)

| Method | Cap at τ | Cost saved % | Runs killed % | Successes lost % | Cost per success (ref. units) |
|---|---|---|---|---|---|
| P1 | 0.90 | … | … | … | … |
| Best + CQR | 0.90 | … | … | … | … |
| … | … | … | … | … | … |

### A6. Required figures

| File | Content |
|---|---|
| `fig_e1_cost_by_model.png` | Distribution of log Y per model |
| `fig_e1_scaling.png` | log Y vs log K with fitted slopes per model |
| `fig_e2_spread_transfer.png` | Heatmap of disattenuated spread correlations between models |
| `fig_e3_pit_grid.png` | PIT histograms, one panel per method |
| `fig_e3_reliability.png` | Nominal vs empirical coverage, one line per method |
| `fig_e4_gap_closed.png` | CRPS of methods between P1 and the cell oracle |
| `fig_e5_crps_vs_k.png` | QCRPS vs k, both populations, with CIs |
| `fig_e7_rolling_coverage.png` | Rolling coverage, static CQR vs ACI |
| `fig_e8_frontier.png` | Cost saved vs successes lost, per method |
