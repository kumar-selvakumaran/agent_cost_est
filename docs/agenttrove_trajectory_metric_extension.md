# Extension 1 — Trajectory-Integrated Evaluation of Online Cost Forecasts

Version 1.0 · 2026-11-01 · Author: Kumar (with Claude)
Extends: **AgentTrove Token-Cost Prediction Benchmark — Implementation Specification, v1.0** (the "base spec"). Section numbers prefixed with "B" refer to the base spec (for example §B5.2).

This document is self-contained as a brief and as a specification, but it is an *extension*: it assumes the base spec has been implemented and run in the same repository. Everything the base spec says about rules (MUST/SHOULD, conservative choices under ambiguity, `DEVIATIONS.md`, seeds, leakage, things you must not do) applies here unchanged.

---

## Part A — Brief for the implementing agent

### A.1 Your task
Add a new evaluation to the existing benchmark: score online cost forecasts **along the whole trajectory of each run**, not only at a few checkpoints. Concretely:
1. Make every online predictor issue a forecast **immediately before every model call** of every test run (§3).
2. Implement the trajectory metrics defined in §4: one primary, proper score (ITS), and the author's original "area between curves" metric (NTA) as a secondary, descriptive metric.
3. Run experiment E9 (§6) and write its section into `results/REPORT.md` (§7), plus a standalone `results/logs/e9.md`.

### A.2 Why this matters (the intent)
The base spec scores online forecasts only after k ∈ {1, 2, 4, 8, 16, 32} turns. That misses what matters in practice: a forecaster used to manage a running agent must be right *throughout* the run, and especially early, when decisions (stop, continue, switch model) are cheap to act on. The author's idea is to compare two curves over the course of a run:
- the **true remaining cost** — total cost minus the cost incurred so far, and
- the **forecast remaining cost** at the same point,

and to measure the **area between them** from the first token to the last. A small area means the forecaster converged early and stayed right. This extension turns that idea into a rigorous metric, keeps the original as a readable descriptive quantity, and adds a distributional, proper version so the metric rewards calibration and cannot be gamed.

### A.3 Prerequisites (check before starting)
The following outputs of the base spec MUST exist. If any is missing, finish that part of the base spec first, or stop and tell the user.
- `data_cache/turns.parquet` with per-turn `I_k, O_k, Cch_k, U_k` (§B5.5).
- `data_cache/runs.parquet`, `data_cache/splits.parquet`.
- Fitted online predictors O1, O2 and O2-noregime, and the best cold pre-run (S1) predictor identified in E3, all with the interface of §B7.1.
- `results/DEVIATIONS.md` — read it first. If the base run deviated from the base spec in a way that affects this extension (for example a different target definition or k grid), adapt this extension consistently and log the adaptation.

### A.4 Steps

| Step | What to do | Exit criterion |
|---|---|---|
| 1. Read | Read this document and `results/DEVIATIONS.md` in full. | — |
| 2. Tests | Write the unit tests of §8. | All pass. |
| 3. Decision-point forecasts | Implement §3: forecasts before every call, for every method in §5. | `results/predictions/e9_decision_points.parquet` written (schema in §3.4). |
| 4. Metrics | Implement ITS, its variants, and NTA (§4). | Unit tests pass; values computed for all methods. |
| 5. Inference | Bootstrap CIs and paired comparisons (§4.6). | Tables written. |
| 6. Figures | Produce the figures of §7.2. | Figures written. |
| 7. Report | Add the E9 section to `REPORT.md` (§7.1); add H7–H9 to the hypothesis table. | Definition of done (§A.6) met. |

### A.5 Communication
When E9 is complete, send the user: the main E9 table, the hypothesis verdicts for H7–H9, and a 3-sentence interpretation. No intermediate checkpoint is required unless a prerequisite is missing or a sanity check (§A.7) fails.

### A.6 Definition of done
- Every method in §5 has a forecast at every decision point of every test run (or a logged, justified exclusion).
- ITS (primary) and all variants and NTA (secondary) are reported with task-cluster bootstrap CIs.
- H7–H9 have verdicts in the report's hypothesis table.
- All figures in §7.2 exist.
- All sanity checks in §A.7 pass or are explained in the report.
- `python -m src.run_all --config config.yaml --only e9` regenerates every E9 output.

### A.7 Sanity checks
1. For the **static** method (§5), ITS equals the pooled weighted mean of the pre-run QCRPS (it never updates). Verify numerically.
2. NTA of a perfect point forecaster (median = true Y at every decision point) is exactly 0 (unit test).
3. With unit weights and a single decision point per run, ITS reduces to the base spec's QCRPS at k = 0 (unit test).
4. ITS of O2 at decision points with k ∈ {1, 2, 4, 8, 16, 32} agrees with E5's per-k QCRPS within the bootstrap CI (it is computed from the same forecasts if O2 is unchanged).
5. Rows in `e9_decision_points.parquet` = Σ over test runs of K (one forecast per call).

### A.8 Things you must not do
Everything in the base spec's Part A §A.9 applies. In addition:
- Do not use NTA (or any metric flagged improper in §4.5) for model selection, hyperparameter tuning, early stopping or recalibration. Use only ITS or the base spec's QCRPS for those.
- Do not use any information from turn k + 1 or later in a forecast issued before call k + 1.
- Do not select example trajectories for figures by looking at errors; use the seeded procedure in §7.2.

---

## 1. Notation

All cost quantities come from §B5.2–§B5.3 and are computed with the same canonical tokenizer and the primary reference-cost weights (λ_c = 0.1, λ_o = 4).

For a run with K model calls (turns), k = 1..K:
- Per-call reference cost: `c_k = U_k + λ_c·Cch_k + λ_o·O_k`.
- Per-call tokens processed: `n_k = I_k + O_k`.
- Cumulative cost after k calls: `C_k = Σ_{j ≤ k} c_j`, with `C_0 = 0`. Total cost `Y = C_K`.
- Cumulative tokens after k calls: `N_k = Σ_{j ≤ k} n_j`, with `N_0 = 0`. Total tokens `N = N_K`.
- True remaining cost after k calls: `R_k = Y − C_k`. Note `R_K = 0` and `R_k ≥ c_{k+1} > 0` for k < K.
- Log total cost: `y = log Y`.

## 2. The trajectory and the two curves

### 2.1 The token clock
The horizontal axis is the **token clock** t ∈ [0, N]: the number of tokens processed so far in the run, as the author specified ("after T tokens of an N-token trajectory"). A normalized clock `t̃ = t / N ∈ [0, 1]` is used for plots and for NTA only.

Within call k (t ∈ (N_{k−1}, N_k]), cost is assumed to accrue linearly in tokens: the true remaining cost is
`R(t) = Y − C_{k−1} − c_k · (t − N_{k−1}) / n_k`.
`R(t)` is continuous, piecewise linear, decreasing from `R(0) = Y` to `R(N) = 0`.

### 2.2 Decision points
Forecasts are issued at **decision points**: immediately before each model call. Decision point d_k is "before call k + 1", for k = 0..K − 1. At d_k the forecaster knows:
- all pre-run information (§B6.1), and
- everything about calls 1..k and the environment's responses to them (§B6.3 prefix features, computed on turns ≤ k), including `C_k`, `N_k`, and the exact input size of the upcoming call `I_{k+1}` (the full history about to be sent, which is known before the call is made).

There is no decision point after call K: the run has ended. A forecast issued at d_k is held until d_{k+1}; over the token interval (N_k, N_{k+1}] the forecast is a step function in information, while the true curve keeps decreasing.

### 2.3 Forecast curve and the key equivalence
A forecaster at d_k outputs 99 quantiles `q^{(k)}_1 ≤ … ≤ q^{(k)}_99` of **log total cost** y (the base spec's representation, §B7.1). Its implied forecast of remaining cost at clock t ∈ (N_k, N_{k+1}] is obtained by subtracting the cost already accrued, which the system observes:
`R̂_j(t) = exp(q^{(k)}_j) − C(t)`, where `C(t) = Y − R(t)` is the observed cost so far.

Because the subtracted quantity `C(t)` is observed by both the forecast and the truth:

**Equivalence.** `R̂(t) − R(t) = Ŷ^{(k)} − Y` for any point summary (and quantile by quantile for the distribution). The error of a remaining-cost forecast equals the error of the total-cost forecast issued at the latest decision point.

Consequences, which the implementation MUST respect:
1. The area between the forecast and true remaining-cost curves equals the area between the total-cost forecast and the constant Y. Both are computed from total-cost forecasts; no separate remaining-cost model is needed.
2. Forecasting remaining cost on a **log** scale is *not* equivalent (log R ≠ log Y − constant). The log-remaining variant in §4.3 is therefore a genuinely different metric, emphasizing relative accuracy late in the run.
3. Remaining-cost quantiles must be clipped at zero: `q^{R}_j = max(exp(q^{(k)}_j) − C_k, 0)`; the forecaster knows `R ≥ 0`. Any forecast quantile with `exp(q^{(k)}_j) < C_k` is a logical error (the forecaster predicted a total below the cost already spent). Count these per method as a diagnostic (§4.7) and enforce `q^{(k)}_j ≥ log C_k` before scoring by replacing violating quantiles with `log C_k` (then re-sort).

## 3. Producing forecasts at every decision point

### 3.1 Coverage of decision points
Every method in §5 MUST produce a forecast at every d_k, k = 0..K − 1, of every test run in the base spec's online test folds (§B8.1). The forecast at d_0 is the pre-run forecast.

### 3.2 Changes to O2 (prefix GBM)
The base spec trains one O2 model per k in {1, 2, 4, 8, 16, 32}. For this extension, additionally train **O2-pooled**: a single quantile LightGBM per quantile level (§B7.5 levels) on rows (run, k) for k = 1..min(K − 1, 64), with `k` and `log k` as features alongside all prefix and pre-run features.
- Row weights: each training run contributes total weight 1, spread equally over its rows (weight `1 / (number of rows of that run)`), so long runs do not dominate.
- For decision points with k > 64, compute the prefix features at the actual k but set the `k` feature to 64 (log k to log 64), and log how many forecasts this affects.
- Hyperparameters: reuse O2's chosen configuration from the base run; do not re-tune on test data.
- Use O2-pooled in E9 for "O2" and "O2-noregime" (the noregime variant drops the loop/stuck features of §B6.3, as in the base spec). Report in `DEVIATIONS.md`-style notes that E9's O2 is the pooled version; sanity check §A.7 item 4 uses the base per-k models.

### 3.3 Changes to O1 (structural)
O1 (§B7.11) works at any k. If the base implementation fit one hazard model per k in the grid, fit one additional pooled hazard model with `k` as a feature for decision points outside the grid (same pooling rule as §3.2). Use the per-k models where they exist and the pooled model elsewhere; record which was used in the predictions file.

### 3.4 Output file
`results/predictions/e9_decision_points.parquet`, one row per (method, run, k):
`method, run_id, task_id, model_id, source, fold, k, K, C_k, N_k, I_next (= I_{k+1}), q01 … q99, y, Y, N, success`.

## 4. Metrics

### 4.1 Requirement: propriety
A score is **proper** if a forecaster minimizes its expected score by reporting its true belief. Only proper scores may be used to compare methods. Two traps apply to trajectory metrics, and the definitions below avoid both:
- **Outcome-dependent normalization.** Dividing an error by the realized outcome (for example `|Ŷ − Y| / Y`) makes the score improper: it rewards systematic under-forecasting.
- **Outcome-dependent weights.** Weighting decision points by quantities not known when the forecast is issued — such as the share `n_{k+1} / N` of the *final* token count, or `1 / K` — makes the weighted score improper, because N and K are correlated with Y. A weighted sum of proper scores is proper when every weight is known at the moment its forecast is issued (is "predictable").

### 4.2 Primary metric: Integrated Trajectory Score (ITS)
For method m and run r, with QCRPS as in §B9.1 (on log total cost):
```
S_rk = QCRPS(q^{(k)}, y_r)        for k = 0..K_r − 1
w_rk = I_{r,k+1}                  (input tokens of the next call; known at d_k)
```
Pooled over the evaluation set:
```
ITS = ( Σ_r Σ_k w_rk · S_rk ) / ( Σ_r Σ_k w_rk )
```
Interpretation: the token-weighted average log-scale CRPS over the whole trajectory — a discretized, distributional version of the author's area between curves, with the area measured on the token clock. Because `w_rk` is known at d_k and the denominator does not depend on the forecasts, ITS is proper in the large-sample sense: no method can improve its expected ITS by misreporting.

**Run-balanced ITS (ITS-rb).** The pooled ITS gives long runs more weight. As a co-primary view, compute the same ratio within each run (`ITS_r = Σ_k w_rk S_rk / Σ_k w_rk`) and average across runs, then across tasks (task-weighted, as in §B9.1). Caveat: the per-run normalization uses the realized number of decision points, so ITS-rb is not strictly proper; use it for reporting only, never for selection. If ITS and ITS-rb disagree on the ranking of two methods, report it explicitly.

### 4.3 Variants (secondary; all proper unless stated)
- **ITS-turn:** `w_rk = 1` (each decision point counts equally). Measures performance per decision rather than per token.
- **ITS-early:** `w_rk = I_{r,k+1}` for k ≤ 7 (the first 8 decision points), 0 otherwise. Emphasizes the early phase where interventions are cheap.
- **ITS-level:** QCRPS on the natural cost scale: `S_rk = QCRPS(exp(q^{(k)}), Y_r)`, reported in reference units. Proper, but dominated by expensive runs; report for completeness.
- **ITS-logR (relative remaining):** `S_rk = QCRPS(log q^{R,(k)}, log R_rk)` with `q^{R}` from §2.3 (quantiles at 0 replaced by `log(1)`; floor R at 1 reference unit). Proper; emphasizes relative accuracy of what is left, so it weights late-run accuracy much more than ITS. It answers a different question ("how good is the estimate of what remains?") and need not rank methods the same way.

### 4.4 The author's metric: Normalized Trajectory Area (NTA)
The literal area between the forecast and true remaining-cost curves, with both axes normalized so runs of different length and cost are comparable. Using the point forecast `Ŷ^{(k)} = exp(q^{(k)}_{50})` (the median) and the equivalence of §2.3:
```
A_r   = ∫_0^N |R̂(t) − R(t)| dt  =  Σ_{k=0}^{K−1} n_{k+1} · |Ŷ^{(k)} − Y|          (units: tokens × reference cost)
NTA_r = A_r / (N · Y)            =  Σ_{k=0}^{K−1} (n_{k+1} / N) · |Ŷ^{(k)} − Y| / Y   (dimensionless)
```
`NTA_r = 0` for a forecaster that is exactly right from the first token; `NTA_r = 0.3` means that, averaged over the run's tokens, the remaining-cost forecast was off by 30% of the run's total cost.

Report NTA as the run-level mean, then task-weighted mean, with CIs; also its median, because it is heavy-tailed.

**NTA is improper on both counts of §4.1** (normalized by the realized Y, weighted by the realized N). It is reported because it is the intuitive, plain-language quantity, and its curves are what the figures show. It MUST NOT be used for selection or tuning. If NTA and ITS rank two methods differently, say so and explain that ITS is the authoritative comparison.

### 4.5 Calibration along the trajectory
- **Pooled PIT:** PIT values (§B9.3) of y under each `q^{(k)}`, pooled over all decision points with weights `w_rk`. Report the weighted KS distance from uniform.
- **Coverage by progress:** coverage of the central 80% interval, binned by the normalized clock position of the decision point (`N_k / N`) into deciles. This uses the realized N for binning only, which is fine for a diagnostic (no method is ranked by it). Figure in §7.2.
- **Sharpness by progress:** mean width `q_90 − q_10` per progress decile.

### 4.6 Statistical inference
Task-cluster bootstrap (B = 1000, seeded as in §B2.3): resample tasks, carry all their runs and all decision points. Report 95% CIs for every metric, and paired differences with two-sided p-values for the comparisons named in §6. Apply Holm–Bonferroni across E9's comparisons.

### 4.7 Diagnostics (reported, not ranked)
- **Infeasible quantiles:** share of decision points where any `q^{(k)}_j < log C_k` before correction (§2.3), per method.
- **Time to confidence (descriptive):** for each run, the smallest normalized clock position `N_k / N` such that the forecast median stays within ±25% of Y (`|q^{(k')}_{50} − y| ≤ log 1.25`) for every later decision point k' ≥ k. Report the median across runs per method, and the share of runs that never reach it. This uses the realized outcome and is descriptive only.

## 5. Methods compared in E9

| Method | Forecast at d_k | Purpose |
|---|---|---|
| Static | The best S1 pre-run forecast, never updated (`q^{(k)} = q^{(0)}` for all k) | The "no updating" reference |
| Static-truncated | The best S1 forecast with quantiles clipped at `log C_k` (§2.3) at each d_k | Isolates the gain from simply knowing the cost so far is a floor |
| O1 | Structural online model (§B7.11, §3.3) | Mechanical extrapolation; uncertainty in remaining turns only |
| O2 | Prefix quantile GBM, pooled (§3.2) | Learned nowcasting |
| O2-noregime | O2 without loop/stuck features | Value of detecting the stuck regime |

At d_0 every method uses the same pre-run forecast, except that O1 and O2 MAY use their own k = 0 behavior if the base implementation defines one; if so, state it.

## 6. Experiment E9 — Trajectory-integrated evaluation

**Population.** All test runs of the base spec's online folds (§B8.1), all decision points. Report results overall, by model, and by source family.

**Pre-registered hypotheses** (registered in this document on 2026-11-01; add them to the base report's hypothesis table):
- **H7 (updating pays).** O2's ITS is lower than Static-truncated's ITS (paired bootstrap difference CI excludes 0).
- **H8 (stuck detection pays).** O2's ITS is lower than O2-noregime's ITS, and the gap is larger under ITS-early than under ITS (compare the relative gaps with bootstrap CIs).
- **H9 (learned beats mechanical).** O2's ITS is lower than O1's ITS.

**Comparisons (Holm-corrected):** O2 vs Static-truncated; O2 vs O2-noregime (ITS and ITS-early); O2 vs O1; O1 vs Static-truncated; Static-truncated vs Static.

**Sensitivity:** repeat ITS for (a) runs flagged for summarization excluded (§B4.4); (b) the secondary cost targets `Y_tok` and `Y_hi` (§B5.3) if the base run computed them; (c) runs with K ≥ 5 only (where the trajectory has room to update).

## 7. Outputs

### 7.1 Report section ("Trajectory-integrated evaluation (E9)")
Insert after the base report's online section. Contents:
1. Two sentences defining ITS and NTA in plain language.
2. Main table (format below).
3. Hypotheses H7–H9 added to the report's hypothesis table, with verdicts.
4. Figures of §7.2 with one-sentence captions.
5. A short paragraph on any disagreement between ITS, ITS-rb, ITS-logR and NTA rankings.

Main table:

| Method | ITS (CI) | ITS-rb | ITS-early | ITS-turn | ITS-logR | NTA mean (CI) | NTA median | PIT KS | Infeasible % | Time to confidence (median) |
|---|---|---|---|---|---|---|---|---|---|---|
| Static | … | … | … | … | … | … | … | … | … | … |
| Static-truncated | … | … | … | … | … | … | … | … | 0 | … |
| O1 | … | … | … | … | … | … | … | … | … | … |
| O2 | … | … | … | … | … | … | … | … | … | … |
| O2-noregime | … | … | … | … | … | … | … | … | … | … |

### 7.2 Figures (PNG, same style rules as §B12)
| File | Content |
|---|---|
| `fig_e9_example_trajectories.png` | 6 example runs: true remaining cost R(t) (black) and each method's forecast remaining-cost median with 10–90% band, against the normalized token clock; the shaded area between the true curve and O2's median is NTA. Selection: with seed `SEED + 9`, draw one test run uniformly from each sextile of K. Never choose by error. |
| `fig_e9_error_vs_progress.png` | Mean QCRPS (log total) vs normalized clock decile, one line per method, with CIs. |
| `fig_e9_coverage_vs_progress.png` | Central 80% coverage vs normalized clock decile, per method, with the nominal line. |
| `fig_e9_width_vs_progress.png` | Mean 10–90 width vs normalized clock decile, per method. |
| `fig_e9_nta_distribution.png` | Distribution of NTA per run, per method (log-scaled axis if needed). |

### 7.3 Tables
`results/tables/e9_main.csv`, `e9_by_model.csv`, `e9_by_source_family.csv`, `e9_comparisons.csv` (difference, CI, raw p, Holm-adjusted p), `e9_sensitivity.csv`, `e9_diagnostics.csv`.

## 8. Unit tests (must pass before E9 runs)
1. **Area identity.** For a synthetic run (K = 4, known `I_k, O_k, c_k`) and a constant point forecast Ŷ, compute `A_r` by numerically integrating `|R̂(t) − R(t)|` on a fine grid over [0, N] (with the linear within-call accrual of §2.1) and check it equals `Σ n_{k+1} |Ŷ − Y|` to 1e-6 relative error.
2. **NTA zero.** A forecaster with `Ŷ^{(k)} = Y` at every k has NTA = 0.
3. **ITS reductions.** (a) Static's ITS equals the weighted mean of its d_0 QCRPS over runs with weights `Σ_k w_rk`; (b) with one decision point per run and unit weights, ITS equals mean QCRPS.
4. **Predictable weights.** A test asserting that the weight column used for ITS is computed only from turns ≤ k (construct a run, then change turn k + 2's tokens and verify w at d_k is unchanged).
5. **No look-ahead in features.** Changing any turn > k must not change the prefix features or the forecast at d_k (run on 50 random real runs with a perturbed future).
6. **Infeasibility correction.** After §2.3's correction, no quantile is below `log C_k`, and quantiles remain non-decreasing.
7. **Propriety sanity (simulation).** Simulate runs where the true conditional distribution of y at each d_k is known (e.g. y ~ N(μ_k, σ_k²) with σ_k shrinking in k). Verify that the true distribution has a lower expected ITS than (i) a version shrunk 20% toward the median and (ii) a version shifted down by 0.1, and that NTA can prefer the downward-shifted forecast (demonstrating why NTA is not used for ranking). Use 10,000 simulated runs.

## 9. Constants (add to `config.yaml` under `e9:`)

| Name | Value |
|---|---|
| Pooled O2 max k | 64 |
| ITS weight | input tokens of next call, `I_{k+1}` |
| ITS-early window | first 8 decision points (k ≤ 7) |
| Time-to-confidence tolerance | ±25% (`log 1.25`) |
| R floor for ITS-logR | 1 reference unit |
| Example-trajectory seed | `SEED + 9` |
| Bootstrap | B = 1000, task-clustered |

## 10. Rationale summary
- **Why integrate over the trajectory:** a forecaster that is right only at the end is useless for steering a run; integrating over the token clock rewards early convergence, exactly as the author intended.
- **Why distributions (CRPS) instead of absolute error:** CRPS reduces to absolute error for a point forecast, so ITS strictly generalizes the area idea while also rewarding honest uncertainty.
- **Why log total cost as the scored quantity:** it is scale-free across runs, consistent with the base spec, and — by the equivalence of §2.3 — scoring total cost at each decision point is the same as scoring remaining cost on the natural scale.
- **Why predictable weights:** they keep the metric proper, so it cannot be improved by systematically biased forecasts.
- **Why keep NTA:** it is the direct, plain-language version of the author's idea and the one the figures show; it is labeled descriptive because it is improper.
