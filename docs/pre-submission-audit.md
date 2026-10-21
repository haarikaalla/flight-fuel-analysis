# Pre-submission audit — 21 October 2026

**Verdict: the code and offline demo are reproducible research prototypes; the complete college submission is not ready.** This is a retrospective *eligibility and representation audit* for fuel estimates on a specific public benchmark, not a deployable live estimator or a new fuel-burn algorithm. The coverage rule and its 50-row aircraft cutoff were revised after seeing October results. September and October policy comparisons are exploratory. The final formatted report, presentation, project diary, guide approval and administrative details are absent from this repository.

## 1. Leakage and split design

| Check | Finding and action |
| --- | --- |
| Target into features | Feature extraction uses only `source == "adsb"` trajectory rows in the supplied half-open interval, duration and aircraft type; no fuel report, label or ACARS fuel-on-board field enters the design matrix. Label is used for fitting and post-selection evaluation. Verified code path, not a guarantee against every source-data artifact. |
| Preprocessing | `design` determines aircraft dummy columns from April–August training, aligns later periods to those columns, and uses no full-dataset scaler or target encoding. Type-rate fallback is calculated from training labels only. Risk model fits on early-September absolute residuals, not late-September/October residuals. |
| Flights and time | April–August training, September 1–9 residual ranking, September 10–18 calibration, September 19–27 validation, October exploratory evaluation. Whole flights remain in a block; 3 boundary flights/23 intervals are excluded. Train/rank/final flight IDs are checked for disjointness. A forward time split suits retrospective temporal shift better than random row k-fold. |
| Historical internal fit | Original Phase 4 `early_stopping="auto"` randomly withheld training *rows* internally, so intervals of a training flight could be on both sides. This did not expose September/October labels but violated strict within-fit flight separation. **Fixed here:** revised exploratory base fit explicitly disables internal early stopping; results regenerated. Historical JSON remains an honest record, not a comparable policy-only control. |
| Remaining overlap | Route and possibly aircraft overlap persists across months; no route-held-out, airframe-held-out or external-operator test. October contains 35,422 seen-route and 2,034 unseen-route intervals; their full-model MAEs are 81.15 and 174.77 kg, respectively. Descriptive and confounded. Supplied ACARS-defined start/end times would be unavailable in a live ADS-B-only service. |

## 2. Model and statistical checks

- The fixed aircraft-type median kg/min rate is a simple training-only baseline/fallback. The HistGradientBoostingRegressor and residual ranker are modest tabular models. Hyperparameters were fixed in earlier phases; there is **no systematic search or nested tuning**. Explain this as a limited computational baseline, not an optimal model. Do not compare the historical Phase 4 score with this revised model as a controlled policy-only result.
- September validation, all 7,619 intervals: MAE **96.07 kg**, RMSE **281.69 kg**, R² **0.872**, WAPE **22.43%**, median absolute error **33.69 kg**, 95th **372.75 kg**, 99th **1,008.65 kg**, signed bias **−19.56 kg**. Large tails and scale heterogeneity make R² an incomplete summary. The **41.88% MAPE** is diagnostic only; 102 labels below 10 kg make percentage errors unstable. The residual JSON and demo include duration strata and risk-score deciles. The score is a rank, not a calibrated prediction interval.
- At a 75.01% selection rate, September accepted MAE is **44.23 kg** for risk only and **54.02 kg** with a 50% group floor. Access for 15–60 min rises **0.60% → 50.12%**; B744 rises **1/66 → 33/66**, but its 33 accepted intervals have **609.90 kg MAE**. Labeled-fuel share rises **29.77% → 41.51%**. The fixed-mask flight-cluster bootstrap gives a descriptive 95% percentile range of **+4.74 to +15.27 kg** for the MAE cost and **+8.50 to +14.74 percentage points** for fuel-share gain. It does not refit models or recompute selection in resamples, and it is not a confirmatory confidence interval for a future batch.
- October policy results are exploratory: risk-only accepted MAE **40.49 kg**, floor **50.07 kg**, both at 75.00% acceptance; 15–60 min access **0.36% → 50.04%**; B744 **3/342 → 171/342**. Since the idea and cutoff followed an earlier inspection of October, calling this an untouched holdout would be false.
- Selective MAE is calculated only on served rows. It can improve just by omitting hard cases. The demo explicitly reports rejected and group errors plus *this sample's* observed-fuel share. That share requires labels and cannot be known for a new unlabeled batch. The floor guarantees representation only within the current batch, never accuracy.

## 3. Aviation validity and novelty

The target is a difference between reported fuel-on-board figures, **not measured fuel-flow**. There are no payload, takeoff weight, true route distance, winds, weather, independently standardized altitude/speed, fuel uplift corrections, or phase-specific aircraft physics. The raw publisher README says 5–60 minute intervals, while the released split features contain **76,723 intervals below five minutes**, including near-second records; this discrepancy is retained and flagged. Raw ADS-B altitude/speed values should not be given SI physical interpretations without a verified schema. The median-rate fallback and nonnegative clipping are pragmatic statistical choices, not a mass-balance law. No per-flight total, operational fuel planning or carbon emissions can be defended from this result.

Related trajectory-fuel and reliability work already exists. The defensible contribution is the **transparent, dataset-specific access/error tradeoff and offline audit interface**, with a negative finding that equal group access is expensive for difficult aircraft. It is an incremental student study; independent data and stronger policy comparisons would be needed for a novelty or generalization claim. See [literature](literature.md).

## 4. Implementation and submission readiness

The repository is modular (`data_audit`, archive downloader, feature extractor, baselines, historical protocol, eligibility analysis), uses verified archive checksums, a feature manifest, deterministic selection and focused unit/integration tests. It has an executable README, an explanatory notebook and a generated browser demo. The dashboard is a **pre-scored labeled batch simulation**, not a model server; its embedded labels are for evaluation and are never used to select rows. Error handling rejects invalid durations/scores and missing route metadata. This is reasonable research code but not production software: no CI, type-checking gate, fully independent replication, privacy/security review, or externally verified aviation units. The bundled result JSON records package versions and source checksums; recreate the generated demo after any model change.

The README and methods now state the architecture, pipeline, exact periods, core figures and limits. A college **report and slides are still missing**. The submission template also requires personal/guide fields and may require certificates and a dated diary. Never invent guide sign-off, diary entries, paper submission or measured environmental impact. Check current institutional requirements in [submission notes](submission-requirements.md).

## Prioritized completion ledger

| Priority | Action | Status |
| --- | --- | --- |
| P0 | Remove row-random internal validation; regenerate results/dashboard and align README/notebook | Done in this audit; verify with tests and notebook run. |
| P0 | Expose residual tails, R², WAPE/MAPE interpretation, route overlap, short intervals and conditional bootstrap limitations | Done in code/results/docs; verify artifact consistency. |
| P0 | Prepare final formatted report and slide deck with actual student/guide details, figure captions, references, diary and required forms | **Open; cannot claim submission-ready without these.** |
| P1 | Obtain a genuinely later, independent labeled batch or other operator; freeze policy and protocol beforehand | Open; necessary for a confirmatory test. |
| P1 | Run route-held-out and airframe-held-out sensitivity, stronger weather/weight-aware baselines if lawful data exist, and label-quality review | Open; necessary before transfer/operational claims. |
| P2 | Register a limited hyperparameter protocol, assess flight-level aggregate error and calibrate uncertainty on an independent block | Open research extension. |

### Viva answers to rehearse

1. **Why not random k-fold?** Adjacent intervals share a flight and time structure. The outer chronological, flight-disjoint blocks reduce direct dependence; route and operator overlap remain.
2. **Why is R² high while errors matter?** Fuel scale varies widely. RMSE 281.69 kg and p99 absolute error 1,008.65 kg reveal tails; subgroup access and error answer the actual research question.
3. **Is this a live fuel predictor?** No. The dataset supplies completed fuel-report interval boundaries and evaluation labels. The browser recalculates a selection over pre-scored benchmark rows.
4. **Is the floor better?** It improves access and labeled-fuel representation in these batches, at a measurable error cost. It cannot guarantee accurate estimates for B744 or future batches.
5. **What is new?** The narrow dataset-specific audit and interactive exposure of selection bias, not a novel estimator or established aviation physics.
