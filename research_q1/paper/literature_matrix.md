# Closest literature and novelty stress test

Search checked on 2026-10-22 against publisher/proceedings pages. No citation
below is intentionally left unverified. This is a focused closest-work search,
not a PRISMA systematic review.

| Work | Established result | What PI-MCS must add empirically | Overlap risk |
| --- | --- | --- | --- |
| Sun, Spinielli & Strohmeier (2026), [PRC dataset](https://doi.org/10.59490/joas.2026.8750) | ACARS fuel-on-board differences aligned with OpenSky trajectories, April–October 2025; 15,761 flights, 193,275 intervals, 27 types. | Reliability-aware decision layer with leakage-safe shift evaluation. | Labels are noisy, anonymized and selected by receiver availability; one year cannot support a multi-year claim. |
| Grigoriou et al. (2026), [physics-guided gradient boosting](https://doi.org/10.59490/joas.2026.8764) | OpenAP, weather/context and phase features combined with boosting on the same challenge. | Calibrated physics residuals, conditional interval evaluation and risk-budget selection—not merely physics features. | **High:** physics+ML itself is not novel and must not be claimed. |
| Kačar & Ćulibrk (2026), [trajectory and meteorological model](https://doi.org/10.59490/joas.2026.8766) | ERA5/OpenAP-like energy and phase features, mass estimation and ML on PRC data. | Shift-aware uncertainty and a constrained downstream selection objective. | **High:** energy-balance mass/fuel features already exist in this application. |
| Stawiarska (2026), [data quality and predictive reliability](https://doi.org/10.3390/en19204797) | Missingness/noise/shift sensitivity, synthetic studies and an external PRC subset. | Full released-benchmark conditional coverage and exact operational trade-off comparisons with honest external confirmation. | **High:** robustness and uncertainty on this benchmark are already studied. |
| Jarry et al. (2025), [unseen-aircraft generalization](https://doi.org/10.1016/j.trc.2025.105143) | Domain-generalized neural fuel-flow models across 101 aircraft types, including 37 unseen types. | Evaluate full estimator and selector under aircraft-family shift, not only a calibrator. | **High:** unseen-aircraft generalization is not new. |
| Romano, Patterson & Candès (2019), [CQR](https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html) | Heteroscedastic conformalized quantile regression with finite-sample marginal coverage under exchangeability. | Compare PI-MCS width at matched coverage; no new CQR theorem. | CQR is a mandatory interval baseline. |
| Tibshirani et al. (2019), [weighted conformal](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html) | Importance-weighted conformal prediction under covariate shift when likelihood ratios are correct. | Measure domain-classifier AUC and sensitivity to estimated weights in aviation shifts. | Estimated weights weaken the oracle guarantee and must be labeled approximate. |
| Gibbs & Candès (2021/2024), [adaptive conformal](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html) | Online adaptation of miscoverage under distribution drift. | Chronological aviation evaluation with delayed labels and fixed update timing. | ACI is a baseline/variant, not a contribution. |
| Gibbs, Cherian & Candès (2023), [conditional guarantees](https://arxiv.org/abs/2305.12616) | Coverage over specified group/shift classes and limits of universal conditional coverage. | Pre-specify aircraft×phase groups and report sparse-cell fallback. | Mondrian grouping alone is not novel and does not guarantee arbitrary conditional coverage. |
| Angelopoulos et al. (2024), [conformal risk control](https://openreview.net/forum?id=33XGfHLtZg) | Distribution-free control of expected monotone losses. | Use a valid risk-control baseline and distinguish it from empirical post-selection coverage. | Plain selection after viewing scores can invalidate marginal-coverage interpretations. |
| Jin & Candès (2023), [selection by conformal p-values](https://arxiv.org/abs/2210.01408) | Model-free candidate selection with FDR control under stated exchangeability/dependence conditions. | Translate the selected item/error event precisely and compare with risk-budget optimization. | Ordinary BH is not automatically valid for dependent conformal p-values; use the paper's conditions/procedure. |
| Jin & Candès (2025), [weighted conformal selection](https://doi.org/10.1093/biomet/asaf066) | Weighted conformal p-values and selection under covariate shift. | Aviation-specific utility/risk constraints beyond FDR, with guarantees kept separate. | Shift-aware conformal selection already exists. |
| Shah et al. (2022), [selective regression fairness](https://proceedings.mlr.press/v162/shah22a.html) | Selective regression can magnify subgroup disparities; proposes fairness criteria. | Quantify aircraft/duration access, utility and error at matched coverage. | Subgroup-aware abstention is established; application evidence must carry the contribution. |
| Bertsimas & Sim (2004), [price of robustness](https://doi.org/10.1287/opre.1030.0065) | Tractable budgeted uncertainty for LP/MIP, including knapsack-like problems. | Calibrate risk-deviation sets and measure the aviation utility/violation trade-off. | Robust ILP formulation is established optimization, not a new theorem. |

## Novelty decision

Drop these claims:

- “first physics-informed aircraft fuel estimator”;
- “new Mondrian conformal method”;
- “new robust optimization method”;
- “guaranteed selected-set coverage” without a theorem matching the actual
  selection rule and exchangeability/shift conditions.

The defensible contribution to test is narrower: **an aviation decision system
that combines training-only physics-residual prediction, hierarchical
aircraft×phase calibration, measured shift diagnostics and exact/robust
risk-budget selection, evaluated jointly for accuracy, conditional coverage,
utility and representation under grouped temporal/route/aircraft shifts.** The
combination is a methodological integration; publication value depends on
real-data effect sizes and negative-result reporting.

