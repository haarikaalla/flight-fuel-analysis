# Phase 1 literature review and research gap

**Search snapshot:** 18 October 2026. This is a critical, focused review for the project proposal, not a systematic review or a claim that all relevant literature has been exhausted. Recheck new publications before submitting a journal manuscript.

## Selection and comparison method

The search focused on (1) aircraft fuel estimation from ADS-B and onboard observations, (2) the EUROCONTROL PRC 2025 dataset and challenge papers, and (3) predictive uncertainty with missing inputs and temporal change. We favored dataset authors, original research articles, and original ML-method papers. Papers below have different targets, data access, and evaluation protocols; reported errors should **not** be ranked against each other as if they were the same benchmark.

| Ref. | Data and method | Contribution that matters here | Limitation or distinction for this project |
| --- | --- | --- | --- |
| [1] Sun et al. (2026) | OpenSky ADS-B + airframes.io ACARS + flight metadata; derived interval fuel labels and challenge splits. | Public benchmark with 193,275 fuel intervals over 15,761 flights, 27 aircraft types, chronological periods, and real coverage gaps. | Fuel is the *difference* between noisy, sparse fuel-on-board reports; ambiguous units were checked with OpenAP. Dataset construction is not itself a model contribution. |
| [2] Grigoriou et al. (2026) | Physics-guided trajectory features, OpenAP, weather and XGBoost on the challenge. | Strong segment-level point-estimation comparator; reports held-out MAE 85.7 kg and RMSE 204.6 kg for its evaluation setup. | Its features and protocol differ from a deliberately ADS-B-only, no-external-performance-model setting. The reported score is not a guaranteed reproducible baseline for our narrower inputs; paper does not establish subgroup calibration under gap patterns. |
| [3] Kačar and Ćulibrk (2026) | ADS-B/ACARS trajectories, ERA5/METAR and aircraft-performance proxies; phase-specific physical features and ML. | Shows that adding weather and physically motivated features has already been investigated on the challenge. | Our scope is reliability and feature availability, not claiming novelty from weather or phase features. Compare methods only after aligning input budgets and splits. |
| [4] Stawiarska (2026) | Data-quality degradation, distribution shift, conformal uncertainty; mainly synthetic experiments with 686 observed fuel intervals from 79 flights in this challenge. | Direct overlap: missing data and uncertainty are *already studied*. Shows aggregate coverage can coexist with very wide intervals as data quality deteriorates. | A full-benchmark, flight-grouped, chronologically held-out study of natural ADS-B gap patterns and aircraft-group calibration could add evidence, but distinctiveness is a **hypothesis** pending deeper method comparison. |
| [5] Zhang et al. (2025) | ADS-B/ACARS trajectory dynamics and deep-learning model of interval fuel; independent 2023 data. | Strong precedent for high-resolution trajectory-based fuel estimation and a DL comparator. | Different data and validation make its percentage error non-comparable to PRC challenge RMSE; a new DNN architecture alone is unlikely to be novel. |
| [6] Jarry et al. (2025) | Deep-learning fuel-flow generalization across unseen aircraft types; BADA-based modeled fuel labels. | Demonstrates that aircraft-type shift matters, even with sophisticated models. | Its target uses simulated BADA fuel flow and a different unit/task; our study uses ACARS-derived interval burn and should separate seen-type subgroup analysis from a truly held-out-type experiment. |
| [7] Romano et al. (2019) | Conformalized quantile regression (CQR). | Model-agnostic, adaptive intervals under suitable exchangeability. | A method reference, not evidence that nominal coverage will hold across the PRC dataset's chronological distribution shift. |
| [8] Zaffran et al. (2023) | Conformal prediction with missing covariates. | Shows marginal coverage can hide undercoverage for specific missingness patterns; motivates conditional/mask-aware analysis. | Its assumptions and missing-feature setting do not automatically transfer to correlated gaps in an ADS-B time series. |
| [9] Tibshirani et al. (2019) | Weighted conformal prediction under covariate shift. | Gives a framework for studying distribution shift. | We will not claim finite-sample guarantees for chronological shift without checking the required likelihood-ratio and exchangeability assumptions. |

## Synthesis

**What is settled:** A meaningful, observed-derived fuel target and a large public benchmark now exist [1]. Competitive point prediction from rich trajectories, aircraft performance and weather has been demonstrated [2], [3], [5]. Generalization beyond familiar aircraft types is a known challenge [6]. Generic conformal prediction and missing-covariate theory already exist [7]–[9].

**What is not settled for this proposed deployment boundary:** The challenge papers emphasize accuracy using rich inputs. The recent data-quality study [4] already examines missingness and uncertainty, so “we add missingness and conformal intervals” is **not** a defensible novelty statement. Our narrower question is whether predictions based on **ADS-B-only trajectory observables plus aircraft metadata**, with no ACARS-derived predictor or proprietary performance tables, remain calibrated in realistic natural coverage gaps and across aircraft groups and later months on the full released benchmark. We must establish that this combined empirical protocol has not already been implemented elsewhere. A rigorous negative result or finding of unacceptably wide intervals would still be a valid academic result.

**Important qualifiers:** The PRC labels were quality controlled using a physics model; they are not direct engine-flow measurements [1]. The benchmark's start/end intervals are defined by ACARS report times, which makes the task retrospective and may not be available to an ordinary live ADS-B-only service. In our proposed research setting, those boundaries define *when* to estimate; they cannot be used to claim a deployable live predictor without a separate interval-definition mechanism. A random split of intervals would leak flight-specific information. Cross-period evaluation gives realism but invalidates automatic exchangeability-based coverage guarantees [7]–[9].

## Refined research question and hypotheses

**RQ:** On the released PRC 2025 benchmark, how do natural and controlled ADS-B observation gaps affect point error and 90% interval coverage for retrospective fuel-burn estimates across aircraft groups and time periods?

- **H1 (empirical):** A model given explicit coverage/gap features will have lower error under sparse trajectories than the same estimator without these features, under a fixed feature budget and matched evaluation. The null is no improvement.
- **H2 (empirical):** A single pooled 90% calibration will show uneven coverage across aircraft groups or gap severity on the later chronological test period. The null is no material difference after accounting for sample sizes.
- **H3 (conditional):** Group- or quality-aware calibration can improve the worst-group coverage/width tradeoff compared with pooled intervals. This must be assessed with enough calibration cases per group; there is no promised finite-sample conditional guarantee under time shift.

**Primary contribution if supported:** A reproducible, leakage-audited empirical reliability benchmark and a useful calibration improvement under a precisely stated ADS-B-only retrospective setting. We will describe H1–H3 as questions until tested. This is substantially more specific than “physics-informed XGBoost + anomaly detection + map.”

## Planned comparisons that follow from the literature

1. Constant fuel rate by aircraft type and interval duration; global fallback for rare types.
2. Tabular tree model with aircraft and duration; then the same model with ADS-B trajectory summaries; then with explicit coverage/gap features.
3. Pooled split-conformal residual intervals and CQR if quantile models are feasible; then stratified/group-aware calibration with documented minimum group sizes and fallback. Do not present theoretical coverage claims under chronological shift.
4. Natural gaps, contiguous simulated outages, and random thinning **as different mechanisms**. Report clean and degraded point error, 90% coverage, average width, worst-group coverage, and counts.
5. Aircraft groups must be defined before seeing final-test outcomes. An unseen-type experiment is separate from ordinary seen-type subgroup reporting.

## References

[1] J. Sun, E. Spinielli, and M. Strohmeier, “Aircraft Fuel Burn Estimation: The EUROCONTROL PRC 2025 Data Challenge,” *Journal of Open Aviation Science*, vol. 4, no. 3, 2026, doi: [10.59490/joas.2026.8750](https://doi.org/10.59490/joas.2026.8750). Dataset: [10.5281/zenodo.19184662](https://doi.org/10.5281/zenodo.19184662).

[2] Y. Grigoriou *et al.*, “A Physics-Guided Gradient Boosting Framework for Open-Source Aviation Fuel Estimation,” *Journal of Open Aviation Science*, vol. 4, no. 3, 2026, doi: [10.59490/joas.2026.8764](https://doi.org/10.59490/joas.2026.8764).

[3] R. Kačar and D. Ćulibrk, “Fuel Burn Prediction from Open Flight Trajectory and Meteorological Data for the PRC Data Challenge 2025,” *Journal of Open Aviation Science*, vol. 4, no. 3, 2026, doi: [10.59490/joas.2026.8766](https://doi.org/10.59490/joas.2026.8766).

[4] E. M. Stawiarska, “Data Quality and Predictive Reliability in AI-Enabled Aviation Energy Management: Physics-Informed Sensitivity Analysis with Open-Data Validation,” *Energies*, vol. 19, no. 20, art. 4797, 2026, doi: [10.3390/en19204797](https://doi.org/10.3390/en19204797).

[5] L. Zhang, L. Bian, C. Jiang, and L. Wu, “A comprehensive framework for estimating aircraft fuel consumption based on flight trajectories,” *Transportation Research Part E: Logistics and Transportation Review*, vol. 203, art. 104339, 2025, doi: [10.1016/j.tre.2025.104339](https://doi.org/10.1016/j.tre.2025.104339).

[6] G. Jarry, R. Dalmau, P. Very, and J. Sun, “On the generalization properties of deep learning for aircraft fuel flow estimation models,” *Transportation Research Part C: Emerging Technologies*, 2025, doi: [10.1016/j.trc.2025.105143](https://doi.org/10.1016/j.trc.2025.105143).

[7] Y. Romano, E. Patterson, and E. J. Candès, “Conformalized Quantile Regression,” *Advances in Neural Information Processing Systems*, vol. 32, 2019. [Publisher](https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html).

[8] M. Zaffran, A. Dieuleveut, J. Josse, and Y. Romano, “Conformal Prediction with Missing Values,” *Proceedings of the 40th International Conference on Machine Learning*, PMLR 202, pp. 40578–40604, 2023. [Publisher](https://proceedings.mlr.press/v202/zaffran23a.html).

[9] R. J. Tibshirani, R. F. Barber, E. J. Candès, and A. Ramdas, “Conformal Prediction Under Covariate Shift,” *Advances in Neural Information Processing Systems*, vol. 32, 2019. [Publisher](https://proceedings.neurips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html).
