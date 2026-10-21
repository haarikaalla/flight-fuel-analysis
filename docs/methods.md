# Method and interpretation

## Research question

For a batch of completed, ACARS-defined flight intervals, can a public-data fuel-estimate eligibility rule preserve access across duration and sufficiently represented aircraft types while showing the error cost of doing so? The intended user is an aviation researcher or dataset curator deciding whether a particular *sample* supports a comparison. This is not a live flight predictor or certified carbon accounting.

The reference is `fuel_kg`, a difference between reported fuel-on-board values, not a direct fuel-flow measurement. Aircraft type and interval start/end are supplied; ADS-B values are drawn only from recorded `source == "adsb"` positions inside `[start, end)`. The unverified raw altitude, speed and climb fields are used as statistical inputs, not presented as standardized physical quantities. All published files were checked against the dataset's MD5 values before feature extraction. The publisher's README describes limitations of ACARS-derived differences; its stated minimum duration disagrees with the released labels, so the analysis retains short intervals and reports them separately.

## Data and time boundaries

| Period | Role |
| --- | --- |
| April–August 2025 | Fit the ADS-B histogram gradient boosting fuel model and aircraft-type median kg/min rates. |
| September 1–9 | Fit a second histogram gradient boosting model on the fuel model's absolute errors. |
| September 10–18 | Display calibration-block behavior; the new batch floor does not learn a threshold here. |
| September 19–27 | Main interactive batch for the new exploratory eligibility audit. |
| October 2025 | Historical Phase 4 evaluation, and **exploratory** evaluation of the new rule. October results were already seen before designing it. |

Three flights spanning September block boundaries (23 intervals) are excluded rather than split. Train/rank/final flight IDs are disjoint. The original Phase 4 protocol and aggregates are retained to show what motivated the new question; its October scores were obtained before the access floor existed. The new October comparison is not a second untouched test.

## Estimation and policy

1. Estimate fuel with the April–August ADS-B model. For intervals with zero usable ADS-B positions, use duration multiplied by the training median kg/min for their aircraft type, or the global median for an unseen type. This fallback was fixed in Phase 4.
2. Rank expected absolute error with an early-September residual model using type, duration, ADS-B position and speed counts, maximum gap fraction, and predicted fuel. This is a **ranking score**, not a calibrated confidence bound.
3. For each input batch, serve the lowest-ranked intervals in each nonempty duration bin until at least the chosen floor is met. Then serve each aircraft type with at least 50 intervals in the batch (smallest groups first) until its floor is met. Finally fill the lowest-ranked remaining intervals until the overall target is met. Stable hashes of IDs break equal scores. If floors exceed the overall target, keep the floors and report the excess.
4. Compare with a zero-floor rule at the same nominal overall target. Because this is a batch policy, the decision for one interval can change when the batch changes. Neither policy uses labels to select intervals. The greedy order is deterministic but not a proof of globally minimum error under constraints. The 50-interval cutoff was revised during exploratory analysis because a 100-interval rule omitted B744 in September (66 rows); it is a service rule, not evidence of precise error measurement for small groups.

The dashboard recomputes the same rule in the browser on the 7,619 validation intervals. It contains labeled public benchmark records for evaluation, so one can see the error cost and the share of observed fuel represented. On an unlabeled batch, that fuel share is unavailable. The browser is a scenario audit of already scored benchmark rows; it does not fit a model or accept a new raw ADS-B flight.

## Measures and limits

Report actual interval acceptance, MAE on accepted and rejected intervals, full-batch MAE, access and accepted MAE per duration/type/observation group, and the fraction of **this labeled sample's** fuel represented. The latter uses reference labels only after selection and must not be extrapolated to an aircraft fleet or global emissions. A group floor guarantees an acceptance fraction only for the evaluated batch, not a maximum error or robust future coverage.

In September validation the access floor raises acceptance among 15–60-minute intervals from 0.72% to 50%, and B744 from 0/66 to 33/66. At the same 75.01% overall acceptance, accepted-case MAE rises from 44.05 to 53.92 kg; the selected B744 intervals alone have 541.84 kg MAE. The sample's labeled-fuel share rises from 29.69% to 41.57%, still well below full representation. These descriptive comparisons are the tradeoff, not a proof that the floor improved fuel estimation.

The dataset contains many very short intervals, heterogeneous aircraft types, sparse ADS-B trajectories, and uncertain fuel-label precision. Group MAE in kg depends strongly on fuel scale, so compare access and absolute error together. The natural missingness pattern is not random. The older controlled-outage experiment is a small selected validation-flight stress test, not an intervention on flight operation. There is no external-operator transfer check, independent post-revision test, or calibrated individual prediction interval. A result that shows a large error increase for wider access is a valid limitation, not a failure to report.

## Related work and positioning

The [dataset publication](https://doi.org/10.59490/joas.2026.8750) describes the benchmark and its ACARS/OpenSky construction. Fuel prediction from trajectories is established, including [physics-guided](https://doi.org/10.59490/joas.2026.8764) and [weather-enhanced](https://doi.org/10.59490/joas.2026.8766) challenge entries. A [related reliability study](https://doi.org/10.3390/en19204797) already examines missingness and uncertainty, and selective regression is a general method. This project claims a *dataset-specific, auditable access/error tradeoff for research use*, not a new estimator family or proven journal novelty. Independent data and stronger comparisons are needed before claiming generality.
