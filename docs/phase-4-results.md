# Phase 4: selective fuel estimation under observation limits

The executable protocol is [`configs/phase-4-protocol.json`](../configs/phase-4-protocol.json) and [`src/flight_fuel/phase4.py`](../src/flight_fuel/phase4.py). Aggregate metrics and subgroup tables are in [`reports/phase-4-results.json`](../reports/phase-4-results.json); [`notebooks/phase-4-walkthrough.ipynb`](../notebooks/phase-4-walkthrough.ipynb) is a Jupyter walkthrough of that report. Python is the source of truth for extracting features and fitting models; the notebook reads the results and makes the validation, October, and outage comparisons explorable without duplicating training logic.

## Frozen evaluation contract

- **Target and inputs:** retrospective fuel consumed (kg) in a completed ACARS-defined interval. The only prediction inputs are aircraft type, interval duration, and ADS-B-only summaries from `[start, end)`. Interval boundaries come from ACARS, so this is not a deployable real-time forecast. Labels are quality-controlled fuel-report differences with known limitations, not ground-truth fuel-flow sensors.
- **Fixed base model:** Phase 3 HistGradientBoostingRegressor fitted on all April–August training intervals, with its original features and hyperparameters. The type-rate baseline uses the April–August median kg/min by aircraft type and a global rate for unseen types.
- **Fallback:** use the type-rate estimate only when the interval has zero valid ADS-B positions. Otherwise use the base ADS-B model. This rule was fixed based on Phase 3, before Phase 4 validation and October scoring.
- **Risk model:** a second fixed HistGB with absolute-error loss learns the fallback's absolute residual on **September 1–9** flights. Its features are aircraft type, log duration, log ADS-B position/speed counts, largest normalized coverage gap, and log base estimate. They are all available at inference. The target fuel is used only to *train* the risk model, never as an inference feature.
- **Calibration and testing:** threshold risk scores at target acceptance of **50%, 75%, 90%, and 100%** using input scores from **September 10–18**, with 75% designated primary. The later **September 19–27** block checks temporal validation; October is a further held-out test. Whole flights stay in one block. Three flights with 23 intervals crossing a block boundary are excluded from the September development/validation analysis. For tied scores, a stable hash of interval/flight IDs breaks ties; IDs are not estimator inputs. The hash is a deterministic tie breaker, not a random seed tuned on labels.
- **Matched selectors:** maximum ADS-B gap fraction, interval duration, and a deterministic hash define comparator acceptance rules, each thresholded on the same calibration block to the same target coverage. Their actual later coverage can vary. All selectors score the **same fallback estimates** on their accepted subsets. No error metric from September validation or October selects the primary risk rule.
- **Diagnostics:** report interval MAE and RMSE, flight-macro metrics, acceptance by aircraft/duration/observation group, and a flight-cluster paired bootstrap for fallback versus the base model. The candidate integer-pound lattice flag is used to stratify *evaluation only*; it never selects a model or estimate.

The protocol was committed as `0a4791a` before October model scoring. October's label **distribution** was inspected descriptively during data preparation, before that commit; no October model errors or selection comparisons were available when the policy was set. This is a temporal holdout, with that provenance limitation, rather than a claim that no final labels were ever opened. After the first October run, a bookkeeping defect was found in the 100% comparator: a calibration-set maximum had rejected a few larger October scores. The code now makes 100% accept every interval. The base model, primary 75% threshold, and all other selectors were unchanged; the report was regenerated with that correction. This is a reproducibility repair, not model selection against October error.

## September validation

The last September block contains **7,619 intervals across 583 flights**. The ADS-B model has **97.49 kg MAE** and the zero-input fallback **96.18 kg MAE**. The fallback minus base MAE is **−1.31 kg**, with a whole-flight bootstrap 95% interval of **−3.36 to +0.75 kg**; the overall gain is inconclusive. RMSE moves **+2.71 kg**, with interval **−8.65 to +16.43 kg**. The fallback is an explicit availability policy, not an established whole-population improvement.

At the calibrated 75% point, the risk selector accepts **75.2%** of later September intervals with **44.28 kg MAE** on accepted intervals. The duration selector accepts **75.6%** with **51.58 kg MAE**; the maximum-gap selector accepts **69.9%** with **66.91 kg MAE**; the hash accepts **74.8%** with **96.03 kg MAE**. These are **different accepted subsets**, so lower accepted-case MAE primarily measures choosing easier intervals. There is no prediction at all for abstained intervals and no defined business or safety cost for doing so. The risk score is a ranking proxy; its numerical value is not a calibrated confidence interval or a guarantee for a particular aircraft.

## Controlled outages on September validation

With seed 42, choose up to 120 eligible whole flights with at least 20 natural ADS-B positions and score their **1,355 originally observed intervals**. For each of two seeds and 20%, 50%, and 80% removal, perturb raw ADS-B observations *before* feature extraction, leaving ACARS rows, interval boundaries, labels, model, and type metadata unchanged. Compare random point thinning with one contiguous **point block** of equal size; the latter is not a constant-duration outage. Recomputed unperturbed features are checked against the full cached features before perturbation.

| Removed ADS-B positions | Random thinning ADS-B MAE, seeds 42/43 | Contiguous point block ADS-B MAE, seeds 42/43 | Block fallback MAE, seeds 42/43 |
| --- | ---: | ---: | ---: |
| 0% (natural) | 78.80 kg | 78.80 kg | — |
| 20% | 78.90 / 79.26 kg | 80.92 / 80.84 kg | 79.81 / 78.65 kg |
| 50% | 80.26 / 80.23 kg | 83.83 / 84.11 kg | 77.77 / 77.83 kg |
| 80% | 80.48 / 80.54 kg | 90.07 / 91.53 kg | 79.77 / 79.77 kg |

For the 50% point-block mask, the ADS-B model's paired MAE rises by **5.03 kg** (95% whole-flight bootstrap **1.78 to 8.35 kg**) under seed 42, and **5.31 kg** (**1.93 to 9.57 kg**) under seed 43. The same number of independently removed points causes a much smaller change. The block mask creates zero-input intervals in about **46–47%** of this *originally observed* sample; the zero-input fallback reduces sample MAE in these conditions. These are interventions on recorded ADS-B rows, not randomized flight operations. They do not establish that naturally sparse flights have the same response, nor that the fallback helps every aircraft.

## October temporal evaluation

The full October period contains **37,456 intervals from 2,836 flights**; **9,557 intervals (25.5%)** have zero valid ADS-B positions. The same April–August base model has **88.49 kg MAE / 240.70 kg RMSE**; the type-rate comparator has **89.30 / 249.36 kg**. The fixed zero-input fallback reaches **86.07 / 241.11 kg**. Its paired MAE change against ADS-B alone is **−2.43 kg** (95% whole-flight bootstrap **−3.47 to −1.41 kg**). Its RMSE change is **+0.40 kg** (**−4.39 to +5.90 kg**); the RMSE difference is inconclusive.

| Selector at calibrated 75% target | October intervals accepted | MAE on accepted intervals | RMSE on accepted intervals |
| --- | ---: | ---: | ---: |
| Learned absolute-error risk | **76.2%** | **41.02 kg** | 82.67 kg |
| Shortest duration | 76.3% | 47.15 kg | 93.81 kg |
| Smallest ADS-B gap fraction | 73.1% | 67.22 kg | 158.23 kg |
| Deterministic hash | 75.2% | 86.07 kg | 245.59 kg |

The full fallback gives **86.07 kg MAE on all October intervals**, and its 100% acceptance point now reproduces that exactly. The risk selector has lower accepted-case MAE than the fixed comparators at a similar overall accepted fraction, but it has not improved estimation *on* the rejected cases. It accepts **93.7% of <5 minute intervals**, **75.8% of 5–15 minute intervals**, and only **0.36% of 15–60 minute intervals**. Aircraft access is also uneven: **93.4% for A20N** (17,095 intervals), **46.5% for A359** (5,592), and **0.29% for B744** (342). Some aircraft groups have very small sample sizes. This is a substantive failure mode for any intended service that needs longer intervals or equal aircraft coverage, despite the aggregate risk–coverage curve. The score is not a guarantee of conditional error or an operational safety criterion.

The candidate integer-pound lattice includes **58.2%** of October labels under the stated numerical tolerance. Fallback MAE is **66.54 kg** in that flagged group and **113.22 kg** in the other group. These groups have different aircraft/duration composition and fuel scale. The observation does not measure label uncertainty or show that one group is intrinsically more reliable.

## Label and deployment limits

Flagging fuel values near integer multiples of 0.453592 kg is a diagnostic of a possible pound conversion lattice, not proof of the source, precision, or error distribution for each label. The report stratifies this flag by aircraft and interval duration; differences in MAE are confounded by aircraft mix and fuel scale. No true label-noise correction or calibrated uncertainty is claimed.

The overlapping 2026 reliability study already examines data quality, missingness, and uncertainty on part of this benchmark. The contribution hypothesis here is a full released benchmark analysis of observation availability, controlled point loss, selective coverage, aircraft-group access, and a later-month test with an ADS-B-only input budget. That empirical combination still needs a systematic prior-work check and independent operational data before a journal novelty or transfer claim. Unequal acceptance across aircraft is a potential failure mode, and the subgroup counts in the JSON must be read alongside aggregate error. There is no quantified cost of abstention, prospective boundary availability, or independent-operator validation.

## Reproduce

From the repository root, install [`requirements.txt`](../requirements.txt) with Python 3.11+, obtain the official metadata/labels and the three MD5-verified ZIPs as described in the [README](../README.md), then run:

```bash
python -m src.flight_fuel.features --splits train rank final
python -m src.flight_fuel.phase4 --rank-only --output reports/phase-4-rank-check.json
python -m src.flight_fuel.phase4 --output reports/phase-4-results.json
python -m unittest discover -s tests
```

The full pipeline extracts one flight at a time and verifies each archive against its official MD5. Raw archives, interval-level features, model objects and row predictions are ignored by Git. The report and notebook contain only aggregate outputs. The October command is an evaluation; avoid iteratively changing the protocol against its results.
