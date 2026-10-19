# Phase 3: reproducible fuel-interval baselines

**Target:** `fuel_kg` for a completed, ACARS-defined interval. **Version:** [EUROCONTROL PRC 2025 release v1](https://doi.org/10.5281/zenodo.19184662), CC BY 4.0. The label is a quality-controlled difference of onboard fuel reports; it is not an error-free engine-flow measurement. See the [Phase 2 data acceptance](phase-2-data-acceptance.md) for the duration and unit caveats.

## Evaluation contract

- Fit on the released **train** period (April–August 2025; 11,037 flights, 131,530 intervals). Score on **rank** (September; 1,888 flights, 24,289 intervals). The **final** period (October) is unopened by the baseline program and reserved for a single locked evaluation after method decisions.
- Interval and flight IDs are join keys, never model inputs. Predictors are aircraft type and duration, optionally ADS-B-only summaries inside `[start, end)`. The code reads only the allowed trajectory columns, filters `source == 'adsb'`, and never loads ACARS Mach/TAS/CAS as predictors. The interval boundaries originate from ACARS and make the task retrospective.
- Retain every published positive-duration interval, including intervals shorter than five minutes and intervals with zero usable ADS-B positions. The latter receive missing trajectory statistics and may still be predicted from type and duration. No label-derived quality flag is passed to the model.
- The raw trajectory altitude, speed and vertical-rate columns are summarized without physical conversion while the [suspected published-unit mismatch](phase-2-data-acceptance.md) is unresolved. Model feature names carry `_raw` for this reason. Do not interpret their coefficients as SI physics.
- Fixed models and hyperparameters are in [`configs/phase-3-baselines.json`](../configs/phase-3-baselines.json). There is no hyperparameter search against ranking labels. Negative tree predictions are clipped to zero kilograms.

## Models

| Model | Inputs | Training rule |
| --- | --- | --- |
| Type median rate | Aircraft type, duration | Training median of `fuel_kg / duration_min` per type, multiplied by holdout duration. Global training median fallback for unseen types. |
| HistGB metadata | Aircraft type, duration | One-hot type, numeric duration; histogram gradient boosting with fixed squared-error settings. |
| HistGB ADS-B | Same metadata plus ADS-B interval summaries | Same estimator/settings as above; median/quantile altitude, ground speed, absolute vertical rate, position count and speed count. Missing native numerical values indicate absent measurements. |

The last two differ only in available predictors, enabling a controlled trajectory-feature comparison. Coverage gap features are *recorded* in the cached feature table but excluded from the Phase 3 predictor list so they can be studied separately in Phase 4. Stronger weather/performance-table models from the PRC challenge have a different input budget and cannot be compared as matched baselines without reproducing their setting.

**Feature meanings:** `duration_min` is interval end minus start in minutes; `aircraft_type` is the flight-list ICAO designator. `adsb_altitude_{median,p10,p90}_raw` and `adsb_speed_{median,p10,p90}_raw` summarize their respective valid ADS-B measurements within the interval. `adsb_vertical_rate_abs_{median,p90}_raw` summarizes the magnitude of climb/descent rate, not its direction. `adsb_position_count` counts timestamped positions with coordinates; `adsb_speed_count` counts those positions that also contain speed. A zero count leaves the corresponding measurement summaries missing; the tree estimator handles missing numeric inputs. The cached edge and maximum-gap features describe observation coverage but are **not predictors in Phase 3**. Percentiles and medians are computed per interval, so a longer interval does not contribute more feature rows than a shorter one.

## Ranking-period results

The committed machine-readable tables are [`reports/phase-3-metadata-rank.json`](../reports/phase-3-metadata-rank.json) and [`reports/phase-3-rank-baselines.json`](../reports/phase-3-rank-baselines.json). MAE/RMSE are kilograms **per interval**. Flight-macro MAE gives each flight equal weight; flight-macro RMSE is the square root of mean flight MSE. The 95% paired intervals resample whole ranking flights 1,000 times, preserving correlated intervals within a flight. They quantify sampling variability for this fixed ranking cohort, not dataset-label error or generalization to every airline. Both full train and rank ZIPs were MD5 verified before feature extraction; the rank model used **all 131,530 training and 24,289 ranking intervals**.

| Model | MAE (kg) | RMSE (kg) | Flight-macro MAE (kg) |
| --- | ---: | ---: | ---: |
| Type median rate | 94.81 | 266.59 | 116.75 |
| HistGB metadata | 100.03 | 263.83 | 123.25 |
| HistGB ADS-B | **92.95** | **256.70** | **112.91** |

Against the type-rate baseline, metadata HistGB has **+5.22 kg MAE** (95% flight-bootstrap interval **+3.71 to +6.88 kg**) and **−2.76 kg RMSE** (interval **−9.78 to +3.72 kg**). Thus it has worse average absolute error, while its small RMSE advantage is inconclusive under this resampling. The simple baseline is a serious comparator, not a straw man.

Against matched metadata HistGB, ADS-B HistGB changes MAE by **−7.08 kg** (95% flight-bootstrap interval **−8.19 to −6.10 kg**) and RMSE by **−7.13 kg** (**−10.85 to −3.70 kg**). Against the type-rate baseline, it changes MAE by **−1.85 kg** (**−3.70 to +0.04 kg**) and RMSE by **−9.89 kg** (**−16.11 to −4.08 kg**). The direct MAE comparison with the simple baseline does **not** establish a clear gain under this bootstrap, despite the lower point estimate.

### Observation and aircraft breakdowns

The full feature extraction found **45,874 / 131,530 (34.9%)** zero-ADS-B training intervals and **6,395 / 24,289 (26.3%)** zero-ADS-B ranking intervals. This is a change in input availability between periods, not evidence by itself that the model caused the different errors. Ranking results are:

| ADS-B in interval | Intervals | Type-rate MAE | Metadata HistGB MAE | ADS-B HistGB MAE |
| --- | ---: | ---: | ---: | ---: |
| At least one position | 17,894 | 70.72 | 77.00 | **67.04** |
| No position | 6,395 | **162.20** | 164.48 | 165.46 |

All values are kg per interval. On zero-input intervals, the ADS-B model cannot use kinematics and has worse MAE than the simple type-rate baseline. The observed/non-observed groups also differ in aircraft and interval characteristics; their error gap is **associational**, not a controlled missingness effect. Phase 4 should test a prespecified fallback and evaluate risk alongside the fraction of intervals accepted.

Aircraft types vary markedly: for A20N (11,989 rank intervals), the ADS-B model has 44.5 kg MAE against 45.0 for type rate; for A320 (4,993), 60.2 against 69.2; for B744 (240), **672.6 against 621.4**. The overall score is dominated by common types, and the model should not be advertised as uniformly better.

The [duration-error figure](../reports/figures/rank-mae-by-duration.svg) displays the large error increase for 15–60 minute intervals; groups have different fuel magnitudes, so cross-bin MAE is not a normalized efficiency measure. The [ADS-B availability figure](../reports/figures/rank-mae-by-adsb-availability.svg) shows the subgroup errors. Neither chart proves that missing observations cause fuel-error differences.

## Reproduce

Install `requirements.txt` and download the small metadata/label files and all three trajectory archives using the commands in the [README](../README.md). Run from the repository root:

```bash
python -m src.flight_fuel.baselines --output reports/phase-3-metadata-rank.json
python -m src.flight_fuel.features --splits train rank
python -m src.flight_fuel.baselines --feature-dir data/processed --output reports/phase-3-rank-baselines.json
python -m src.flight_fuel.plots --report reports/phase-3-rank-baselines.json
python -m unittest discover -s tests
```

Full ZIP files and extracted feature Parquet are ignored by Git. Feature extraction verifies each official ZIP MD5 and matches archive members to flight-list IDs. The JSON metrics and SVG figures are tracked; no test labels, predictions, trained models or raw trajectories are published here. The baseline script refuses `final`, and model/feature decisions should be frozen before any October evaluation.

## Interpretation and next gate

These results measure agreement with ACARS-derived labels for the released sample. They do not estimate operational fuel-saving opportunities or prove that a surveillance-only estimator works on unseen operators. Phase 4 should compare selective use, metadata fallback and abstention under explicitly defined costs and acceptance rates, with duration and aircraft stratification. The paper-level novelty claim remains conditional on a full prior-work comparison and convincing external or temporal validation.
