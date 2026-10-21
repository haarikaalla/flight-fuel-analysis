# Who Gets a Fuel Estimate?

**An auditable eligibility study for fuel estimates from incomplete public flight data.**

Open ADS-B trajectories do not contain measured fuel burn. A model can look accurate by withholding estimates on difficult intervals, yet leave long intervals and some aircraft types nearly unrepresented. This project asks a more useful question for **aviation researchers and public-data curators**: *Which intervals can our estimate cover, what error does that selection have, and which groups disappear from the analysis?*

The primary deliverable is a [self-contained interactive dashboard](demo/index.html). Open it directly after cloning; adjust overall acceptance and a minimum access floor. It recalculates a selection on **7,619 labeled late-September intervals** and shows accepted and rejected counts, group access, accepted-case error, and the fraction of this sample's observed fuel represented. It is a real dataset audit, not live inference on a new flight.

## What the project actually does

- **Target:** fuel consumed in a completed, predefined flight interval, in kg, inferred from differences between reported fuel-on-board values. These are imperfect reference labels, not direct fuel-flow measurements.
- **Input:** aircraft type, supplied interval duration, and summaries of ADS-B observations within the interval. A fixed aircraft-type median-rate fallback is used if there is no usable ADS-B position.
- **Decision:** rank intervals by predicted absolute error; optionally guarantee a chosen access floor in each duration bin and each aircraft type with at least 50 intervals in the **current batch**. Serve the lowest-ranked intervals needed for the floors before filling the overall target. The greedy rule is deterministic, and it discloses when the floors force acceptance beyond the target. Error estimates for groups near this size are especially uncertain.
- **Output:** actual acceptance and error overall and by group. The dashboard also evaluates how much *labeled fuel in this sample* is represented. Fuel labels do not choose accepted intervals. A new unlabeled batch cannot know its true fuel share.

### Key result

The previous Phase 4 risk rule achieved **41.02 kg accepted-case MAE at 76.17% acceptance in October**, but accepted only **0.36% of 15–60-minute intervals** and **0.29% of B744 intervals**. This is the problem the revised project makes visible. The new table uses an exact **batch-level 75%** target, so its risk-only score is slightly different from that earlier thresholded rule.

| Batch / policy | Accepted | Accepted MAE | 15–60 min accepted | B744 accepted | Sample's labeled fuel represented |
| --- | ---: | ---: | ---: | ---: | ---: |
| September validation / risk only | 75.01% | 44.23 kg | 0.60% | 1 of 66 | 29.77% |
| September validation / 50% group floor | 75.01% | 54.02 kg | 50.12% | 33 of 66 | 41.51% |
| October **exploratory** / risk only | 75.00% | 40.49 kg | 0.36% | 3 of 342 | 28.14% |
| October **exploratory** / 50% group floor | 75.00% | 50.07 kg | 50.04% | 171 of 342 | 41.64% |

The 33 selected B744 intervals in September have **609.90 kg MAE**; the floor improves *access*, not accuracy on that group. The 50-row aircraft eligibility cutoff was revised during this exploratory work after a 100-row cutoff left September B744 unprotected. Neither the September nor the post-revision October comparisons are independent confirmation of that choice. This audit also disabled the fuel model's earlier row-random automatic internal early stopping, so the new scores cannot be directly compared as a controlled policy-only experiment with historical Phase 4 scores.

On all September validation intervals, MAE is **96.07 kg**, RMSE **281.69 kg**, R² **0.872**, WAPE **22.43%**, and the 95th percentile absolute error is **372.75 kg**. R² reflects variation explained in this particular sample; it is not a per-interval accuracy guarantee. A naive MAPE is **41.88%**, but small fuel labels make it unstable. In October, the revised model's error is **81.15 kg MAE on routes seen in training** and **174.77 kg on unseen routes**; route composition and aircraft mix confound this comparison.

Accepted-case MAE scores only the accepted subset. Higher error when serving difficult groups is a transparent tradeoff; the fuel predictor itself is unchanged. There is no accuracy guarantee for any particular interval or aircraft.

## Try it

1. Open [`demo/index.html`](demo/index.html) in a browser. No server or dependency is required.
2. Start with **75% overall / 0% group floor**, then set the floor to **50%**. Compare the 15–60-minute row, aircraft access, accepted MAE, and fraction of labeled fuel represented.
3. Use [the notebook](notebooks/eligibility-walkthrough.ipynb) for a short explanation and presentation figure based on the committed result file. The Python module is the repeatable analysis.

## Reproduce from the published data

Python 3.11+ and about 5 GB for the three trajectory ZIPs, plus space for extraction and temporary downloads:

```bash
python -m pip install -r requirements.txt
python -m src.flight_fuel.data_audit --download
python -m src.flight_fuel.download_archives
python -m src.flight_fuel.features --splits train rank final
python -m src.flight_fuel.eligibility
python -m unittest discover -s tests
```

Raw data and interval features are ignored by Git. The downloader verifies the publisher's MD5 for each file; extraction reads one flight at a time. `reports/eligibility-results.json` and `demo/index.html` are generated by the final command. The retained `reports/phase-4-results.json` records the previous frozen experiment, not a directly comparable refit. For exact split rules, scores and caveats see [methods](docs/methods.md); [the literature review](docs/literature.md) records related work; [the pre-submission audit](docs/pre-submission-audit.md) records the fixes and remaining gaps; [submission notes](docs/submission-requirements.md) preserve the college template and rubric constraints for the report and slides.

## Evidence boundary

This policy was proposed **after we saw the earlier October evaluation**, so the new September and October comparisons are **exploratory**, even though the revised fuel predictor and residual ranker retain their earlier training periods. October is not a fresh independent test of the access floor. We need a later independently labeled period or another operator before claiming transfer. The sample is not a fleet-level emissions inventory; its accepted reference-fuel share cannot be generalized to worldwide aviation. The error rank is not a calibrated confidence interval, and future group coverage is not guaranteed outside each audited batch.

The dataset supplies completed interval boundaries; this demo cannot decide the end of an interval from ADS-B alone. No payload, aircraft weight, wind, weather, or independently verified altitude/speed units enter the model. The released data also contain **76,723 intervals under five minutes** across the three splits despite the publisher README's 5-minute lower-bound description. We retain and flag them rather than silently treating every interval as a typical cruise segment. The project does **not yet include the required final report, slides, project diary, or institutional sign-off**; verify these before submission.

The [EUROCONTROL PRC 2025 Aircraft Fuel Burn Estimation Data Challenge](https://doi.org/10.5281/zenodo.19184662) supplies the ACARS-derived labels, OpenSky ADS-B records and metadata under CC BY 4.0. Cite its [dataset paper](https://doi.org/10.59490/joas.2026.8750). Earlier trajectory-fuel studies and a related reliability study already address aspects of fuel prediction and missingness; this project makes a narrower, testable **coverage and error audit** contribution, not a claim of a novel fuel-prediction algorithm or proven journal novelty.

## Small project map

- `src/flight_fuel/`: verified ingestion, feature extraction, original baselines, historical Phase 4 protocol, and the new `eligibility.py` audit.
- `reports/`: prior Phase 4 aggregate evidence and the new result JSON.
- `demo/`: maintained HTML template and generated offline dashboard.
- `notebooks/`: explanatory walkthrough; `tests/`: decision and accounting checks.
- `docs/`: methodology, literature context and college submission notes.

The earlier [flight-fuel-efficiency-analysis](https://github.com/milantony05/flight-fuel-efficiency-analysis) used physics-generated targets and is not evidence for measured fuel accuracy here.
