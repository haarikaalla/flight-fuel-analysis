# Flight Fuel Analysis: Coverage and Error Audit

An interactive data science project that estimates fuel use for **completed flight intervals** from public flight tracking data, then checks **which flights those estimates cover and how accurate they are**. [Try the offline dashboard](demo/index.html) or [walk through the results in Jupyter](notebooks/eligibility-walkthrough.ipynb).

## Journal study (branch `q1-research-extension`)

A pre-registered study adds a **physics-informed fuel model**, **cell-level conformal uncertainty** and **representation-aware release** on the full EUROCONTROL PRC 2025 benchmark. It is written up as a Springer LNCS manuscript: [`research_q1/paper/lncs/main.tex`](research_q1/paper/lncs/main.tex) (compiled: [`main.pdf`](research_q1/paper/lncs/main.pdf)). Every number in the paper is generated from [`research_q1/outputs/journal-study.json`](research_q1/outputs/journal-study.json); a test fails if they drift apart.

**Protocol.** All models are fit on April–August 2025 (131,530 intervals). September (24,289) is split into flight-disjoint halves for error-model fitting and conformal calibration. October (37,456) is a forward test. The configuration ([`research_q1/configs/journal_study.json`](research_q1/configs/journal_study.json)) with four hypotheses was committed (`de64954`) before the run. October aggregates were seen in earlier phases, so it is a forward temporal test on development-era data, not a sealed confirmation.

**Data correction.** The dataset README documents trajectory altitude/speed/vertical rate as metres and m/s; the released values are **feet, knots and ft/min** (median per-flight cruise ≈ 37,000 and ≈ 449). Features convert to SI before any physics.

**Key October 2025 results** (paired flight-cluster bootstrap, 2,000 repetitions, Holm-adjusted):

| Estimator | MAE kg [95% CI] | ΔMAE vs reference [95% CI] |
| --- | ---: | ---: |
| Type median rate | 89.30 [84.25, 95.05] | +12.78 [10.59, 14.91] |
| OpenAP physics, type-scaled | 85.60 [80.71, 91.27] | +9.08 [6.06, 12.48] |
| HGB metadata + ADS-B (reference) | 76.52 [71.66, 82.05] | — |
| HGB full (+ SI kinematics + OpenAP) | 73.12 [68.95, 77.98] | −3.40 [−5.70, −1.03] |
| Physics-residual HGB | **72.68** [68.53, 77.41] | **−3.84** [−6.27, −1.35] |

- **Ablation:** the gain comes from three OpenAP physics features (+2.58 kg if removed); the 14 SI kinematic and 14 original ADS-B summaries are redundant given physics (< 0.25 kg each).
- **Unseen aircraft types (leave-one-type-out, 14 types):** pooled MAE falls from 174.68 kg (reference) to 111.49 (HGB full) and 104.40 (physics-residual); HGB full wins on 10/14 types. Types with close training siblings (A21N, B738, A359) do not benefit.
- **Uncertainty (nominal 90%):** all methods are ≈90% marginally, but constant-width split conformal under-covers 15 of 28 aircraft-type × duration cells (worst 25.6%). Error-scaled and hierarchical Mondrian intervals reduce this to 5 and 7 cells; **no method reaches 85% in every cell** under the September→October shift (hypothesis H3 not supported).
- **Release at 75%:** risk-only release keeps ≤ 2.4% of 15–60-min intervals for every estimator. A 50% group floor restores access and raises the labelled-fuel share from 26.3–26.9% to 40.6–42.0% at a cost of 7.83–9.09 kg accepted MAE.
- **Inventory:** labelled-fuel totals are within ≈ 1% for the learned models (bootstrap CIs include zero); the uncalibrated OpenAP prior overestimates by 17.8%.

**Reproduce** (≈ 5 GB download; extraction ≈ 4 min, study ≈ 3–4 min on 8 cores):

```bash
python -m pip install -r research_q1/configs/requirements.txt openap
python -m src.flight_fuel.data_audit --download
python -m src.flight_fuel.download_archives
python -m research_q1.journal.extract              # ADS-B, SI kinematic and OpenAP features
python -m research_q1.journal.study                # -> research_q1/outputs/journal-study.json
python -m research_q1.journal.artifacts            # -> paper tables, figures, macros
python -m research_q1.dashboard.build              # -> research_q1/dashboard/index.html
python -m unittest discover -s research_q1/tests -t .
```

Compile the paper with any TeX distribution (e.g. `tectonic main.tex` or `pdflatex`+`bibtex`) in `research_q1/paper/lncs/`. LNCS is Springer's proceedings format; a journal submission would usually move to the journal's own class. Author names, affiliations and declarations are placeholders.

The results dashboard [`research_q1/dashboard/index.html`](research_q1/dashboard/index.html) has a journal-study panel (model comparison, paired tests, LOTO wins, interval coverage, release policy switch).

## Optimization and reliability workspace

The `research-extension` branch preserves the original analysis and adds an
**exact coverage-constrained selector**, five-fold out-of-fold isotonic risk
calibration and scaled conformal prediction intervals. The professional
workspace compares four policies, exposes bootstrap uncertainty and subgroup
reliability, supports multiple coverage and replay filters, and remains fully
self-contained in [`demo/advanced.html`](demo/advanced.html).

The publication-oriented layer additionally compares global, duration-Mondrian
and aircraft-Mondrian uncertainty, evaluates risk-calibrator transfer when each
sufficiently large aircraft type is held out, audits natural ADS-B availability,
and stress-tests the selector under score perturbations. A frozen prospective
protocol in [`configs/confirmatory-protocol.json`](configs/confirmatory-protocol.json)
defines the hypotheses and evidence required for a later untouched batch.

The replay uses committed historical benchmark rows; it is not a live aircraft
feed. A live system would require rolling-window retraining, approved telemetry
access, calibrated uncertainty and a new independently frozen evaluation period.
See [`UPSTREAM_ATTRIBUTION.md`](UPSTREAM_ATTRIBUTION.md) for the boundary between
the upstream project and this fork's additions.

### New robustness findings

- Global cross-fitted intervals reach **89.95%** empirical coverage with a
  **449.97 kg** mean width.
- Duration-Mondrian intervals reach **90.25%** coverage with a narrower
  **422.54 kg** mean width and only a **0.23 percentage-point** eligible-group
  coverage gap on this exploratory benchmark.
- Aircraft-Mondrian intervals are wider (**579.05 kg**) and still reach only
  **87.88%** coverage for the worst eligible aircraft group.
- When the risk calibrator is trained without each tested aircraft type in
  turn, pooled coverage is **90.45%**, but B744 coverage falls to **75.76%**.
  The fuel estimator itself is not held out in this diagnostic.
- Zero-position intervals have **163.22 kg** MAE versus **68.84 kg** for
  intervals with at least 51 ADS-B positions. Natural missingness is not random,
  so this is an association rather than a causal outage effect.

## The problem

Public ADS-B flight tracking data contain aircraft positions, but no measured fuel consumption. Researchers can estimate fuel use with a model; if they keep only its easiest predictions, the reported error looks better while longer intervals and some aircraft types largely disappear from the sample. That can distort conclusions drawn from the selected data.

**Research question:** For a batch of completed flight intervals, how does restricting estimates to lower-risk intervals affect prediction error, coverage across aircraft and interval lengths, and the share of the sample's observed fuel represented?

**Who can use this:** aviation researchers and public flight-data curators evaluating whether an estimated-fuel dataset adequately covers the flights they intend to study. The project also demonstrates to data science reviewers how model selection, missing data, subgroup analysis and honest evaluation fit into a reproducible workflow. It is not an airline fuel-planning tool.

## What I built

1. Verify the public dataset downloads and extract ADS-B summaries within each supplied flight interval. The reference fuel value comes from the difference between fuel-on-board reports; it is an imperfect label, not a fuel-flow sensor reading.
2. Train a fuel estimator on April–August 2025 data using aircraft type, interval duration and ADS-B summaries. Use a training-only aircraft-type median-rate fallback when an interval has no usable ADS-B positions.
3. Train a separate error-ranking model on early-September residuals. For a batch, rank intervals by predicted error and select an overall percentage. An optional **group floor** ensures a minimum accepted fraction in each duration group and aircraft type with at least 50 intervals in that batch.
4. Compare prediction error and access on later labeled intervals. The browser dashboard recalculates the selection locally as you change the overall target or group floor; labels are used to *evaluate* its choices, never to choose intervals.

The dashboard contains pre-scored benchmark intervals and works without a server. It does not accept new raw flights or run live inference.

## Main finding

At the same **75.01%** overall acceptance on **7,619 September validation intervals**, giving each eligible group at least 50% access changes the sample substantially:

| Measure | Lowest predicted error only | With 50% group floor |
| --- | ---: | ---: |
| Mean absolute error on accepted intervals | 44.23 kg | 54.02 kg |
| 15–60-minute intervals accepted | 0.60% | 50.12% |
| Boeing 747-400 (B744) intervals accepted | 1 of 66 | 33 of 66 |
| Share of this sample's labeled fuel represented | 29.77% | 41.51% |

**Interpretation:** broader access costs accuracy. The 33 accepted B744 intervals under the floor have **609.90 kg** mean absolute error. This is an important warning, not a success claim for those estimates. On *all* September validation intervals, the fuel model's MAE is **96.07 kg**, RMSE **281.69 kg** and R² **0.872**; its 95th percentile absolute error is **372.75 kg**. R² and selective MAE alone conceal large individual and subgroup errors. [Full results and residual breakdowns](reports/eligibility-results.json) are committed for inspection.

## Explore and reproduce

- Open [`demo/index.html`](demo/index.html) in a browser. Set **overall target = 75%**, then switch the group floor between **0%** and **50%**. Compare the duration and aircraft tables, accepted error and labeled-fuel share.
- Open the [Jupyter walkthrough](notebooks/eligibility-walkthrough.ipynb) for the result tables, access chart, residual diagnostics and route-overlap check.
- To rebuild the analysis from the publisher's data, use Python 3.11+ and allow roughly 5 GB for the trajectory archives plus extraction space:

```bash
python -m pip install -r requirements.txt
python -m src.flight_fuel.data_audit --download
python -m src.flight_fuel.download_archives
python -m src.flight_fuel.features --splits train rank final
python -m src.flight_fuel.eligibility
python -m unittest discover -s tests
```

To regenerate the reliability workspace without downloading the raw trajectory archives:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m flight_fuel.advanced
.venv/bin/python -m http.server 8000
```

Then open `http://localhost:8000/demo/advanced.html`, or open the generated file
directly because it is self-contained. The command reads the committed benchmark
payload and regenerates both `reports/advanced-results.json` and
`demo/advanced.html`.

To inspect the provider-safe event contract without revealing historical
labels, run:

```bash
.venv/bin/python -m flight_fuel.streaming --limit 20 --withhold-labels
```

The live-provider class intentionally refuses to operate without an explicitly
approved provider. It never invents a real-time reference-fuel value.

Raw data and extracted features are excluded from Git. Downloads are checked against publisher MD5 values. The final analysis command regenerates `reports/eligibility-results.json` and `demo/index.html`.

## Scope and limitations

- The policy was developed **after earlier October results were examined**. Its September and October comparisons are exploratory; October is not a fresh independent confirmation. The revised model also disables a row-random internal early-stopping split used in historical Phase 4, so the historical score is not a controlled policy-only comparison.
- Aircraft routes overlap across periods. In the exploratory October batch, MAE is **81.15 kg** on intervals from routes seen in training and **174.77 kg** on unseen routes. This comparison is affected by aircraft and duration mix, but warns against assuming transfer.
- Interval boundaries come from the completed fuel-report dataset. Payload, weight, wind and weather are absent, and raw altitude/speed units are unverified. The labels are fuel-on-board differences; neither per-flight operational fuel needs nor emissions inventories are validated here.
- The labeled-fuel share is known only for this evaluation sample. The error score ranks cases; it is not a calibrated confidence interval or an individual accuracy guarantee. The dataset contains many intervals shorter than the publisher README's stated five-minute lower bound.

The contribution is a **dataset-specific, transparent coverage/error audit**, not a novel fuel-prediction algorithm. See the [methodology](docs/methods.md), [related work](docs/literature.md) and [self-critical audit](docs/pre-submission-audit.md) for design choices, checks and outstanding research work.

## Project structure and data source

| Path | Purpose |
| --- | --- |
| [`src/flight_fuel/`](src/flight_fuel/) | Download checks, feature extraction, baselines, evaluation and batch selection. |
| [`demo/`](demo/) | Self-contained interactive dashboard and its maintained template. |
| [`notebooks/`](notebooks/) | Presentation-friendly analysis of committed results. |
| [`reports/`](reports/) | Generated result JSON and the historical Phase 4 record. |
| [`tests/`](tests/) | Decision-rule, accounting and dashboard checks. |
| [`paper/`](paper/) | Publication-stage LaTeX manuscript source (opens with live PDF preview in Codex). |

Data: [EUROCONTROL PRC 2025 Aircraft Fuel Burn Estimation Data Challenge](https://doi.org/10.5281/zenodo.19184662), combining ACARS-derived fuel labels with OpenSky ADS-B records and metadata, licensed CC BY 4.0. Cite the [dataset paper](https://doi.org/10.59490/joas.2026.8750). This repository does not contain the raw dataset.
