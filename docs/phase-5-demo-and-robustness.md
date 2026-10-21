# Phase 5: offline demo and robustness audit

## Deliverables

- [`demo/index.html`](../demo/index.html) is a self-contained interactive walkthrough of the committed September validation and October aggregate results. Open the file in a browser; it does not need a server, notebook kernel, network connection, or the raw multi-gigabyte archives.
- [`src/flight_fuel/phase5.py`](../src/flight_fuel/phase5.py) checks the Phase 4 aggregate report, writes the compact demo data into the HTML template, and generates [`reports/phase-5-robustness.json`](../reports/phase-5-robustness.json).
- [`demo/template.html`](../demo/template.html) is the maintained source of the interface; generated `index.html` is committed so a presentation works directly from a clone. No row-level label or trajectory is embedded. The earlier [Jupyter walkthrough](../notebooks/phase-4-walkthrough.ipynb) remains the more technical presentation of the same study.

The demo has period, acceptance-rule and target-coverage controls. It displays accepted-case MAE, actual acceptance, a risk–coverage chart, duration and aircraft access, full-fallback duration errors, and the paired controlled-outage results. It explicitly distinguishes *abstention* from improved prediction of rejected intervals. It does **not** calculate a new flight's fuel estimate: the raw dataset, fitted estimator and per-flight predictions are intentionally absent from Git, and the aggregate report cannot supply such an estimate.

## Checks and sensitivity

The audit requires every published 100% selection point to equal the full fallback score, verifies count and acceptance partitions and monotone coverage, and reconstructs fallback MAE from the **disjoint** integer-pound-lattice/non-lattice diagnostics within each duration bin. The reconstruction must match the published overall fallback MAE before the sensitivity results are written. This is a consistency audit of committed aggregate data, not a second model fit or an independent replication of the October evaluation.

| Period | All intervals | Under 5 min | At least 5 min | 15–60 min |
| --- | ---: | ---: | ---: | ---: |
| Late September validation | 96.18 kg (7,619) | 57.25 kg (3,447) | 128.35 kg (4,172) | 347.87 kg (838) |
| October final | 86.07 kg (37,456) | 50.60 kg (17,318) | 116.56 kg (20,138) | 302.14 kg (3,943) |

Values are fallback MAE per interval; counts are in parentheses. The **at-least-5-minute** score directly addresses the dataset README's five-minute claim, which conflicts with many released shorter intervals. Excluding them raises the observed MAE because those short intervals have smaller fuel amounts and lower absolute errors. This filter is applied after evaluation to disjoint group statistics; it is not a refit and has no new confidence interval. Errors across different duration groups also differ in fuel scale.

The October duration mix has 46.24% intervals under five minutes, against 45.24% in the September validation block. Reweighting October group MAEs to the September validation duration mix gives **87.36 kg** instead of the observed **86.07 kg**. Conversely, reweighting September validation group MAEs to October's duration mix gives **94.73 kg** instead of **96.18 kg**. These descriptive direct-standardization calculations show that duration composition explains *some* of the period-level difference, but they cannot identify causes of the remaining difference or adjust for aircraft and observation coverage jointly.

## Selective-coverage failure mode

| Risk target | October actually accepted | October accepted-case MAE | 15–60 min accepted | B744 accepted |
| --- | ---: | ---: | ---: | ---: |
| 50% | 49.35% | 27.42 kg | 0.00% | 0.00% |
| **75% primary** | **76.17%** | **41.02 kg** | **0.36%** | **0.29%** |
| 90% | 90.57% | 58.59 kg | 29.80% | 6.14% |
| 100% | 100% | 86.07 kg | 100% | 100% |

The long-interval acceptance collapse also occurs in late September: **0.84%** at the primary 75% target. Even at a 90% October overall acceptance target, most long intervals and B744 intervals receive no estimate. Selection is useful as an error-ranking study; the present policy fails a plausible requirement to serve longer intervals and different aircraft types. We did not retroactively optimize a fairness floor against October outcomes. A future constrained selector would need its target access requirements set and tested on genuinely new data.

The Phase 4 outage experiment already repeats two random seeds over three removal levels and contrasts scattered with contiguous point loss on the same originally observed September intervals. The demo exposes those existing paired numbers and bootstrap intervals. This Phase 5 aggregate audit neither reruns corrupted trajectories nor claims robustness on a newly sampled cohort. Independent operators, prospective interval availability, the cost of abstention, and label quality remain open.

## Presenting the demo

1. Open [`demo/index.html`](../demo/index.html). Start with **October / lowest predicted error / 75%**: explain the 76.2% acceptance and 41.02 kg accepted-case MAE, then scroll to the duration and aircraft bars. The missing longer intervals are the central limitation.
2. Switch the target to **100%**: explain that the full fallback produces 86.07 kg MAE on all intervals. Switch to **shortest duration** or **fixed random comparator** at 75% to explain the comparison of different accepted sets.
3. In the outage panel select **50% removed** and compare scattered versus contiguous point loss. The target label and model were held fixed while raw ADS-B positions were removed before feature extraction.
4. Show the [Phase 4 Jupyter notebook](../notebooks/phase-4-walkthrough.ipynb) if asked about the exact data blocks, estimator inputs, or figures. The Python modules are the repeatable implementation; the notebook explains their results.

## Reproduce and verify

From the repository root with Python 3.11+:

```bash
python -m pip install -r requirements.txt
python -m src.flight_fuel.phase5
python -m unittest discover -s tests
```

The Phase 5 command needs only the tracked Phase 4 JSON report and Python's standard library. The full test suite additionally needs the declared Parquet dependency. The generated JSON and HTML should be byte-for-byte identical to the committed versions. To rerun the underlying Phase 3/4 fits from source, first obtain the verified raw archives and metadata using the [README](../README.md) commands; Phase 5's aggregate reconstruction is **not** a substitute for that full rerun.
