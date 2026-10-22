# Research extension roadmap

## Proposed research question

Can coverage-constrained selective regression preserve access across aircraft
and interval-duration groups while controlling prediction risk under missing
ADS-B observations and temporal or route distribution shift?

## Completed first milestone

- Preserved the upstream analysis without changing its repository.
- Added a mixed-integer minimum-risk selector with explicit overall, duration
  and sufficiently represented aircraft-group constraints.
- Added decision explanations that distinguish low-risk inclusion from
  constraint-driven inclusion.
- Added an advanced offline dashboard with policy comparison, historical
  streaming replay, per-row decisions and an accuracy–coverage frontier.
- Added a generated advanced result report, Python packaging, a one-command
  workflow and continuous integration.

## Next scientific milestone: calibrated uncertainty

1. Freeze a chronological calibration block that is not used for fitting the
   fuel model or the risk model.
2. Add split conformal residual calibration for marginal prediction intervals.
3. Report empirical coverage and interval width overall and by aircraft,
   duration, route familiarity and ADS-B availability.
4. Compare global, Mondrian/group-conditional and risk-adaptive conformal
   variants. Clearly report small-group instability.
5. Never calibrate intervals on the committed September validation labels used
   for the headline selector comparison.

## Next generalization milestone

- Register the protocol before opening the final labels.
- Evaluate time-disjoint, route-disjoint and—where identifiers permit—airframe-
  disjoint splits.
- Add stronger tabular baselines and ablations for type, duration, trajectory
  summaries and missingness indicators.
- Repeat selection after every model refit; do not reuse fixed masks when making
  generalization claims.
- Report flight-cluster uncertainty and multiple-comparison-aware subgroup
  analysis.

## Real-time architecture

The current dataset supplies completed ACARS-defined interval boundaries. A
real-time adapter therefore cannot reuse the retrospective inference contract
unchanged.

Build two adapters behind one interface:

1. **Historical replay** — the default reproducible mode. Stream cached ADS-B
   records through rolling windows, reveal labels only after each window closes,
   and compute delayed evaluation metrics.
2. **Approved live telemetry** — an optional adapter enabled only after the
   provider's authentication, rate-limit and licensing requirements are met.
   Show predictions, uncertainty and abstention decisions, but never invent a
   real-time reference fuel value.

Both adapters should feed the same components:

```text
state source -> flight state store -> rolling feature extractor
             -> versioned model bundle -> uncertainty + drift checks
             -> constrained decision service -> WebSocket dashboard
```

## Publication boundary

The advanced selector and dashboard are meaningful engineering and experimental
extensions, not a guarantee of journal acceptance. A strong paper requires a
clear comparison with selective-regression and fairness-constrained baselines,
an untouched confirmatory dataset, calibrated uncertainty, independent
reproduction and appropriately narrow aviation claims.
