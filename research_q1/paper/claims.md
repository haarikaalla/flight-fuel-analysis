# Pre-specified claims and decision rules

## Research questions

1. Does a training-fold-only physics prior improve prediction when aircraft or
   routes are absent from training?
2. Do physics-residual aircraft×phase Mondrian intervals reduce width while
   retaining coverage under temporal and covariate shift?
3. At the same risk budget and minimum coverage, does exact or robust selection
   improve utility relative to risk ranking, confidence thresholding, greedy
   selection and conformal selection?

## Hypothesis decisions

- **H1 coverage:** pass only when the cluster-bootstrap lower confidence bound
  for PI-MCS coverage is at least `1-alpha-tolerance` and standard conformal is
  below that threshold on the same aircraft-held-out observations.
- **H2 width:** pass only when paired cluster bootstrap shows smaller width for
  PI-MCS than both standard conformal and CQR while their coverage difference is
  inside the pre-specified tolerance.
- **H3 optimization:** pass only when exact selection has strictly greater
  utility at a matched feasible risk budget, solver gap is reported, and the
  paired confidence interval excludes zero. Otherwise fail, including ties.
- **H4 physics:** pass only when removing the physics prior increases held-out
  aircraft MAE and RMSE with Holm-adjusted p < 0.05 and a practically reported
  effect size.

Each hypothesis is reported as PASS, FAIL or INCONCLUSIVE. No substitution of a
different split, metric or subgroup is allowed after inspecting the result.

## Guarantee boundary

- Split/Mondrian conformal: marginal or pre-specified group coverage under the
  relevant exchangeability assumption; no guarantee after arbitrary selection.
- Weighted conformal: target-distribution marginal coverage only if density
  ratios are correct; estimated ratios are a sensitivity analysis.
- Adaptive conformal: long-run online coverage-frequency statement under its
  update protocol, not per-flight conditional coverage.
- Conformal selection: FDR control only under the exact conformal p-value and
  dependence assumptions of the chosen procedure.
- Exact/robust ILP: optimization feasibility/optimality for supplied calibrated
  risks; statistical validity is not created by the solver.

