# Evidence status

## Safety baseline

- Clean baseline: 22/22 original tests passed.
- Existing offline dashboard returned HTTP 200 and retained its committed hash.
- Existing September evidence and all committed output hashes were recorded.
- The new work is isolated on `q1-research-extension`.

## Verified new real-data evidence

- PRC metadata/labels were downloaded from the published record and verified.
- The audit contains 15,761 flights, 193,275 intervals and 27 aircraft types.
- Seven real prediction baselines were trained on 131,530 April--August
  intervals, calibrated on 24,289 September intervals and evaluated on 37,456
  October intervals.
- Histogram gradient boosting obtained 79.22 kg MAE, with a 67.15--95.25 kg
  route-cluster bootstrap 95% confidence interval. Its paired MAE difference
  from the histogram boosting ensemble was 0.26 kg with CI -0.17--0.74 kg, so superiority is
  not supported.
- Only four October intervals were from a type absent in training (B734), which
  is inadequate for an unseen-aircraft claim.

These October results are exploratory because those labels were previously
examined by the inherited project.

## Implemented research machinery

Physics and hybrid residual models, hierarchical Mondrian, split/weighted/
adaptive conformal, CQR, three risk calibrators, exact/robust/greedy/Lagrangian
selection, conformal-selection boundaries, temporal/route/aircraft/family/
seasonal splits, grouped nested CV, shift diagnostics, cluster inference,
ablation, permutation importance, optional SHAP and one-command paper artifacts
are implemented under `research_q1/`.

## Evidence boundary

The committed dashboard rows contain
aircraft type, interval duration, a precomputed prediction, a reference fuel
difference, an error score, and an ADS-B position count. They do not contain the
trajectory vectors, winds, atmospheric state, aircraft geometry, take-off mass,
flight phase, route identifiers, operator identifiers, or multi-year labels
needed to fit or fairly evaluate PI-MCS. Metadata-only evidence is therefore
kept separate from full PI-MCS claims.

