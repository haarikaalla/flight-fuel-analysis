# Evidence status

## Completed before PI-MCS experiments

- Clean baseline: 22/22 original tests passed.
- Existing offline dashboard returned HTTP 200 and retained its committed hash.
- Existing September evidence and all committed output hashes were recorded.
- The new work is isolated on `q1-research-extension`.

## Not yet evidence

PI-MCS, weighted conformal, adaptive conformal, CQR, robust optimization,
feature ablation, SHAP, route/aircraft/family holdouts, and multi-year validation
have **no reported performance numbers yet**. Their implementations can be
unit-tested, but research results require the approved tables listed in
`../external/DATA_REQUIREMENTS.md`.

## Evidence currently available

At the baseline audit, the checkout had no `data/` directory. The committed dashboard rows contain
aircraft type, interval duration, a precomputed prediction, a reference fuel
difference, an error score, and an ADS-B position count. They do not contain the
trajectory vectors, winds, atmospheric state, aircraft geometry, take-off mass,
flight phase, route identifiers, operator identifiers, or multi-year labels
needed to fit or fairly evaluate PI-MCS.

