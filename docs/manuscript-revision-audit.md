# Manuscript revision audit

## What was completed

The supplied PDF was read in full and retained unchanged. A separate editable
review manuscript follows its numbered journal structure, with ten tables,
four new vector/raster figures, methods, five equations, results, discussion,
limitations, conclusion, declarations and seven verified-source references.
It is a Word reconstruction, not a compiled official `sn-jnl` LaTeX document:
the matching source package was not supplied and the local TeX files are older
article-class drafts. The new manuscript narrows the title to the actual evidence.

## Evidence and reproduction

Run from the repository root:

```sh
.venv/bin/python -m unittest discover -s research_q1/tests -v
.venv/bin/python -m research_q1.validation.manuscript_audit
.venv/bin/python -m research_q1.paper.audit_figures
```

The review manuscript and its document builder are retained locally and are not
distributed in this branch. The results, experiment code and figure generator
remain available above.

`research_q1/configs/development_audit.json` was committed as `d648709` before
the added experiment. The runner and tests were committed as `3b990be` before
execution. The resulting `research_q1/outputs/manuscript-audit.json` records
input, code and configuration SHA-256 values, seed, environment, calibration
counts and timings. The result is exploratory; no `protocol-v1` confirmatory
tag was created or backdated. Existing metadata baseline values were reproduced
in the preceding audit; changed key order and PDF creation metadata are not new
scientific results.

| Manuscript evidence | Artifact key or source |
| --- | --- |
| Seven point and interval baselines | `metadata-baselines.json:models` |
| Feature availability 79.22 vs 83.37 kg | `manuscript-audit.json:point` |
| Paired 4.15 kg difference and CI | `manuscript-audit.json:paired` |
| Coverage, widths and Winkler scores | `manuscript-audit.json:interval` |
| All cells and flight-bootstrap intervals | `manuscript-audit.json:cells` |
| Release curve and threshold 0.38 | `manuscript-audit.json:selection`, `crc` |
| Four inventory variants and bootstrap | `manuscript-audit.json:inventory` |
| CO2 scenario | Fuel totals multiplied by ICAO Jet-A factor 3.16 |
| Historical feature ablations | `metadata-interpretability.json` |
| Compute observations | `manuscript-audit.json:point`, `hardware` |

## Novelty decision and self review

No candidate passed all requirements of the original T6 gates. Aggregate
guarantees, few-shot cold-start, adaptive shift robustness and anomaly detection
were not evaluated. The two-stage duration-cell calibration has a narrower
empirical benefit: worst eligible coverage rises from 28.95% to 58.78%, but mean
width increases 14.39% relative to global split. It does not justify a new
phase-conditional coverage theorem or matched-width superiority claim.

The additional flight-block CRC construction controls the stated joint loss
only under its assumptions. October's observed joint loss is 4.37%, while the
conditional released error fraction is 17.13%. They have different denominators;
the original H3 language must not conflate them. No temporal-shift theorem follows.

The inventory identity is exact: imputing with the same prediction cannot change
the total. Type-median replacement worsens this cohort's aggregate bias. The
negative result is useful scientific evidence, but is not claimed as a new theorem.

The statistically supported metadata gain over a type-median rate is 10.08 kg,
paired route-bootstrap interval 5.29-14.70 kg. Overlapping standalone confidence
intervals did not establish equivalence in the original draft.

## Remaining submission requirements

- Intact trajectories and weather for the full mass-rollout physics experiment.
- Independently protected data for confirmation and sufficiently supported
  route/type/family holdouts; October remains development only.
- Matched-budget full-system baselines, phase-cell calibration, and independent
  evaluation of any chosen novelty upgrade.
- A systematic literature review, original-paper checks beyond this focused
  source set, and journal-specific scope/quartile/guideline verification.
- Complete Springer LaTeX source package if exact `sn-jnl` reproduction is required.
- Human author names, affiliations, funding, conflicts and contribution approval.
- Upstream code licensing resolution and an archived software DOI.
- Independent expert statistical review of any final guarantee claims.

This revision completes a substantive development manuscript, not the original
T0-T10 confirmatory program. It is not represented as Q1 submission-ready.

## Source verification log

Checked 27 October 2026 using publisher, proceedings or institutional sources:

- PRC dataset publication: https://journals.open.tudelft.nl/joas/article/download/8750/6543/35462
- OpenAP: https://www.mdpi.com/2226-4310/7/8/104/notes
- Grigoriou et al.: https://journals.open.tudelft.nl/joas/article/view/8764
- CQR: https://proceedings.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html
- CRC: https://openreview.net/pdf?id=33XGfHLtZg
- Selective regression: https://proceedings.mlr.press/v162/shah22a.html
- ICAO: https://www.icao.int/sites/default/files/environmental-protection/CORSIA/CORSIA-FAQ.html

Searches used exact DOI identifiers and exact method titles. No Scopus or Web of
Science access was available in this run; no exhaustive search count is claimed.
