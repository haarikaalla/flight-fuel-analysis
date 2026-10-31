# Q1 research extension

## Journal study (physics-informed model, cell-level conformal, representation-aware release)

Code: [`journal/`](journal/) (`extract`, `study`, `artifacts`). Protocol: [`configs/journal_study.json`](configs/journal_study.json).
Results: [`outputs/journal-study.json`](outputs/journal-study.json). Manuscript (Springer LNCS):
[`paper/lncs/main.tex`](paper/lncs/main.tex) and [`paper/lncs/main.pdf`](paper/lncs/main.pdf). See the root README for
headline numbers and reproduction commands.

## Additional development experiments

The additional experiments evaluate flight-block selective-release, aircraft-by-duration calibration,
landing-time availability and inventory-accounting experiments. Results and
all cell-level diagnostics are in [`outputs/manuscript-audit.json`](outputs/manuscript-audit.json).
The experiments are exploratory and do not establish full PI-MCS validation or
Q1 submission readiness. See [`../docs/manuscript-revision-audit.md`](../docs/manuscript-revision-audit.md)
for reproduction, provenance, novelty assessment and remaining requirements.

This directory is an additive, non-destructive research layer. It imports the
existing project but does not modify its source, reports, tests, or dashboard.

The proposed contribution is **PI-MCS**: Physics-Informed Mondrian Conformal
Selection with an Exact Risk Budget. The method and hypotheses are
pre-specified in `configs/study.json`. Results are generated only from approved,
locally supplied data that pass `validation/data_contract.py`.

No PI-MCS performance result is currently claimed. A real metadata-only
development benchmark is committed in `outputs/metadata-baselines.json`; it is
separate from PI-MCS evidence. Raw public trajectories are checksum-verified in
the ignored `data/raw/` workspace during acquisition. Weather,
aircraft-performance, operator, multi-year and independent-reference fields must
satisfy the explicit data contract before broader claims are evaluated.

## Safety boundary

- The frozen future-confirmation labels are not read by this extension.
- Splits are grouped by time, route, aircraft type, or aircraft family.
- Test labels never determine a split, tune a model, or calibrate an interval.
- Synthetic data are permitted only in unit tests and are never reported as
  research evidence.
- The existing dashboard remains unchanged.

## Reproduce verified development evidence

```bash
.venv/bin/python -m research_q1.paper.reproduce
```

Run a quick smoke check with `--quick`. Run a complete grouped experiment with:

```bash
.venv/bin/python -m research_q1.validation.full_study \
  --data /approved/path/analytical.parquet \
  --split route \
  --output research_q1/outputs/full-route-study.json
```

The protected confirmation filename guard remains active for every runner.

## Editable journal manuscript

`paper/PI-MCS_Springer_Nature_manuscript.docx` is a seven-page Springer Nature
style development manuscript. It contains the method equations, verified
metadata results, four tables, three figures, references and declarations.
It is an anonymized review copy and does not claim that PI-MCS has been
empirically validated or accepted by a Q1 journal. Rebuild it after the real
result JSON files are updated:

```bash
.venv/bin/python -m research_q1.paper.generate_artifacts
.venv/bin/python -m research_q1.paper.render_equations
python -m research_q1.paper.build_docx
```

The final command requires `python-docx`. Author details, funding and competing
interest statements must be confirmed by the submitting authors.

## View the verified research results

Open [`dashboard/index.html`](dashboard/index.html) directly in a browser. It is
self-contained and works without a server or downloaded raw data. The model
selector updates charts instantly from committed real-result JSON; the export
button downloads the comparison table. The page also links to the existing
recorded-flight replay and the editable journal manuscript. Historical replay
is not a live flight feed.

To run the repository locally with one URL:

```bash
python -m http.server 8765
```

Then visit `http://localhost:8765/research_q1/dashboard/index.html`. To rebuild
the page after verified result JSON changes, run:

```bash
.venv/bin/python -m research_q1.dashboard.build
```

