# Data required before research claims

The research runner must receive approved local files; it never scrapes or
silently downloads data.

## Required analytical table

One row per completed flight or non-overlapping flight segment, with:

- stable `flight_id`, UTC start/end, year and season;
- origin/destination ICAO, operator identifier, aircraft ICAO type and family;
- reference fuel in kg and its provenance (`ACARS_FOB_DIFFERENCE`, `QAR`, or
  another documented source);
- synchronized ADS-B ground-velocity components, altitude, vertical rate and
  sampling-quality fields;
- ERA5 or another licensed wind/temperature/pressure source joined by time and
  position, with the retrieval request and license recorded;
- flight phase, aircraft wing area, aspect ratio and tank capacity;
- a flag separating train, development and a never-opened external confirmation
  source.

Take-off mass, TSFC, zero-lift drag and Oswald efficiency may be unknown. PI-MCS
fits them by aircraft type using training folds only and records fitted bounds.

## Breadth needed for the planned claims

- At least three calendar years so a past-to-future evaluation is possible.
- Multiple operators and route networks with documented permission.
- Enough observations per aircraft-phase Mondrian cell, or explicit hierarchical
  fallback to aircraft, phase and global calibration.
- A genuinely independent fuel reference for a subset. Carrier-level BTS totals
  are useful as aggregate context but are not flight-level ground truth.

## Legitimate candidate sources

| Source | Intended use | Access/license boundary |
| --- | --- | --- |
| EUROCONTROL PRC 2025 challenge, DOI 10.5281/zenodo.19184662 | Current benchmark labels, ADS-B and metadata | Dataset is CC BY 4.0; current checkout contains no raw files. |
| OpenSky Network historical data | Multi-year trajectories | Research access is available to eligible institutions; commercial/operational use requires permission or a license. |
| ERA5 via Copernicus Climate Data Store | Wind and atmospheric covariates | Free/open subject to account, dataset terms and recorded retrieval request. |
| US BTS airline fuel cost and consumption | Carrier-month aggregate plausibility check | Public aggregate table; not a flight-level label. |
| EASA Data4Safety | Potential external European validation | Partnership/controlled platform; not treated as open data. |

Operator identity must not be reverse-engineered from anonymized records. No
external-confirmation file may be opened until the protocol, software version
and hashes are frozen and the user explicitly authorizes the one-time run.

