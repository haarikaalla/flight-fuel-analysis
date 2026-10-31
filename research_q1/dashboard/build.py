"""Build a self-contained dashboard from committed, verified result JSON."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def build() -> Path:
    outputs = ROOT / "research_q1/outputs"
    benchmark = json.loads((outputs / "metadata-baselines.json").read_text(encoding="utf-8"))
    interpretation = json.loads((outputs / "metadata-interpretability.json").read_text(encoding="utf-8"))
    audit = json.loads((outputs / "prc-metadata-audit.json").read_text(encoding="utf-8"))
    readiness = json.loads((outputs / "q1-readiness.json").read_text(encoding="utf-8"))
    journal_path = outputs / "journal-study.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8")) if journal_path.exists() else None
    payload = {
        "benchmark": benchmark,
        "interpretation": interpretation,
        "audit": audit,
        "readiness": readiness,
        "journal": journal and {
            "git_commit": journal["git_commit"], "counts": journal["counts"],
            "october": {m: {"mae_kg": v["mae_kg"], "rmse_kg": v["rmse_kg"], "r2": v["r2"], "ci": v["mae_flight_ci95"]}
                        for m, v in journal["periods"]["october"]["models"].items()},
            "paired": journal["periods"]["october"]["paired_vs_reference"],
            "loto": journal.get("leave_one_type_out", {}).get("types_where_model_beats_reference"),
            "loto_n": len(journal.get("leave_one_type_out", {}).get("eligible_types", [])),
            "intervals": {m: {k: {"coverage": iv["coverage"], "width": iv["mean_width_kg"], "worst": iv["worst_cell_coverage"]}
                              for k, iv in v["intervals"].items()}
                          for m, v in journal["uncertainty_and_selection"]["models"].items()},
            "selection": {m: v["selection_frontier"] for m, v in journal["uncertainty_and_selection"]["models"].items()},
        },
    }
    serialized = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    source = (HERE / "template.html").read_text(encoding="utf-8")
    marker = "/* VERIFIED_RESULTS_JSON */ null"
    if source.count(marker) != 1:
        raise ValueError("Dashboard template data marker must occur exactly once")
    output = HERE / "index.html"
    output.write_text(source.replace(marker, serialized), encoding="utf-8")
    return output


if __name__ == "__main__":
    print(build())
