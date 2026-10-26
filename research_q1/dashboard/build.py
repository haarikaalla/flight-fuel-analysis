"""Build a self-contained dashboard from committed, verified result JSON."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def build() -> Path:
    outputs = ROOT / "research_q1/outputs"
    benchmark = json.loads((outputs / "metadata-baselines.json").read_text())
    interpretation = json.loads((outputs / "metadata-interpretability.json").read_text())
    audit = json.loads((outputs / "prc-metadata-audit.json").read_text())
    readiness = json.loads((outputs / "q1-readiness.json").read_text())
    payload = {
        "benchmark": benchmark,
        "interpretation": interpretation,
        "audit": audit,
        "readiness": readiness,
    }
    serialized = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    source = (HERE / "template.html").read_text()
    marker = "/* VERIFIED_RESULTS_JSON */ null"
    if source.count(marker) != 1:
        raise ValueError("Dashboard template data marker must occur exactly once")
    output = HERE / "index.html"
    output.write_text(source.replace(marker, serialized))
    return output


if __name__ == "__main__":
    print(build())
