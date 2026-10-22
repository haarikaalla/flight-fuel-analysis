"""One-command reproduction for all evidence that is legitimately available."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from research_q1.baselines.metadata_benchmark import run
from research_q1.paper.generate_artifacts import generate


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--quick", action="store_true", help="Use 200 bootstrap draws and 80 model iterations for a smoke run")
    args = parser.parse_args()
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-v", "-s", "research_q1/tests"],
                   cwd=ROOT, check=True)
    result = run(args.raw_dir, bootstrap_repetitions=200 if args.quick else 2000,
                 max_iterations=80 if args.quick else 300)
    output = ROOT / "research_q1/outputs/metadata-baselines.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    generate(output, ROOT / "research_q1/paper/generated")
    print("Reproduced tested real-data development evidence. Protected confirmation data were not accessed.")


if __name__ == "__main__":
    main()

