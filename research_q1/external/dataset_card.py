"""Generate a reproducible dataset card from a validated analytical table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_q1.validation.data_contract import frame_fingerprint, read_table, validate_frame
from research_q1.validation.labels import label_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = read_table(args.data, access_log=Path("research_q1/external/future-confirmation-access.log"))
    payload = {"manifest": validate_frame(frame), "sha256": frame_fingerprint(frame),
               "label_audit": label_audit(frame)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

