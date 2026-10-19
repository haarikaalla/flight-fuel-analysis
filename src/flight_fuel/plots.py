"""Generate report-ready SVG figures from the fixed ranking-period metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(report: dict, directory: Path) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    palette = ["#274c77", "#d17a22", "#3a8f7b"]
    models = [name for name in ("median_type_rate", "histgb_metadata", "histgb_adsb") if name in report["models"]]
    labels = {"median_type_rate": "Type median rate", "histgb_metadata": "Gradient boosting: metadata",
              "histgb_adsb": "Gradient boosting: ADS-B"}
    bins = ["<5 min", "5–15 min", "15–60 min"]
    x = np.arange(len(bins))
    width = .72 / len(models)
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for i, name in enumerate(models):
        data = report["models"][name]["subgroups"]["duration_bin"]
        ax.bar(x + (i-(len(models)-1)/2)*width, [data[b]["mae_kg"] for b in bins],
               width=width, label=labels.get(name, name), color=palette[i])
    ax.set_xticks(x, bins)
    ax.set_ylabel("Mean absolute error (kg per interval)")
    ax.set_title("September 2025 held-out flights · error by interval duration")
    ax.grid(axis="y", alpha=.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    first = directory / "rank-mae-by-duration.svg"
    fig.savefig(first, format="svg")
    plt.close(fig)
    first.write_text("\n".join(line.rstrip() for line in first.read_text().splitlines()) + "\n")
    outputs = [str(first)]
    if all("adsb_input" in report["models"][name]["subgroups"] for name in models):
        groups = ["zero ADS-B", "at least one ADS-B"]
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        x = np.arange(len(groups))
        for i, name in enumerate(models):
            data = report["models"][name]["subgroups"]["adsb_input"]
            ax.bar(x + (i-(len(models)-1)/2)*width, [data[g]["mae_kg"] for g in groups],
                   width=width, label=labels.get(name, name), color=palette[i])
        ax.set_xticks(x, groups)
        ax.set_ylabel("Mean absolute error (kg per interval)")
        ax.set_title("September 2025 held-out flights · ADS-B availability")
        ax.grid(axis="y", alpha=.25)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        second = directory / "rank-mae-by-adsb-availability.svg"
        fig.savefig(second, format="svg")
        plt.close(fig)
        second.write_text("\n".join(line.rstrip() for line in second.read_text().splitlines()) + "\n")
        outputs.append(str(second))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("reports/phase-3-rank-baselines.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    print("\n".join(render(json.loads(args.report.read_text()), args.output_dir)))


if __name__ == "__main__":
    main()
