"""Generate LaTeX tables and publication figures only from saved real results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/flight-fuel-q1-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/flight-fuel-q1-cache")

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "aircraft_type_median_rate": "Aircraft-type median rate",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "hist_gradient_boosting": "Histogram gradient boosting",
    "mlp": "MLP",
    "deep_ensemble": "Deep ensemble",
    "quantile_regression": "Quantile regression",
}


def generate(metadata_path: Path, output_dir: Path) -> list[Path]:
    if not metadata_path.exists():
        raise FileNotFoundError("Real metadata benchmark result is required; no placeholder table is generated")
    data = json.loads(metadata_path.read_text())
    if data.get("status") != "REAL_DATA_EXPLORATORY_METADATA_BENCHMARK":
        raise ValueError("Unexpected result status")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    ordered = sorted(data["models"], key=lambda name: data["models"][name]["regression"]["mae_kg"])
    lines = [
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Method & MAE (kg) & Route-cluster 95\% CI & RMSE (kg) & $R^2$ \\", r"\midrule",
    ]
    for name in ordered:
        values = data["models"][name]
        regression = values["regression"]
        ci = values["mae_route_cluster_ci"]["cluster_bootstrap_95pct_ci"]
        lines.append(f"{LABELS[name]} & {regression['mae_kg']:.2f} & [{ci[0]:.2f}, {ci[1]:.2f}] & "
                     f"{regression['rmse_kg']:.2f} & {regression['r2']:.3f} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    table = output_dir / "metadata_baselines.tex"
    table.write_text("\n".join(lines) + "\n")

    interval_lines = [
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Method & Coverage & Mean width (kg) & Winkler score \\", r"\midrule",
    ]
    for name in ordered:
        interval = data["models"][name]["split_conformal_90"]
        interval_lines.append(f"{LABELS[name]} & {100*interval['coverage']:.2f}\\% & "
                              f"{interval['mean_width_kg']:.2f} & {interval['winkler_score']:.2f} " + r"\\")
    cqr = data["models"]["quantile_regression"]["cqr_90"]
    interval_lines.append(f"CQR & {100*cqr['coverage']:.2f}\\% & {cqr['mean_width_kg']:.2f} & "
                          f"{cqr['winkler_score']:.2f} " + r"\\")
    interval_lines.extend([r"\bottomrule", r"\end{tabular}"])
    interval_table = output_dir / "metadata_intervals.tex"
    interval_table.write_text("\n".join(interval_lines) + "\n")

    mae = np.array([data["models"][name]["regression"]["mae_kg"] for name in ordered])
    ci = np.array([data["models"][name]["mae_route_cluster_ci"]["cluster_bootstrap_95pct_ci"]
                   for name in ordered])
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.errorbar(mae, y, xerr=np.vstack([mae - ci[:, 0], ci[:, 1] - mae]), fmt="o",
                color="#3157a4", ecolor="#6f88bd", capsize=3)
    ax.set_yticks(y, [LABELS[name] for name in ordered])
    ax.invert_yaxis()
    ax.set_xlabel("October 2025 MAE (kg), 95% route-cluster bootstrap CI")
    ax.set_title("Exploratory metadata-only baseline comparison")
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    figure_pdf = figure_dir / "metadata_baselines.pdf"
    figure_png = figure_dir / "metadata_baselines.png"
    fig.savefig(figure_pdf, bbox_inches="tight")
    fig.savefig(figure_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [table, interval_table, figure_pdf, figure_png]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("research_q1/outputs/metadata-baselines.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("research_q1/paper/generated"))
    args = parser.parse_args()
    for path in generate(args.metadata, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()

