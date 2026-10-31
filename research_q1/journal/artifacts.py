"""Generate LaTeX tables, figures and a macro file from research_q1/outputs/journal-study.json.

The manuscript includes only generated numbers; nothing is typed by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL_LABELS = {
    "type_median_rate": "Type median rate",
    "openap_physics": "OpenAP physics (uncalibrated)",
    "openap_calibrated": "OpenAP physics (type-scaled)",
    "hgb_metadata": "HGB metadata",
    "hgb_metadata_adsb": "HGB metadata + ADS-B summaries",
    "hgb_full": "HGB full (+ kinematics + physics)",
    "hgb_residual_physics": "Physics-residual HGB",
}
MODEL_ORDER = list(MODEL_LABELS)
INTERVAL_LABELS = {"split_constant": "Split conformal (constant width)",
                   "split_scaled": "Split conformal ($\\hat\\sigma$-scaled)",
                   "mondrian_type_duration": "Hierarchical Mondrian (type $\\times$ duration)"}


def f(value, digits=2):
    return "--" if value is None else f"{value:,.{digits}f}"


def pct(value, digits=1):
    return "--" if value is None else f"{100 * value:.{digits}f}"


def ci(values, digits=2):
    return f"[{f(values[0], digits)}, {f(values[1], digits)}]"


def macro_name(key: str) -> str:
    out = []
    for part in key.replace("-", "_").split("_"):
        out.append(part[:1].upper() + part[1:])
    return "".join(out)


def write_macros(result: dict, out: Path) -> None:
    oct_ = result["periods"]["october"]
    sep = result["periods"]["september"]
    ref = result["config"]["reference_model"]
    unc = result["uncertainty_and_selection"]
    macros = {
        "nTrain": f(result["counts"]["train"]["intervals"], 0), "nTrainFlights": f(result["counts"]["train"]["flights"], 0),
        "nSep": f(result["counts"]["september"]["intervals"], 0), "nSepFlights": f(result["counts"]["september"]["flights"], 0),
        "nOct": f(result["counts"]["october"]["intervals"], 0), "nOctFlights": f(result["counts"]["october"]["flights"], 0),
        "nTypes": f(result["counts"]["train"]["aircraft_types"], 0),
        "zeroAdsbOct": pct(result["counts"]["october"]["zero_adsb_intervals"] / result["counts"]["october"]["intervals"]),
        "eligibleCells": f(unc["eligible_test_cells"], 0),
        "gitCommit": result["git_commit"][:10], "runtimeMin": f(result["runtime_seconds"] / 60, 1),
        "bootReps": f(result["config"]["bootstrap_repetitions"], 0),
    }
    for model in MODEL_ORDER:
        key = macro_name(model)
        for period, data in (("Oct", oct_), ("Sep", sep)):
            m = data["models"][model]
            macros[f"mae{key}{period}"] = f(m["mae_kg"])
            macros[f"rmse{key}{period}"] = f(m["rmse_kg"])
            macros[f"rsq{key}{period}"] = f(m["r2"], 3)
            macros[f"maeCi{key}{period}"] = ci(m["mae_flight_ci95"])
            if model != ref:
                p = data["paired_vs_reference"][model]
                macros[f"delta{key}{period}"] = f(p["mae_difference_kg"])
                macros[f"deltaCi{key}{period}"] = ci(p["ci95"])
                macros[f"holm{key}{period}"] = f"{p['holm_p']:.3f}" if p["holm_p"] >= .001 else "<0.001"
        ft = oct_["flight_totals"]["models"][model]
        macros[f"invBias{key}"] = pct(ft["inventory_relative_bias"])
        macros[f"invBiasCi{key}"] = f"[{pct(ft['inventory_bias_ci95'][0])}, {pct(ft['inventory_bias_ci95'][1])}]"
        macros[f"flightMape{key}"] = pct(ft["flight_total_mape"])
        rt = oct_["route_transfer"]["mae_kg"][model]
        macros[f"seenRoute{key}"] = f(rt["seen"])
        macros[f"unseenRoute{key}"] = f(rt["unseen"])
    macros["unseenRouteN"] = f(oct_["route_transfer"]["unseen_route_intervals"], 0)
    macros["seenRouteN"] = f(oct_["route_transfer"]["seen_route_intervals"], 0)
    for model, data in unc["models"].items():
        key = macro_name(model)
        macros[f"spearman{key}"] = f(data["sigma_spearman_with_error"], 3)
        for method, iv in data["intervals"].items():
            mk = macro_name(method)
            macros[f"cov{key}{mk}"] = pct(iv["coverage"])
            macros[f"width{key}{mk}"] = f(iv["mean_width_kg"], 0)
            macros[f"worst{key}{mk}"] = pct(iv["worst_cell_coverage"])
            macros[f"below{key}{mk}"] = f(iv["cells_below_085"], 0)
            macros[f"winkler{key}{mk}"] = f(iv["winkler_score"], 0)
        prim = data["selection_primary"]
        for policy in ("risk_only", "representation_floor"):
            pk = macro_name(policy)
            macros[f"accMae{key}{pk}"] = f(prim[policy]["accepted_mae_kg"])
            macros[f"fuelShare{key}{pk}"] = pct(prim[policy]["fuel_share"])
            macros[f"longCov{key}{pk}"] = pct(prim[policy]["long_interval_coverage"])
            macros[f"bSevenCov{key}{pk}"] = pct(prim[policy]["b744_coverage"])
            macros[f"coverage{key}{pk}"] = pct(prim[policy]["coverage"])
        macros[f"floorCost{key}"] = f(prim["floor_cost_kg"])
    loto = result.get("leave_one_type_out")
    if loto:
        macros["lotoTypes"] = f(len(loto["eligible_types"]), 0)
        for model, wins in loto["types_where_model_beats_reference"].items():
            macros[f"lotoWins{macro_name(model)}"] = f(wins, 0)
            macros[f"lotoPooled{macro_name(model)}"] = f(loto["pooled_mae_kg"][model])
    abl = result["ablation_october"]
    macros["ablFull"] = f(abl["full_mae_kg"])
    for group, data in abl["drop"].items():
        macros[f"ablDrop{macro_name(group)}"] = f(data["mae_kg"])
        macros[f"ablDelta{macro_name(group)}"] = f"{data['delta_vs_full_kg']:+.2f}"
    unit = result["feature_manifest"]["final"].get("unit_check", {})
    macros["unitAlt"] = f(unit.get("median_flight_p90_raw_altitude_when_groundspeed_gt_300"), 0)
    macros["unitGs"] = f(unit.get("median_raw_groundspeed_when_gt_300"), 0)
    lines = [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in sorted(macros.items())]
    out.write_text("% Generated by research_q1.journal.artifacts; do not edit.\n" + "\n".join(lines) + "\n", encoding="utf-8")


def table_models(result: dict, out: Path) -> None:
    ref = result["config"]["reference_model"]
    rows = []
    for model in MODEL_ORDER:
        o = result["periods"]["october"]["models"][model]
        s = result["periods"]["september"]["models"][model]
        if model == ref:
            delta = "reference"
        else:
            p = result["periods"]["october"]["paired_vs_reference"][model]
            delta = f"{p['mae_difference_kg']:+.2f} {ci(p['ci95'])}"
        rows.append(f"{MODEL_LABELS[model]} & {f(s['mae_kg'])} & {f(o['mae_kg'])} & {ci(o['mae_flight_ci95'])} & "
                    f"{f(o['rmse_kg'])} & {f(o['r2'], 3)} & {delta} \\\\")
    out.write_text("\\begin{tabular}{lrrlrrl}\n\\toprule\nModel & Sep.\\ MAE & Oct.\\ MAE & Oct.\\ MAE 95\\% CI & Oct.\\ RMSE & Oct.\\ $R^2$ & $\\Delta$MAE vs.\\ ref.\\ [95\\% CI] \\\\\n"
                   " & (kg) & (kg) & (kg) & (kg) & & (kg) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def table_subgroups(result: dict, out: Path) -> None:
    sub = result["periods"]["october"]["subgroups"]
    models = ["type_median_rate", "openap_calibrated", "hgb_metadata_adsb", "hgb_full", "hgb_residual_physics"]
    head = " & ".join(["Group", "$n$"] + [MODEL_LABELS[m].replace("HGB metadata + ADS-B summaries", "HGB meta+ADS-B")
                                            .replace("HGB full (+ kinematics + physics)", "HGB full")
                                            .replace("OpenAP physics (type-scaled)", "OpenAP scaled") for m in models])
    rows = []
    for label, title in (("duration", "Duration"), ("adsb_availability", "ADS-B positions")):
        rows.append(f"\\multicolumn{{{2 + len(models)}}}{{l}}{{\\emph{{{title}}}}} \\\\")
        order = ["<5 min", "5–15 min", "15–60 min", "60+ min"] if label == "duration" else ["zero", "1-50", "51+"]
        for name in order:
            if name not in sub[label]:
                continue
            g = sub[label][name]
            rows.append(f"\\quad {name.replace('–', '--').replace('<', '$<$')} & {f(g['n'], 0)} & " +
                        " & ".join(f(g["mae_kg"][m]) for m in models) + " \\\\")
    rows.append(f"\\multicolumn{{{2 + len(models)}}}{{l}}{{\\emph{{Aircraft type (October $n \\geq 200$)}}}} \\\\")
    for name, g in sorted(sub["aircraft_type"].items(), key=lambda kv: -kv[1]["n"]):
        if g["n"] >= 200:
            rows.append(f"\\quad {name} & {f(g['n'], 0)} & " + " & ".join(f(g["mae_kg"][m]) for m in models) + " \\\\")
    out.write_text(f"\\begin{{tabular}}{{l{'r' * (1 + len(models))}}}\n\\toprule\n{head} \\\\\n\\midrule\n"
                   + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def table_intervals(result: dict, out: Path) -> None:
    unc = result["uncertainty_and_selection"]["models"]
    rows = []
    for model, data in unc.items():
        for method, iv in data["intervals"].items():
            rows.append(f"{MODEL_LABELS[model]} & {INTERVAL_LABELS[method]} & {pct(iv['coverage'])} & {f(iv['mean_width_kg'], 0)} & "
                        f"{f(iv['winkler_score'], 0)} & {pct(iv['worst_cell_coverage'])} & {iv['cells_below_085']} \\\\")
    out.write_text("\\begin{tabular}{llrrrrr}\n\\toprule\nPoint model & Interval method & Coverage & Width & Winkler & Worst cell & Cells \\\\\n"
                   " & & (\\%) & (kg) & (kg) & cov.\\ (\\%) & $<85\\%$ \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def table_selection(result: dict, out: Path) -> None:
    unc = result["uncertainty_and_selection"]["models"]
    rows = []
    for model, data in unc.items():
        p = data["selection_primary"]
        for policy, label in (("risk_only", "Risk ranking only"), ("representation_floor", "50\\% group floor")):
            r = p[policy]
            rows.append(f"{MODEL_LABELS[model]} & {label} & {pct(r['coverage'])} & {f(r['accepted_mae_kg'])} & {f(r['rejected_mae_kg'])} & "
                        f"{pct(r['fuel_share'])} & {pct(r['long_interval_coverage'])} & {pct(r['b744_coverage'])} \\\\")
    out.write_text("\\begin{tabular}{llrrrrrr}\n\\toprule\nPoint model & Policy & Released & Acc.\\ MAE & Rej.\\ MAE & Fuel share & 15--60 min & B744 \\\\\n"
                   " & & (\\%) & (kg) & (kg) & (\\%) & released (\\%) & released (\\%) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def table_loto(result: dict, out: Path) -> None:
    loto = result.get("leave_one_type_out")
    if not loto:
        out.write_text("% leave-one-type-out not run\n")
        return
    models = ["type_median_rate", "openap_physics", "openap_calibrated", "hgb_metadata_adsb", "hgb_full", "hgb_residual_physics"]
    rows = []
    for t, r in loto["types"].items():
        best = min(models, key=lambda m: r["mae_kg"][m])
        cells = [(f"\\textbf{{{f(r['mae_kg'][m])}}}" if m == best else f(r["mae_kg"][m])) for m in models]
        rows.append(f"{t} & {f(r['n'], 0)} & " + " & ".join(cells) + " \\\\")
    rows.append("\\midrule")
    rows.append("Pooled & " + f(sum(r["n"] for r in loto["types"].values()), 0) + " & " + " & ".join(f(loto["pooled_mae_kg"][m]) for m in models) + " \\\\")
    rows.append("Types beating reference & & " + " & ".join(
        ("--" if m == "hgb_metadata_adsb" else f(loto["types_where_model_beats_reference"][m], 0)) for m in models) + " \\\\")
    head = "Held-out type & $n$ & Type rate & OpenAP & OpenAP scaled & HGB meta+ADS-B & HGB full & Physics-residual"
    out.write_text(f"\\begin{{tabular}}{{lr{'r' * len(models)}}}\n\\toprule\n{head} \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def table_ablation(result: dict, out: Path) -> None:
    abl = result["ablation_october"]
    names = {"meta": "Flight metadata (duration, timing, route distance)", "adsb": "Original ADS-B summaries (14)",
             "kin": "SI kinematics and energy (14)", "phys": "OpenAP physics (3)"}
    rows = [f"{names[g]} & {f(d['mae_kg'])} & {d['delta_vs_full_kg']:+.2f} \\\\" for g, d in abl["drop"].items()]
    out.write_text("\\begin{tabular}{lrr}\n\\toprule\nRemoved feature group & Oct.\\ MAE (kg) & $\\Delta$ vs.\\ full (kg) \\\\\n\\midrule\n"
                   f"None (full model) & {f(abl['full_mae_kg'])} & 0.00 \\\\\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def figures(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    oct_ = result["periods"]["october"]["models"]
    # 1. Model comparison with flight-cluster CIs.
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    names = MODEL_ORDER
    mae = [oct_[m]["mae_kg"] for m in names]
    lo = [oct_[m]["mae_kg"] - oct_[m]["mae_flight_ci95"][0] for m in names]
    hi = [oct_[m]["mae_flight_ci95"][1] - oct_[m]["mae_kg"] for m in names]
    ax.barh(range(len(names)), mae, xerr=[lo, hi], color="#4477aa", capsize=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("October 2025 MAE (kg), 95% flight-cluster bootstrap CI")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"model_comparison.{ext}", dpi=200)
    plt.close(fig)
    # 2. Release frontier.
    unc = result["uncertainty_and_selection"]["models"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for model, style in (("hgb_metadata_adsb", "--"), ("hgb_full", "-"), ("hgb_residual_physics", ":")):
        if model not in unc:
            continue
        fr = unc[model]["selection_frontier"]
        for policy, marker, color in (("risk_only", "o", "#cc3311"), ("representation_floor", "s", "#228833")):
            x = [100 * r[policy]["coverage"] for r in fr]
            axes[0].plot(x, [r[policy]["accepted_mae_kg"] for r in fr], style, marker=marker, color=color, ms=3,
                         label=f"{MODEL_LABELS[model].split(' (')[0]}, {policy.replace('_', ' ')}")
            axes[1].plot(x, [100 * r[policy]["fuel_share"] for r in fr], style, marker=marker, color=color, ms=3)
    axes[0].set_xlabel("Released intervals (%)")
    axes[0].set_ylabel("Accepted MAE (kg)")
    axes[1].set_xlabel("Released intervals (%)")
    axes[1].set_ylabel("Labelled fuel represented (%)")
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"release_frontier.{ext}", dpi=200)
    plt.close(fig)
    # 3. Cell coverage.
    model = "hgb_full" if "hgb_full" in unc else next(iter(unc))
    cells = unc[model]["intervals"]["mondrian_type_duration"]["cells"]
    const = unc[model]["intervals"]["split_constant"]["cells"]
    names = sorted(cells, key=lambda c: cells[c]["coverage"])
    fig, ax = plt.subplots(figsize=(6.4, 0.18 * len(names) + 1.2))
    y = np.arange(len(names))
    ax.scatter([100 * const[c]["coverage"] for c in names], y, marker="x", color="#cc3311", label="Split conformal, constant width", s=18)
    ax.scatter([100 * cells[c]["coverage"] for c in names], y, marker="o", color="#4477aa", label="Hierarchical Mondrian", s=18)
    ax.axvline(90, color="k", lw=.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("October empirical coverage (%) of nominal 90% intervals, by aircraft type / duration cell")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"cell_coverage.{ext}", dpi=200)
    plt.close(fig)
    # 4. Risk deciles.
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for model, marker in (("hgb_metadata_adsb", "x"), ("hgb_full", "o")):
        if model in unc:
            d = unc[model]["sigma_deciles"]
            ax.plot([r["mean_sigma_kg"] for r in d], [r["observed_mae_kg"] for r in d], marker=marker, ms=4,
                    label=MODEL_LABELS[model].split(" (")[0])
    lim = ax.get_xlim()
    ax.plot(lim, lim, "k--", lw=.8)
    ax.set_xlabel("Mean predicted error $\\hat\\sigma$ (kg), decile")
    ax.set_ylabel("Observed MAE (kg)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"risk_deciles.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "research_q1/outputs/journal-study.json")
    parser.add_argument("--out", type=Path, default=ROOT / "research_q1/paper/lncs/generated")
    args = parser.parse_args()
    result = json.loads(args.results.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    write_macros(result, args.out / "macros.tex")
    table_models(result, args.out / "table_models.tex")
    table_subgroups(result, args.out / "table_subgroups.tex")
    table_intervals(result, args.out / "table_intervals.tex")
    table_selection(result, args.out / "table_selection.tex")
    table_loto(result, args.out / "table_loto.tex")
    table_ablation(result, args.out / "table_ablation.tex")
    figures(result, args.out / "figures")
    print(f"Wrote artifacts to {args.out}")


if __name__ == "__main__":
    main()
