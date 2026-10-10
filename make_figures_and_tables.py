"""Create manuscript tables and boxplots from completed analysis outputs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


METHODS = ("POR1", "POR2", "PH", "PIPW", "PMR")
LABELS = {"POR1": r"$\mathrm{POR}_1$", "POR2": r"$\mathrm{POR}_2$",
          "PH": "PH", "PIPW": "PIPW", "PMR": "PMR", "naive": "Naive", "MR": "MR"}
SIZES = (1000, 2000, 4000, 8000)
COLORS = ("#E69F00", "#56B4E9", "#009E73", "#6A5ACD")


def write(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fixed(value, digits):
    """Format table entries without displaying negative rounded zero."""
    if abs(value) < 0.5 * 10**(-digits):
        value = 0.0
    return f"{value:.{digits}f}"


def completed_summary(path):
    frame = pd.read_csv(path)
    if frame.empty or (frame["pending"] != 0).any() or (frame["completed"] < 1).any():
        raise ValueError(f"Simulation is incomplete: {path}")
    if "failed" in frame and (frame["failed"] != 0).any():
        raise ValueError(f"Simulation contains failed fits: {path}")
    return frame


def parametric_table(source, output):
    points = completed_summary(source / "summary.csv")
    intervals = completed_summary(source / "inference_summary.csv")
    counts = points["completed"].unique()
    if len(counts) != 1:
        raise ValueError("Parametric table requires the same repetition count in each cell")
    repetitions = int(counts[0])
    points = points.set_index(["n", "scenario", "estimator"], verify_integrity=True)
    intervals = intervals.set_index(["n", "scenario"], verify_integrity=True)
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Empirical bias and root mean squared error (RMSE) of the parametric estimators, "
        r"and coverage and mean length of nominal 95\% PMR confidence intervals, over "
        + str(repetitions) + " replications.}",
        r"\label{tab:parametric_bias_rmse}", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{rl*{12}{c}}", r"\toprule",
        r"$n$ & Scenario & \multicolumn{2}{c}{POR$_1$} & \multicolumn{2}{c}{POR$_2$} "
        r"& \multicolumn{2}{c}{PH} & \multicolumn{2}{c}{PIPW} & \multicolumn{4}{c}{PMR} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-14}",
        r"& & Bias & RMSE & Bias & RMSE & Bias & RMSE & Bias & RMSE & Bias & RMSE & Coverage & Length \\",
        r"\midrule"]
    for index, n in enumerate(sorted(points.index.get_level_values("n").unique())):
        if index:
            lines.append(r"\addlinespace")
        for scenario in range(1, 6):
            cells = [str(n) if scenario == 1 else "", str(scenario)]
            for method in METHODS:
                row = points.loc[(n, scenario, method)]
                bold = scenario == 5 or method == "PMR" or method == METHODS[scenario - 1]
                for metric in ("bias", "rmse"):
                    value = fixed(row[metric], 3)
                    cells.append(r"\textbf{" + value + "}" if bold else value)
            ci = intervals.loc[(n, scenario)]
            cells.extend([fixed(100 * ci['coverage'], 1), fixed(ci['mean_ci_width'], 3)])
            lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
        r"\begin{minipage}{\textwidth}\footnotesize",
        r"Bias is the mean estimation error; RMSE is the square root of the mean squared error. "
        r"Coverage is reported as a percentage; length is the mean interval width. "
        r"Bold bias and RMSE entries denote estimators whose consistency conditions hold in the corresponding scenario. "
        r"Scenarios 1--4 correspond to $\mathcal M_1$--$\mathcal M_4$, respectively; all working models are correctly specified in Scenario 5. "
        r"Confidence intervals use the original EIF-based standard error, without a correction for nuisance estimation under misspecification.",
        r"\end{minipage}", r"\end{table}"])
    write(output / "parametric_table.tex", lines)


def nonparametric_figure(source, output):
    frames, repetitions = [], []
    for case in (1, 2):
        folder = source / f"case{case}"
        summary = completed_summary(folder / "summary.csv")
        frame = pd.read_csv(folder / "detailed.csv")
        counts = summary["completed"].unique()
        if len(counts) != 1 or frame.duplicated(["n", "replicate"]).any():
            raise ValueError(f"Invalid repetition counts in Case {case}")
        for n in summary["n"].unique():
            if len(frame[frame.n == n]) != int(counts[0]):
                raise ValueError(f"Detailed records and summaries disagree in Case {case}")
        frames.append(frame)
        repetitions.append(int(counts[0]))
    sizes = sorted(frames[0]["n"].unique())
    if sizes != sorted(frames[1]["n"].unique()) or repetitions[0] != repetitions[1]:
        raise ValueError("Both cases must have matching sample sizes and repetition counts")
    methods = (*METHODS, "naive")
    colors = [COLORS[SIZES.index(n)] for n in sizes]
    offsets = np.linspace(-0.30, 0.30, len(sizes)) if len(sizes) > 1 else [0.0]
    all_errors = np.concatenate([frame[[f"{m}_residual" for m in methods]].to_numpy().ravel()
                                 for frame in frames])
    if not np.isfinite(all_errors).all():
        raise ValueError("Estimation errors must be finite")
    lower, upper = min(-1.12, all_errors.min()), max(1.55, all_errors.max())
    padding = 0.03 * (upper - lower)
    plt.rcParams.update({"font.family": "serif", "font.size": 11, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.7), sharey=True)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.18, top=0.775, wspace=0.12)
    for case, (ax, frame) in enumerate(zip(axes, frames), start=1):
        for n, offset, color in zip(sizes, offsets, colors):
            values = [frame.loc[frame.n == n, f"{m}_residual"].to_numpy() for m in methods]
            ax.boxplot(values, positions=np.arange(len(methods)) + offset, widths=0.12,
                whis=1.5, showfliers=True, patch_artist=True, manage_ticks=False,
                boxprops=dict(facecolor=color, edgecolor="#333333", linewidth=0.65, alpha=0.85),
                medianprops=dict(color="#171717", linewidth=0.95),
                whiskerprops=dict(color="#555555", linewidth=0.65),
                capprops=dict(color="#555555", linewidth=0.65),
                flierprops=dict(marker="o", markersize=1.7, markerfacecolor=color,
                                markeredgecolor=color, alpha=0.6))
        ax.axhline(0, color="#B22222", linestyle="--", linewidth=0.85)
        ax.set_xticks(range(len(methods)), [LABELS[m] for m in methods], fontsize=10)
        ax.set_xlim(-0.65, len(methods) - 0.35)
        ax.set_ylim(lower - padding, upper + padding)
        ax.set_title(f"({'ab'[case - 1]}) Case {case}", fontsize=13, loc="left")
        ax.grid(axis="y", color="#E1E3E6", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supylabel(r"Estimation error ($\hat\psi-\psi$)", fontsize=12, x=0.012)
    fig.legend([Patch(facecolor=c, edgecolor="#333333") for c in colors],
               [f"$n={n}$" for n in sizes], loc="upper center", ncol=len(sizes),
               frameon=False, fontsize=11, bbox_to_anchor=(0.54, 1.0))
    filename = "nonparametric_errors"
    fig.savefig(output / f"{filename}.pdf")
    fig.savefig(output / f"{filename}.png", dpi=300)
    plt.close(fig)
    write(output / "nonparametric_figure.tex", [r"\begin{figure}[htbp]", r"\centering",
        r"\includegraphics[width=\textwidth]{" + filename + ".pdf}",
        r"\caption{Boxplots of estimation errors $\hat\psi-\psi$ in (a) Case 1 and (b) Case 2, over "
        + str(repetitions[0]) + r" replications at each sample size. The dashed line marks zero estimation error.}",
        r"\label{fig:nonparametric_errors}", r"\end{figure}"])


def application_table(source, output):
    result = json.loads((source / "result.json").read_text(encoding="utf-8"))
    points = {row["method"]: row["estimate"] for row in result["results"]}
    methods = (*METHODS, "MR")
    values = [result["generated_population_truth"], *(points[m] for m in methods)]
    if not np.isfinite(values).all():
        raise ValueError("Application estimates must be finite")
    write(output / "application_table.tex", [r"\begin{table}[htbp]", r"\centering",
        r"\caption{Path-specific effect estimates in the ACTG175-based application.}",
        r"\label{tab:actg175_application}", r"\begin{tabular}{l*{7}{c}}", r"\toprule",
        r"& True value & POR$_1$ & POR$_2$ & PH & PIPW & PMR & MR \\", r"\midrule",
        "Estimate & " + " & ".join(fixed(value, 4) for value in values) + r" \\",
        r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
        r"\begin{minipage}{\textwidth}\footnotesize",
        r"Estimates are on the $\log(1+\mathrm{CD4})$ scale. MR uses the generated $M_1$; the other estimators use proxy information with $M_1$ hidden.",
        r"\end{minipage}", r"\end{table}"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/manuscript"))
    parser.add_argument("--study", choices=("all", "parametric", "nonparametric", "application"), default="all")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exporters = {"parametric": parametric_table, "nonparametric": nonparametric_figure,
                 "application": application_table}
    for name, export in exporters.items():
        if args.study in ("all", name):
            export(args.results_dir / name, args.output_dir)
            print(f"Exported {name} to {args.output_dir}")


if __name__ == "__main__":
    main()
