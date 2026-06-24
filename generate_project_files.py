from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(".")
SUMMARY_CSV = ROOT / "summary_results.csv"
SINGLE_EVAL_CSV = ROOT / "single_dataset_comparisons.csv"
MULTIPLE_EVAL_CSV = ROOT / "multiple_dataset_comparisons.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt_pct(value: str | float) -> str:
    return f"{float(value) * 100:.2f}"


def fmt_float(value: str | float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def nbc_vs_bagging_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["baseline"] == "NBC"]


def bagging_size_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["baseline"] != "NBC"]


def make_markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| Dataset | Model | Base models | Mean accuracy (%) | Std. dev. (%) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['base_models']} | "
            f"{fmt_pct(row['mean_accuracy'])} | {fmt_pct(row['std_accuracy'])} |"
        )
    return "\n".join(lines)


def make_markdown_eval_table(rows: list[dict[str, str]], include_dataset: bool) -> str:
    if include_dataset:
        lines = [
            "| Dataset | Baseline | Comparison | Dataset diff. (%) | Dataset z | Dataset p | Fold diff. (%) | Matched t | Matched p |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    else:
        lines = [
            "| Baseline | Comparison | Dataset-average diff. (%) | Dataset z | Dataset p | Fold-average diff. (%) | Matched t | Matched p |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    for row in rows:
        cells = []
        if include_dataset:
            cells.append(row["dataset"])
        if include_dataset:
            cells.extend([
                row["baseline"], row["comparison"],
                fmt_pct(row["dataset_difference"]), fmt_float(row["dataset_z"]),
                fmt_float(row["dataset_p_value"]), fmt_pct(row["fold_mean_difference"]),
                fmt_float(row["fold_matched_t"]), fmt_float(row["fold_matched_p_value"]),
            ])
        else:
            cells.extend([
                row["baseline"], row["comparison"],
                fmt_pct(row["dataset_average_difference"]), fmt_float(row["dataset_average_z"]),
                fmt_float(row["dataset_average_p_value"]), fmt_pct(row["fold_average_difference"]),
                fmt_float(row["fold_matched_t"]), fmt_float(row["fold_matched_p_value"]),
            ])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def make_latex_summary_table(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Five-fold cross-validation accuracy.}",
        r"\label{tab:summary}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Dataset & Model & Base models & Accuracy (\%) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['dataset']} & {row['model']} & {row['base_models']} & "
            f"{fmt_pct(row['mean_accuracy'])} $\\pm$ {fmt_pct(row['std_accuracy'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def make_latex_eval_table(
    rows: list[dict[str, str]],
    caption: str,
    label: str,
    include_dataset: bool,
) -> str:
    if include_dataset:
        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\scriptsize",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{lllrrrrrr}",
            r"\toprule",
            r"Dataset & Baseline & Comparison & Data diff. (\%) & $z$ & $p_z$ & Fold diff. (\%) & $t$ & $p_t$ \\",
            r"\midrule",
        ]
    else:
        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{llrrrrrr}",
            r"\toprule",
            r"Baseline & Comparison & Data avg. diff. (\%) & $z$ & $p_z$ & Fold avg. diff. (\%) & $t$ & $p_t$ \\",
            r"\midrule",
        ]
    for row in rows:
        prefix = f"{row['dataset']} & " if include_dataset else ""
        if include_dataset:
            values = [row["dataset_difference"], row["dataset_z"], row["dataset_p_value"],
                      row["fold_mean_difference"], row["fold_matched_t"], row["fold_matched_p_value"]]
        else:
            values = [row["dataset_average_difference"], row["dataset_average_z"], row["dataset_average_p_value"],
                      row["fold_average_difference"], row["fold_matched_t"], row["fold_matched_p_value"]]
        lines.append(
            f"{prefix}{row['baseline']} & {row['comparison']} & {fmt_pct(values[0])} & "
            f"{fmt_float(values[1])} & {fmt_float(values[2])} & {fmt_pct(values[3])} & "
            f"{fmt_float(values[4])} & {fmt_float(values[5])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def write_notebook(
    summary_rows: list[dict[str, str]],
    single_rows: list[dict[str, str]],
    multiple_rows: list[dict[str, str]],
) -> None:
    implementation = Path("run_experiments.py").read_text(encoding="utf-8")
    implementation = implementation.replace('\n\nif __name__ == "__main__":\n    main()\n', "\n")
    result_table = make_markdown_table(summary_rows)
    single_nbc_table = make_markdown_eval_table(nbc_vs_bagging_rows(single_rows), True)
    single_bagging_table = make_markdown_eval_table(bagging_size_rows(single_rows), True)
    multiple_table = make_markdown_eval_table(multiple_rows, False)

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Final Project: Naive Bayesian Classifier and Bagging\n",
                    "\n",
                    "This notebook implements the final project experiment for three UCI datasets: Banknote Authentication, Glass Identification, and Image Segmentation. Continuous attributes are discretized by equal-width binning with ten bins. Missing values are ignored when estimating conditional probabilities, and Laplace estimates are used for the Naive Bayes probabilities.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Dataset Loading\n",
                    "\n",
                    "The assignment's image dataset snippet repeats the Glass dataset code. The UCI Image Segmentation dataset is loaded with `fetch_ucirepo(id=50)`.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": implementation.splitlines(keepends=True),
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Run All Experiments\n",
                    "\n",
                    "Executing the next cell fetches the UCI datasets, evaluates the original NBC and bagging ensembles with 10, 20, and 30 base models, and writes `fold_results.csv`, `summary_results.csv`, `single_dataset_comparisons.csv`, and `multiple_dataset_comparisons.csv`.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["main()\n"],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Results From This Run\n", "\n", result_table, "\n"],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Result Evaluation\n",
                    "\n",
                    "The cross validation split follows Chapter 4 method III: each instance receives a uniform random number, instances are sorted by that number, and the sorted instances are assigned to the five folds cyclically. This keeps fold sizes as equal as possible, with any two folds differing by at most one instance.\n",
                    "\n",
                    "Chapter 5 defines two aggregation levels for a single dataset. Dataset aggregation pools the correct predictions across all folds and applies Theorem 5.1's two-proportion z test. Fold aggregation pairs the two model accuracies from each identical test fold and applies Theorem 5.2's matched-sample t test. For multiple datasets, dataset averaging follows Theorem 5.4, while matched fold averaging follows Theorems 5.5 and 5.6.\n",
                    "\n",
                    "### Single Dataset: NBC vs. Bagging\n",
                    "\n",
                    single_nbc_table,
                    "\n\n",
                    "Positive differences mean the comparison model is more accurate than the baseline. In the current run, none of the NBC-versus-bagging comparisons is significant at 0.05 under either single-dataset level. The largest gain is Bagging-30 over NBC on Glass Identification: 4.21 percentage points for dataset aggregation (p = 0.3692) and 4.23 points for matched fold aggregation (p = 0.2930).\n",
                    "\n",
                    "### Single Dataset: Bagging Size Comparison\n",
                    "\n",
                    single_bagging_table,
                    "\n\n",
                    "The single-dataset bagging-size tests report both levels. The only p-value below 0.05 is the matched fold comparison of Bagging-10 with Bagging-30 on Glass Identification (difference = 2.79 points, p = 0.0327); its dataset-aggregation test is not significant (p = 0.5479).\n",
                    "\n",
                    "### Multiple Dataset Comparison\n",
                    "\n",
                    multiple_table,
                    "\n\n",
                    "Across all three datasets, Bagging-30 has the largest average advantage over NBC: 1.50 percentage points at dataset averaging (p = 0.4683) and 1.51 points at matched fold averaging (p = 0.3002). Neither result is significant at 0.05, and no multiple-dataset comparison among bagging sizes is significant.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Summary\n",
                    "\n",
                    "Bagging-30 has the best observed mean accuracy on Banknote Authentication and Glass Identification and ties NBC on Image Segmentation. However, no NBC-versus-bagging comparison is significant at alpha = 0.05 at either single- or multiple-dataset aggregation levels. The evidence therefore supports a descriptive advantage for Bagging-30 in this run, but not a statistically confirmed general advantage.\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    Path("Final-Project.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def write_report_tex(
    summary_rows: list[dict[str, str]],
    single_rows: list[dict[str, str]],
    multiple_rows: list[dict[str, str]],
) -> None:
    summary_table = make_latex_summary_table(summary_rows)
    single_nbc_table = make_latex_eval_table(
        nbc_vs_bagging_rows(single_rows),
        "Single-dataset NBC versus bagging comparisons. Positive differences favor bagging.",
        "tab:single-nbc",
        True,
    )
    single_bagging_table = make_latex_eval_table(
        bagging_size_rows(single_rows),
        "Single-dataset comparisons among bagging ensemble sizes.",
        "tab:single-bagging",
        True,
    )
    multiple_table = make_latex_eval_table(
        multiple_rows,
        "Multiple-dataset comparisons at the data-set-averaging and matched fold-averaging levels.",
        "tab:multiple",
        False,
    )
    content = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{hyperref}}
\usepackage{{array}}
\usepackage{{float}}
\title{{Final Project: Naive Bayesian Classifier and Bagging}}
\author{{}}
\date{{}}

\begin{{document}}
\maketitle

\section{{Introduction}}
This project evaluates a Naive Bayesian classifier (NBC) and bagging ensembles of NBC models on three UCI Machine Learning Repository datasets: Banknote Authentication, Glass Identification, and Image Segmentation. The goal is to compare the original NBC with bagging ensembles containing 10, 20, and 30 base models.

\section{{Methodology}}
All datasets were downloaded by the \texttt{{ucimlrepo}} package. The dataset identifiers were 267 for Banknote Authentication, 42 for Glass Identification, and 50 for Image Segmentation. Continuous attributes were discretized with equal-width discretization into ten bins. The discretizer was fit only on the training portion of each fold, then applied to the corresponding test portion.

The NBC model estimates class priors and feature conditional probabilities with Laplace estimation. Missing feature values are ignored when estimating conditional probabilities and are also skipped during prediction. For bagging, each base NBC was trained on a bootstrap sample of the training fold. Final predictions were produced by majority vote. Ties were broken deterministically by the sorted class labels.

Performance was measured by five-fold cross validation using only train and test splits. The fold assignment follows Chapter 4 method III. First, each instance is assigned a random number from the uniform interval [0,1]. Second, instances are sorted by the random number. Third, the sorted instances are read sequentially and assigned cyclically to folds 1 through 5. This makes the fold sizes approximately equal, and the size difference between any two folds is at most one. The random seed was fixed at 42 for reproducibility.

Four Chapter 5 comparisons were computed. For one data set, data-set aggregation pools correct predictions over all test folds and applies Theorem 5.1's two-proportion statistic
\[
z=\frac{{\hat p_2-\hat p_1}}{{\sqrt{{2\hat p(1-\hat p)/n}}}},
\qquad \hat p=\frac{{\hat p_1+\hat p_2}}{{2}}.
\]
Fold aggregation uses the matched differences $d_l=\hat p_{{2l}}-\hat p_{{1l}}$ from the same folds and Theorem 5.2's statistic $t=\bar d/\sqrt{{s_d^2/k}}$ with $k-1$ degrees of freedom. For multiple data sets, Theorem 5.4 averages each data set's pooled accuracy difference with equal data-set weight and estimates its normal variance from each data set's size. Theorems 5.5 and 5.6 average matched fold differences inside each data set and then across data sets; their $t$ statistic combines the within-data-set difference variances, and the degrees of freedom are rounded down according to Theorem 5.6. Positive differences in all tables mean that the comparison model is more accurate than the baseline. All data-set-level counts satisfy Chapter 5's large-sample condition.

\section{{Experimental Results}}
{summary_table}

\section{{Analysis}}
The current run produced the same deterministic results whenever it was repeated with seed 42. For Banknote Authentication, NBC obtained 89.80\% mean fold accuracy and Bagging-30 obtained the highest value, 90.09\%, an increase of only 0.29 percentage points at the data-set aggregation level.

Glass Identification showed the largest numerical benefit. NBC obtained 60.29\% mean fold accuracy, compared with 61.73\%, 62.60\%, and 64.52\% for Bagging-10, Bagging-20, and Bagging-30. Thus, Bagging-30 improved the data-set-aggregated accuracy over NBC by 4.21 percentage points.

For Image Segmentation, NBC, Bagging-20, and Bagging-30 all produced 80.00\% mean accuracy, while Bagging-10 obtained 79.05\%. Bagging therefore did not improve the final accuracy on this data set under the selected preprocessing and model settings.

\section{{Result Evaluation}}
Table~\ref{{tab:single-nbc}} reports both single-data-set aggregation levels. None of the NBC-versus-bagging comparisons is significant at $\alpha=0.05$. The largest observed gain is NBC versus Bagging-30 on Glass Identification: the pooled data-set difference is 4.21 percentage points ($z=0.8980$, $p=0.3692$), and the matched fold difference is 4.23 points ($t=1.2097$, $df=4$, $p=0.2930$).

{single_nbc_table}

Table~\ref{{tab:single-bagging}} compares ensemble sizes. The matched fold test finds one significant result: on Glass Identification, Bagging-30 exceeds Bagging-10 by 2.79 points ($t=3.2071$, $df=4$, $p=0.0327$). The data-set aggregation test for the same comparison is not significant ($z=0.6010$, $p=0.5479$). This difference illustrates that the aggregation level changes the sampling variance and therefore the inference.

{single_bagging_table}

Table~\ref{{tab:multiple}} reports the two multiple-data-set levels. Bagging-30 has the largest average advantage over NBC: 1.50 percentage points at the data-set-averaging level ($z=0.7253$, $p=0.4683$) and 1.51 points at the matched fold-averaging level ($t=1.1336$, $df=6$, $p=0.3002$). Neither result is significant. No comparison among ensemble sizes is significant under either multiple-data-set test; the smallest such p-value is 0.1708 for the matched Bagging-10 versus Bagging-30 comparison.

{multiple_table}

\section{{Conclusion}}
The current run shows a modest descriptive advantage for larger bagging ensembles: Bagging-30 has the highest mean accuracy on Banknote Authentication and Glass Identification and ties NBC on Image Segmentation. However, every NBC-versus-bagging test has $p>0.05$ at both the single-data-set and multiple-data-set levels. The only significant result is the matched fold comparison between Bagging-10 and Bagging-30 on Glass Identification, and its pooled data-set test is not significant. Consequently, this experiment does not provide consistent statistical evidence that bagging outperforms NBC or that 30 estimators is universally superior. It supports only the narrower conclusion that Bagging-30 has the best observed average performance for these settings and this fixed run.

\end{{document}}
"""
    Path("report.tex").write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def add_pdf_page(pdf: PdfPages, title: str, body: str, fontsize: int = 10) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.95, title, fontsize=16, weight="bold", va="top")
    y = 0.89
    for paragraph in body.split("\n\n"):
        for line in textwrap.wrap(paragraph, width=98):
            fig.text(0.08, y, line, fontsize=fontsize, va="top")
            y -= 0.024
        y -= 0.018
    plt.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_report_pdf(
    summary_rows: list[dict[str, str]],
    single_rows: list[dict[str, str]],
    multiple_rows: list[dict[str, str]],
) -> None:
    table_lines = [
        "Dataset                         Model              Base  Accuracy",
        "------------------------------------------------------------------",
    ]
    for row in summary_rows:
        table_lines.append(
            f"{row['dataset'][:29]:29} {row['model'][:18]:18} "
            f"{row['base_models']:>4}  {fmt_pct(row['mean_accuracy'])}% +/- {fmt_pct(row['std_accuracy'])}%"
        )

    single_lines = [
        "Dataset              Comparison       Data diff  Data z  Data p  Fold diff  Match t  Match p",
        "--------------------------------------------------------------------------------",
    ]
    for row in nbc_vs_bagging_rows(single_rows):
        single_lines.append(
            f"{row['dataset'][:20]:20} {row['comparison'][:18]:18} "
            f"{fmt_pct(row['dataset_difference']):>6}%  "
            f"{fmt_float(row['dataset_z']):>7}  {fmt_float(row['dataset_p_value']):>7}  "
            f"{fmt_pct(row['fold_mean_difference']):>6}%  "
            f"{fmt_float(row['fold_matched_t']):>7}  {fmt_float(row['fold_matched_p_value']):>7}"
        )
    single_bagging_lines = [
        "Dataset              Baseline       Comparison   Data diff Data z Data p Fold diff Match t Match p",
        "-------------------------------------------------------------------------------------------",
    ]
    for row in bagging_size_rows(single_rows):
        single_bagging_lines.append(
            f"{row['dataset'][:20]:20} {row['baseline'][:13]:13} {row['comparison'][:13]:13} "
            f"{fmt_pct(row['dataset_difference']):>6}%  "
            f"{fmt_float(row['dataset_z']):>7}  {fmt_float(row['dataset_p_value']):>7}  "
            f"{fmt_pct(row['fold_mean_difference']):>6}%  "
            f"{fmt_float(row['fold_matched_t']):>7}  {fmt_float(row['fold_matched_p_value']):>7}"
        )

    multiple_lines = [
        "Baseline            Comparison       Data avg  Data z  Data p  Fold avg  Match t  Match p",
        "-------------------------------------------------------------------------------",
    ]
    for row in multiple_rows:
        multiple_lines.append(
            f"{row['baseline'][:18]:18} {row['comparison'][:18]:18} "
            f"{fmt_pct(row['dataset_average_difference']):>6}%  "
            f"{fmt_float(row['dataset_average_z']):>7}  {fmt_float(row['dataset_average_p_value']):>7}  "
            f"{fmt_pct(row['fold_average_difference']):>6}%  "
            f"{fmt_float(row['fold_matched_t']):>7}  {fmt_float(row['fold_matched_p_value']):>7}"
        )

    page1 = (
        "Introduction\n"
        "This project evaluates a Naive Bayesian classifier and bagging ensembles on Banknote Authentication, Glass Identification, and Image Segmentation from the UCI Machine Learning Repository.\n\n"
        "Methodology\n"
        "Continuous attributes are discretized by equal-width binning into ten bins. The discretizer is fit only on each training fold. Missing values are ignored for conditional probability estimation and skipped during prediction. Laplace estimates are used for the Naive Bayes probabilities. Bagging trains each base model on a bootstrap sample and combines predictions with majority vote. Performance is measured by five-fold cross validation using Chapter 4 method III: random numbers, sorting, and cyclic fold assignment.\n\n"
        "Results\n"
        + "\n".join(table_lines)
    )
    page2 = (
        "Single Dataset Evaluation\n"
        "Chapter 5 matched-sample tests compare fold-by-fold differences on the same folds. Independent-sample tests are also shown as secondary comparisons.\n\n"
        + "\n".join(single_lines)
        + "\n\nSingle Dataset Bagging Size Comparison\n"
        + "\n".join(single_bagging_lines)
    )
    page3 = (
        "Multiple Dataset Evaluation\n"
        "Across all three datasets, Bagging-30 has the largest average improvement over NBC, but neither the matched-sample nor independent-sample Chapter 5 test is significant at alpha = 0.05.\n\n"
        + "\n".join(multiple_lines)
        + "\n\nConclusion\n"
        "Bagging improves mean accuracy for Banknote Authentication and Glass Identification and ties NBC on Image Segmentation for 20 and 30 base models. The improvements are positive but not statistically confirmed at the 0.05 level."
    )
    with PdfPages("report.pdf") as pdf:
        add_pdf_page(pdf, "Final Project: Naive Bayesian Classifier and Bagging", page1, fontsize=9)
        add_pdf_page(pdf, "Chapter 5 Single-Dataset Evaluation", page2, fontsize=7)
        add_pdf_page(pdf, "Chapter 5 Multiple-Dataset Evaluation", page3, fontsize=8)


def main() -> None:
    summary_rows = read_csv(SUMMARY_CSV)
    single_rows = read_csv(SINGLE_EVAL_CSV)
    multiple_rows = read_csv(MULTIPLE_EVAL_CSV)
    write_notebook(summary_rows, single_rows, multiple_rows)
    write_report_tex(summary_rows, single_rows, multiple_rows)
    write_report_pdf(summary_rows, single_rows, multiple_rows)


if __name__ == "__main__":
    main()
