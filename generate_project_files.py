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
            "| Dataset | Baseline | Comparison | Mean diff. (%) | Matched t | Matched p | Independent t | Independent p |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    else:
        lines = [
            "| Baseline | Comparison | Mean diff. (%) | Matched t | Matched p | Independent t | Independent p |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    for row in rows:
        cells = []
        if include_dataset:
            cells.append(row["dataset"])
        cells.extend(
            [
                row["baseline"],
                row["comparison"],
                fmt_pct(row["mean_difference"]),
                fmt_float(row["matched_t"]),
                fmt_float(row["matched_p_value"]),
                fmt_float(row["independent_t"]),
                fmt_float(row["independent_p_value"]),
            ]
        )
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
            r"\begin{tabular}{lllrrrrr}",
            r"\toprule",
            r"Dataset & Baseline & Comparison & Diff. (\%) & Matched $t$ & Matched $p$ & Indep. $t$ & Indep. $p$ \\",
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
            r"\begin{tabular}{llrrrrr}",
            r"\toprule",
            r"Baseline & Comparison & Diff. (\%) & Matched $t$ & Matched $p$ & Indep. $t$ & Indep. $p$ \\",
            r"\midrule",
        ]
    for row in rows:
        prefix = f"{row['dataset']} & " if include_dataset else ""
        lines.append(
            f"{prefix}{row['baseline']} & {row['comparison']} & "
            f"{fmt_pct(row['mean_difference'])} & "
            f"{fmt_float(row['matched_t'])} & "
            f"{fmt_float(row['matched_p_value'])} & "
            f"{fmt_float(row['independent_t'])} & "
            f"{fmt_float(row['independent_p_value'])} \\\\"
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
                    "Chapter 5 gives two ways to compare two classification algorithms. For a single dataset, the matched-sample method compares fold-by-fold differences because both algorithms are tested on the same folds. The independent-sample method treats the two sets of fold accuracies as independent and uses a Welch-style t statistic. In this project, the matched-sample method is the main reference because every model uses the same folds; independent-sample results are included for comparison.\n",
                    "\n",
                    "### Single Dataset: NBC vs. Bagging\n",
                    "\n",
                    single_nbc_table,
                    "\n\n",
                    "Positive mean difference means the comparison model is more accurate than the baseline. At the 0.05 level, none of the NBC-vs-bagging differences are significant under the matched-sample test, although the best bagging mean is higher than NBC for Banknote Authentication and Glass Identification.\n",
                    "\n",
                    "### Single Dataset: Bagging Size Comparison\n",
                    "\n",
                    single_bagging_table,
                    "\n\n",
                    "The single-dataset bagging-size tests show one matched-sample significant result: on Glass Identification, Bagging-30 is higher than Bagging-10 with p = 0.0327. The corresponding independent-sample p-value is 0.7207, which shows how much less sensitive the independent-sample method can be when the same folds were actually used.\n",
                    "\n",
                    "### Multiple Dataset Comparison\n",
                    "\n",
                    multiple_table,
                    "\n\n",
                    "Across all three datasets, Bagging-30 has the largest average improvement over NBC, but the matched-sample p-value is 0.3002 and the independent-sample p-value is 0.5800. Therefore, using the Chapter 5 tests, the observed overall advantage is not statistically significant at alpha = 0.05.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Summary\n",
                    "\n",
                    "Bagging improved the mean accuracy for Banknote Authentication and Glass Identification when the best ensemble size was selected. Image Segmentation tied the original NBC with 20 and 30 base models. The largest bagging gain over NBC was on Glass Identification with 30 base models. However, the Chapter 5 matched-sample tests do not show a statistically significant NBC-vs-bagging improvement at alpha = 0.05.\n",
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
        "Multiple-dataset comparisons using Chapter 5 fold-averaging methods.",
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

\section{{Experimental Results}}
{summary_table}

\section{{Analysis}}
Bagging helped at least one ensemble size on Banknote Authentication and Glass Identification, but the best number of base models was not identical across datasets. For Banknote Authentication, the original NBC reached 89.80\% accuracy. Bagging with 30 base models obtained the best result, 90.09\%, but the improvement was small.

Glass Identification benefited the most from bagging. The original NBC reached 60.29\%, while bagging with 30 base models reached 64.52\%. The 10-model and 20-model ensembles were also above NBC, so bagging was consistently better on this dataset in mean accuracy.

For Image Segmentation, NBC, Bagging-20, and Bagging-30 all produced 80.00\% mean accuracy, while Bagging-10 decreased to 79.05\%. This suggests that bagging did not materially improve this dataset under the chosen discretization.

\section{{Result Evaluation}}
Chapter 5 describes matched-sample and independent-sample approaches for comparing two classification algorithms. In a single dataset, the matched-sample method compares the fold-by-fold accuracy differences. Its statistic is $t=\bar{{d}}/\sqrt{{s_d^2/k}}$ with $k-1$ degrees of freedom. The independent-sample method compares the two fold-accuracy samples as if they were independent, using $t=(\bar{{x}}_1-\bar{{x}}_2)/\sqrt{{s_1^2/k+s_2^2/k}}$ with Welch degrees of freedom. Because all models in this project are tested on the same folds, the matched-sample test is the more appropriate primary test; the independent-sample values are reported as secondary checks.

{single_nbc_table}

Table~\ref{{tab:single-bagging}} compares different bagging sizes inside each dataset. One matched-sample result is significant at $\alpha=0.05$: for Glass Identification, Bagging-30 is better than Bagging-10 with $p=0.0327$. The corresponding independent-sample $p$-value is 0.7207, showing that the independent-sample method is much less sensitive when the same folds are actually shared.

{single_bagging_table}

For multiple datasets, Chapter 5 extends the fold-averaging approach. The matched-sample method first averages the fold differences inside each dataset and then averages these differences across datasets. The independent-sample method averages each algorithm's fold accuracies inside each dataset and compares the global means. Table~\ref{{tab:multiple}} reports both methods for NBC versus bagging and for different bagging sizes.

{multiple_table}

The statistical comparison supports a cautious conclusion. Bagging-30 has the largest average improvement over NBC across datasets, but its matched-sample $p$-value is 0.3002 and its independent-sample $p$-value is 0.5800, so the improvement is not statistically significant at $\alpha=0.05$. Comparing different bagging sizes, Bagging-30 has the highest average accuracy among ensembles, but none of the multiple-dataset bagging-size comparisons are significant at $\alpha=0.05$.

\section{{Conclusion}}
Overall, bagging was better than the original NBC in mean accuracy for Banknote Authentication and Glass Identification, and it tied NBC on Image Segmentation when 20 or 30 base models were used. The best ensemble size was 30 for Banknote Authentication and Glass Identification, while Image Segmentation showed no improvement beyond the NBC baseline. According to the Chapter 5 statistical tests, these improvements are not significant at the 0.05 level, so the final conclusion is that bagging shows a positive but not statistically confirmed advantage in this experiment.

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
        "Dataset              Comparison          Diff.   Match t  Match p  Indep t  Indep p",
        "--------------------------------------------------------------------------------",
    ]
    for row in nbc_vs_bagging_rows(single_rows):
        single_lines.append(
            f"{row['dataset'][:20]:20} {row['comparison'][:18]:18} "
            f"{fmt_pct(row['mean_difference']):>6}%  "
            f"{fmt_float(row['matched_t']):>7}  {fmt_float(row['matched_p_value']):>7}  "
            f"{fmt_float(row['independent_t']):>7}  {fmt_float(row['independent_p_value']):>7}"
        )
    single_bagging_lines = [
        "Dataset              Baseline       Comparison     Diff.   Match t  Match p  Indep t  Indep p",
        "-------------------------------------------------------------------------------------------",
    ]
    for row in bagging_size_rows(single_rows):
        single_bagging_lines.append(
            f"{row['dataset'][:20]:20} {row['baseline'][:13]:13} {row['comparison'][:13]:13} "
            f"{fmt_pct(row['mean_difference']):>6}%  "
            f"{fmt_float(row['matched_t']):>7}  {fmt_float(row['matched_p_value']):>7}  "
            f"{fmt_float(row['independent_t']):>7}  {fmt_float(row['independent_p_value']):>7}"
        )

    multiple_lines = [
        "Baseline            Comparison          Diff.   Match t  Match p  Indep t  Indep p",
        "-------------------------------------------------------------------------------",
    ]
    for row in multiple_rows:
        multiple_lines.append(
            f"{row['baseline'][:18]:18} {row['comparison'][:18]:18} "
            f"{fmt_pct(row['mean_difference']):>6}%  "
            f"{fmt_float(row['matched_t']):>7}  {fmt_float(row['matched_p_value']):>7}  "
            f"{fmt_float(row['independent_t']):>7}  {fmt_float(row['independent_p_value']):>7}"
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
