from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from ucimlrepo import fetch_ucirepo


# Reusing one seed makes every random split and bootstrap sample reproducible.
# The remaining constants keep the experimental settings in one place.
RANDOM_STATE = 42
N_BINS = 10
N_SPLITS = 5


DATASETS = {
    "Banknote Authentication": 267,
    "Glass Identification": 42,
    "Image Segmentation": 50,
}

MODEL_ORDER = ["NBC", "Bagging NBC (10)", "Bagging NBC (20)", "Bagging NBC (30)"]
MODEL_BASE_MODELS = {
    "NBC": 1,
    "Bagging NBC (10)": 10,
    "Bagging NBC (20)": 20,
    "Bagging NBC (30)": 30,
}
COMPARISON_PAIRS = [
    ("NBC", "Bagging NBC (10)"),
    ("NBC", "Bagging NBC (20)"),
    ("NBC", "Bagging NBC (30)"),
    ("Bagging NBC (10)", "Bagging NBC (20)"),
    ("Bagging NBC (10)", "Bagging NBC (30)"),
    ("Bagging NBC (20)", "Bagging NBC (30)"),
]


@dataclass
class EqualWidthDiscretizer:
    """Learn and apply equal-width bins independently for every feature.

    ``@dataclass`` creates the initializer automatically. After ``fit``,
    ``edges_`` contains the nine internal boundaries required for ten bins.
    """

    n_bins: int = N_BINS
    edges_: list[np.ndarray] | None = None

    def fit(self, X: pd.DataFrame | np.ndarray) -> "EqualWidthDiscretizer":
        """Learn ten-bin boundaries from the training set only."""
        # np.asarray accepts either a DataFrame or ndarray and gives the rest of
        # this method one predictable floating-point representation.
        X_arr = np.asarray(X, dtype=float)
        edges: list[np.ndarray] = []
        for j in range(X_arr.shape[1]):
            col = X_arr[:, j]
            # np.isnan identifies missing values; ``~`` reverses the mask so
            # only observed values influence the learned minimum and maximum.
            observed = col[~np.isnan(col)]
            if observed.size == 0:
                # Keep an all-missing feature valid; transform() will leave it as -1.
                # np.full creates all nine placeholder boundaries at once.
                edges.append(np.full(self.n_bins - 1, np.nan))
                continue
            # np.min/np.max determine the range of this training feature.
            lo = float(np.min(observed))
            hi = float(np.max(observed))
            if np.isclose(lo, hi):
                # np.isclose safely detects a constant (or nearly constant)
                # feature without relying on exact floating-point equality.
                edges.append(np.full(self.n_bins - 1, lo))
            else:
                # np.linspace creates 11 equally spaced endpoints. Removing
                # the first and last leaves the nine internal split points.
                edges.append(np.linspace(lo, hi, self.n_bins + 1)[1:-1])
        self.edges_ = edges
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Apply learned boundaries, encoding missing values with the sentinel -1."""
        if self.edges_ is None:
            raise RuntimeError("Discretizer must be fit before transform.")
        X_arr = np.asarray(X, dtype=float)
        # Start every entry at -1, the sentinel reserved for missing data.
        X_disc = np.full(X_arr.shape, -1, dtype=int)
        for j, edges in enumerate(self.edges_):
            col = X_arr[:, j]
            mask = ~np.isnan(col)
            # np.digitize converts each observed value to its boundary index;
            # with right=False, a value equal to an edge enters the higher bin.
            X_disc[mask, j] = np.digitize(col[mask], edges, right=False)
            # np.clip also assigns test values outside the training range to
            # the first or last valid bin instead of producing invalid indices.
            X_disc[mask, j] = np.clip(X_disc[mask, j], 0, self.n_bins - 1)
        return X_disc

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Learn boundaries from ``X`` and immediately discretize the same data."""
        # Method chaining ensures transform uses the boundaries just learned.
        return self.fit(X).transform(X)


class NaiveBayesClassifier:
    """Categorical Naive Bayes with Laplace-smoothed probabilities.

    Probabilities are stored as logarithms so prediction can add small values
    safely instead of multiplying many values and risking numeric underflow.
    """

    def __init__(self, n_bins: int = N_BINS):
        self.n_bins = n_bins
        self.classes_: np.ndarray | None = None
        self.log_priors_: dict[object, float] = {}
        self.log_cond_probs_: dict[object, np.ndarray] = {}

    def fit(self, X_disc: np.ndarray, y: Iterable) -> "NaiveBayesClassifier":
        """Estimate class priors and feature-bin probabilities in log space."""
        # ravel flattens Series, lists, or column arrays into one label vector.
        y_arr = np.asarray(y).ravel()
        # np.unique returns the distinct class labels in deterministic order.
        self.classes_ = np.unique(y_arr)
        n_samples, n_features = X_disc.shape
        n_classes = len(self.classes_)

        for cls in self.classes_:
            class_mask = y_arr == cls
            X_cls = X_disc[class_mask]
            # Summing a Boolean mask counts the training rows in this class.
            class_count = int(np.sum(class_mask))
            # Add one pseudo-count to every class when estimating its prior.
            # np.log stores the Laplace-smoothed prior as a log probability.
            self.log_priors_[cls] = np.log((class_count + 1) / (n_samples + n_classes))

            probs = np.zeros((n_features, self.n_bins), dtype=float)
            for j in range(n_features):
                observed = X_cls[:, j] != -1
                col = X_cls[observed, j]
                # Missing entries are excluded; each of the ten bins gets one pseudo-count.
                denom = col.size + self.n_bins
                # np.bincount efficiently counts how often each bin occurs;
                # minlength guarantees an entry even for bins with zero cases.
                counts = np.bincount(col, minlength=self.n_bins)
                probs[j] = (counts + 1) / denom
            self.log_cond_probs_[cls] = np.log(probs)
        return self

    def predict(self, X_disc: np.ndarray) -> np.ndarray:
        """Predict the class with the largest Naive Bayes log-posterior score."""
        if self.classes_ is None:
            raise RuntimeError("Classifier must be fit before predict.")
        predictions = []
        for row in X_disc:
            scores = {}
            # A missing feature contributes no conditional-probability term.
            # np.where returns the column positions that are not missing.
            observed_features = np.where(row != -1)[0]
            for cls in self.classes_:
                score = self.log_priors_[cls]
                cond = self.log_cond_probs_[cls]
                for j in observed_features:
                    score += cond[j, row[j]]
                scores[cls] = score
            # max(..., key=scores.get) selects the label with the greatest
            # log-posterior score rather than the largest label value.
            predictions.append(max(scores, key=scores.get))
        return np.asarray(predictions)


def fit_predict_nbc(X_train, y_train, X_test) -> np.ndarray:
    """Fit fold-specific preprocessing and one NBC, then predict ``X_test``.

    The discretizer is fitted only on training data to prevent test-fold
    information from leaking into the learned bin boundaries.
    """
    discretizer = EqualWidthDiscretizer(n_bins=N_BINS)
    X_train_disc = discretizer.fit_transform(X_train)
    X_test_disc = discretizer.transform(X_test)
    model = NaiveBayesClassifier(n_bins=N_BINS).fit(X_train_disc, y_train)
    return model.predict(X_test_disc)


def fit_predict_bagging(X_train, y_train, X_test, n_estimators: int, seed: int) -> np.ndarray:
    """Train NBCs on bootstrap samples and combine them by majority vote."""
    # default_rng creates an isolated, reproducible random-number generator.
    rng = np.random.default_rng(seed)
    # DataFrame construction standardizes inputs; reset_index makes the rows
    # selectable by the zero-based bootstrap positions generated below.
    X_train_df = pd.DataFrame(X_train).reset_index(drop=True)
    y_train_arr = np.asarray(y_train).ravel()
    X_test_df = pd.DataFrame(X_test).reset_index(drop=True)
    n_samples = len(X_train_df)
    predictions = []

    for _ in range(n_estimators):
        # Bootstrap sampling draws n training instances with replacement.
        # rng.integers permits repeated positions, so some training rows appear
        # more than once and others are omitted—the defining bootstrap behavior.
        sample_idx = rng.integers(0, n_samples, size=n_samples)
        pred = fit_predict_nbc(
            X_train_df.iloc[sample_idx],
            y_train_arr[sample_idx],
            X_test_df,
        )
        predictions.append(pred)

    # vstack first makes one row per estimator; transpose changes this to one
    # row per test instance so each row contains all ensemble votes.
    pred_matrix = np.vstack(predictions).T
    final_predictions = []
    for row in pred_matrix:
        # np.unique sorts labels, making ties deterministic.
        # return_counts=True returns each candidate label and its vote total.
        values, counts = np.unique(row, return_counts=True)
        # np.argmax locates the largest vote count; indexing values retrieves
        # the corresponding class label.
        final_predictions.append(values[np.argmax(counts)])
    return np.asarray(final_predictions)


def method_iii_cv_indices(n_samples: int, n_splits: int, seed: int) -> list[np.ndarray]:
    """Create Chapter 4 method-III folds by cyclically assigning sorted instances."""
    rng = np.random.default_rng(seed)
    # Give every instance a U(0, 1) random number and sort by that number.
    # rng.random draws one reproducible U(0, 1) value for every instance.
    random_numbers = rng.random(n_samples)
    # argsort returns row positions ordered by those values. Stable mergesort
    # preserves original order in the extremely unlikely event of a tie.
    sorted_indices = np.argsort(random_numbers, kind="mergesort")
    folds = [[] for _ in range(n_splits)]
    # Sequential cyclic assignment makes all fold sizes differ by at most one.
    for rank, idx in enumerate(sorted_indices):
        folds[rank % n_splits].append(int(idx))
    return [np.asarray(fold, dtype=int) for fold in folds]


def add_result_row(rows, dataset_name, model_name, fold, predictions, y_test) -> None:
    """Calculate one fold's accuracy and append a tidy result record."""
    y_test_arr = np.asarray(y_test).ravel()
    # Elementwise comparison produces Booleans and np.sum counts the True ones.
    correct = int(np.sum(predictions == y_test_arr))
    total = int(len(y_test_arr))
    rows.append(
        {
            "dataset": dataset_name,
            "model": model_name,
            "base_models": MODEL_BASE_MODELS[model_name],
            "fold": fold,
            "correct": correct,
            "total": total,
            "accuracy": correct / total,
        }
    )


def evaluate_dataset(dataset_name: str, dataset_id: int) -> list[dict[str, object]]:
    """Run five-fold CV for NBC and the three requested bagging sizes."""
    # fetch_ucirepo downloads the dataset identified by its UCI repository ID
    # and returns an object containing separate feature and target tables.
    repo = fetch_ucirepo(id=dataset_id)
    # copy prevents later cleaning operations from modifying the fetched object.
    X = repo.data.features.copy()
    # iloc[:, 0] selects the first target column as a one-dimensional Series.
    y = repo.data.targets.iloc[:, 0].copy()

    # apply calls to_numeric on every feature column. Invalid text becomes NaN
    # (rather than raising an exception) because errors="coerce" is specified.
    X = X.apply(pd.to_numeric, errors="coerce")
    y = y.astype(str)

    folds = method_iii_cv_indices(
        n_samples=len(X),
        n_splits=N_SPLITS,
        seed=RANDOM_STATE + dataset_id,
    )
    rows: list[dict[str, object]] = []

    # arange creates every valid row position: 0, 1, ..., len(X)-1.
    all_indices = np.arange(len(X))
    for fold, test_idx in enumerate(folds, start=1):
        # Each instance is test data once and training data in the other four folds.
        # setdiff1d returns all positions not assigned to the current test fold.
        train_idx = np.setdiff1d(all_indices, test_idx, assume_unique=False)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        nbc_pred = fit_predict_nbc(X_train, y_train, X_test)
        add_result_row(rows, dataset_name, "NBC", fold, nbc_pred, y_test)

        for n_estimators in (10, 20, 30):
            bagging_pred = fit_predict_bagging(
                X_train,
                y_train,
                X_test,
                n_estimators=n_estimators,
                seed=RANDOM_STATE + dataset_id * 100 + fold * 10 + n_estimators,
            )
            add_result_row(
                rows,
                dataset_name,
                f"Bagging NBC ({n_estimators})",
                fold,
                bagging_pred,
                y_test,
            )
    return rows


def two_sided_pvalue(t_value: float, df: float) -> float:
    """Convert a t statistic and degrees of freedom to a two-sided p-value."""
    # isfinite rejects NaN and infinity, for which a p-value is not meaningful.
    if not np.isfinite(t_value) or not np.isfinite(df) or df <= 0:
        return np.nan
    # stats.t.sf gives the upper-tail probability. abs handles either sign and
    # multiplication by two includes equally extreme outcomes in both tails.
    return float(2 * stats.t.sf(abs(t_value), df))


def two_sided_normal_pvalue(z_value: float) -> float:
    """Convert a z statistic to its two-sided standard-normal p-value."""
    if not np.isfinite(z_value):
        return np.nan
    return float(2 * stats.norm.sf(abs(z_value)))


def single_dataset_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    """Compare every model pair at both Chapter 5 single-data-set levels.

    Data-set aggregation follows Theorem 5.1: pool the correct predictions over
    all folds, calculate each model's accuracy as a sample proportion, and use
    the pooled two-proportion z statistic.

    Fold aggregation follows Theorem 5.2's matched-sample approach: pair the two
    model accuracies from each identical test fold, then test the mean of those
    fold differences with t = mean(d) / sqrt(s_d^2 / k), df = k - 1.
    """
    rows = []
    # groupby processes each dataset independently while preserving input order.
    for dataset_name, dataset_df in results.groupby("dataset", sort=False):
        # pivot aligns all model accuracies by fold, which is required for the
        # matched test to subtract scores from the exact same test samples.
        wide = dataset_df.pivot(index="fold", columns="model", values="accuracy")
        for baseline, comparison in COMPARISON_PAIRS:
            base_rows = dataset_df[dataset_df["model"] == baseline]
            comp_rows = dataset_df[dataset_df["model"] == comparison]

            # Theorem 5.1 data-set aggregation: sum correct predictions and test
            # the two resulting proportions with their pooled accuracy estimate.
            base_correct = int(base_rows["correct"].sum())
            comp_correct = int(comp_rows["correct"].sum())
            base_total = int(base_rows["total"].sum())
            comp_total = int(comp_rows["total"].sum())
            if base_total != comp_total:
                raise ValueError("Data-set aggregation requires the same test instances.")
            n = base_total
            base_dataset_accuracy = base_correct / n
            comp_dataset_accuracy = comp_correct / n
            dataset_diff = comp_dataset_accuracy - base_dataset_accuracy
            pooled_accuracy = (base_dataset_accuracy + comp_dataset_accuracy) / 2
            dataset_se = np.sqrt(2 * pooled_accuracy * (1 - pooled_accuracy) / n)
            dataset_z = dataset_diff / dataset_se if dataset_se > 0 else np.nan
            dataset_large_sample = min(
                base_correct,
                n - base_correct,
                comp_correct,
                n - comp_correct,
            ) >= 5

            # Theorem 5.2 matched fold aggregation: subtract baseline from the
            # comparison within each fold before estimating the sampling error.
            fold_diff = wide[comparison] - wide[baseline]
            k = int(fold_diff.shape[0])
            fold_mean_diff = float(fold_diff.mean())
            fold_diff_var = float(fold_diff.var(ddof=1))
            fold_matched_se = np.sqrt(fold_diff_var / k)
            fold_matched_t = (
                fold_mean_diff / fold_matched_se if fold_matched_se > 0 else np.nan
            )
            fold_matched_df = k - 1

            rows.append(
                {
                    "dataset": dataset_name,
                    "baseline": baseline,
                    "comparison": comparison,
                    "dataset_baseline_accuracy": base_dataset_accuracy,
                    "dataset_comparison_accuracy": comp_dataset_accuracy,
                    "dataset_difference": dataset_diff,
                    "dataset_z": dataset_z,
                    "dataset_p_value": two_sided_normal_pvalue(dataset_z),
                    "dataset_large_sample": dataset_large_sample,
                    "fold_mean_difference": fold_mean_diff,
                    "fold_matched_t": fold_matched_t,
                    "fold_matched_df": fold_matched_df,
                    "fold_matched_p_value": two_sided_pvalue(
                        fold_matched_t, fold_matched_df
                    ),
                }
            )
    return pd.DataFrame(rows)


def multiple_dataset_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    """Compare every model pair at both Chapter 5 multiple-data-set levels.

    Data-set averaging follows Theorem 5.4. Each data set contributes one pooled
    accuracy per model, so data sets have equal weight, while each proportion's
    variance still uses that data set's own number of instances.

    Fold averaging follows the matched approach in Theorems 5.5 and 5.6. Fold
    differences estimate a variance separately inside each data set; those
    variances are combined without treating data-set means as one random sample.
    """
    rows = []
    # nunique counts how many datasets contribute to the combined comparison.
    L = int(results["dataset"].nunique())
    k = N_SPLITS
    for baseline, comparison in COMPARISON_PAIRS:
        dataset_diffs = []
        dataset_variance_terms = []
        fold_dataset_diffs = []
        fold_matched_vars = []
        all_large_sample = True
        for _, dataset_df in results.groupby("dataset", sort=False):
            base_rows = dataset_df[dataset_df["model"] == baseline]
            comp_rows = dataset_df[dataset_df["model"] == comparison]
            base_correct = int(base_rows["correct"].sum())
            comp_correct = int(comp_rows["correct"].sum())
            base_total = int(base_rows["total"].sum())
            comp_total = int(comp_rows["total"].sum())
            if base_total != comp_total:
                raise ValueError("Data-set averaging requires the same test instances.")
            dataset_size = base_total
            base_accuracy = base_correct / dataset_size
            comp_accuracy = comp_correct / dataset_size
            dataset_diffs.append(comp_accuracy - base_accuracy)
            dataset_variance_terms.append(
                (
                    base_accuracy * (1 - base_accuracy)
                    + comp_accuracy * (1 - comp_accuracy)
                )
                / dataset_size
            )
            all_large_sample = all_large_sample and min(
                base_correct,
                dataset_size - base_correct,
                comp_correct,
                dataset_size - comp_correct,
            ) >= 5

            wide = dataset_df.pivot(index="fold", columns="model", values="accuracy")
            fold_diff = wide[comparison] - wide[baseline]
            if len(fold_diff) != k:
                raise ValueError(f"Expected {k} folds for every data set.")
            fold_dataset_diffs.append(float(fold_diff.mean()))
            fold_matched_vars.append(float(fold_diff.var(ddof=1)))

        # Theorem 5.4: average data-set proportions, then use the sum of their
        # binomial variance estimates divided by L squared.
        dataset_average_diff = float(np.mean(dataset_diffs))
        dataset_average_se = np.sqrt(np.sum(dataset_variance_terms) / (L**2))
        dataset_average_z = (
            dataset_average_diff / dataset_average_se
            if dataset_average_se > 0
            else np.nan
        )

        # Theorems 5.5-5.6: average matched fold differences across data sets and
        # combine the within-data-set difference variances.
        fold_average_diff = float(np.mean(fold_dataset_diffs))
        fold_vars_arr = np.asarray(fold_matched_vars, dtype=float)
        fold_matched_se = np.sqrt(np.sum(fold_vars_arr / k) / (L**2))
        fold_matched_t = (
            fold_average_diff / fold_matched_se if fold_matched_se > 0 else np.nan
        )
        fold_matched_df_raw = (
            (k - 1) * (np.sum(fold_vars_arr) ** 2) / np.sum(fold_vars_arr**2)
            if np.sum(fold_vars_arr**2) > 0
            else np.nan
        )
        # The chapter's approximation uses the conservative integer below the
        # calculated degrees of freedom, obtained with np.floor.
        fold_matched_df = (
            float(np.floor(fold_matched_df_raw))
            if np.isfinite(fold_matched_df_raw)
            else np.nan
        )

        rows.append(
            {
                "baseline": baseline,
                "comparison": comparison,
                "dataset_average_difference": dataset_average_diff,
                "dataset_average_z": dataset_average_z,
                "dataset_average_p_value": two_sided_normal_pvalue(dataset_average_z),
                "dataset_large_sample": all_large_sample,
                "fold_average_difference": fold_average_diff,
                "fold_matched_t": fold_matched_t,
                "fold_matched_df": fold_matched_df,
                "fold_matched_p_value": two_sided_pvalue(
                    fold_matched_t, fold_matched_df
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run experiments, summarize accuracies, and save all comparison tables."""
    all_rows = []
    for dataset_name, dataset_id in DATASETS.items():
        print(f"Running {dataset_name}...")
        # extend adds every fold/model record returned for this dataset.
        all_rows.extend(evaluate_dataset(dataset_name, dataset_id))

    # DataFrame converts the list of equally structured dictionaries into a
    # table suitable for aggregation, statistical comparison, and CSV export.
    results = pd.DataFrame(all_rows)
    summary = (
        # groupby creates one group per dataset/model setting; agg then computes
        # named summary columns from the fold-level measurements in each group.
        results.groupby(["dataset", "model", "base_models"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            total_correct=("correct", "sum"),
            total_instances=("total", "sum"),
        )
        # sort_values presents NBC first, followed by increasing ensemble size.
        .sort_values(["dataset", "base_models"])
    )
    single_eval = single_dataset_comparisons(results)
    multiple_eval = multiple_dataset_comparisons(results)

    # index=False avoids writing the DataFrame's internal row numbers to disk.
    results.to_csv("fold_results.csv", index=False)
    summary.to_csv("summary_results.csv", index=False)
    single_eval.to_csv("single_dataset_comparisons.csv", index=False)
    multiple_eval.to_csv("multiple_dataset_comparisons.csv", index=False)
    print(summary.to_string(index=False))
    print("\nSingle-dataset comparisons:")
    print(single_eval.to_string(index=False))
    print("\nMultiple-dataset comparisons:")
    print(multiple_eval.to_string(index=False))


if __name__ == "__main__":
    main()
