from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from ucimlrepo import fetch_ucirepo


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
    """Discretize each continuous feature into equal-width intervals."""

    n_bins: int = N_BINS
    edges_: list[np.ndarray] | None = None

    def fit(self, X: pd.DataFrame | np.ndarray) -> "EqualWidthDiscretizer":
        """Learn ten-bin boundaries from the training set only."""
        X_arr = np.asarray(X, dtype=float)
        edges: list[np.ndarray] = []
        for j in range(X_arr.shape[1]):
            col = X_arr[:, j]
            observed = col[~np.isnan(col)]
            if observed.size == 0:
                # Keep an all-missing feature valid; transform() will leave it as -1.
                edges.append(np.full(self.n_bins - 1, np.nan))
                continue
            lo = float(np.min(observed))
            hi = float(np.max(observed))
            if np.isclose(lo, hi):
                edges.append(np.full(self.n_bins - 1, lo))
            else:
                edges.append(np.linspace(lo, hi, self.n_bins + 1)[1:-1])
        self.edges_ = edges
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Apply learned boundaries, encoding missing values with the sentinel -1."""
        if self.edges_ is None:
            raise RuntimeError("Discretizer must be fit before transform.")
        X_arr = np.asarray(X, dtype=float)
        X_disc = np.full(X_arr.shape, -1, dtype=int)
        for j, edges in enumerate(self.edges_):
            col = X_arr[:, j]
            mask = ~np.isnan(col)
            X_disc[mask, j] = np.digitize(col[mask], edges, right=False)
            X_disc[mask, j] = np.clip(X_disc[mask, j], 0, self.n_bins - 1)
        return X_disc

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class NaiveBayesClassifier:
    """Categorical Naive Bayes with Laplace-smoothed probability estimates."""

    def __init__(self, n_bins: int = N_BINS):
        self.n_bins = n_bins
        self.classes_: np.ndarray | None = None
        self.log_priors_: dict[object, float] = {}
        self.log_cond_probs_: dict[object, np.ndarray] = {}

    def fit(self, X_disc: np.ndarray, y: Iterable) -> "NaiveBayesClassifier":
        """Estimate class priors and feature-bin probabilities in log space."""
        y_arr = np.asarray(y).ravel()
        self.classes_ = np.unique(y_arr)
        n_samples, n_features = X_disc.shape
        n_classes = len(self.classes_)

        for cls in self.classes_:
            class_mask = y_arr == cls
            X_cls = X_disc[class_mask]
            class_count = int(np.sum(class_mask))
            # Add one pseudo-count to every class when estimating its prior.
            self.log_priors_[cls] = np.log((class_count + 1) / (n_samples + n_classes))

            probs = np.zeros((n_features, self.n_bins), dtype=float)
            for j in range(n_features):
                observed = X_cls[:, j] != -1
                col = X_cls[observed, j]
                # Missing entries are excluded; each of the ten bins gets one pseudo-count.
                denom = col.size + self.n_bins
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
            observed_features = np.where(row != -1)[0]
            for cls in self.classes_:
                score = self.log_priors_[cls]
                cond = self.log_cond_probs_[cls]
                for j in observed_features:
                    score += cond[j, row[j]]
                scores[cls] = score
            predictions.append(max(scores, key=scores.get))
        return np.asarray(predictions)


def fit_predict_nbc(X_train, y_train, X_test) -> np.ndarray:
    """Fit preprocessing and one NBC using only the current training fold."""
    discretizer = EqualWidthDiscretizer(n_bins=N_BINS)
    X_train_disc = discretizer.fit_transform(X_train)
    X_test_disc = discretizer.transform(X_test)
    model = NaiveBayesClassifier(n_bins=N_BINS).fit(X_train_disc, y_train)
    return model.predict(X_test_disc)


def fit_predict_bagging(X_train, y_train, X_test, n_estimators: int, seed: int) -> np.ndarray:
    """Train NBCs on bootstrap samples and combine them by majority vote."""
    rng = np.random.default_rng(seed)
    X_train_df = pd.DataFrame(X_train).reset_index(drop=True)
    y_train_arr = np.asarray(y_train).ravel()
    X_test_df = pd.DataFrame(X_test).reset_index(drop=True)
    n_samples = len(X_train_df)
    predictions = []

    for _ in range(n_estimators):
        # Bootstrap sampling draws n training instances with replacement.
        sample_idx = rng.integers(0, n_samples, size=n_samples)
        pred = fit_predict_nbc(
            X_train_df.iloc[sample_idx],
            y_train_arr[sample_idx],
            X_test_df,
        )
        predictions.append(pred)

    pred_matrix = np.vstack(predictions).T
    final_predictions = []
    for row in pred_matrix:
        # np.unique sorts labels, making ties deterministic.
        values, counts = np.unique(row, return_counts=True)
        final_predictions.append(values[np.argmax(counts)])
    return np.asarray(final_predictions)


def method_iii_cv_indices(n_samples: int, n_splits: int, seed: int) -> list[np.ndarray]:
    """Create Chapter 4 method-III folds by cyclically assigning sorted instances."""
    rng = np.random.default_rng(seed)
    # Give every instance a U(0, 1) random number and sort by that number.
    random_numbers = rng.random(n_samples)
    sorted_indices = np.argsort(random_numbers, kind="mergesort")
    folds = [[] for _ in range(n_splits)]
    # Sequential cyclic assignment makes all fold sizes differ by at most one.
    for rank, idx in enumerate(sorted_indices):
        folds[rank % n_splits].append(int(idx))
    return [np.asarray(fold, dtype=int) for fold in folds]


def add_result_row(rows, dataset_name, model_name, fold, predictions, y_test) -> None:
    y_test_arr = np.asarray(y_test).ravel()
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
    repo = fetch_ucirepo(id=dataset_id)
    X = repo.data.features.copy()
    y = repo.data.targets.iloc[:, 0].copy()

    X = X.apply(pd.to_numeric, errors="coerce")
    y = y.astype(str)

    folds = method_iii_cv_indices(
        n_samples=len(X),
        n_splits=N_SPLITS,
        seed=RANDOM_STATE + dataset_id,
    )
    rows: list[dict[str, object]] = []

    all_indices = np.arange(len(X))
    for fold, test_idx in enumerate(folds, start=1):
        # Each instance is test data once and training data in the other four folds.
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
    if not np.isfinite(t_value) or not np.isfinite(df) or df <= 0:
        return np.nan
    return float(2 * stats.t.sf(abs(t_value), df))


def single_dataset_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    """Apply Chapter 5 matched and independent tests within each dataset."""
    rows = []
    for dataset_name, dataset_df in results.groupby("dataset", sort=False):
        wide = dataset_df.pivot(index="fold", columns="model", values="accuracy")
        for baseline, comparison in COMPARISON_PAIRS:
            # Matched samples pair model accuracies from the exact same test fold.
            diff = wide[comparison] - wide[baseline]
            k = int(diff.shape[0])
            mean_diff = float(diff.mean())
            diff_var = float(diff.var(ddof=1))
            matched_se = np.sqrt(diff_var / k)
            matched_t = mean_diff / matched_se if matched_se > 0 else np.nan
            matched_df = k - 1

            base_values = wide[baseline]
            comp_values = wide[comparison]
            # The independent calculation ignores pairing and uses Welch's standard error.
            base_var = float(base_values.var(ddof=1))
            comp_var = float(comp_values.var(ddof=1))
            indep_se = np.sqrt(base_var / k + comp_var / k)
            indep_t = mean_diff / indep_se if indep_se > 0 else np.nan
            numerator = (base_var / k + comp_var / k) ** 2
            denominator = ((base_var / k) ** 2 / (k - 1)) + ((comp_var / k) ** 2 / (k - 1))
            indep_df = numerator / denominator if denominator > 0 else np.nan

            rows.append(
                {
                    "dataset": dataset_name,
                    "baseline": baseline,
                    "comparison": comparison,
                    "mean_difference": mean_diff,
                    "matched_t": matched_t,
                    "matched_df": matched_df,
                    "matched_p_value": two_sided_pvalue(matched_t, matched_df),
                    "independent_t": indep_t,
                    "independent_df": indep_df,
                    "independent_p_value": two_sided_pvalue(indep_t, indep_df),
                }
            )
    return pd.DataFrame(rows)


def multiple_dataset_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    """Combine fold-level comparisons across datasets using Chapter 5 formulas."""
    rows = []
    L = int(results["dataset"].nunique())
    k = N_SPLITS
    for baseline, comparison in COMPARISON_PAIRS:
        dataset_diffs = []
        matched_vars = []
        indep_vars = []
        for _, dataset_df in results.groupby("dataset", sort=False):
            wide = dataset_df.pivot(index="fold", columns="model", values="accuracy")
            diff = wide[comparison] - wide[baseline]
            dataset_diffs.append(float(diff.mean()))
            matched_vars.append(float(diff.var(ddof=1)))
            indep_vars.append(
                float(wide[comparison].var(ddof=1)) + float(wide[baseline].var(ddof=1))
            )

        # Every dataset has equal weight, regardless of its number of instances.
        mean_diff = float(np.mean(dataset_diffs))
        matched_vars_arr = np.asarray(matched_vars, dtype=float)
        matched_se = np.sqrt(np.sum(matched_vars_arr / k) / (L**2))
        matched_t = mean_diff / matched_se if matched_se > 0 else np.nan
        matched_df_raw = (
            (k - 1) * (np.sum(matched_vars_arr) ** 2) / np.sum(matched_vars_arr**2)
            if np.sum(matched_vars_arr**2) > 0
            else np.nan
        )
        matched_df = float(np.floor(matched_df_raw)) if np.isfinite(matched_df_raw) else np.nan

        indep_vars_arr = np.asarray(indep_vars, dtype=float)
        indep_se = np.sqrt(np.sum(indep_vars_arr / k) / (L**2))
        indep_t = mean_diff / indep_se if indep_se > 0 else np.nan
        indep_denominator = np.sum(indep_vars_arr**2)
        indep_df_raw = (
            (k - 1) * (np.sum(indep_vars_arr) ** 2) / indep_denominator
            if indep_denominator > 0
            else np.nan
        )
        indep_df = float(np.floor(indep_df_raw)) if np.isfinite(indep_df_raw) else np.nan

        rows.append(
            {
                "baseline": baseline,
                "comparison": comparison,
                "mean_difference": mean_diff,
                "matched_t": matched_t,
                "matched_df": matched_df,
                "matched_p_value": two_sided_pvalue(matched_t, matched_df),
                "independent_t": indep_t,
                "independent_df": indep_df,
                "independent_p_value": two_sided_pvalue(indep_t, indep_df),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run experiments, summarize accuracies, and save all comparison tables."""
    all_rows = []
    for dataset_name, dataset_id in DATASETS.items():
        print(f"Running {dataset_name}...")
        all_rows.extend(evaluate_dataset(dataset_name, dataset_id))

    results = pd.DataFrame(all_rows)
    summary = (
        results.groupby(["dataset", "model", "base_models"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            total_correct=("correct", "sum"),
            total_instances=("total", "sum"),
        )
        .sort_values(["dataset", "base_models"])
    )
    single_eval = single_dataset_comparisons(results)
    multiple_eval = multiple_dataset_comparisons(results)

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
