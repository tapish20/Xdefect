"""Within-project defect prediction baselines.

Runs stratified 5-fold and 10-fold cross-validation for static-only and
hybrid feature sets, applying SMOTE to each training fold only.
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from preprocessing import load_and_clean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "data" / "static"
HYBRID_DIR = PROJECT_ROOT / "data" / "hybrid"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "within_project"
LABEL_COL = "defects"
RANDOM_STATE = 42

IDENTIFIER_COLUMNS = {
    "classname",
    "class_name",
    "file_module_id",
    "module_id",
    "module",
    "file",
    "filename",
    "path",
    "name",
}
LEAKAGE_COLUMNS = {
    "bugs",
    "nonTrivialBugs",
    "majorBugs",
    "criticalBugs",
    "highPriorityBugs",
}
NON_FEATURE_COLUMNS = {
    LABEL_COL,
    "repo_metrics_available",
    "hybrid_join_status",
    "dataset_family",
    "dataset_family_x",
    "dataset_family_y",
    *IDENTIFIER_COLUMNS,
    *LEAKAGE_COLUMNS,
}
METRIC_NAMES = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc_roc",
    "mcc",
    "balanced_acc",
    "pr_auc",
)


@dataclass(frozen=True)
class DatasetBundle:
    dataset: str
    feature_set: str
    df: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names without .csv. Defaults to every data/static CSV.",
    )
    parser.add_argument(
        "--cv-folds",
        nargs="*",
        type=int,
        default=[5, 10],
        choices=[5, 10],
        help="Cross-validation fold counts to run.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="*",
        default=["static", "hybrid"],
        choices=["static", "hybrid"],
        help="Feature sets to run.",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")


def model_factories() -> dict[str, object]:
    return {
        "logistic_regression": lambda: LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": lambda: XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "lightgbm": lambda: LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            class_weight="balanced",
            n_jobs=1,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
    }


def load_datasets(dataset_names: list[str] | None, feature_sets: list[str]) -> list[DatasetBundle]:
    names = dataset_names or sorted(path.stem for path in STATIC_DIR.glob("*.csv"))
    bundles: list[DatasetBundle] = []

    for dataset in names:
        if "static" in feature_sets:
            static_path = STATIC_DIR / f"{dataset}.csv"
            if static_path.exists():
                bundles.append(
                    DatasetBundle(dataset=dataset, feature_set="static", df=load_and_clean(static_path))
                )

        if "hybrid" in feature_sets:
            hybrid_path = HYBRID_DIR / f"{dataset}_hybrid.csv"
            if not hybrid_path.exists():
                continue
            hybrid = load_and_clean(hybrid_path)
            if "repo_metrics_available" in hybrid.columns:
                hybrid = hybrid[hybrid["repo_metrics_available"].astype(bool)].copy()
            if not hybrid.empty:
                bundles.append(DatasetBundle(dataset=dataset, feature_set="hybrid", df=hybrid))

    return bundles


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {col for col in df.columns if col in NON_FEATURE_COLUMNS}
    excluded.update(
        col
        for col in df.columns
        if col.endswith("_repo")
        or col.endswith("_x")
        or col.endswith("_y")
        or col.lower() in {name.lower() for name in NON_FEATURE_COLUMNS}
    )
    numeric = df.select_dtypes(include=[np.number]).columns
    return [col for col in numeric if col not in excluded]


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing required label column: {LABEL_COL}")
    features = feature_columns(df)
    if not features:
        raise ValueError("No numeric feature columns remain after excluding identifiers/leakage.")
    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)
    y = df[LABEL_COL].astype(int)
    return X, y


def resample_training_fold(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    counts = y_train.value_counts()
    if len(counts) < 2:
        return X_train, y_train
    minority_count = int(counts.min())
    if minority_count < 2:
        return X_train, y_train
    smote = SMOTE(k_neighbors=min(5, minority_count - 1), random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)


def fit_predict(
    model_name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "logistic_regression":
        scaler = StandardScaler()
        X_train_model = scaler.fit_transform(X_train)
        X_test_model = scaler.transform(X_test)
    else:
        X_train_model = X_train
        X_test_model = X_test

    model.fit(X_train_model, y_train)
    y_pred = model.predict(X_test_model)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_model)[:, 1]
    else:
        y_score = y_pred
    return np.asarray(y_pred, dtype=int), np.asarray(y_score, dtype=float)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
    }
    if y_true.nunique() == 2:
        metrics["auc_roc"] = roc_auc_score(y_true, y_score)
        metrics["pr_auc"] = average_precision_score(y_true, y_score)
    else:
        metrics["auc_roc"] = np.nan
        metrics["pr_auc"] = np.nan
    return metrics


def run_cv(bundle: DatasetBundle, cv_folds: int) -> list[dict[str, float | str | int]]:
    X, y = prepare_xy(bundle.df)
    min_class_count = int(y.value_counts().min())
    if min_class_count < cv_folds:
        raise ValueError(
            f"{bundle.dataset} has only {min_class_count} rows in its minority class, "
            f"which is too small for {cv_folds}-fold stratified CV."
        )

    splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    model_builders = model_factories()
    summaries = []

    for model_name, build_model in model_builders.items():
        fold_metrics = []
        prediction_rows = []
        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            X_resampled, y_resampled = resample_training_fold(X_train, y_train)
            y_pred, y_score = fit_predict(
                model_name,
                build_model(),
                X_resampled,
                y_resampled,
                X_test,
            )
            metrics = compute_metrics(y_test, y_pred, y_score)
            fold_metrics.append(metrics)

            for row_index, actual, pred, score in zip(
                X_test.index,
                y_test.to_numpy(),
                y_pred,
                y_score,
                strict=True,
            ):
                prediction_rows.append(
                    {
                        "dataset": bundle.dataset,
                        "feature_set": bundle.feature_set,
                        "model": model_name,
                        "cv_folds": cv_folds,
                        "fold": fold_id,
                        "row_index": row_index,
                        "y_true": int(actual),
                        "y_pred": int(pred),
                        "y_score": float(score),
                    }
                )

        predictions = pd.DataFrame(prediction_rows)
        prediction_path = (
            OUTPUT_DIR
            / f"{safe_name(bundle.dataset)}_{bundle.feature_set}_{model_name}_{cv_folds}fold_predictions.csv"
        )
        predictions.to_csv(prediction_path, index=False)

        summary: dict[str, float | str | int] = {
            "dataset": bundle.dataset,
            "feature_set": bundle.feature_set,
            "model": model_name,
            "cv_folds": cv_folds,
        }
        metrics_df = pd.DataFrame(fold_metrics)
        for metric in METRIC_NAMES:
            summary[f"{metric}_mean"] = metrics_df[metric].mean()
            summary[f"{metric}_std"] = metrics_df[metric].std(ddof=1)
        summaries.append(summary)
    return summaries


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bundles = load_datasets(args.datasets, args.feature_sets)
    if not bundles:
        raise SystemExit("No datasets found for the requested feature sets.")

    all_summaries = []
    skipped = []
    for bundle in bundles:
        for cv_folds in args.cv_folds:
            try:
                print(
                    f"Running {bundle.dataset} [{bundle.feature_set}] {cv_folds}-fold CV",
                    flush=True,
                )
                all_summaries.extend(run_cv(bundle, cv_folds))
            except ValueError as exc:
                skipped.append(
                    {
                        "dataset": bundle.dataset,
                        "feature_set": bundle.feature_set,
                        "cv_folds": cv_folds,
                        "reason": str(exc),
                    }
                )

    results = pd.DataFrame(all_summaries)
    results_path = OUTPUT_DIR / "within_project_results.csv"
    results.to_csv(results_path, index=False)

    if skipped:
        pd.DataFrame(skipped).to_csv(OUTPUT_DIR / "within_project_skipped.csv", index=False)
    print(results_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
