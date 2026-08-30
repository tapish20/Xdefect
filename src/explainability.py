"""SHAP explainability for defect prediction models.

Computes TreeExplainer SHAP values for the best-performing tree model per dataset,
saves beeswarm summary plots, global feature importance bar plots, and full,
un-truncated feature rankings to JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from train_within_project import (
    HYBRID_DIR,
    OUTPUT_DIR as WITHIN_OUTPUT_DIR,
    STATIC_DIR,
    load_and_clean,
    model_factories,
    prepare_xy,
    resample_training_fold,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHAP_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shap"
TREE_MODELS = {"random_forest", "xgboost", "lightgbm"}


def get_best_tree_models() -> dict[tuple[str, str], str]:
    results_path = WITHIN_OUTPUT_DIR / "within_project_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing Phase 3 results file: {results_path}")

    df = pd.read_csv(results_path)
    tree_df = df[df["model"].isin(TREE_MODELS)].copy()

    # Average AUC across CV fold configurations
    avg_scores = tree_df.groupby(["dataset", "feature_set", "model"], as_index=False)["auc_roc_mean"].mean()

    best_models = {}
    for (dataset, feature_set), group in avg_scores.groupby(["dataset", "feature_set"]):
        best_row = group.sort_values(by="auc_roc_mean", ascending=False).iloc[0]
        best_models[(dataset, feature_set)] = str(best_row["model"])

    return best_models


def extract_positive_shap(shap_vals: np.ndarray | list[np.ndarray]) -> np.ndarray:
    if isinstance(shap_vals, list):
        # List of arrays per class -> class 1
        arr = shap_vals[1]
    elif isinstance(shap_vals, np.ndarray):
        if shap_vals.ndim == 3:
            # Shape (N, M, 2) -> class 1
            arr = shap_vals[:, :, 1]
        else:
            arr = shap_vals
    else:
        arr = np.asarray(shap_vals)
    return arr


def process_dataset(dataset: str, feature_set: str, model_name: str) -> None:
    if feature_set == "static":
        data_path = STATIC_DIR / f"{dataset}.csv"
        dataset_key = dataset
    else:
        data_path = HYBRID_DIR / f"{dataset}_hybrid.csv"
        dataset_key = f"{dataset}_hybrid"

    if not data_path.exists():
        print(f"[SKIP] Data file does not exist: {data_path}", flush=True)
        return

    df = load_and_clean(data_path)
    if feature_set == "hybrid" and "repo_metrics_available" in df.columns:
        df = df[df["repo_metrics_available"].astype(bool)].copy()

    if df.empty:
        print(f"[SKIP] Empty dataframe for {dataset_key}", flush=True)
        return

    print(f"[START] Processing {dataset_key} using {model_name}...", flush=True)

    X, y = prepare_xy(df)

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Resample training fold only
    X_train_res, y_train_res = resample_training_fold(X_train, y_train)

    # Build and fit model (use 100 estimators for fast SHAP computation)
    builders = model_factories()
    if model_name not in builders:
        raise ValueError(f"Unknown tree model: {model_name}")

    if model_name == "random_forest":
        model = RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42)
    else:
        model = builders[model_name]()

    model.fit(X_train_res, y_train_res)

    # Compute SHAP values using TreeExplainer
    explainer = shap.TreeExplainer(model)
    try:
        raw_shap_vals = explainer.shap_values(X_test, check_additivity=False)
    except Exception:
        raw_shap_vals = explainer.shap_values(X_test)
    shap_matrix = extract_positive_shap(raw_shap_vals)

    # 1. Save Summary / Beeswarm Plot
    summary_plot_path = SHAP_OUTPUT_DIR / f"{dataset_key}_summary.png"
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_matrix, X_test, show=False)
    plt.title(f"SHAP Summary ({dataset_key} - {model_name})", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Save Global Importance Bar Plot
    importance_plot_path = SHAP_OUTPUT_DIR / f"{dataset_key}_importance.png"
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_matrix, X_test, plot_type="bar", show=False)
    plt.title(f"SHAP Global Importance ({dataset_key} - {model_name})", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(importance_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Compute FULL ranked feature list with mean |SHAP value|
    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    feature_names = list(X_test.columns)

    # Pair features with mean |SHAP| and sort descending
    paired = sorted(
        zip(feature_names, mean_abs_shap, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )

    ranking_list = [
        {
            "rank": rank + 1,
            "feature": feat,
            "mean_abs_shap": float(val),
        }
        for rank, (feat, val) in enumerate(paired)
    ]

    ranking_json_path = SHAP_OUTPUT_DIR / f"{dataset_key}_feature_ranking.json"
    with open(ranking_json_path, "w", encoding="utf-8") as f:
        json.dump(ranking_list, f, indent=2)

    print(f"[SUCCESS] Processed {dataset_key} ({model_name}): {len(ranking_list)} features ranked.", flush=True)


def main() -> int:
    SHAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_models = get_best_tree_models()
    print(f"Found best tree models for {len(best_models)} dataset configurations:", flush=True)
    for (dataset, feature_set), model_name in sorted(best_models.items()):
        print(f"  - {dataset} [{feature_set}]: {model_name}", flush=True)

    for (dataset, feature_set), model_name in sorted(best_models.items()):
        process_dataset(dataset, feature_set, model_name)

    print(f"\nAll SHAP explainability artifacts saved to {SHAP_OUTPUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
