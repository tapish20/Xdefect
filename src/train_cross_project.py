"""Cross-project defect prediction evaluation.

Models are trained on one full source project, after applying SMOTE to that
source training data only, then evaluated once on the full target project.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_within_project import (
    HYBRID_DIR,
    LABEL_COL,
    OUTPUT_DIR as WITHIN_OUTPUT_DIR,
    STATIC_DIR,
    compute_metrics,
    feature_columns,
    fit_predict,
    load_and_clean,
    model_factories,
    prepare_xy,
    resample_training_fold,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CROSS_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cross_project"
CROSS_RESULTS_PATH = CROSS_OUTPUT_DIR / "cross_project_results.csv"
SKIPPED_PATH = CROSS_OUTPUT_DIR / "cross_project_skipped.csv"
GAP_CHART_PATH = CROSS_OUTPUT_DIR / "generalization_gap.png"

NASA_REQUESTED_PAIRS = [
    ("kc1", "pc1"),
    ("pc1", "cm1"),
    ("kc1", "cm1"),
    ("pc1", "pc3"),
]
NASA_DATASETS = ["cm1", "kc1", "pc1", "pc3"]
AEEEM_DATASETS = ["aeeem_eclipse", "aeeem_equinox", "aeeem_lucene", "aeeem_mylyn", "aeeem_pde"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-sets",
        nargs="*",
        default=["static", "hybrid"],
        choices=["static", "hybrid"],
        help="Feature sets to evaluate.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum parallel thread workers for pair evaluation.",
    )
    return parser.parse_args()


def ordered_pairs(datasets: list[str]) -> list[tuple[str, str]]:
    return [(source, target) for source, target in itertools.permutations(datasets, 2)]


def default_pairs(feature_set: str) -> list[tuple[str, str]]:
    pairs = set()
    if feature_set == "static":
        pairs.update(NASA_REQUESTED_PAIRS)
        pairs.update(ordered_pairs(NASA_DATASETS))
        pairs.update(ordered_pairs(AEEEM_DATASETS))
    elif feature_set == "hybrid":
        pairs.update(ordered_pairs(AEEEM_DATASETS))
    return sorted(pairs)


def load_dataset(dataset: str, feature_set: str) -> pd.DataFrame | None:
    if feature_set == "static":
        path = STATIC_DIR / f"{dataset}.csv"
    else:
        path = HYBRID_DIR / f"{dataset}_hybrid.csv"
    if not path.exists():
        return None
    df = load_and_clean(path)
    if feature_set == "hybrid" and "repo_metrics_available" in df.columns:
        df = df[df["repo_metrics_available"].astype(bool)].copy()
    return df if not df.empty else None


def aligned_xy(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    source_features = set(feature_columns(source_df))
    target_features = set(feature_columns(target_df))
    common_features = sorted(source_features & target_features)
    if not common_features:
        raise ValueError("No common numeric feature columns between source and target.")

    source_aligned = source_df[[*common_features, LABEL_COL]].copy()
    target_aligned = target_df[[*common_features, LABEL_COL]].copy()
    X_source, y_source = prepare_xy(source_aligned)
    X_target, y_target = prepare_xy(target_aligned)
    return X_source, y_source, X_target, y_target, common_features


def run_pair(source: str, target: str, feature_set: str) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    source_df = load_dataset(source, feature_set)
    target_df = load_dataset(target, feature_set)
    if source_df is None:
        return [], {
            "source_dataset": source,
            "target_dataset": target,
            "feature_set": feature_set,
            "reason": f"Source dataset not available for {feature_set}: {source}",
        }
    if target_df is None:
        return [], {
            "source_dataset": source,
            "target_dataset": target,
            "feature_set": feature_set,
            "reason": f"Target dataset not available for {feature_set}: {target}",
        }

    try:
        X_source, y_source, X_target, y_target, common_features = aligned_xy(source_df, target_df)
    except ValueError as exc:
        return [], {
            "source_dataset": source,
            "target_dataset": target,
            "feature_set": feature_set,
            "reason": str(exc),
        }

    X_resampled, y_resampled = resample_training_fold(X_source, y_source)

    rows = []
    for model_name, build_model in model_factories().items():
        y_pred, y_score = fit_predict(
            model_name,
            build_model(),
            X_resampled,
            y_resampled,
            X_target,
        )
        metrics = compute_metrics(y_target, y_pred, y_score)
        rows.append(
            {
                "source_dataset": source,
                "target_dataset": target,
                "feature_set": feature_set,
                "model": model_name,
                "n_source_rows": len(source_df),
                "n_target_rows": len(target_df),
                "n_common_features": len(common_features),
                **metrics,
            }
        )
    return rows, None


def plot_generalization_gap(cross_results: pd.DataFrame) -> None:
    within_path = WITHIN_OUTPUT_DIR / "within_project_results.csv"
    if not within_path.exists() or cross_results.empty:
        return

    within = pd.read_csv(within_path)
    within_summary = (
        within.groupby("model", as_index=False)["auc_roc_mean"]
        .mean()
        .rename(columns={"auc_roc_mean": "auc_roc"})
    )
    within_summary["evaluation"] = "Within-Project"

    cross_summary = (
        cross_results.groupby("model", as_index=False)["auc_roc"]
        .mean()
    )
    cross_summary["evaluation"] = "Cross-Project"

    plot_df = pd.concat([within_summary, cross_summary], ignore_index=True)

    model_display_names = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
    }

    models = sorted(plot_df["model"].unique())
    x_labels = [model_display_names.get(m, m) for m in models]
    x_positions = np.arange(len(models))
    width = 0.35

    colors = {"Within-Project": "#1f77b4", "Cross-Project": "#ff7f0e"}

    fig, ax = plt.subplots(figsize=(10, 6))

    for offset, evaluation in [(-width / 2, "Within-Project"), (width / 2, "Cross-Project")]:
        values = []
        for model in models:
            match = plot_df[(plot_df["model"] == model) & (plot_df["evaluation"] == evaluation)]
            values.append(float(match["auc_roc"].iloc[0]) if not match.empty else 0.0)

        rects = ax.bar(
            x_positions + offset,
            values,
            width=width,
            label=evaluation,
            color=colors[evaluation],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )

        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Average AUC-ROC", fontsize=12, fontweight="bold")
    ax.set_title("Within-Project vs Cross-Project Generalization Gap", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="gray", fontsize=11)

    fig.tight_layout()
    fig.savefig(GAP_CHART_PATH, dpi=300)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    CROSS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for feature_set in args.feature_sets:
        for source, target in default_pairs(feature_set):
            tasks.append((source, target, feature_set))

    results = []
    skipped = []

    print(f"Executing {len(tasks)} cross-project pairs using ThreadPoolExecutor...", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {
            executor.submit(run_pair, source, target, feature_set): (source, target, feature_set)
            for source, target, feature_set in tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            source, target, feature_set = future_to_task[future]
            try:
                pair_rows, skip_info = future.result()
                if pair_rows:
                    results.extend(pair_rows)
                    print(f"[SUCCESS] {source} -> {target} [{feature_set}]", flush=True)
                if skip_info:
                    skipped.append(skip_info)
                    print(f"[SKIPPED] {source} -> {target} [{feature_set}]: {skip_info['reason']}", flush=True)
            except Exception as exc:
                skipped.append(
                    {
                        "source_dataset": source,
                        "target_dataset": target,
                        "feature_set": feature_set,
                        "reason": str(exc),
                    }
                )
                print(f"[ERROR] {source} -> {target} [{feature_set}]: {exc}", flush=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(CROSS_RESULTS_PATH, index=False)
    print(f"Saved results to {CROSS_RESULTS_PATH}")

    if skipped:
        pd.DataFrame(skipped).to_csv(SKIPPED_PATH, index=False)
        print(f"Saved skipped pairs to {SKIPPED_PATH}")

    plot_generalization_gap(results_df)
    print(f"Saved generalization gap chart to {GAP_CHART_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
