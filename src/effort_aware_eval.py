"""Effort-aware defect prediction evaluator.

Calculates P_opt20 (Defect Recall at 20% LOC inspected) and Effort-Aware Inspection
Cost-Benefit Curves across datasets.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from train_within_project import (
    HYBRID_DIR,
    STATIC_DIR,
    load_and_clean,
    model_factories,
    prepare_xy,
    resample_training_fold,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EFFORT_DIR = PROJECT_ROOT / "outputs" / "effort_aware"
CSV_PATH = EFFORT_DIR / "effort_aware_summary.csv"
PLOT_PATH = EFFORT_DIR / "popt_inspection_curve.png"


def compute_popt(y_true: np.ndarray, y_score: np.ndarray, loc: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    df_eval = pd.DataFrame({"y_true": y_true, "y_score": y_score, "loc": loc})
    df_eval["density"] = df_eval["y_score"] / (df_eval["loc"] + 1e-5)

    # Sort by risk density descending
    df_model = df_eval.sort_values(by="density", ascending=False).reset_index(drop=True)
    df_model["cum_loc"] = df_model["loc"].cumsum() / df_model["loc"].sum()
    df_model["cum_defects"] = df_model["y_true"].cumsum() / max(df_model["y_true"].sum(), 1)

    # Optimal curve (sorted by actual defects / loc)
    df_opt = df_eval.sort_values(by=["y_true", "loc"], ascending=[False, True]).reset_index(drop=True)
    df_opt["cum_loc"] = df_opt["loc"].cumsum() / df_opt["loc"].sum()
    df_opt["cum_defects"] = df_opt["y_true"].cumsum() / max(df_opt["y_true"].sum(), 1)

    # P_opt at 20% LOC
    idx_20 = np.searchsorted(df_model["cum_loc"].values, 0.20)
    popt20 = float(df_model["cum_defects"].iloc[min(idx_20, len(df_model) - 1)])

    return round(popt20, 4), df_model["cum_loc"].values, df_model["cum_defects"].values


def run_effort_evaluation() -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    configs = [
        ("cm1", "static", "loc"),
        ("kc1", "static", "loc"),
        ("pc1", "static", "loc"),
        ("pc3", "static", "loc"),
        ("aeeem_eclipse", "hybrid", "numberOfLinesOfCode"),
        ("aeeem_equinox", "hybrid", "numberOfLinesOfCode"),
        ("aeeem_lucene", "hybrid", "numberOfLinesOfCode"),
    ]

    results = []
    curves = {}
    builders = model_factories()

    for dataset, feature_set, loc_col in configs:
        data_path = STATIC_DIR / f"{dataset}.csv" if feature_set == "static" else HYBRID_DIR / f"{dataset}_hybrid.csv"
        if not data_path.exists():
            continue

        df = load_and_clean(data_path)
        if feature_set == "hybrid" and "repo_metrics_available" in df.columns:
            df = df[df["repo_metrics_available"].astype(bool)].copy()

        if loc_col not in df.columns:
            # Fallback to any line count column
            loc_candidates = [c for c in df.columns if "loc" in c.lower() or "line" in c.lower()]
            loc_col = loc_candidates[0] if loc_candidates else df.columns[0]

        loc_values = df[loc_col].values
        X, y = prepare_xy(df)

        X_train, X_test, y_train, y_test, loc_tr, loc_te = train_test_split(
            X, y, loc_values, test_size=0.3, random_state=42, stratify=y
        )
        X_tr_res, y_tr_res = resample_training_fold(X_train, y_train)

        for model_name, factory in builders.items():
            clf = factory()
            clf.fit(X_tr_res, y_tr_res)
            y_score = clf.predict_proba(X_test)[:, 1]

            popt20, cum_loc, cum_defects = compute_popt(y_test.values, y_score, loc_te)
            results.append(
                {
                    "dataset": dataset,
                    "feature_set": feature_set,
                    "model": model_name,
                    "popt20": popt20,
                }
            )

            if model_name == "random_forest":
                curves[f"{dataset}_{feature_set}"] = (cum_loc, cum_defects)

    return pd.DataFrame(results), curves


def plot_effort_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    plt.figure(figsize=(9, 6))
    for key, (loc, defects) in curves.items():
        plt.plot(loc * 100, defects * 100, label=key, linewidth=2)

    plt.axvline(x=20, color="red", linestyle="--", alpha=0.7, label="20% Inspection Limit")
    plt.plot([0, 100], [0, 100], "k--", alpha=0.4, label="Random Inspection Baseline")
    plt.title("Effort-Aware Defect Inspection Cost-Benefit Curves (P_opt)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("% Lines of Code Inspected", fontsize=11, fontweight="bold")
    plt.ylabel("% Defect Recall Achieved", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> int:
    EFFORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Running Effort-Aware Defect Prediction Evaluation (P_opt20)...", flush=True)
    df_effort, curves = run_effort_evaluation()

    df_effort.to_csv(CSV_PATH, index=False)
    print(f"Saved Effort-Aware CSV to {CSV_PATH}")

    plot_effort_curves(curves)
    print(f"Saved P_opt Inspection Curve Plot to {PLOT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
