"""Refactoring recommendation engine driven by SHAP explainability for RQ4.

Maps stable high-importance features to actionable, plain-English refactoring
recommendations using dataset-specific dynamic 75th percentile thresholds.
Provides end-to-end traceability: Prediction -> SHAP Explanation -> Recommendation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
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
SHAP_DIR = PROJECT_ROOT / "outputs" / "shap"
RECOMMENDATIONS_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "recommendations"
JSON_REPORT_PATH = RECOMMENDATIONS_OUTPUT_DIR / "sample_reports.json"
MD_REPORT_PATH = RECOMMENDATIONS_OUTPUT_DIR / "sample_reports.md"

TREE_MODELS = {"random_forest", "xgboost", "lightgbm"}

# Stable feature categories identified in Phase 7
STABLE_CATEGORIES = {
    "size": {"numberOfLinesOfCode", "NUMBER_OF_LINES", "avgLOC_EXECUTABLE", "sumLOC_EXECUTABLE", "LOC_BLANK", "LOC_EXECUTABLE"},
    "complexity": {"wmc", "CYCLOMATIC_COMPLEXITY", "avgCYCLOMATIC_COMPLEXITY", "sumCYCLOMATIC_COMPLEXITY"},
    "coupling": {"rfc", "cbo", "fanOut", "COUPLING_BETWEEN_OBJECTS", "RESPONSE_FOR_CLASS"},
    "churn": {"CvsExpEntropy", "CvsLogEntropy", "CvsLinEntropy", "code_churn", "commit_count", "exp_churn"},
    "hierarchy": {"numberOfPublicMethods", "numberOfMethodsInherited", "numberOfMethods"},
}


def get_best_tree_models() -> dict[tuple[str, str], str]:
    results_path = WITHIN_OUTPUT_DIR / "within_project_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing Phase 3 results file: {results_path}")

    df = pd.read_csv(results_path)
    tree_df = df[df["model"].isin(TREE_MODELS)].copy()
    avg_scores = tree_df.groupby(["dataset", "feature_set", "model"], as_index=False)["auc_roc_mean"].mean()

    best_models = {}
    for (dataset, feature_set), group in avg_scores.groupby(["dataset", "feature_set"]):
        best_row = group.sort_values(by="auc_roc_mean", ascending=False).iloc[0]
        best_models[(dataset, feature_set)] = str(best_row["model"])

    return best_models


def extract_positive_shap(shap_vals: np.ndarray | list[np.ndarray]) -> np.ndarray:
    if isinstance(shap_vals, list):
        arr = shap_vals[1]
    elif isinstance(shap_vals, np.ndarray):
        if shap_vals.ndim == 3:
            arr = shap_vals[:, :, 1]
        else:
            arr = shap_vals
    else:
        arr = np.asarray(shap_vals)
    return arr


def evaluate_refactoring_rules(
    feature_dict: dict[str, float],
    shap_dict: dict[str, float],
    p75_dict: dict[str, float],
) -> list[dict[str, object]]:
    recommendations = []

    # Check for high features above 75th percentile
    high_feats = {f: val for f, val in feature_dict.items() if val > p75_dict.get(f, float("inf"))}

    # 1. Complexity & Size Rule
    high_comp = [f for f in high_feats if any(f in STABLE_CATEGORIES["complexity"] for c in [f])]
    high_size = [f for f in high_feats if any(f in STABLE_CATEGORIES["size"] for c in [f])]

    if high_comp or high_size:
        triggered = []
        for f in high_comp + high_size:
            triggered.append(
                {
                    "feature": f,
                    "value": float(feature_dict[f]),
                    "p75_threshold": float(p75_dict[f]),
                    "shap_contribution": float(shap_dict.get(f, 0.0)),
                }
            )
        comp_str = ", ".join([f"{t['feature']}={t['value']:.1f} (P75={t['p75_threshold']:.1f})" for t in triggered if t['feature'] in high_comp])
        size_str = ", ".join([f"{t['feature']}={t['value']:.1f} (P75={t['p75_threshold']:.1f})" for t in triggered if t['feature'] in high_size])

        rec_text = "High code complexity and/or size detected."
        if comp_str and size_str:
            rec_text = f"High complexity ({comp_str}) combined with large module size ({size_str}). Consider splitting this module/class into smaller, decoupled units using 'Extract Method' or 'Extract Class' refactorings."
        elif comp_str:
            rec_text = f"High cyclomatic complexity ({comp_str}). Simplify control flow structures and break complex conditional branches using 'Decompose Conditional'."
        elif size_str:
            rec_text = f"Large module size ({size_str}). Decompose class into focused sub-components to adhere to Single Responsibility Principle (SRP)."

        recommendations.append(
            {
                "rule": "Complexity & Size Management",
                "recommendation": rec_text,
                "triggered_by_features": triggered,
            }
        )

    # 2. Coupling & Dependency Rule
    high_coupling = [f for f in high_feats if any(f in STABLE_CATEGORIES["coupling"] for c in [f])]
    if high_coupling:
        triggered = [
            {
                "feature": f,
                "value": float(feature_dict[f]),
                "p75_threshold": float(p75_dict[f]),
                "shap_contribution": float(shap_dict.get(f, 0.0)),
            }
            for f in high_coupling
        ]
        coup_str = ", ".join([f"{t['feature']}={t['value']:.1f} (P75={t['p75_threshold']:.1f})" for t in triggered])
        recommendations.append(
            {
                "rule": "Coupling & Dependency Reduction",
                "recommendation": f"Excessive inter-module coupling and dependency fan-out ({coup_str}). Reduce tight coupling by applying Dependency Inversion (DIP) or introducing Facade/Adapter patterns.",
                "triggered_by_features": triggered,
            }
        )

    # 3. Churn & Process Risk Rule
    high_churn = [f for f in high_feats if any(f in STABLE_CATEGORIES["churn"] for c in [f])]
    if high_churn:
        triggered = [
            {
                "feature": f,
                "value": float(feature_dict[f]),
                "p75_threshold": float(p75_dict[f]),
                "shap_contribution": float(shap_dict.get(f, 0.0)),
            }
            for f in high_churn
        ]
        churn_str = ", ".join([f"{t['feature']}={t['value']:.2f} (P75={t['p75_threshold']:.2f})" for t in triggered])
        recommendations.append(
            {
                "rule": "Process Churn & Modification Risk",
                "recommendation": f"High code churn and modification entropy ({churn_str}). This module experiences frequent, volatile changes; prioritize for mandatory senior code review and expanded regression test suites.",
                "triggered_by_features": triggered,
            }
        )

    # 4. Hierarchy & API Surface Rule
    high_hier = [f for f in high_feats if any(f in STABLE_CATEGORIES["hierarchy"] for c in [f])]
    if high_hier:
        triggered = [
            {
                "feature": f,
                "value": float(feature_dict[f]),
                "p75_threshold": float(p75_dict[f]),
                "shap_contribution": float(shap_dict.get(f, 0.0)),
            }
            for f in high_hier
        ]
        hier_str = ", ".join([f"{t['feature']}={t['value']:.1f} (P75={t['p75_threshold']:.1f})" for t in triggered])
        recommendations.append(
            {
                "rule": "API Surface & Hierarchy Refactoring",
                "recommendation": f"Large method API surface / inherited complexity ({hier_str}). Consider favoring composition over inheritance and segregating interfaces (Interface Segregation Principle).",
                "triggered_by_features": triggered,
            }
        )

    # Default recommendation if no specific rule triggered
    if not recommendations:
        top_shap_feat = max(shap_dict.items(), key=lambda x: x[1])[0] if shap_dict else "unknown"
        recommendations.append(
            {
                "rule": "General Defect Mitigation",
                "recommendation": f"Elevated defect risk driven by {top_shap_feat} (SHAP={shap_dict.get(top_shap_feat, 0.0):.4f}). Conduct focused code audit and increase automated test coverage.",
                "triggered_by_features": [
                    {
                        "feature": top_shap_feat,
                        "value": float(feature_dict.get(top_shap_feat, 0.0)),
                        "p75_threshold": float(p75_dict.get(top_shap_feat, 0.0)),
                        "shap_contribution": float(shap_dict.get(top_shap_feat, 0.0)),
                    }
                ],
            }
        )

    return recommendations


def generate_sample_reports() -> list[dict[str, object]]:
    best_models = get_best_tree_models()
    reports = []

    # Target diverse datasets for sample reports
    target_configs = [
        ("cm1", "static"),
        ("kc1", "static"),
        ("pc1", "static"),
        ("pc3", "static"),
        ("aeeem_eclipse", "static"),
        ("aeeem_eclipse", "hybrid"),
        ("aeeem_equinox", "hybrid"),
        ("aeeem_mylyn", "hybrid"),
    ]

    for dataset, feature_set in target_configs:
        if (dataset, feature_set) not in best_models:
            continue
        model_name = best_models[(dataset, feature_set)]

        if feature_set == "static":
            data_path = STATIC_DIR / f"{dataset}.csv"
            dataset_key = dataset
        else:
            data_path = HYBRID_DIR / f"{dataset}_hybrid.csv"
            dataset_key = f"{dataset}_hybrid"

        if not data_path.exists():
            continue

        df = load_and_clean(data_path)
        if feature_set == "hybrid" and "repo_metrics_available" in df.columns:
            df = df[df["repo_metrics_available"].astype(bool)].copy()

        if df.empty:
            continue

        X, y = prepare_xy(df)

        # 75th Percentile threshold dictionary
        p75_dict = X.quantile(0.75).to_dict()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train_res, y_train_res = resample_training_fold(X_train, y_train)

        builders = model_factories()
        model = builders[model_name]()
        model.fit(X_train_res, y_train_res)

        y_score = model.predict_proba(X_test)[:, 1]

        explainer = shap.TreeExplainer(model)
        try:
            raw_shap_vals = explainer.shap_values(X_test, check_additivity=False)
        except Exception:
            raw_shap_vals = explainer.shap_values(X_test)
        shap_matrix = extract_positive_shap(raw_shap_vals)

        # Pick top high-risk module in test set
        high_risk_indices = np.argsort(y_score)[::-1]
        for idx in high_risk_indices[:1]:  # Pick top highest risk module per dataset
            risk_score = float(y_score[idx])
            if risk_score < 0.50:
                continue

            instance_x = X_test.iloc[idx].to_dict()
            instance_shap = {feat: float(shap_matrix[idx, col_i]) for col_i, feat in enumerate(X_test.columns)}

            # Top SHAP features by magnitude
            top_shap_feats = sorted(instance_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            top_shap_list = [
                {
                    "feature": feat,
                    "value": float(instance_x[feat]),
                    "shap_value": float(val),
                }
                for feat, val in top_shap_feats
            ]

            recs = evaluate_refactoring_rules(instance_x, instance_shap, p75_dict)

            reports.append(
                {
                    "module_id": f"{dataset_key}_test_row_{X_test.index[idx]}",
                    "dataset": dataset_key,
                    "feature_set": feature_set,
                    "model_used": model_name,
                    "predicted_risk_score": round(risk_score, 4),
                    "predicted_risk_level": "High Risk" if risk_score >= 0.70 else "Medium Risk",
                    "top_shap_contributions": top_shap_list,
                    "refactoring_recommendations": recs,
                }
            )

    return reports


def export_markdown_report(reports: list[dict[str, object]]) -> None:
    lines = [
        "# Refactoring Recommendation Reports (Sample High-Risk Modules)",
        "",
        "This report demonstrates end-to-end traceability from **Defect Risk Prediction $\\rightarrow$ SHAP Feature Attribution $\\rightarrow$ Actionable Refactoring Recommendations** to address **Research Question 4 (RQ4)**.",
        "",
        "---",
        "",
    ]

    for idx, report in enumerate(reports, start=1):
        lines.append(f"## Module Report #{idx}: `{report['module_id']}`")
        lines.append(f"- **Dataset**: `{report['dataset']}` ({report['feature_set']} feature set)")
        lines.append(f"- **Model Used**: `{report['model_used']}`")
        lines.append(f"- **Predicted Risk Score**: `{report['predicted_risk_score']}` ({report['predicted_risk_level']})")
        lines.append("")
        lines.append("### Top SHAP-Contributing Features")
        lines.append("| Feature | Actual Value | SHAP Contribution |")
        lines.append("|---|---|---|")
        for item in report["top_shap_contributions"]:
            lines.append(f"| `{item['feature']}` | `{item['value']:.4f}` | `{item['shap_value']:+.4f}` |")
        lines.append("")
        lines.append("### Actionable Refactoring Recommendations")
        for rec in report["refactoring_recommendations"]:
            lines.append(f"#### Category: {rec['rule']}")
            lines.append(f"> **Recommendation**: {rec['recommendation']}")
            lines.append("")
            lines.append("*Traceability (Triggered Features)*:")
            for trig in rec["triggered_by_features"]:
                lines.append(
                    f"- Feature `{trig['feature']}` = `{trig['value']:.2f}` exceeding 75th percentile threshold (`{trig['p75_threshold']:.2f}`) with SHAP impact `{trig['shap_contribution']:+.4f}`"
                )
            lines.append("")
        lines.append("---")
        lines.append("")

    with open(MD_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    RECOMMENDATIONS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating refactoring recommendation sample reports...", flush=True)
    reports = generate_sample_reports()

    with open(JSON_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    print(f"Saved JSON report to {JSON_REPORT_PATH}")

    export_markdown_report(reports)
    print(f"Saved Markdown report to {MD_REPORT_PATH}")

    print(f"\nGenerated {len(reports)} high-risk module reports with complete SHAP traceability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
