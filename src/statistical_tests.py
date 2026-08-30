"""Statistical significance testing for within-project and cross-project defect prediction.

Includes:
1. Wilcoxon signed-rank tests across folds for all model pairs (F1 & AUC-ROC).
2. McNemar's test comparing instance-level prediction accuracy between top 2 models per dataset.
3. Shapiro-Wilk normality testing on metric differences and secondary paired t-tests.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WITHIN_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "within_project"
CROSS_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cross_project"
STATS_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stats_tests"
SUMMARY_CSV_PATH = STATS_OUTPUT_DIR / "significance_summary.csv"


def run_wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = a - b
    if np.all(diff == 0):
        return 0.0, 1.0
    try:
        res = stats.wilcoxon(a, b, zero_method="pratt")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return 0.0, 1.0


def run_mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> tuple[float, float]:
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    # Contingency table:
    # n00: both incorrect, n01: A incorrect & B correct
    # n10: A correct & B incorrect, n11: both correct
    n01 = int(np.sum(~correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))

    if n01 + n10 == 0:
        return 0.0, 1.0

    # For small sample of discordant pairs, use exact binomial test
    if n01 + n10 < 25:
        btest = stats.binomtest(n01, n01 + n10, 0.5)
        return float(n01), float(btest.pvalue)
    else:
        # McNemar continuity-corrected chi-square statistic
        stat = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10)
        p_val = stats.chi2.sf(stat, df=1)
        return float(stat), float(p_val)


def run_shapiro(diff: np.ndarray) -> tuple[float, float]:
    if len(diff) < 3 or np.all(diff == diff[0]):
        return 0.0, 1.0
    try:
        res = stats.shapiro(diff)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return 0.0, 1.0


def run_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = a - b
    if np.all(diff == 0):
        return 0.0, 1.0
    try:
        res = stats.ttest_rel(a, b)
        stat = 0.0 if np.isnan(res.statistic) else float(res.statistic)
        pval = 1.0 if np.isnan(res.pvalue) else float(res.pvalue)
        return stat, pval
    except Exception:
        return 0.0, 1.0


def compute_fold_metrics(df: pd.DataFrame) -> dict[int, dict[str, float]]:
    fold_metrics = {}
    for fold, fold_df in df.groupby("fold"):
        y_true = fold_df["y_true"].to_numpy()
        y_pred = fold_df["y_pred"].to_numpy()
        y_score = fold_df["y_score"].to_numpy()

        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_score))
        else:
            auc = 0.5
        fold_metrics[int(fold)] = {"f1": f1, "auc_roc": auc}
    return fold_metrics


def process_within_project() -> tuple[list[dict[str, object]], list[str]]:
    summary_rows = []
    plain_summaries = []

    # Find prediction files
    pred_files = list(WITHIN_OUTPUT_DIR.glob("*_predictions.csv"))
    if not pred_files:
        return summary_rows, plain_summaries

    # Group files by dataset_feature_cv
    groups: dict[str, dict[str, pd.DataFrame]] = {}
    for filepath in pred_files:
        df = pd.read_csv(filepath)
        if df.empty:
            continue
        dataset = df["dataset"].iloc[0]
        feature_set = df["feature_set"].iloc[0]
        cv_folds = df["cv_folds"].iloc[0]
        model = df["model"].iloc[0]

        group_key = f"{dataset}_{feature_set}_{cv_folds}fold"
        if group_key not in groups:
            groups[group_key] = {}
        groups[group_key][model] = df

    for group_key, model_dfs in sorted(groups.items()):
        models = sorted(model_dfs.keys())
        if len(models) < 2:
            continue

        # 1. Fold-level metrics for each model
        model_fold_metrics: dict[str, dict[int, dict[str, float]]] = {}
        model_avg_auc: dict[str, float] = {}
        model_avg_f1: dict[str, float] = {}

        for model_name in models:
            metrics = compute_fold_metrics(model_dfs[model_name])
            model_fold_metrics[model_name] = metrics
            folds = sorted(metrics.keys())
            model_avg_auc[model_name] = float(np.mean([metrics[f]["auc_roc"] for f in folds]))
            model_avg_f1[model_name] = float(np.mean([metrics[f]["f1"] for f in folds]))

        # 2. Rank models by AUC-ROC to find Top 2 for McNemar test
        ranked_models = sorted(models, key=lambda m: model_avg_auc[m], reverse=True)
        top1, top2 = ranked_models[0], ranked_models[1]

        # Combine instance predictions for McNemar
        df1 = model_dfs[top1].sort_values(by=["fold", "row_index"])
        df2 = model_dfs[top2].sort_values(by=["fold", "row_index"])

        merged = pd.merge(
            df1[["fold", "row_index", "y_true", "y_pred"]],
            df2[["fold", "row_index", "y_pred"]],
            on=["fold", "row_index"],
            suffixes=(f"_{top1}", f"_{top2}"),
        )
        y_true_arr = merged["y_true"].to_numpy()
        y_pred1_arr = merged[f"y_pred_{top1}"].to_numpy()
        y_pred2_arr = merged[f"y_pred_{top2}"].to_numpy()

        mc_stat, mc_pval = run_mcnemar(y_true_arr, y_pred1_arr, y_pred2_arr)
        summary_rows.append(
            {
                "comparison": f"{top1} vs {top2} (instance accuracy)",
                "dataset": group_key,
                "test": "mcnemar",
                "statistic": mc_stat,
                "p_value": mc_pval,
                "significant_at_0.05": mc_pval < 0.05,
            }
        )

        # 3. Model pair comparisons (Wilcoxon, Shapiro-Wilk, paired t-test)
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m1, m2 = models[i], models[j]
                folds = sorted(set(model_fold_metrics[m1].keys()) & set(model_fold_metrics[m2].keys()))
                if not folds:
                    continue

                for metric in ["auc_roc", "f1"]:
                    arr1 = np.array([model_fold_metrics[m1][f][metric] for f in folds])
                    arr2 = np.array([model_fold_metrics[m2][f][metric] for f in folds])
                    diff = arr1 - arr2

                    # Shapiro-Wilk on metric difference
                    sh_stat, sh_pval = run_shapiro(diff)
                    is_norm_violated = sh_pval < 0.05
                    summary_rows.append(
                        {
                            "comparison": f"{m1} vs {m2} ({metric} diff normality)",
                            "dataset": group_key,
                            "test": "shapiro_wilk",
                            "statistic": sh_stat,
                            "p_value": sh_pval,
                            "significant_at_0.05": is_norm_violated,
                        }
                    )

                    # Wilcoxon signed-rank test
                    w_stat, w_pval = run_wilcoxon(arr1, arr2)
                    summary_rows.append(
                        {
                            "comparison": f"{m1} vs {m2} ({metric})",
                            "dataset": group_key,
                            "test": f"wilcoxon_{metric}",
                            "statistic": w_stat,
                            "p_value": w_pval,
                            "significant_at_0.05": w_pval < 0.05,
                        }
                    )

                    # Paired t-test
                    t_stat, t_pval = run_ttest(arr1, arr2)
                    summary_rows.append(
                        {
                            "comparison": f"{m1} vs {m2} ({metric})",
                            "dataset": group_key,
                            "test": f"paired_ttest_{metric}{'_normality_violated' if is_norm_violated else ''}",
                            "statistic": t_stat,
                            "p_value": t_pval,
                            "significant_at_0.05": t_pval < 0.05,
                        }
                    )

                    # Formulate plain language summary for key comparisons (e.g. top1 vs logistic regression)
                    if (m1 == top1 and m2 == "logistic_regression") or (m2 == top1 and m1 == "logistic_regression"):
                        better_model = m1 if np.mean(arr1) > np.mean(arr2) else m2
                        worse_model = m2 if better_model == m1 else m1
                        sig_str = "significantly outperformed" if w_pval < 0.05 else "did not significantly outperform"
                        plain_summaries.append(
                            f"On {group_key.upper()}, {better_model.upper()} {sig_str} {worse_model.upper()} on {metric.upper()} (Wilcoxon p={w_pval:.4f})."
                        )

    return summary_rows, plain_summaries


def process_cross_project() -> list[dict[str, object]]:
    summary_rows = []
    results_path = CROSS_OUTPUT_DIR / "cross_project_results.csv"
    if not results_path.exists():
        return summary_rows

    df = pd.read_csv(results_path)
    if df.empty:
        return summary_rows

    # Group by feature_set
    for feature_set, fs_df in df.groupby("feature_set"):
        models = sorted(fs_df["model"].unique())
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m1, m2 = models[i], models[j]

                m1_df = fs_df[fs_df["model"] == m1].sort_values(by=["source_dataset", "target_dataset"])
                m2_df = fs_df[fs_df["model"] == m2].sort_values(by=["source_dataset", "target_dataset"])

                merged = pd.merge(
                    m1_df[["source_dataset", "target_dataset", "f1", "auc_roc"]],
                    m2_df[["source_dataset", "target_dataset", "f1", "auc_roc"]],
                    on=["source_dataset", "target_dataset"],
                    suffixes=(f"_{m1}", f"_{m2}"),
                )

                if merged.empty:
                    continue

                for metric in ["auc_roc", "f1"]:
                    arr1 = merged[f"{metric}_{m1}"].to_numpy()
                    arr2 = merged[f"{metric}_{m2}"].to_numpy()
                    diff = arr1 - arr2

                    sh_stat, sh_pval = run_shapiro(diff)
                    w_stat, w_pval = run_wilcoxon(arr1, arr2)
                    t_stat, t_pval = run_ttest(arr1, arr2)

                    dataset_label = f"cross_project_{feature_set}"

                    summary_rows.append(
                        {
                            "comparison": f"{m1} vs {m2} ({metric} diff normality)",
                            "dataset": dataset_label,
                            "test": "shapiro_wilk",
                            "statistic": sh_stat,
                            "p_value": sh_pval,
                            "significant_at_0.05": sh_pval < 0.05,
                        }
                    )
                    summary_rows.append(
                        {
                            "comparison": f"{m1} vs {m2} ({metric})",
                            "dataset": dataset_label,
                            "test": f"wilcoxon_{metric}",
                            "statistic": w_stat,
                            "p_value": w_pval,
                            "significant_at_0.05": w_pval < 0.05,
                        }
                    )
                    summary_rows.append(
                        {
                            "comparison": f"{m1} vs {m2} ({metric})",
                            "dataset": dataset_label,
                            "test": f"paired_ttest_{metric}",
                            "statistic": t_stat,
                            "p_value": t_pval,
                            "significant_at_0.05": t_pval < 0.05,
                        }
                    )
    return summary_rows


def main() -> int:
    warnings.filterwarnings("ignore")
    STATS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    within_rows, plain_summaries = process_within_project()
    cross_rows = process_cross_project()

    all_rows = within_rows + cross_rows
    summary_df = pd.DataFrame(all_rows)

    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
    print(f"Saved statistical significance summary table to {SUMMARY_CSV_PATH}")

    print("\n--- PLAIN-LANGUAGE STATISTICAL SUMMARIES ---")
    for summary in sorted(set(plain_summaries)):
        print(summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
