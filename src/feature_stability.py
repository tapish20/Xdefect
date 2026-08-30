"""Feature stability analysis across projects for RQ2.

Loads SHAP feature rankings from Phase 6 and computes three stability metrics
between each dataset pair:
1. Jaccard similarity of top-k features (k=5 and k=10)
2. Spearman rank correlation across shared features
3. Kendall's Tau correlation across shared features

Exports:
- Pairwise stability heatmaps (outputs/stability/{metric}_heatmap.png)
- Combined CSV (outputs/stability/cross_project_feature_stability.csv)
- Plain-language interpretation of stable vs dataset-specific features.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHAP_DIR = PROJECT_ROOT / "outputs" / "shap"
STABILITY_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stability"
STABILITY_CSV_PATH = STABILITY_OUTPUT_DIR / "cross_project_feature_stability.csv"


def load_rankings() -> dict[str, list[dict[str, object]]]:
    json_files = list(SHAP_DIR.glob("*_feature_ranking.json"))
    rankings = {}
    for filepath in sorted(json_files):
        dataset_name = filepath.name.replace("_feature_ranking.json", "")
        with open(filepath, "r", encoding="utf-8") as f:
            rankings[dataset_name] = json.load(f)
    return rankings


def compute_jaccard(set_a: set[str], set_b: set[str]) -> float:
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def plot_heatmap(df_matrix: pd.DataFrame, metric_name: str, title: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar=True,
        linewidths=0.5,
        square=True,
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9, fontweight="bold")
    plt.yticks(rotation=0, fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def process_stability() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rankings = load_rankings()
    dataset_names = sorted(rankings.keys())

    parsed_data = {}
    for name, item_list in rankings.items():
        feat_ranks = {item["feature"]: item["rank"] for item in item_list}
        feat_shaps = {item["feature"]: item["mean_abs_shap"] for item in item_list}
        sorted_feats = [item["feature"] for item in item_list]
        parsed_data[name] = {
            "ranks": feat_ranks,
            "shaps": feat_shaps,
            "top5": set(sorted_feats[:5]),
            "top10": set(sorted_feats[:10]),
        }

    rows = []
    pairs = list(itertools.combinations(dataset_names, 2))

    for da, db in pairs:
        info_a = parsed_data[da]
        info_b = parsed_data[db]

        jacc_5 = compute_jaccard(info_a["top5"], info_b["top5"])
        jacc_10 = compute_jaccard(info_a["top10"], info_b["top10"])

        common_feats = sorted(set(info_a["ranks"].keys()) & set(info_b["ranks"].keys()))

        if len(common_feats) >= 3:
            ranks_a = [info_a["ranks"][f] for f in common_feats]
            ranks_b = [info_b["ranks"][f] for f in common_feats]

            sp_res = stats.spearmanr(ranks_a, ranks_b)
            sp_corr = float(sp_res.statistic) if not np.isnan(sp_res.statistic) else 0.0
            sp_pval = float(sp_res.pvalue) if not np.isnan(sp_res.pvalue) else 1.0

            kt_res = stats.kendalltau(ranks_a, ranks_b)
            kt_tau = float(kt_res.statistic) if not np.isnan(kt_res.statistic) else 0.0
            kt_pval = float(kt_res.pvalue) if not np.isnan(kt_res.pvalue) else 1.0
        else:
            sp_corr, sp_pval = np.nan, np.nan
            kt_tau, kt_pval = np.nan, np.nan

        rows.append(
            {
                "dataset_a": da,
                "dataset_b": db,
                "jaccard_top5": jacc_5,
                "jaccard_top10": jacc_10,
                "spearman_corr": sp_corr,
                "spearman_pvalue": sp_pval,
                "kendall_tau": kt_tau,
                "kendall_pvalue": kt_pval,
            }
        )

    results_df = pd.DataFrame(rows)

    # Construct symmetric matrices for heatmaps
    metrics_to_plot = ["jaccard_top5", "jaccard_top10", "spearman_corr", "kendall_tau"]
    matrices = {}

    for metric in metrics_to_plot:
        matrix = pd.DataFrame(1.0, index=dataset_names, columns=dataset_names)
        for _, row in results_df.iterrows():
            val = row[metric]
            matrix.loc[row["dataset_a"], row["dataset_b"]] = val
            matrix.loc[row["dataset_b"], row["dataset_a"]] = val
        matrices[metric] = matrix

    return results_df, matrices


def print_feature_interpretation(rankings: dict[str, list[dict[str, object]]]) -> None:
    # Frequency of features appearing in Top 5 and Top 10 across all datasets
    top5_counts: dict[str, int] = {}
    top10_counts: dict[str, int] = {}
    total_datasets = len(rankings)

    for item_list in rankings.values():
        feats = [item["feature"] for item in item_list]
        for f in feats[:5]:
            top5_counts[f] = top5_counts.get(f, 0) + 1
        for f in feats[:10]:
            top10_counts[f] = top10_counts.get(f, 0) + 1

    sorted_top5 = sorted(top5_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_top10 = sorted(top10_counts.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 70)
    print("FEATURE STABILITY INTERPRETATION (RESEARCH QUESTION 2)")
    print("=" * 70)

    print("\n1. CONSISTENTLY STABLE FEATURES (High Overlap Across Projects):")
    for feat, count in sorted_top5:
        if count >= 3:
            pct = (count / total_datasets) * 100
            print(f"   - {feat}: Top-5 in {count}/{total_datasets} datasets ({pct:.1f}%)")

    for feat, count in sorted_top10:
        if count >= 4 and feat not in [f for f, c in sorted_top5 if c >= 3]:
            pct = (count / total_datasets) * 100
            print(f"   - {feat}: Top-10 in {count}/{total_datasets} datasets ({pct:.1f}%)")

    print("\n2. DATASET-SPECIFIC FEATURES (High Importance in 1 Dataset, Low/Absent in Others):")
    for feat, count in sorted_top5:
        if count == 1:
            print(f"   - {feat}: Top-5 in only 1 dataset")

    print("=" * 70 + "\n")


def main() -> int:
    STABILITY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_df, matrices = process_stability()
    results_df.to_csv(STABILITY_CSV_PATH, index=False)
    print(f"Saved feature stability summary CSV to {STABILITY_CSV_PATH}")

    # Plot heatmaps
    plot_heatmap(
        matrices["jaccard_top5"],
        "jaccard_top5",
        "Top-5 Jaccard Feature Similarity Across Projects",
        STABILITY_OUTPUT_DIR / "jaccard_top5_heatmap.png",
    )
    plot_heatmap(
        matrices["jaccard_top10"],
        "jaccard_top10",
        "Top-10 Jaccard Feature Similarity Across Projects",
        STABILITY_OUTPUT_DIR / "jaccard_top10_heatmap.png",
    )
    plot_heatmap(
        matrices["spearman_corr"],
        "spearman_corr",
        "Spearman Rank Correlation Across Projects",
        STABILITY_OUTPUT_DIR / "spearman_corr_heatmap.png",
    )
    plot_heatmap(
        matrices["kendall_tau"],
        "kendall_tau",
        "Kendall's Tau Correlation Across Projects",
        STABILITY_OUTPUT_DIR / "kendall_tau_heatmap.png",
    )
    print(f"Saved stability heatmaps to {STABILITY_OUTPUT_DIR}")

    rankings = load_rankings()
    print_feature_interpretation(rankings)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
