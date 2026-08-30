"""Feature distribution drift profiler for cross-project defect prediction.

Computes Kolmogorov-Smirnov (KS) test statistics and p-values to quantify domain
shift between source and target software metric distributions prior to transfer.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from train_within_project import STATIC_DIR, load_and_clean, prepare_xy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STABILITY_DIR = PROJECT_ROOT / "outputs" / "stability"
DRIFT_CSV_PATH = STABILITY_DIR / "feature_drift_analysis.csv"
DRIFT_PLOT_PATH = STABILITY_DIR / "ks_drift_heatmap.png"


def process_feature_drift() -> pd.DataFrame:
    pairs = [
        ("kc1", "pc1"),
        ("pc1", "cm1"),
        ("kc1", "cm1"),
        ("pc1", "pc3"),
        ("cm1", "pc3"),
        ("aeeem_eclipse", "aeeem_equinox"),
        ("aeeem_eclipse", "aeeem_lucene"),
        ("aeeem_equinox", "aeeem_mylyn"),
        ("aeeem_lucene", "aeeem_pde"),
    ]

    rows = []

    for src_name, tgt_name in pairs:
        src_path = STATIC_DIR / f"{src_name}.csv"
        tgt_path = STATIC_DIR / f"{tgt_name}.csv"

        if not src_path.exists() or not tgt_path.exists():
            continue

        df_src = load_and_clean(src_path)
        df_tgt = load_and_clean(tgt_path)

        Xs, _ = prepare_xy(df_src)
        Xt, _ = prepare_xy(df_tgt)

        common_cols = sorted(set(Xs.columns) & set(Xt.columns))
        if not common_cols:
            continue

        for col in common_cols:
            vals_s = Xs[col].dropna().values
            vals_t = Xt[col].dropna().values

            ks_res = stats.ks_2samp(vals_s, vals_t)
            ks_stat = float(ks_res.statistic)
            ks_pval = float(ks_res.pvalue)

            is_drifted = (ks_stat > 0.30) and (ks_pval < 0.05)

            rows.append(
                {
                    "source": src_name,
                    "target": tgt_name,
                    "feature": col,
                    "ks_statistic": round(ks_stat, 4),
                    "ks_pvalue": round(ks_pval, 6),
                    "severe_drift": is_drifted,
                }
            )

    return pd.DataFrame(rows)


def plot_ks_heatmap(df: pd.DataFrame) -> None:
    # Average KS statistic per dataset pair
    pair_ks = df.groupby(["source", "target"], as_index=False)["ks_statistic"].mean()
    datasets = sorted(set(pair_ks["source"]).union(set(pair_ks["target"])))

    matrix = pd.DataFrame(0.0, index=datasets, columns=datasets)
    for _, r in pair_ks.iterrows():
        val = r["ks_statistic"]
        matrix.loc[r["source"], r["target"]] = val
        matrix.loc[r["target"], r["source"]] = val

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        cbar=True,
        linewidths=0.5,
        square=True,
        vmin=0.0,
        vmax=0.8,
    )
    plt.title("Cross-Project Feature Distribution Drift (Average KS Statistic)", fontsize=12, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9, fontweight="bold")
    plt.yticks(rotation=0, fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(DRIFT_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> int:
    STABILITY_DIR.mkdir(parents=True, exist_ok=True)

    print("Quantifying feature distribution drift via Kolmogorov-Smirnov (KS) tests...", flush=True)
    df_drift = process_feature_drift()

    df_drift.to_csv(DRIFT_CSV_PATH, index=False)
    print(f"Saved Feature Drift CSV to {DRIFT_CSV_PATH}")

    plot_ks_heatmap(df_drift)
    print(f"Saved KS Drift Heatmap to {DRIFT_PLOT_PATH}")

    # Print top severe drift features
    severe_df = df_drift[df_drift["severe_drift"]]
    print(f"\nIdentified {len(severe_df)} feature-pair instances with severe domain drift (KS > 0.30, p < 0.05).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
