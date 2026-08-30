"""Domain adaptation module using Transfer Component Analysis (TCA) for cross-project defect prediction.

Reduces domain shift between source and target projects by projecting features
into a shared latent subspace that minimizes Maximum Mean Discrepancy (MMD).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.preprocessing import StandardScaler

from train_within_project import (
    HYBRID_DIR,
    STATIC_DIR,
    compute_metrics,
    load_and_clean,
    model_factories,
    prepare_xy,
    resample_training_fold,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DA_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "domain_adaptation"
TCA_CSV_PATH = DA_OUTPUT_DIR / "tca_cross_project_results.csv"
TCA_PLOT_PATH = DA_OUTPUT_DIR / "tca_generalization_gain.png"


def compute_tca(Xs: np.ndarray, Xt: np.ndarray, num_components: int = 5, mu: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Fast Transfer Component Analysis (TCA) using Symmetric Generalized Eigen-decomposition."""
    ns, num_feats = Xs.shape
    nt, _ = Xt.shape
    n = ns + nt

    X = np.vstack((Xs, Xt))

    # Kernel matrix (Linear Kernel)
    K = np.dot(X, X.T)

    # MMD Matrix L
    L = np.zeros((n, n), dtype=np.float64)
    L[:ns, :ns] = 1.0 / (ns * ns)
    L[ns:, ns:] = 1.0 / (nt * nt)
    L[:ns, ns:] = -1.0 / (ns * nt)
    L[ns:, :ns] = -1.0 / (ns * nt)

    # Centering Matrix H
    H = np.eye(n, dtype=np.float64) - (1.0 / n) * np.ones((n, n), dtype=np.float64)

    # Left: K L K + mu * I, Right: K H K
    KLK = np.dot(np.dot(K, L), K)
    KHK = np.dot(np.dot(K, H), K)

    left = (KLK + KLK.T) / 2.0 + mu * np.eye(n, dtype=np.float64)
    right = (KHK + KHK.T) / 2.0 + 1e-5 * np.eye(n, dtype=np.float64)

    # Use fast symmetric generalized eigenvalue solver
    eigenvalues, eigenvectors = linalg.eigh(left, right)
    idx = np.argsort(np.abs(eigenvalues))
    A = eigenvectors[:, idx[:num_components]].real

    # Project kernel
    Z = np.dot(K, A)
    Zs = Z[:ns, :]
    Zt = Z[ns:, :]

    return Zs, Zt


def process_domain_adaptation() -> pd.DataFrame:
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

    models_to_test = ["random_forest", "xgboost"]
    builders = model_factories()
    rows = []

    for src_name, tgt_name in pairs:
        src_path = STATIC_DIR / f"{src_name}.csv"
        tgt_path = STATIC_DIR / f"{tgt_name}.csv"

        if not src_path.exists() or not tgt_path.exists():
            continue

        df_src = load_and_clean(src_path)
        df_tgt = load_and_clean(tgt_path)

        Xs, ys = prepare_xy(df_src)
        Xt, yt = prepare_xy(df_tgt)

        # Common features
        common_cols = sorted(set(Xs.columns) & set(Xt.columns))
        if len(common_cols) < 3:
            continue

        Xs = Xs[common_cols]
        Xt = Xt[common_cols]

        scaler = StandardScaler()
        Xs_scaled = scaler.fit_transform(Xs)
        Xt_scaled = scaler.transform(Xt)

        # Apply TCA domain adaptation
        num_comp = min(5, len(common_cols))
        Zs, Zt = compute_tca(Xs_scaled, Xt_scaled, num_components=num_comp, mu=1.0)

        for model_name in models_to_test:
            # 1. Baseline: Raw Cross-Project Model
            Xs_res, ys_res = resample_training_fold(pd.DataFrame(Xs_scaled), ys)
            model_raw = builders[model_name]()
            model_raw.fit(Xs_res, ys_res)

            y_pred_raw = model_raw.predict(Xt_scaled)
            y_prob_raw = model_raw.predict_proba(Xt_scaled)[:, 1]
            raw_metrics = compute_metrics(yt, y_pred_raw, y_prob_raw)

            # 2. TCA Adapted Model
            Zs_res, ys_tca_res = resample_training_fold(pd.DataFrame(Zs), ys)
            model_tca = builders[model_name]()
            model_tca.fit(Zs_res, ys_tca_res)

            y_pred_tca = model_tca.predict(Zt)
            y_prob_tca = model_tca.predict_proba(Zt)[:, 1]
            tca_metrics = compute_metrics(yt, y_pred_tca, y_prob_tca)

            rows.append(
                {
                    "source": src_name,
                    "target": tgt_name,
                    "model": model_name,
                    "raw_auc": round(float(raw_metrics["auc_roc"]), 4),
                    "tca_auc": round(float(tca_metrics["auc_roc"]), 4),
                    "auc_gain": round(float(tca_metrics["auc_roc"] - raw_metrics["auc_roc"]), 4),
                    "raw_mcc": round(float(raw_metrics["mcc"]), 4),
                    "tca_mcc": round(float(tca_metrics["mcc"]), 4),
                    "mcc_gain": round(float(tca_metrics["mcc"] - raw_metrics["mcc"]), 4),
                    "raw_f1": round(float(raw_metrics["f1"]), 4),
                    "tca_f1": round(float(tca_metrics["f1"]), 4),
                }
            )

    return pd.DataFrame(rows)


def plot_tca_gain(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))

    pairs_labels = [f"{r['source']} -> {r['target']}\n({r['model']})" for _, r in df.iterrows()]
    x = np.arange(len(pairs_labels))
    width = 0.35

    rects1 = plt.bar(x - width / 2, df["raw_auc"], width, label="Baseline (No Adaptation)", color="#ff7f0e", edgecolor="black")
    rects2 = plt.bar(x + width / 2, df["tca_auc"], width, label="TCA Domain Adapted", color="#2ca02c", edgecolor="black")

    for rect in rects1:
        h = rect.get_height()
        plt.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        plt.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    plt.xticks(x, pairs_labels, rotation=45, ha="right", fontsize=8.5, fontweight="bold")
    plt.ylabel("AUC-ROC Score", fontsize=11, fontweight="bold")
    plt.title("Cross-Project Domain Adaptation Gain via Transfer Component Analysis (TCA)", fontsize=12, fontweight="bold", pad=12)
    plt.ylim(0, 1.1)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(TCA_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> int:
    DA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Running Transfer Component Analysis (TCA) Domain Adaptation...", flush=True)
    df_results = process_domain_adaptation()

    df_results.to_csv(TCA_CSV_PATH, index=False)
    print(f"Saved TCA results CSV to {TCA_CSV_PATH}")

    plot_tca_gain(df_results)
    print(f"Saved TCA generalization gain plot to {TCA_PLOT_PATH}")

    print("\nSummary of TCA Domain Adaptation Gains:")
    print(df_results[["source", "target", "model", "raw_auc", "tca_auc", "auc_gain"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
