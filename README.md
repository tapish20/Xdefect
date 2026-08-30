# Explainable Cross-Project Software Defect Prediction Using Hybrid Static Code Metrics and Repository Mining

This repository presents a comprehensive, empirical framework for explainable software defect prediction, combining static code metrics with Git repository historical metrics (churn, entropy, commit history). It evaluates machine learning classifiers across within-project and cross-project settings, assesses SHAP feature stability, provides statistical significance testing, and delivers end-to-end traceable refactoring recommendations.

---

## 🚀 Quick Start & Execution

### 1. Run Complete Pipeline (Phases 3–9)
To execute all experimental phases sequentially:

```powershell
cd explainable-cross-project-defect-prediction
..\.venv_xcpdp\Scripts\python.exe src/run_all.py
```

### 2. Launch Interactive Web Dashboard
To launch the Streamlit visual dashboard:

```powershell
..\.venv_xcpdp\Scripts\streamlit.exe run app.py
```
Open your browser at: **`http://localhost:8501`**

---

## 📁 Repository Structure

```text
explainable-cross-project-defect-prediction/
│
├── data/
│   ├── static/                Cleaned static code metric CSVs (NASA & AEEEM)
│   ├── repo_mined/            Repository mining CSV outputs
│   └── hybrid/                Fused static + process metric CSVs
│
├── src/                       Python Source Pipeline
│   ├── train_within_project.py  Phase 3: 10-Fold Stratified CV Training
│   ├── train_cross_project.py   Phase 4: 46 Cross-Project Evaluation Pairs
│   ├── statistical_tests.py    Phase 5: Wilcoxon, McNemar & Normality Tests
│   ├── explainability.py       Phase 6: SHAP TreeExplainer Extraction & Plots
│   ├── feature_stability.py    Phase 7: Jaccard, Spearman & Kendall Stability
│   ├── recommendation_engine.py Phase 8: Dynamic P75 Refactoring Recommendation Engine
│   ├── compile_results.py      Phase 9: Master Tables & 300 DPI Publication Figures
│   ├── domain_adaptation.py    Enhancement: Transfer Component Analysis (TCA)
│   └── run_all.py              Master Pipeline Runner
│
├── app.py                     Interactive Dark Glassmorphism Streamlit Dashboard
│
├── outputs/                   Generated Artifacts & Results
│   ├── within_project/        Phase 3 cross-validation metrics & per-fold predictions
│   ├── cross_project/         Phase 4 cross-project results & generalization gap plot
│   ├── stats_tests/           Phase 5 statistical significance summary CSV
│   ├── shap/                  Phase 6 SHAP summary/importance plots & feature ranking JSONs
│   ├── stability/             Phase 7 stability metrics CSV & 4 heatmaps
│   ├── recommendations/       Phase 8 traceable refactoring sample reports (JSON & MD)
│   ├── domain_adaptation/     TCA cross-project domain adaptation results & plot
│   ├── master_comparison_table.csv  Annotated Master Performance Table
│   ├── master_comparison_table.md   Formatted Markdown Master Table
│   ├── fig1_within_vs_cross_auc_mcc.png   300 DPI Figure 1
│   ├── fig2_static_vs_hybrid_comparison.png 300 DPI Figure 2
│   ├── fig3_shap_stability_heatmap.png     300 DPI Figure 3
│   ├── fig4_recommendation_case_study.png  300 DPI Figure 4
│   └── final_summary.txt      Written research answers for RQ1, RQ2, RQ3, and RQ4
│
├── requirements.txt
└── README.md
```

---

## 📊 Summary of Research Findings (Answering RQ1–RQ4)

- **RQ1 (Within-Project Performance)**:
  Tree ensemble classifiers significantly outperform baseline Logistic Regression ($p < 0.05$). **Random Forest** achieved the peak within-project AUC-ROC of **0.831** (hybrid) and **0.797** (static), closely followed by XGBoost (**0.823** / **0.790**) and LightGBM (**0.822** / **0.787**).

- **RQ2 (Feature Ranking Stability)**:
  SHAP feature importance ordering displays moderate overall cross-project stability (Spearman $r = 0.391, p < 0.01$; Kendall $\tau = 0.287, p < 0.01$). Static metrics (`numberOfLinesOfCode`, `wmc`, `rfc`, `fanOut`) are **consistently stable** (Top-5 in 28.6%–42.9% of datasets), whereas process metrics (`CvsEntropy`, `commit_count`) are dataset-specific.

- **RQ3 (Generalization Gap & Hybrid Metric Value)**:
  Zero-shot cross-project evaluation suffers a **~14%–16% AUC performance drop** due to domain shift across repositories. Repository-mined hybrid features provide a **+6.6% relative AUC boost** in within-project maintenance settings (0.831 vs 0.771 AUC on AEEEM).

- **RQ4 (Explainable Refactoring Traceability)**:
  By coupling dataset-specific 75th percentile ($P_{75}$) dynamic thresholds with SHAP feature attributions, the framework generates actionable refactoring recommendations with **100% feature-to-recommendation traceability**:
  $$\text{Prediction Risk Score} \longrightarrow \text{SHAP Feature Impact} \longrightarrow \text{Dynamic P75 Trigger} \longrightarrow \text{Refactoring Action}$$

---

## 🛠️ Environment Setup

Activate the project's Python virtual environment:

```powershell
..\.venv_xcpdp\Scripts\Activate.ps1
```

Verify dependencies:
```powershell
python -c "import pandas, numpy, sklearn, xgboost, lightgbm, shap, scipy, streamlit; print('All dependencies operational!')"
```
