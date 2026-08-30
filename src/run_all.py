"""Master runner script for Explainable Cross-Project Defect Prediction.

Executes all pipeline phases sequentially:
1. Within-Project Training (Phase 3)
2. Cross-Project Evaluation (Phase 4)
3. Statistical Significance Testing (Phase 5)
4. SHAP Explainability (Phase 6)
5. Feature Stability Analysis (Phase 7)
6. Refactoring Recommendation Engine (Phase 8)
7. Final Results Compilation (Phase 9)
"""

import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent

PHASE_SCRIPTS = [
    ("Phase 3: Within-Project Training", "train_within_project.py"),
    ("Phase 4: Cross-Project Evaluation", "train_cross_project.py"),
    ("Phase 5: Statistical Significance Testing", "statistical_tests.py"),
    ("Phase 6: SHAP Explainability", "explainability.py"),
    ("Phase 7: Feature Stability Analysis", "feature_stability.py"),
    ("Phase 8: Refactoring Recommendation Engine", "recommendation_engine.py"),
    ("Phase 9: Final Results Compilation", "compile_results.py"),
]


def run_phase(title: str, script_name: str) -> None:
    print("\n" + "=" * 80)
    print(f"RUNNING {title.upper()}")
    print("=" * 80, flush=True)

    script_path = SRC_DIR / script_name
    cmd = [sys.executable, str(script_path)]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] {title} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"[SUCCESS] Finished {title}")


def main() -> None:
    print("Starting Complete Pipeline Execution...")
    for title, script in PHASE_SCRIPTS:
        run_phase(title, script)

    print("\n" + "=" * 80)
    print("ALL PIPELINE PHASES COMPLETED SUCCESSFULLY!")
    print("Outputs available in explainable-cross-project-defect-prediction/outputs/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
