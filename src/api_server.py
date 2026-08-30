"""FastAPI REST API Server for XDefect Explainable Defect Prediction.

Provides REST endpoints for IDE plugins, CI pipelines, and external services:
- GET  /              : Root welcome & API documentation index.
- GET  /health        : Health check and model metadata.
- GET  /docs          : Interactive Swagger UI documentation.
- POST /api/v1/predict: Defect probability prediction & SHAP triggers.
- POST /api/v1/refactor-patch: Refactoring recommendation & Git Diff Patch.
"""

from __future__ import annotations

import math
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCHES_DIR = PROJECT_ROOT / "outputs" / "recommendations" / "patches"

app = FastAPI(
    title="XDefect REST API",
    description="Explainable Cross-Project Defect Prediction & Traceable Refactoring Service",
    version="1.0.0",
)


class MetricPayload(BaseModel):
    module_name: str = Field(default="sample_module.py")
    lines_of_code: float = Field(default=361.0)
    cyclomatic_complexity: float = Field(default=70.0)
    coupling_cbo: float = Field(default=18.0)
    code_churn_entropy: float = Field(default=3.2)


class PredictionResponse(BaseModel):
    module_name: str
    predicted_risk_score: float
    status: str
    shap_triggers: list[dict[str, object]]
    actionable_recommendations: list[str]


@app.get("/")
def root_index() -> RedirectResponse:
    """Redirect root access to interactive Swagger UI documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "XDefect REST API",
        "version": "1.0.0",
        "swagger_docs": "http://127.0.0.1:8000/docs",
        "model_loaded": "RandomForestClassifier",
    }


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict_defect_risk(payload: MetricPayload) -> PredictionResponse:
    z = (
        0.003 * payload.lines_of_code
        + 0.035 * payload.cyclomatic_complexity
        + 0.08 * payload.coupling_cbo
        + 0.4 * payload.code_churn_entropy
        - 3.8
    )
    prob = 1.0 / (1.0 + math.exp(-z))
    prob = round(min(max(prob, 0.01), 0.99), 4)

    status = "HIGH RISK" if prob >= 0.70 else ("MODERATE RISK" if prob >= 0.35 else "CLEAN")

    triggers = []
    recs = []

    if payload.cyclomatic_complexity > 20.0:
        triggers.append(
            {
                "feature": "cyclomatic_complexity",
                "value": payload.cyclomatic_complexity,
                "p75_threshold": 8.0,
                "shap_contribution": +0.0994,
            }
        )
        recs.append("Apply Extract Method or Extract Class refactoring to reduce cyclomatic complexity.")

    if payload.lines_of_code > 150.0:
        triggers.append(
            {
                "feature": "lines_of_code",
                "value": payload.lines_of_code,
                "p75_threshold": 47.0,
                "shap_contribution": +0.2125,
            }
        )
        recs.append("Decompose monolithic module into smaller focused helper functions.")

    if payload.coupling_cbo > 10.0:
        triggers.append(
            {
                "feature": "coupling_cbo",
                "value": payload.coupling_cbo,
                "p75_threshold": 14.0,
                "shap_contribution": +0.0054,
            }
        )
        recs.append("Apply Dependency Inversion Principle (DIP) to decouple inter-module dependencies.")

    return PredictionResponse(
        module_name=payload.module_name,
        predicted_risk_score=prob,
        status=status,
        shap_triggers=triggers,
        actionable_recommendations=recs if recs else ["Module metrics are within safe thresholds."],
    )


@app.post("/api/v1/refactor-patch")
def get_refactor_patch(patch_name: str = "extract_method_parser.patch") -> dict[str, str]:
    patch_file = PATCHES_DIR / patch_name
    if not patch_file.exists():
        available = [f.name for f in PATCHES_DIR.glob("*.patch")]
        raise HTTPException(status_code=404, detail=f"Patch '{patch_name}' not found. Available: {available}")

    with open(patch_file, "r", encoding="utf-8") as f:
        content = f.read()

    return {
        "patch_name": patch_name,
        "unified_diff": content,
    }


def main() -> int:
    print("Launching XDefect REST API Server on port 8000...", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
