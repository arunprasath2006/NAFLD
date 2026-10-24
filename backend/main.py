"""
Phase 10 — Backend API (FastAPI)
Includes NAFLD model inference, Medical Report Analyzer (file upload & OCR extraction),
PDF clinical report generator, and Assessment History.
"""
import sys
import uuid
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.predict import predict_single, InputValidationError
from backend.report_extractor import extract_raw_text, parse_clinical_parameters
from backend.pdf_generator import generate_assessment_pdf

app = FastAPI(
    title="NAFLD Risk Prediction API",
    description="Research/screening prototype for AI-based NAFLD risk prediction. "
                "Not a medical diagnostic tool.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# In-memory storage for assessment history
ASSESSMENT_HISTORY = []

# Upload file constraints
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB


class PatientInput(BaseModel):
    model_config = {"protected_namespaces": ()}

    age: float = Field(..., ge=0, le=120, description="Age in years")
    gender: int = Field(..., ge=0, le=1, description="0 = Male, 1 = Female")
    bmi: float = Field(..., ge=10, le=70, description="Body Mass Index (kg/m^2)")
    alcohol_status: int = Field(..., ge=0, le=1, description="0 = No, 1 = Yes")
    smoking_status: int = Field(..., ge=0, le=1, description="0 = No, 1 = Yes")
    alt: float = Field(..., ge=0, le=1000, description="ALT (U/L)")
    ast: float = Field(..., ge=0, le=1000, description="AST (U/L)")
    ggt: float = Field(..., ge=0, le=1000, description="GGT (U/L)")
    bilirubin_total: float = Field(..., ge=0, le=30, description="Total bilirubin (mg/dL)")
    triglycerides: float = Field(..., ge=0, le=2000, description="Triglycerides (mg/dL)")
    hdl_cholestrol: float = Field(..., ge=0, le=200, description="HDL cholesterol (mg/dL)")
    ldl_cholestrol: float = Field(..., ge=0, le=400, description="LDL cholesterol (mg/dL)")
    platelet_count: float = Field(..., ge=0, le=1000, description="Platelet count (x10^3/uL)")
    fasting_blood_sugar: float = Field(..., ge=0, le=600, description="Fasting blood sugar (mg/dL)")
    tsh: float = Field(..., ge=0, le=100, description="TSH (uIU/mL)")
    diabetes_type_2: int = Field(..., ge=0, le=1, description="0 = No, 1 = Yes")
    hypertension: int = Field(..., ge=0, le=1, description="0 = No, 1 = Yes")
    pcos: int = Field(..., ge=0, le=2, description="0 = No, 1 = Yes, 2 = Not Applicable (male)")
    model_type: Optional[str] = Field("improved_rf", description="Model architecture: improved_rf, xgboost, random_forest, ensemble")


class GeneratePdfRequest(BaseModel):
    assessment_id: Optional[str] = None
    patient_inputs: Dict[str, Any]
    prediction_result: Dict[str, Any]


@app.get("/")
def root():
    return FileResponse(frontend_dir / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/download-verification-report")
def download_verification_report():
    """Download the 50-patient clinical verification comparison PDF."""
    report_path = Path(__file__).resolve().parent.parent / "reports" / "NAFLD_50_Patients_Verification_Report.pdf"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Verification report not found")
    return FileResponse(
        str(report_path),
        media_type="application/pdf",
        filename="NAFLD_50_Patients_Verification_Report.pdf"
    )


@app.get("/api/models")
def get_available_models():
    """Return available model architectures and their benchmarks."""
    return {
        "models": [
            {
                "id": "improved_rf",
                "name": "Improved Random Forest Pipeline (Notebook Model)",
                "description": "SimpleImputer + StandardScaler + RandomForestClassifier (250 trees) from Kaggle notebook",
                "accuracy": "78.95% (4-Class Grade), 86.55% (Binary NAFLD Detection, 93.5% AUC, 92.5% Specificity)",
                "is_default": True
            },
            {
                "id": "xgboost",
                "name": "XGBoost Multi-Class Classifier",
                "description": "Gradient Boosted Decision Trees (300 estimators, max depth 5)",
                "accuracy": "79.30% (4-Class Grade), 93.12% Macro ROC-AUC",
                "is_default": False
            },
            {
                "id": "ensemble",
                "name": "Consensus Ensemble (RF + XGBoost)",
                "description": "Soft-voting blended probabilities between Improved RF and XGBoost",
                "accuracy": "79.25% (4-Class Grade)",
                "is_default": False
            }
        ]
    }


@app.post("/predict")
def predict(payload: PatientInput):
    try:
        data = payload.model_dump()
        model_type = data.pop("model_type", "improved_rf")
        result = predict_single(data, model_name=model_type)
        return result
    except InputValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/api/extract-report")
async def extract_report(files: list[UploadFile] = File(...)):
    """Upload multiple laboratory reports (PDFs and/or images) and merge extracted clinical parameters."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    combined_text = ""
    processed_files = []
    methods_used = set()

    for file in files:
        filename = file.filename or "medical_report"
        ext = Path(filename).suffix.lower()

        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{ext}' in file '{filename}'. Supported: PDF, JPG, JPEG, PNG, WEBP."
            )

        file_bytes = await file.read()
        if len(file_bytes) == 0:
            continue
        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File '{filename}' exceeds 15MB limit.")

        raw_text, method = extract_raw_text(file_bytes, filename)
        if raw_text:
            combined_text += f"\n--- DOCUMENT: {filename} ---\n" + raw_text + "\n"
        methods_used.add(method)
        processed_files.append({
            "name": filename,
            "size": len(file_bytes),
            "method": method
        })

    extracted_parameters = parse_clinical_parameters(combined_text)

    # Compute statistics
    total_fields = len(extracted_parameters)
    extracted_count = sum(1 for p in extracted_parameters.values() if p["status"] == "extracted")
    missing_count = total_fields - extracted_count

    return {
        "success": True,
        "files_count": len(processed_files),
        "files": processed_files,
        "extraction_method": ", ".join(methods_used) if methods_used else "none",
        "total_fields": total_fields,
        "extracted_count": extracted_count,
        "missing_count": missing_count,
        "raw_text_length": len(combined_text),
        "parameters": extracted_parameters
    }


@app.post("/api/generate-pdf")
def create_pdf_report(req: GeneratePdfRequest):
    """Generate and return a downloadable PDF clinical summary."""
    assessment_id = req.assessment_id or f"NAFLD-{uuid.uuid4().hex[:8].upper()}"
    try:
        pdf_bytes = generate_assessment_pdf(
            assessment_id=assessment_id,
            patient_inputs=req.patient_inputs,
            prediction_result=req.prediction_result
        )

        # Save to assessment history
        pred_label = req.prediction_result.get("prediction", "Unknown")
        pred_prob = req.prediction_result.get("probability", 0.0)
        ASSESSMENT_HISTORY.insert(0, {
            "id": assessment_id,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prediction": pred_label,
            "probability": pred_prob,
            "patient_inputs": req.patient_inputs,
            "prediction_result": req.prediction_result
        })
        if len(ASSESSMENT_HISTORY) > 50:
            ASSESSMENT_HISTORY.pop()

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=NAFLD_Assessment_{assessment_id}.pdf"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF report: {e}")


@app.get("/api/assessments/history")
def get_assessment_history():
    """Retrieve saved clinical assessments."""
    return {"history": ASSESSMENT_HISTORY}


@app.delete("/api/assessments/history/{assessment_id}")
def delete_assessment(assessment_id: str):
    """Delete a specific assessment from history."""
    global ASSESSMENT_HISTORY
    ASSESSMENT_HISTORY = [a for a in ASSESSMENT_HISTORY if a["id"] != assessment_id]
    return {"success": True, "message": f"Assessment {assessment_id} removed"}
