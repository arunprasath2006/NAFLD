# AI-Based Prediction of Non-Alcoholic Fatty Liver Disease (NAFLD) Risk Using Clinical and Biochemical Parameters

**Final-Year Biotechnology Project — Machine Learning Prototype**

## 1. Project Objective

Predict a patient's fatty-liver disease severity grade (None / Mild / Moderate / Severe)
from routine clinical and biochemical parameters, using a trained ML model served through
a web interface. This is an **educational/research screening prototype**, not a diagnostic
tool.

## 2. Problem Statement

NAFLD is often asymptomatic and detected late. Routine blood work (liver enzymes, lipid
panel, metabolic markers) combined with anthropometric data (BMI) can carry a detectable
signal about disease severity even before imaging is ordered. This project explores
whether a classifier trained on such variables can flag risk level for further clinical
follow-up.

## 3. Dataset Description

- **Source:** Kaggle fatty-liver-disease dataset, 10,000 patient records, 21 raw columns.
- **Records:** 10,000, 0 duplicate rows, 0 duplicate IDs.
- **Missingness:** `pcos` (77.7%, structural — not applicable to male patients),
  `diabetes_type_2` (21.5%), `hypertension` (17.9%), `smoking_status` (16.7%),
  `alcohol_status` (3.0%). All numeric biochemical columns are complete.

## 4. Target Variable

`fatty_liver_grade` — an **ordinal 4-class severity label**:

| Grade | Meaning | Count | % |
|---|---|---|---|
| 0 | None | 5,922 | 59.2% |
| 1 | Mild | 2,280 | 22.8% |
| 2 | Moderate | 1,113 | 11.1% |
| 3 | Severe | 685 | 6.9% |

Verified (not assumed) by checking that mean BMI, ALT, AST, GGT, and triglycerides rise
monotonically across grades 0→3 while HDL and platelet count fall — consistent with known
NAFLD pathophysiology.

## 5. Features Used (18)

`age, gender, bmi, alcohol_status, smoking_status, alt, ast, ggt, bilirubin_total,
triglycerides, hdl_cholestrol, ldl_cholestrol, platelet_count, fasting_blood_sugar, tsh,
diabetes_type_2, hypertension, pcos`

**Excluded:** `id` (identifier, no signal) and **`liver_cirrhosis`** — excluded as a
**data-leakage risk**. Cirrhosis prevalence is 2.5% overall but jumps to 28.7% within
grade-3 patients; it is a downstream consequence of advanced disease, not an independent
pre-diagnostic input, so including it would let the model partially "see" the answer.

## 6. Data Preprocessing (see `src/data_preprocessing.py`)

- No rows deleted (no duplicates found; no records dropped for missingness).
- `pcos`: recoded to 3 explicit categories {No, Yes, Not Applicable} instead of imputing
  a value for patients (males) for whom the field structurally does not apply.
- `alcohol_status`, `smoking_status`, `diabetes_type_2`, `hypertension`: genuine missing
  values imputed with the column mode (documented as a simple, explainable choice
  appropriate for a screening prototype).
- Enzyme/lipid outliers (right-skewed, high maxima) were **not removed** — they are
  clinically plausible in a disease-severity population and correspond to the very
  patients (grade 2–3) the model most needs to learn to detect.
- `gender` label-encoded (0=Male, 1=Female).

## 7. Exploratory Data Analysis

See `reports/figures/`:
- `01_target_distribution.png` — class imbalance (see table above)
- `02_variable_distributions.png` — age, BMI, glucose, ALT/AST/GGT, triglycerides
- `03_correlation_heatmap.png` — numeric feature correlations
- `04_feature_vs_target.png` — boxplots of top features by grade
- `05_confusion_matrices.png` — all 3 models
- `06_feature_importance.png` — final model's feature importances

Strongest linear correlations with `fatty_liver_grade`: BMI (r=0.78), triglycerides
(r=0.54), AST (r=0.53), ALT (r=0.51), GGT (r=0.41).

## 8. Machine Learning Models

Stratified 80/20 train/test split (seed=42), class-weighted training to address
imbalance (Logistic Regression / Random Forest via `class_weight="balanced"`, XGBoost via
computed sample weights). Test set was never used for model selection or tuning.

- **Logistic Regression** (baseline, features standardized)
- **Random Forest** (300 trees)
- **XGBoost** (300 trees, max_depth=5)

## 9. Evaluation Metrics (actual test-set results)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (macro, OVR) |
|:---|---:|---:|---:|---:|---:|
| LogisticRegression | 0.7805 | 0.7277 | 0.7638 | 0.7431 | 0.9329 |
| RandomForest | 0.7865 | 0.7272 | 0.6931 | 0.7057 | 0.9336 |
| **XGBoost** | **0.7940** | **0.7472** | **0.7600** | **0.7528** | 0.9312 |

*(Values are from a single run with a fixed seed; see `reports/evaluation_results.json`
and `reports/model_comparison.csv` for the exact source of every number above — nothing
here is estimated.)*

## 10. Model Comparison & Final Model Selection

**Selected: XGBoost.** Selection was based on **macro F1** (best of the three, 0.7528),
with macro recall as a tie-breaker — prioritized because in a screening context missing
a true positive (a patient who does have significant fatty liver disease) is more costly
than a false alarm. XGBoost also had the highest accuracy and highest macro precision.
Random Forest had the highest ROC-AUC by a negligible margin (0.9336 vs 0.9312) but its
macro recall was noticeably lower (0.693) — it was under-detecting the Mild class
specifically — so it was not selected. Logistic Regression is retained as the
interpretable baseline for comparison. **Random Forest was not assumed to win — it did
not, matching the project's Rule 3.**

Per-class performance of the final model (XGBoost) on the held-out test set:

| Grade | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| None | 0.907 | 0.861 | 0.883 | 1,184 |
| Mild | 0.596 | 0.669 | 0.630 | 456 |
| Moderate | 0.664 | 0.664 | 0.664 | 223 |
| Severe | 0.823 | 0.847 | 0.835 | 137 |

The model detects Severe cases with 84.7% recall — the class most important not to miss
— though Mild/Moderate (the harder-to-separate middle grades) show more confusion with
each other, visible in `05_confusion_matrices.png`.

## 11. Explainability

Feature importance (gain-based) for the final XGBoost model — top contributors: BMI
(33.7%), alcohol status, triglycerides, AST, GGT, ALT, hypertension, diabetes type 2.
Consistent with the EDA correlations. Displayed on the results page and served via the
API's `important_features` field. **Framed only as statistical influence on the model's
prediction, never as proven biological causation**, per the project's Rule 9.

## 12. Backend

FastAPI service (`backend/main.py`) exposing:

- `GET /health` — liveness check
- `POST /predict` — accepts the 18 patient parameters (Pydantic-validated, range-checked),
  runs the saved inference pipeline (`src/predict.py`), and returns prediction, per-class
  probabilities, top influential features, and the mandatory disclaimer.

Run it:
```bash
cd nafld-ai-project
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```
Interactive API docs: `http://127.0.0.1:8000/docs`

## 13. Frontend

Single-page app at `frontend/index.html` (plain HTML/CSS/JS, no build step). Three views:
Home → Patient Input form (only the 18 features the model actually uses) → Result (grade,
gauge, per-class probabilities, top influential features, disclaimer).

To run: open `frontend/index.html` in a browser while the backend is running on
`http://127.0.0.1:8000` (default `API_BASE` in the file — edit
`window.NAFLD_API_BASE` before the script tag to point elsewhere, e.g. for deployment).

## 14. How to Run the Full Project

```bash
cd nafld-ai-project
pip install -r requirements.txt

# 1. Clean data + retrain from scratch (optional — models/ already contains trained artifacts)
python3 src/data_preprocessing.py
python3 src/eda.py
python3 src/train.py
python3 src/interpret.py

# 2. Run tests
python3 -m pytest src/test_pipeline.py -v

# 3. Start backend
uvicorn backend.main:app --reload --port 8000

# 4. Open frontend/index.html in a browser
```

## 15. API Usage Example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45, "gender": 0, "bmi": 31.2, "alcohol_status": 1,
    "smoking_status": 0, "alt": 55, "ast": 48, "ggt": 90,
    "bilirubin_total": 1.1, "triglycerides": 190, "hdl_cholestrol": 42,
    "ldl_cholestrol": 130, "platelet_count": 220, "fasting_blood_sugar": 115,
    "tsh": 2.1, "diabetes_type_2": 1, "hypertension": 1, "pcos": 2
  }'
```

## 16. Architecture

```
┌─────────────┐     patient params (JSON)     ┌──────────────────┐
│  Frontend    │ ─────────────────────────────▶ │  FastAPI Backend │
│ (HTML/CSS/JS)│ ◀───────────────────────────── │   /predict       │
└─────────────┘     prediction + explainability └────────┬──────────┘
                                                            │
                                                            ▼
                                       ┌────────────────────────────┐
                                       │ src/predict.py              │
                                       │  - validate input            │
                                       │  - apply saved feature order │
                                       │  - load final_model.pkl      │
                                       │  - (scaler.pkl if needed)    │
                                       └────────────────────────────┘
```

Training pipeline (offline, run once): `data_preprocessing.py → eda.py → train.py →
interpret.py`, producing everything under `models/` and `reports/` that the backend loads.

## 17. Project Structure

```
nafld-ai-project/
├── data/
│   ├── fatty_liver_dataset.csv       # raw
│   └── fatty_liver_cleaned.csv       # after Phase 2 cleaning
├── src/
│   ├── data_preprocessing.py         # Phase 2
│   ├── eda.py                        # Phase 4
│   ├── train.py                      # Phase 5 & 6
│   ├── interpret.py                  # Phase 8
│   ├── predict.py                    # Phase 9 inference pipeline
│   └── test_pipeline.py              # Phase 14
├── models/
│   ├── final_model.pkl               # XGBoost (selected)
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   ├── feature_names.json
│   └── feature_importance.json
├── reports/
│   ├── figures/                      # Phase 4 + 6 + 8 plots
│   ├── model_comparison.csv
│   └── evaluation_results.json
├── backend/
│   └── main.py                       # Phase 10 FastAPI app
├── frontend/
│   └── index.html                    # Phase 11-12 UI
├── requirements.txt
└── README.md
```

## 18. Limitations

- Trained on a single dataset; no external validation cohort.
- Mild/Moderate grades are harder to separate than None/Severe (see confusion matrix) —
  the model is more reliable at the extremes than at the middle grades.
- Comorbidity fields (diabetes, hypertension) had real missingness imputed with mode,
  which may bias those specific flags toward the majority class.
- No imaging or histology ground truth used — target is a grade label present in the
  dataset, not a biopsy-confirmed diagnosis.

## 19. Future Improvements

- External/multi-site validation.
- SHAP-based per-patient explanations (SHAP is installed and available — the current
  build uses global gain-based importance for simplicity and viva-explainability).
- Hyperparameter tuning via cross-validated grid/random search.
- Ordinal-aware modeling (e.g., ordinal regression) given the grade is inherently ordered.

## 20. Medical Disclaimer

> This application is an AI-based research/screening prototype and is **not** intended
> to diagnose, treat, or replace evaluation by a qualified healthcare professional.
> All predictions should be interpreted as a possible indicator for further clinical
> investigation, not as a medical diagnosis.
#   N A F L D  
 