"""
Phase 14 — Testing
Covers: valid input, missing input, invalid input, boundary values,
multiple samples, and consistency (same input -> same output).
Run with: python3 -m pytest src/test_pipeline.py -v   (from project root)
"""
import pytest
from src.predict import predict_single, InputValidationError

VALID_SAMPLE = {
    "age": 45, "gender": 0, "bmi": 31.2, "alcohol_status": 1,
    "smoking_status": 0, "alt": 55, "ast": 48, "ggt": 90,
    "bilirubin_total": 1.1, "triglycerides": 190, "hdl_cholestrol": 42,
    "ldl_cholestrol": 130, "platelet_count": 220, "fasting_blood_sugar": 115,
    "tsh": 2.1, "diabetes_type_2": 1, "hypertension": 1, "pcos": 2
}


def test_valid_input_returns_expected_structure():
    result = predict_single(VALID_SAMPLE)
    assert result["prediction"] in ["None", "Mild", "Moderate", "Severe"]
    assert 0.0 <= result["probability"] <= 1.0
    assert "important_features" in result
    assert "disclaimer" in result
    assert len(result["class_probabilities"]) == 4


def test_missing_field_raises():
    incomplete = VALID_SAMPLE.copy()
    del incomplete["bmi"]
    with pytest.raises(InputValidationError):
        predict_single(incomplete)


def test_invalid_type_raises():
    bad = VALID_SAMPLE.copy()
    bad["bmi"] = "not_a_number"
    with pytest.raises(InputValidationError):
        predict_single(bad)


def test_out_of_range_raises():
    bad = VALID_SAMPLE.copy()
    bad["age"] = 500  # implausible
    with pytest.raises(InputValidationError):
        predict_single(bad)


def test_boundary_values_low():
    low = VALID_SAMPLE.copy()
    low.update({"age": 0, "bmi": 10, "alt": 0, "ast": 0, "ggt": 0,
                "bilirubin_total": 0, "triglycerides": 0, "hdl_cholestrol": 0,
                "ldl_cholestrol": 0, "platelet_count": 0,
                "fasting_blood_sugar": 0, "tsh": 0})
    result = predict_single(low)
    assert result["prediction"] in ["None", "Mild", "Moderate", "Severe"]


def test_boundary_values_high():
    high = VALID_SAMPLE.copy()
    high.update({"age": 120, "bmi": 70, "alt": 1000, "ast": 1000, "ggt": 1000,
                 "bilirubin_total": 30, "triglycerides": 2000,
                 "hdl_cholestrol": 200, "ldl_cholestrol": 400,
                 "platelet_count": 1000, "fasting_blood_sugar": 600, "tsh": 100})
    result = predict_single(high)
    assert result["prediction"] in ["None", "Mild", "Moderate", "Severe"]


def test_consistency_same_input_same_output():
    r1 = predict_single(VALID_SAMPLE)
    r2 = predict_single(VALID_SAMPLE)
    assert r1["prediction"] == r2["prediction"]
    assert r1["probability"] == r2["probability"]


def test_multiple_sample_cases():
    samples = [
        {**VALID_SAMPLE, "bmi": 20, "alt": 15, "ast": 12, "ggt": 15,
         "triglycerides": 80, "diabetes_type_2": 0, "hypertension": 0,
         "alcohol_status": 0},  # expect low risk profile
        {**VALID_SAMPLE, "bmi": 38, "alt": 110, "ast": 95, "ggt": 200,
         "triglycerides": 320, "diabetes_type_2": 1, "hypertension": 1},  # expect high risk profile
    ]
    predictions = [predict_single(s)["prediction"] for s in samples]
    # Not asserting exact class (that's the model's job) — just that both run
    # end-to-end and produce valid categories, and that the low-risk profile
    # is not predicted as Severe while the high-risk one is not predicted as None.
    assert predictions[0] != "Severe"
    assert predictions[1] != "None"


def test_improved_rf_and_multi_model_support():
    # Verify Notebook Improved Random Forest Pipeline
    r_rf = predict_single(VALID_SAMPLE, model_name="improved_rf")
    assert r_rf["model_info"]["model_key"] == "improved_rf"
    assert r_rf["model_info"]["is_notebook_architecture"] is True
    assert "binary_screening" in r_rf
    assert r_rf["binary_screening"]["is_positive"] is True

    # Verify XGBoost
    r_xgb = predict_single(VALID_SAMPLE, model_name="xgboost")
    assert r_xgb["model_info"]["model_key"] == "xgboost"

    # Verify Ensemble
    r_ens = predict_single(VALID_SAMPLE, model_name="ensemble")
    assert r_ens["model_info"]["model_key"] == "ensemble"
    assert 0.0 <= r_ens["probability"] <= 1.0
