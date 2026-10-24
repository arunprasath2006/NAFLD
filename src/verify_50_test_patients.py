"""
Comprehensive 50-Patient Batch Test, Verification & Comparison Script
Evaluates the provided 50 test patients against reference labels
and generates detailed verification metrics and a clinical comparison PDF.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.predict import predict_single


def run_batch_verification():
    csv_path = BASE_DIR / "data" / "test_50_patients.csv"
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} test patient records from {csv_path.name}")

    results = []

    for idx, row in df.iterrows():
        # Clean gender and PCOS
        g_str = str(row["gender"]).strip().upper()
        gender_code = 1 if g_str == "F" else 0
        pcos_val = int(row["pcos"])
        # In our model training, male pcos is coded as 2 (Not Applicable)
        pcos_input = 2 if (gender_code == 0 and pcos_val == 0) else pcos_val

        patient_input = {
            "age": float(row["age"]),
            "gender": gender_code,
            "bmi": float(row["bmi"]),
            "alcohol_status": int(row["alcohol_status"]),
            "smoking_status": int(row["smoking_status"]),
            "alt": float(row["alt"]),
            "ast": float(row["ast"]),
            "ggt": float(row["ggt"]),
            "bilirubin_total": float(row["bilirubin_total"]),
            "triglycerides": float(row["triglycerides"]),
            "hdl_cholestrol": float(row["hdl_cholestrol"]),
            "ldl_cholestrol": float(row["ldl_cholestrol"]),
            "platelet_count": float(row["platelet_count"]),
            "fasting_blood_sugar": float(row["fasting_blood_sugar"]),
            "tsh": float(row["tsh"]),
            "diabetes_type_2": int(row["diabetes_type_2"]),
            "hypertension": int(row["hypertension"]),
            "pcos": pcos_input
        }

        # Run Notebook Improved RF
        res_rf = predict_single(patient_input, model_name="improved_rf")
        # Run XGBoost
        res_xgb = predict_single(patient_input, model_name="xgboost")
        # Run Ensemble
        res_ens = predict_single(patient_input, model_name="ensemble")

        ref_grade = int(row["Reference_Risk_Grade_Number"])
        ref_type = str(row["Reference_Risk_Grade_Type"]).strip()

        grade_names = {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe"}

        pred_grade_rf = res_rf["prediction_class"]
        pred_label_rf = res_rf["prediction"]
        prob_rf = res_rf["probability"]

        pred_grade_xgb = res_xgb["prediction_class"]
        pred_label_xgb = res_xgb["prediction"]
        prob_xgb = res_xgb["probability"]

        pred_grade_ens = res_ens["prediction_class"]

        is_match_rf = (pred_grade_rf == ref_grade)
        is_match_xgb = (pred_grade_xgb == ref_grade)
        is_match_ens = (pred_grade_ens == ref_grade)

        # Binary check (Healthy vs NAFLD)
        ref_binary = 0 if ref_grade == 0 else 1
        pred_binary_rf = 0 if pred_grade_rf == 0 else 1
        binary_match_rf = (ref_binary == pred_binary_rf)

        # Distance error (exact, adjacent by 1 grade, or major)
        distance = abs(pred_grade_rf - ref_grade)

        results.append({
            "Test_ID": row["Test_ID"],
            "Patient_Name": row["Patient_Name"],
            "Age": int(row["age"]),
            "Gender": "F" if gender_code == 1 else "M",
            "BMI": float(row["bmi"]),
            "ALT": float(row["alt"]),
            "AST": float(row["ast"]),
            "GGT": float(row["ggt"]),
            "Triglycerides": float(row["triglycerides"]),
            "Ref_Grade": ref_grade,
            "Ref_Type": ref_type,
            "Pred_Grade_RF": pred_grade_rf,
            "Pred_Label_RF": pred_label_rf,
            "Prob_RF": prob_rf,
            "Match_RF": is_match_rf,
            "Pred_Grade_XGB": pred_grade_xgb,
            "Pred_Label_XGB": pred_label_xgb,
            "Match_XGB": is_match_xgb,
            "Pred_Grade_ENS": pred_grade_ens,
            "Match_ENS": is_match_ens,
            "Binary_Match": binary_match_rf,
            "Grade_Distance": distance,
            "Binary_Screening_Status": res_rf.get("binary_screening", {}).get("status", ""),
            "Binary_Confidence": res_rf.get("binary_screening", {}).get("confidence", 0.0)
        })

    res_df = pd.DataFrame(results)

    n_total = len(res_df)
    n_match_rf = res_df["Match_RF"].sum()
    n_match_xgb = res_df["Match_XGB"].sum()
    n_match_ens = res_df["Match_ENS"].sum()
    n_binary_match = res_df["Binary_Match"].sum()

    acc_rf = (n_match_rf / n_total) * 100
    acc_xgb = (n_match_xgb / n_total) * 100
    acc_ens = (n_match_ens / n_total) * 100
    acc_binary = (n_binary_match / n_total) * 100

    n_exact = (res_df["Grade_Distance"] == 0).sum()
    n_adjacent = (res_df["Grade_Distance"] == 1).sum()
    n_distant = (res_df["Grade_Distance"] > 1).sum()

    print("\n" + "=" * 60)
    print("50-PATIENT VERIFICATION BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total Test Samples Evaluated:            {n_total}")
    print(f"Improved Random Forest (Notebook Model): {acc_rf:.1f}% ({n_match_rf}/{n_total} exact matches)")
    print(f"XGBoost Multi-Class Classifier:          {acc_xgb:.1f}% ({n_match_xgb}/{n_total} exact matches)")
    print(f"Consensus Ensemble (RF + XGBoost):       {acc_ens:.1f}% ({n_match_ens}/{n_total} exact matches)")
    print(f"Binary NAFLD Screening (Healthy vs Dis): {acc_binary:.1f}% ({n_binary_match}/{n_total} correct)")
    print("-" * 60)
    print(f"Exact Matches (Diff = 0):                {n_exact} ({n_exact/n_total*100:.1f}%)")
    print(f"Adjacent Grade (Diff = 1 stage):         {n_adjacent} ({n_adjacent/n_total*100:.1f}%)")
    print(f"Clinical Tolerance (Within <= 1 Grade):  {(n_exact + n_adjacent)} ({(n_exact + n_adjacent)/n_total*100:.1f}%)")
    print(f"Major Discrepancies (Diff > 1):          {n_distant} ({n_distant/n_total*100:.1f}%)")
    print("=" * 60)

    # Save verification results table to CSV
    res_df.to_csv(BASE_DIR / "data" / "verification_50_results.csv", index=False)
    print(f"Saved results dataframe to data/verification_50_results.csv")

    return res_df


if __name__ == "__main__":
    run_batch_verification()
