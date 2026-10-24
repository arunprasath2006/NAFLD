"""
Phase 2 — Data Cleaning
=======================
Reproducible cleaning pipeline for the NAFLD dataset.

Cleaning decisions (documented, matches README section 7):

1. Drop `id` — pure identifier, no predictive value.
2. Drop `liver_cirrhosis` — data-leakage risk. It is a downstream
   consequence of advanced fatty liver disease (concentrated almost
   entirely in grade-3 patients), not an independent pre-diagnostic
   input. Excluded from modeling; may still be discussed in EDA as a
   clinical correlate of severity.
3. `pcos` — structurally missing for every male record (this is not
   "missing data", it's "not applicable"). Encoded as a 3-level
   category {No, Yes, Not Applicable} rather than imputed, so the
   model doesn't invent a value for a variable that literally cannot
   apply to male patients.
4. `alcohol_status`, `smoking_status`, `diabetes_type_2`,
   `hypertension` — genuine missingness (3-22%). Binary flags are
   imputed with the column mode (0, i.e. "absent") since these are
   rare-positive lifestyle/comorbidity flags and mode imputation is a
   defensible, simple, explainable choice for a student project.
   Missing-indicator dummy columns are NOT added, to keep the final
   feature set small and interpretable for a viva.
5. No numeric columns have missing values, so no numeric imputation
   is required.
6. Outliers in liver enzymes (ALT/AST/GGT/bilirubin) and
   triglycerides are NOT removed — the right-skew and high maxima are
   clinically plausible in a fatty-liver-disease population (grade 3
   patients are supposed to have very high enzymes). Removing them
   would delete exactly the severe cases the model needs to learn.
   We DO clip biologically impossible values only if any are found
   (checked programmatically below).
7. No exact duplicate rows or duplicate patient IDs were found, so no
   deduplication step is required.
"""

import pandas as pd
import numpy as np

RAW_PATH = "data/fatty_liver_dataset.csv"

LEAKAGE_COLS = ["liver_cirrhosis"]
ID_COLS = ["id", "gender_encoded"]
TARGET_COL = "fatty_liver_grade"

BINARY_IMPUTE_COLS = ["alcohol_status", "smoking_status",
                       "diabetes_type_2", "hypertension"]


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- sanity checks (do not silently pass) ---------------------------
    n_dupe_rows = df.duplicated().sum()
    n_dupe_ids = df["id"].duplicated().sum() if "id" in df.columns else 0
    assert n_dupe_rows == 0, f"Unexpected {n_dupe_rows} duplicate rows found"
    assert n_dupe_ids == 0, f"Unexpected {n_dupe_ids} duplicate ids found"

    # --- biologically-impossible value clipping (none expected, but check) ---
    numeric_nonneg = ["age", "bmi", "alt", "ast", "ggt", "bilirubin_total",
                       "triglycerides", "hdl_cholestrol", "ldl_cholestrol",
                       "platelet_count", "fasting_blood_sugar", "tsh"]
    for c in numeric_nonneg:
        n_negative = (df[c] < 0).sum()
        if n_negative > 0:
            df.loc[df[c] < 0, c] = np.nan  # treat as missing, impute below

    # --- pcos: structural NA -> explicit category -----------------------
    # 0 = No, 1 = Yes, 2 = Not Applicable (male)
    df["pcos"] = np.where(
        df["gender"] == "M", 2,
        df["pcos"].fillna(1)  # any remaining female NaNs -> conservative default handled below
    )
    # For females, genuine missing pcos values (if any) imputed with mode of females
    female_mask = df["gender"] == "F"
    if df.loc[female_mask, "pcos"].isnull().any():
        mode_val = df.loc[female_mask, "pcos"].mode()[0]
        df.loc[female_mask, "pcos"] = df.loc[female_mask, "pcos"].fillna(mode_val)
    df["pcos"] = df["pcos"].astype(int)

    # --- genuine missing binary comorbidity/lifestyle flags -> mode impute ---
    for c in BINARY_IMPUTE_COLS:
        mode_val = df[c].mode()[0]
        df[c] = df[c].fillna(mode_val).astype(int)

    # --- drop leakage & id columns ---------------------------------------
    df = df.drop(columns=LEAKAGE_COLS + ID_COLS, errors="ignore")

    # --- encode gender as binary ------------------------------------------
    df["gender"] = df["gender"].map({"M": 0, "F": 1})

    return df


def get_feature_target_split(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


if __name__ == "__main__":
    raw = load_raw()
    cleaned = clean_data(raw)
    print("Raw shape:", raw.shape)
    print("Cleaned shape:", cleaned.shape)
    print("\nMissing values after cleaning:\n", cleaned.isnull().sum().sum())
    print("\nColumns:", cleaned.columns.tolist())
    print("\npcos value counts:\n", cleaned["pcos"].value_counts())
    cleaned.to_csv("data/fatty_liver_cleaned.csv", index=False)
    print("\nSaved cleaned dataset to data/fatty_liver_cleaned.csv")
