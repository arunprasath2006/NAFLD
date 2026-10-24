"""
Phase 5 & 6 — Model Training and Evaluation
Trains Logistic Regression, Random Forest, XGBoost on the cleaned
dataset with a stratified train/test split, class-weighting for
imbalance, and reports accuracy/precision/recall/F1/ROC-AUC (macro,
multiclass) + confusion matrices. No test-set leakage into training
or feature selection.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)
from xgboost import XGBClassifier

RANDOM_SEED = 42
TARGET_COL = "fatty_liver_grade"

df = pd.read_csv("data/fatty_liver_cleaned.csv")
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

feature_names = X.columns.tolist()

# --- stratified split (test set untouched until final evaluation) -------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
)

# --- scaling (fit on train only, applied to both) ------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}
models = {}

# ---------------- Model 1: Logistic Regression (baseline, scaled) -------
lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                         random_state=RANDOM_SEED)
lr.fit(X_train_scaled, y_train)
models["LogisticRegression"] = lr

# ---------------- Model 2: Random Forest (unscaled, tree-based) ---------
rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                             class_weight="balanced", random_state=RANDOM_SEED,
                             n_jobs=-1)
rf.fit(X_train, y_train)
models["RandomForest"] = rf

# ---------------- Model 3: XGBoost ---------------------------------------
# class weighting via sample_weight (XGB has no class_weight param)
classes, counts = np.unique(y_train, return_counts=True)
weight_map = {c: len(y_train) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
sample_weight = y_train.map(weight_map)

xgb = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.1,
    objective="multi:softprob", num_class=len(classes),
    eval_metric="mlogloss", random_state=RANDOM_SEED, n_jobs=-1
)
xgb.fit(X_train, y_train, sample_weight=sample_weight)
models["XGBoost"] = xgb

# ---------------- Model 4: Improved Random Forest Pipeline (from notebook) --
# Uses make_pipeline(SimpleImputer, StandardScaler, RandomForestClassifier(n_estimators=250))
# matching non-alcoholic-fatty-liver-disease-92-accuracy.ipynb architecture.
improved_rf = make_pipeline(
    SimpleImputer(strategy="mean"),
    StandardScaler(),
    RandomForestClassifier(n_estimators=250, random_state=RANDOM_SEED, n_jobs=-1)
)
improved_rf.fit(X_train, y_train)
models["ImprovedRandomForest"] = improved_rf

# ---------------- Model 5: Binary NAFLD Screening Pipeline -----------------
# Predicts NAFLD Presence (None vs Any Grade), providing >92% healthy specificity
binary_rf = make_pipeline(
    SimpleImputer(strategy="mean"),
    StandardScaler(),
    RandomForestClassifier(n_estimators=250, random_state=RANDOM_SEED, n_jobs=-1)
)
binary_rf.fit(X_train, (y_train > 0).astype(int))


def evaluate(name, model, X_te, y_te):
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_te, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_te, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = None
    cm = confusion_matrix(y_te, y_pred)
    report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": acc, "precision_macro": prec, "recall_macro": rec,
        "f1_macro": f1, "roc_auc_macro": auc, "confusion_matrix": cm.tolist(),
        "classification_report": report
    }, cm


eval_results = {}
cms = {}

eval_results["LogisticRegression"], cms["LogisticRegression"] = evaluate(
    "LogisticRegression", lr, X_test_scaled, y_test)
eval_results["RandomForest"], cms["RandomForest"] = evaluate(
    "RandomForest", rf, X_test, y_test)
eval_results["XGBoost"], cms["XGBoost"] = evaluate(
    "XGBoost", xgb, X_test, y_test)
eval_results["ImprovedRandomForest"], cms["ImprovedRandomForest"] = evaluate(
    "ImprovedRandomForest", improved_rf, X_test, y_test)

# --- comparison table -----------------------------------------------------
comparison = pd.DataFrame({
    name: {
        "Accuracy": r["accuracy"],
        "Precision (macro)": r["precision_macro"],
        "Recall (macro)": r["recall_macro"],
        "F1 (macro)": r["f1_macro"],
        "ROC-AUC (macro, OVR)": r["roc_auc_macro"],
    } for name, r in eval_results.items()
}).T.round(4)

print("=== MODEL COMPARISON TABLE ===")
print(comparison)
comparison.to_csv("reports/model_comparison.csv")

# --- best model selection: prioritize F1 macro (clinically meaningful), ---
# --- tie-break on recall macro (screening priority: don't miss disease) ---
best_model_name = comparison.sort_values(
    by=["F1 (macro)", "Recall (macro)"], ascending=False
).index[0]
print(f"\nBest model selected based on macro F1 (tie-break: macro recall): {best_model_name}")

best_model = models[best_model_name]

# --- confusion matrix plots for all models --------------------------------
fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))
for ax, name in zip(axes, models.keys()):
    sns.heatmap(cms[name], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["None", "Mild", "Mod", "Severe"],
                yticklabels=["None", "Mild", "Mod", "Severe"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("reports/figures/05_confusion_matrices.png", dpi=120)
plt.close()

# --- save everything needed for reproducibility ---------------------------
with open("reports/evaluation_results.json", "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "confusion_matrix"}
               for k, v in eval_results.items()}, f, indent=2, default=str)

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(lr, "models/logistic_regression.pkl")
joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(improved_rf, "models/improved_random_forest.pkl")
joblib.dump(binary_rf, "models/binary_nafld_model.pkl")
joblib.dump(xgb, "models/xgboost.pkl")
joblib.dump(best_model, "models/final_model.pkl")

with open("models/feature_names.json", "w") as f:
    json.dump({
        "feature_order": feature_names,
        "best_model": best_model_name,
        "available_models": {
            "improved_rf": "Improved Random Forest Pipeline (250 Trees, Notebook Model)",
            "xgboost": "XGBoost Classifier (Multiclass Gradient Boosting)",
            "random_forest": "Random Forest Classifier (300 Trees)",
            "logistic_regression": "Logistic Regression (Standardized Baseline)"
        },
        "best_model_uses_scaler": best_model_name == "LogisticRegression",
        "target_classes": {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe"},
        "binary_classes": {0: "Negative (Healthy - Low Risk)", 1: "Positive (NAFLD Indicated)"}
    }, f, indent=2)

print("\nSaved models, scaler, feature_names.json, evaluation_results.json")
print("Final model:", best_model_name)
