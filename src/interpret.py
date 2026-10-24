"""
Phase 8 — Model Interpretability
Computes and saves feature importance for the final selected model
(XGBoost). Wording is kept strictly correlational, not causal.
"""
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

model = joblib.load("models/final_model.pkl")
with open("models/feature_names.json") as f:
    meta = json.load(f)
feature_names = meta["feature_order"]

importances = model.feature_importances_
imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)

print("=== Feature importance (XGBoost, gain-based) ===")
print(imp_df)

plt.figure(figsize=(8, 6))
plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="teal")
plt.xlabel("Importance (relative influence on model prediction)")
plt.title("Feature Importance — Final Model (XGBoost)\n"
          "Note: reflects statistical influence on the model, not proven biological causation")
plt.tight_layout()
plt.savefig("reports/figures/06_feature_importance.png", dpi=120)
plt.close()

# Save as JSON for backend to serve
top_features = imp_df.head(8).to_dict(orient="records")
with open("models/feature_importance.json", "w") as f:
    json.dump(top_features, f, indent=2)

print("\nSaved reports/figures/06_feature_importance.png and models/feature_importance.json")
