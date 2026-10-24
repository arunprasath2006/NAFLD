"""
Phase 4 — Exploratory Data Analysis
Generates figures into reports/figures/ using the CLEANED dataset.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
FIG_DIR = "reports/figures"

df = pd.read_csv("data/fatty_liver_cleaned.csv")

grade_labels = {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe"}
df["grade_label"] = df["fatty_liver_grade"].map(grade_labels)
order = ["None", "Mild", "Moderate", "Severe"]

# 1. Target class distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="grade_label", order=order, hue="grade_label",
              palette="viridis", legend=False)
plt.title("Fatty Liver Grade Distribution")
plt.xlabel("Grade")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_target_distribution.png", dpi=120)
plt.close()

# 2-8. Distributions of key clinical/biochemical variables
dist_vars = ["age", "bmi", "fasting_blood_sugar", "alt", "ast", "ggt",
             "triglycerides"]
fig, axes = plt.subplots(4, 2, figsize=(11, 14))
axes = axes.flatten()
for i, v in enumerate(dist_vars):
    sns.histplot(df[v], kde=True, ax=axes[i], color="steelblue")
    axes[i].set_title(f"{v} distribution")
for j in range(len(dist_vars), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_variable_distributions.png", dpi=120)
plt.close()

# 9. Correlation heatmap
num_cols = ["age", "bmi", "alt", "ast", "ggt", "bilirubin_total",
            "triglycerides", "hdl_cholestrol", "ldl_cholestrol",
            "platelet_count", "fasting_blood_sugar", "tsh",
            "fatty_liver_grade"]
plt.figure(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_correlation_heatmap.png", dpi=120)
plt.close()

# 10. Important features vs target (boxplots)
top_feats = ["bmi", "triglycerides", "ast", "alt", "ggt", "hdl_cholestrol"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, v in enumerate(top_feats):
    sns.boxplot(data=df, x="grade_label", y=v, order=order, hue="grade_label",
                palette="viridis", legend=False, ax=axes[i])
    axes[i].set_title(f"{v} vs Fatty Liver Grade")
    axes[i].set_xlabel("Grade")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_feature_vs_target.png", dpi=120)
plt.close()

print("EDA figures saved to", FIG_DIR)
print("Files:", ["01_target_distribution.png", "02_variable_distributions.png",
                  "03_correlation_heatmap.png", "04_feature_vs_target.png"])
