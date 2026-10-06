import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# STEP 11 — FINAL RESULTS SUMMARY + EXPLAINABILITY
#
# Inputs:
#   models/v2_incident_classifier.joblib
#   data/v2_ml_training_dataset.csv
#   data/v2_sari_scored.csv
#   results/model_comparison.csv
#   results/best_model_classification_report.csv
#
# Outputs:
#   results/v2_final_summary.txt
#   results/v1_vs_v2_comparison.csv
#   results/top_features_by_class.csv
#   visualization/final_top_features.png
#   visualization/final_top_features.pdf
# ============================================================

Path("results").mkdir(exist_ok=True)
Path("visualization").mkdir(exist_ok=True)

MODEL_FILE = "models/v2_incident_classifier.joblib"
TRAIN_FILE = "data/v2_ml_training_dataset.csv"
SARI_FILE = "data/v2_sari_scored.csv"
MODEL_RESULTS = "results/model_comparison.csv"
REPORT_FILE = "results/best_model_classification_report.csv"

model = joblib.load(MODEL_FILE)
train = pd.read_csv(TRAIN_FILE)
sari = pd.read_csv(SARI_FILE)
model_results = pd.read_csv(MODEL_RESULTS)
report = pd.read_csv(REPORT_FILE)

# ------------------------------------------------------------
# 1. V1 vs V2 comparison table
# ------------------------------------------------------------
# V1 values are from the completed V1 pipeline/paper.
# V2 values are from the current pipeline.
v1_v2 = pd.DataFrame([
    {
        "Aspect": "Usable / analyzed articles",
        "V1": "463",
        "V2": f"{len(sari)} final ML-corpus articles (from 1,094 expanded unique records)"
    },
    {
        "Aspect": "Geographic scope handling",
        "V1": "No dedicated Bangladesh scope-audit stage",
        "V2": "Explicit Bangladesh-primary / foreign-background / out-of-scope separation"
    },
    {
        "Aspect": "Text used",
        "V1": "Primarily article title",
        "V2": "Title + available RSS description"
    },
    {
        "Aspect": "Incident classification",
        "V1": "Rule-based keyword mapping",
        "V2": "Weak supervision + assisted review + classical NLP ML"
    },
    {
        "Aspect": "Best ML model",
        "V1": "Not applicable",
        "V2": "Linear SVM"
    },
    {
        "Aspect": "Evaluation",
        "V1": "No ML validation",
        "V2": "Stratified 2-fold CV on weakly supervised + assisted-reviewed labels"
    },
    {
        "Aspect": "Best Macro F1",
        "V1": "Not applicable",
        "V2": "0.6068"
    },
    {
        "Aspect": "Best Weighted F1",
        "V1": "Not applicable",
        "V2": "0.7517"
    },
    {
        "Aspect": "Risk scoring",
        "V1": "Incident-type-driven fixed score",
        "V2": "Interpretable multi-factor SARI V2 heuristic index"
    },
    {
        "Aspect": "Risk levels observed",
        "V1": "Low / Medium / High",
        "V2": "Low / Medium; no High-risk article under current threshold"
    }
])
v1_v2.to_csv("results/v1_vs_v2_comparison.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 2. Extract top Linear SVM features
# ------------------------------------------------------------
# The saved object is expected to be a scikit-learn Pipeline.
if not hasattr(model, "named_steps"):
    raise ValueError("Saved model is not a scikit-learn Pipeline.")

# Find vectorizer and classifier automatically
vectorizer = None
classifier = None

for name, step in model.named_steps.items():
    if hasattr(step, "get_feature_names_out"):
        vectorizer = step
    if hasattr(step, "coef_") and hasattr(step, "classes_"):
        classifier = step

if vectorizer is None or classifier is None:
    raise ValueError("Could not find TF-IDF vectorizer and linear classifier in saved pipeline.")

features = np.array(vectorizer.get_feature_names_out())
classes = classifier.classes_
coef = classifier.coef_

top_rows = []
TOP_N = 12

for class_idx, class_name in enumerate(classes):
    top_idx = np.argsort(coef[class_idx])[-TOP_N:][::-1]
    for rank, idx in enumerate(top_idx, 1):
        top_rows.append({
            "class": class_name,
            "rank": rank,
            "feature": features[idx],
            "coefficient": float(coef[class_idx, idx])
        })

top_df = pd.DataFrame(top_rows)
top_df.to_csv("results/top_features_by_class.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 3. Paper-ready explainability figure
#    Use four classes with the most support to avoid clutter.
# ------------------------------------------------------------
support = train["ml_label"].value_counts()
plot_classes = support.head(4).index.tolist()

plot_df = (
    top_df[top_df["class"].isin(plot_classes)]
    .groupby("class", group_keys=False)
    .head(8)
)

# Build a compact multi-class horizontal chart in one figure
fig, axes = plt.subplots(len(plot_classes), 1, figsize=(10, 3.1 * len(plot_classes)))

if len(plot_classes) == 1:
    axes = [axes]

colors = ["#2878B5", "#2A9D8F", "#F28E2B", "#8E6BBE"]

for ax, class_name, color in zip(axes, plot_classes, colors):
    sub = plot_df[plot_df["class"] == class_name].sort_values("coefficient")
    ax.barh(sub["feature"], sub["coefficient"], color=color)
    ax.set_title(f"Top TF-IDF Features — {class_name}", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Linear SVM coefficient")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.18, linestyle="--")
    ax.set_axisbelow(True)

fig.suptitle(
    "Interpretable Text Features Learned by the Best V2 Model",
    fontsize=15,
    fontweight="bold",
    y=1.01
)
fig.tight_layout()
fig.savefig("visualization/final_top_features.png", dpi=350, bbox_inches="tight")
fig.savefig("visualization/final_top_features.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------
# 4. Build final human-readable summary
# ------------------------------------------------------------
best = model_results.sort_values("Macro F1", ascending=False).iloc[0]

risk_counts = sari["sari_v2_level"].value_counts()
mean_sari = sari["sari_v2_score"].mean()
max_sari = sari["sari_v2_score"].max()

incident_counts = sari["ml_label"].value_counts()

lines = []
lines.append("===== V2 FINAL RESULTS SUMMARY =====")
lines.append("")
lines.append(f"Final ML corpus: {len(train)} articles")
lines.append(f"Best model: {best['Model']}")
lines.append(f"Accuracy: {best['Accuracy']:.4f}")
lines.append(f"Macro Precision: {best['Macro Precision']:.4f}")
lines.append(f"Macro Recall: {best['Macro Recall']:.4f}")
lines.append(f"Macro F1: {best['Macro F1']:.4f}")
lines.append(f"Weighted F1: {best['Weighted F1']:.4f}")
lines.append("")
lines.append("Incident distribution:")
for k, v in incident_counts.items():
    lines.append(f"  {k}: {v}")
lines.append("")
lines.append("SARI V2:")
for level in ["Low", "Medium", "High"]:
    lines.append(f"  {level}: {int(risk_counts.get(level, 0))}")
lines.append(f"  Mean score: {mean_sari:.3f}")
lines.append(f"  Maximum observed score: {max_sari:.3f}")
lines.append("")
lines.append("Interpretation note:")
lines.append(
    "ML scores represent cross-validated predictive agreement with the current "
    "weakly supervised + assisted-reviewed label set; they are not a fully "
    "independent human-ground-truth benchmark."
)
lines.append(
    "SARI V2 is an interpretable heuristic risk index, not a clinically validated "
    "or empirically calibrated probability."
)
lines.append(
    "Rare classes, especially Rescue and Poisoning, have very small support and "
    "therefore unstable class-level estimates."
)

Path("results/v2_final_summary.txt").write_text("\n".join(lines), encoding="utf-8")

print("===== STEP 11 COMPLETE =====")
print("Created:")
print("results/v2_final_summary.txt")
print("results/v1_vs_v2_comparison.csv")
print("results/top_features_by_class.csv")
print("visualization/final_top_features.png")
print("visualization/final_top_features.pdf")

print("\nPaper-critical summary:")
print(f"Best model: {best['Model']}")
print(f"Macro F1: {best['Macro F1']:.4f}")
print(f"Weighted F1: {best['Weighted F1']:.4f}")
print(f"Mean SARI V2: {mean_sari:.3f}")
print(f"Max SARI V2: {max_sari:.3f}")
print("Risk counts:", {k: int(risk_counts.get(k, 0)) for k in ['Low','Medium','High']})
print("\nIMPORTANT: Do not describe CV metrics as fully independent human-ground-truth accuracy.")
