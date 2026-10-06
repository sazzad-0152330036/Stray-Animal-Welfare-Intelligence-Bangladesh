import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# STEP 8 — V2 MACHINE LEARNING TRAINING
#
# Inputs:
#   data/v2_bangladesh_weak_labels.csv
#   data/v2_bangladesh_validation_assisted_review.csv
#
# Outputs:
#   data/v2_ml_training_dataset.csv
#   results/model_comparison.csv
#   results/best_model_classification_report.csv
#   results/best_model_confusion_matrix.csv
#   visualization/model_comparison.png
#   visualization/confusion_matrix_best_model.png
#   models/v2_incident_classifier.joblib
#
# IMPORTANT SCIENTIFIC NOTE:
# This is a weakly supervised / assisted-label experiment.
# Cross-validation metrics measure agreement with the current label set,
# NOT fully independent human-ground-truth performance.
# ============================================================

WEAK_FILE = "data/v2_bangladesh_weak_labels.csv"
REVIEW_FILE = "data/v2_bangladesh_validation_assisted_review.csv"

TRAINING_FILE = "data/v2_ml_training_dataset.csv"

Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
Path("visualization").mkdir(exist_ok=True)

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------
weak = pd.read_csv(WEAK_FILE)
review = pd.read_csv(REVIEW_FILE)

required_weak = ["v2_article_id", "suggested_label"]
missing = [c for c in required_weak if c not in weak.columns]
if missing:
    raise ValueError(f"Missing columns in weak-label file: {missing}")

required_review = ["v2_article_id", "final_label", "review_status"]
missing = [c for c in required_review if c not in review.columns]
if missing:
    raise ValueError(f"Missing columns in review file: {missing}")

if "article_text" not in weak.columns:
    if "description" not in weak.columns:
        weak["description"] = ""
    weak["article_text"] = (
        weak["title"].fillna("").astype(str) + " " +
        weak["description"].fillna("").astype(str)
    )

weak["article_text"] = weak["article_text"].fillna("").astype(str).str.strip()

# ------------------------------------------------------------
# 2. Use assisted review as quality control
#
# EXCLUDE / AMBIGUOUS rows are removed from ML corpus.
# KEEP rows override weak labels with reviewed final_label.
# All other rows retain suggested_label.
# ------------------------------------------------------------
# The weak-label CSV may already contain blank final_label/review_note columns.
# Drop them before merging so pandas does not create final_label_x/final_label_y.
weak = weak.drop(
    columns=["final_label", "review_note", "review_status"],
    errors="ignore"
)

review_small = review[
    ["v2_article_id", "final_label", "review_status"]
].copy()

df = weak.merge(
    review_small,
    on="v2_article_id",
    how="left"
)

df["review_status"] = df["review_status"].fillna("NOT_REVIEWED")
df["final_label"] = df["final_label"].fillna("").astype(str).str.strip()

before = len(df)

df = df[
    ~df["review_status"].isin(["EXCLUDE", "AMBIGUOUS"])
].copy()

df["ml_label"] = np.where(
    (df["review_status"] == "KEEP") & (df["final_label"] != ""),
    df["final_label"],
    df["suggested_label"]
)

df = df[
    (df["article_text"].str.len() > 2) &
    (df["ml_label"].notna())
].copy()

df.to_csv(TRAINING_FILE, index=False, encoding="utf-8-sig")

print("===== V2 ML DATASET PREPARED =====")
print("Original Bangladesh-primary weak-label rows:", before)
print("Rows removed by assisted review:", before - len(df))
print("Final ML corpus rows:", len(df))

print("\nML label distribution:")
print(df["ml_label"].value_counts())

# ------------------------------------------------------------
# 3. Choose valid number of CV folds automatically
# ------------------------------------------------------------
min_class_count = int(df["ml_label"].value_counts().min())

if min_class_count < 2:
    raise ValueError(
        "At least one class has fewer than 2 examples. "
        "More data or class merging is required before CV."
    )

n_splits = min(3, min_class_count)

print(f"\nUsing Stratified {n_splits}-fold cross-validation.")
print("Reason: the rarest class has", min_class_count, "rows.")

cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=42
)

X = df["article_text"]
y = df["ml_label"]

# ------------------------------------------------------------
# 4. Define models
# ------------------------------------------------------------
models = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            max_features=12000,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            max_features=12000,
            sublinear_tf=True
        )),
        ("clf", LinearSVC(
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Complement Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            max_features=12000,
            sublinear_tf=True
        )),
        ("clf", ComplementNB())
    ])
}

# ------------------------------------------------------------
# 5. Cross-validated model comparison
# ------------------------------------------------------------
comparison_rows = []
predictions = {}

for name, model in models.items():
    print(f"\nTraining/evaluating: {name}")

    y_pred = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        n_jobs=None
    )

    predictions[name] = y_pred

    row = {
        "Model": name,
        "Accuracy": accuracy_score(y, y_pred),
        "Macro Precision": precision_score(
            y, y_pred, average="macro", zero_division=0
        ),
        "Macro Recall": recall_score(
            y, y_pred, average="macro", zero_division=0
        ),
        "Macro F1": f1_score(
            y, y_pred, average="macro", zero_division=0
        ),
        "Weighted F1": f1_score(
            y, y_pred, average="weighted", zero_division=0
        )
    }

    comparison_rows.append(row)

comparison = pd.DataFrame(comparison_rows)
comparison = comparison.sort_values(
    by="Macro F1",
    ascending=False
).reset_index(drop=True)

comparison.to_csv(
    "results/model_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== MODEL COMPARISON =====")
print(comparison.to_string(index=False))

# ------------------------------------------------------------
# 6. Select best model using Macro F1
# ------------------------------------------------------------
best_name = comparison.iloc[0]["Model"]
best_model = models[best_name]
best_pred = predictions[best_name]

print("\nBEST MODEL:", best_name)
print("Selection criterion: highest Macro F1")

# ------------------------------------------------------------
# 7. Classification report
# ------------------------------------------------------------
report = classification_report(
    y,
    best_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(
    "results/best_model_classification_report.csv",
    encoding="utf-8-sig"
)

print("\n===== BEST MODEL CLASSIFICATION REPORT =====")
print(report_df.round(4).to_string())

# ------------------------------------------------------------
# 8. Confusion matrix
# ------------------------------------------------------------
labels_sorted = sorted(y.unique())

cm = confusion_matrix(
    y,
    best_pred,
    labels=labels_sorted
)

cm_df = pd.DataFrame(
    cm,
    index=labels_sorted,
    columns=labels_sorted
)

cm_df.to_csv(
    "results/best_model_confusion_matrix.csv",
    encoding="utf-8-sig"
)

fig, ax = plt.subplots(figsize=(11, 9))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels_sorted
)
disp.plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d"
)
ax.set_title(
    f"V2 Incident Classification — {best_name}\n"
    f"{n_splits}-Fold Cross-Validation"
)
fig.tight_layout()
fig.savefig(
    "visualization/confusion_matrix_best_model.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# ------------------------------------------------------------
# 9. Model comparison chart
# ------------------------------------------------------------
plot_df = comparison.sort_values("Macro F1", ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(plot_df["Model"], plot_df["Macro F1"])
ax.set_xlabel("Macro F1 Score")
ax.set_ylabel("Model")
ax.set_title("V2 ML Model Comparison")
ax.set_xlim(0, 1)

for i, value in enumerate(plot_df["Macro F1"]):
    ax.text(
        min(value + 0.015, 0.97),
        i,
        f"{value:.3f}",
        va="center"
    )

fig.tight_layout()
fig.savefig(
    "visualization/model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# ------------------------------------------------------------
# 10. Fit best model on complete available ML corpus
# ------------------------------------------------------------
best_model.fit(X, y)

joblib.dump(
    best_model,
    "models/v2_incident_classifier.joblib"
)

# Save model predictions on training corpus for later analysis.
df["model_prediction_full_fit"] = best_model.predict(X)
df.to_csv(
    "data/v2_ml_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# 11. Final summary
# ------------------------------------------------------------
best_row = comparison.iloc[0]

print("\n===== V2 MACHINE LEARNING COMPLETE =====")
print("Best model:", best_name)
print(f"Accuracy: {best_row['Accuracy']:.4f}")
print(f"Macro Precision: {best_row['Macro Precision']:.4f}")
print(f"Macro Recall: {best_row['Macro Recall']:.4f}")
print(f"Macro F1: {best_row['Macro F1']:.4f}")
print(f"Weighted F1: {best_row['Weighted F1']:.4f}")

print("\nFiles saved:")
print("data/v2_ml_training_dataset.csv")
print("data/v2_ml_predictions.csv")
print("results/model_comparison.csv")
print("results/best_model_classification_report.csv")
print("results/best_model_confusion_matrix.csv")
print("visualization/model_comparison.png")
print("visualization/confusion_matrix_best_model.png")
print("models/v2_incident_classifier.joblib")

print("\nSCIENTIFIC INTERPRETATION:")
print(
    "These CV scores measure predictive agreement with the current "
    "weakly supervised + assisted-reviewed label set."
)
print(
    "Do NOT describe them as fully independent human-ground-truth accuracy."
)
