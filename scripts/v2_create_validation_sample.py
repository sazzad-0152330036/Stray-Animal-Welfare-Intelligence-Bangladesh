import pandas as pd
from pathlib import Path

INPUT_FILE = "data/v2_label_validation.csv"
OUTPUT_FILE = "data/v2_manual_validation_sample.csv"
RANDOM_STATE = 42

Path("data").mkdir(exist_ok=True)

df = pd.read_csv(INPUT_FILE)

required = ["article_id", "title", "source", "suggested_label"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

allowed_labels = [
    "Abuse",
    "Killing",
    "Poisoning",
    "Disease",
    "Rescue",
    "Human-Animal Conflict",
    "Conflict",
    "Other"
]

samples = []

# More "Other" rows because this is the largest/most ambiguous class
for label in allowed_labels:
    group = df[df["suggested_label"] == label].copy()

    if label == "Other":
        n = min(30, len(group))
    else:
        n = min(15, len(group))

    if n > 0:
        samples.append(group.sample(n=n, random_state=RANDOM_STATE))

sample = pd.concat(samples, ignore_index=True)

# Shuffle the rows so labels are not grouped together
sample = sample.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# Blank human-reviewed columns
sample["final_label"] = ""
sample["review_note"] = ""

# Keep only useful columns
sample = sample[
    [
        "article_id",
        "title",
        "source",
        "suggested_label",
        "final_label",
        "review_note"
    ]
]

sample.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("===== MANUAL VALIDATION SAMPLE CREATED =====")
print("Rows to review:", len(sample))
print("\nSuggested-label distribution in sample:")
print(sample["suggested_label"].value_counts())
print("\nSaved:", OUTPUT_FILE)
print("\nFill ONLY the final_label column.")
print("Allowed labels:")
print("Abuse | Killing | Poisoning | Disease | Rescue | Human-Animal Conflict | Conflict | Other")
