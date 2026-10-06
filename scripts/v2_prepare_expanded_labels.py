import pandas as pd
import re
from pathlib import Path

INPUT_FILE = "data/v2_expanded_master_dataset.csv"
FULL_LABEL_FILE = "data/v2_expanded_weak_labels.csv"
REVIEW_SAMPLE_FILE = "data/v2_human_validation_sample.csv"

Path("data").mkdir(exist_ok=True)

df = pd.read_csv(INPUT_FILE)

required = ["v2_article_id", "title", "source", "article_text"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

def has(text, terms):
    text = str(text).lower()
    return any(re.search(r"\b" + re.escape(term) + r"\b", text) for term in terms)

def weak_label(text):
    t = str(text).lower()

    # Highest-specificity categories first
    if has(t, ["poison", "poisoned", "poisoning", "toxic", "toxicity"]):
        return "Poisoning"

    if has(t, ["abuse", "abused", "cruelty", "torture", "tortured", "beaten", "brutality"]):
        return "Abuse"

    if has(t, ["dog bite", "animal bite", "bite", "bitten", "attack", "attacked",
               "aggressive", "injury", "injured", "maul", "mauled"]):
        return "Human-Animal Conflict"

    if has(t, ["rabies", "disease", "infection", "infected", "sick", "ill",
               "vaccination", "vaccine", "vaccinated"]):
        return "Disease"

    if has(t, ["rescue", "rescued", "saved", "shelter", "adopt", "adopted",
               "adoption", "rehabilitation", "treated", "treatment"]):
        return "Rescue"

    if has(t, ["kill", "killed", "killing", "dead", "death", "slaughter"]):
        return "Killing"

    if has(t, ["relocate", "relocation", "evict", "eviction", "protest",
               "rights", "conflict", "ban", "policy"]):
        return "Conflict"

    return "Other"

labels = df.copy()
labels["suggested_label"] = labels["article_text"].apply(weak_label)
labels["final_label"] = ""
labels["review_note"] = ""

labels.to_csv(FULL_LABEL_FILE, index=False, encoding="utf-8-sig")

# ----------------------------------------------------------
# Human-validation sample
# Target: ~80 rows, stratified where possible
# ----------------------------------------------------------
RANDOM_STATE = 42
target_per_class = 10
samples = []

for label, group in labels.groupby("suggested_label"):
    n = min(target_per_class, len(group))
    samples.append(group.sample(n=n, random_state=RANDOM_STATE))

sample = pd.concat(samples, ignore_index=True)

# Fill remaining slots up to 80 from unsampled rows
TARGET_TOTAL = 80
if len(sample) < TARGET_TOTAL:
    used_ids = set(sample["v2_article_id"])
    remaining = labels[~labels["v2_article_id"].isin(used_ids)]
    extra_n = min(TARGET_TOTAL - len(sample), len(remaining))
    if extra_n > 0:
        extra = remaining.sample(n=extra_n, random_state=RANDOM_STATE)
        sample = pd.concat([sample, extra], ignore_index=True)

sample = sample.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

review_cols = [
    "v2_article_id",
    "title",
    "description",
    "source",
    "suggested_label",
    "final_label",
    "review_note"
]
review_cols = [c for c in review_cols if c in sample.columns]
sample = sample[review_cols]

sample.to_csv(REVIEW_SAMPLE_FILE, index=False, encoding="utf-8-sig")

print("===== V2 LABEL PREPARATION COMPLETE =====")
print("Total relevant articles:", len(labels))

print("\nSuggested-label distribution:")
print(labels["suggested_label"].value_counts())

print("\nHuman-validation sample rows:", len(sample))
print("Validation sample distribution:")
print(sample["suggested_label"].value_counts())

print("\nSaved:")
print(FULL_LABEL_FILE)
print(REVIEW_SAMPLE_FILE)

print("\nIMPORTANT:")
print("Do not treat suggested_label as human ground truth.")
print("Only final_label in the validation sample will be human-reviewed.")
