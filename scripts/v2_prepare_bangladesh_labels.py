import pandas as pd
import re
from pathlib import Path

# ============================================================
# STEP 7 — BANGLADESH PRIMARY LABEL PREPARATION
#
# Input:
#   data/v2_bangladesh_primary.csv
#
# Outputs:
#   data/v2_bangladesh_weak_labels.csv
#   data/v2_bangladesh_validation_sample.csv
#
# NOTE:
# suggested_label = weak / rule-assisted label
# final_label     = human-reviewed label (left blank)
# ============================================================

INPUT_FILE = "data/v2_bangladesh_primary.csv"
WEAK_LABEL_FILE = "data/v2_bangladesh_weak_labels.csv"
VALIDATION_FILE = "data/v2_bangladesh_validation_sample.csv"

Path("data").mkdir(exist_ok=True)

df = pd.read_csv(INPUT_FILE)

required = ["v2_article_id", "title", "source"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if "description" not in df.columns:
    df["description"] = ""

if "article_text" not in df.columns:
    df["article_text"] = (
        df["title"].fillna("").astype(str)
        + " "
        + df["description"].fillna("").astype(str)
    )

# ------------------------------------------------------------
# 1. Helper
# ------------------------------------------------------------
def has(text, terms):
    text = str(text).lower()
    return any(re.search(r"\b" + re.escape(term) + r"\b", text) for term in terms)

# ------------------------------------------------------------
# 2. Weak-label logic
# Specific categories checked before broader categories.
# ------------------------------------------------------------
def weak_label(text):
    t = str(text).lower()

    if has(t, [
        "poison", "poisoned", "poisoning", "toxic", "toxicity"
    ]):
        return "Poisoning"

    if has(t, [
        "abuse", "abused", "cruelty", "torture", "tortured",
        "beaten", "brutality", "mutilated", "violence against"
    ]):
        return "Abuse"

    if has(t, [
        "dog bite", "animal bite", "bite", "bitten", "attack",
        "attacked", "aggressive", "injury", "injured",
        "maul", "mauled", "chased"
    ]):
        return "Human-Animal Conflict"

    if has(t, [
        "rabies", "disease", "infection", "infected", "sick",
        "ill", "virus", "vaccination", "vaccine", "vaccinated",
        "anti-rabies", "anti rabies"
    ]):
        return "Disease"

    if has(t, [
        "rescue", "rescued", "saved", "shelter", "adopt",
        "adopted", "adoption", "rehabilitation",
        "treated", "treatment"
    ]):
        return "Rescue"

    if has(t, [
        "kill", "killed", "killing", "dead", "death",
        "slaughter", "culled", "culling"
    ]):
        return "Killing"

    if has(t, [
        "relocate", "relocation", "evict", "eviction",
        "protest", "rights", "conflict", "ban", "policy",
        "remove", "removal"
    ]):
        return "Conflict"

    return "Other"

labels = df.copy()
labels["suggested_label"] = labels["article_text"].apply(weak_label)
labels["final_label"] = ""
labels["review_note"] = ""

labels.to_csv(WEAK_LABEL_FILE, index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 3. Build a compact human-validation sample
#    Up to 8 rows per class, then fill to 64 rows if needed.
# ------------------------------------------------------------
RANDOM_STATE = 42
PER_CLASS = 8
TARGET_TOTAL = 64

samples = []

for label_name, group in labels.groupby("suggested_label"):
    n = min(PER_CLASS, len(group))
    if n > 0:
        samples.append(group.sample(n=n, random_state=RANDOM_STATE))

sample = pd.concat(samples, ignore_index=True) if samples else labels.head(0).copy()

if len(sample) < TARGET_TOTAL:
    used_ids = set(sample["v2_article_id"].tolist())
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

sample.to_csv(VALIDATION_FILE, index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 4. Summary
# ------------------------------------------------------------
print("===== BANGLADESH LABEL PREPARATION COMPLETE =====")
print("Bangladesh-primary articles:", len(labels))

print("\nSuggested-label distribution:")
print(labels["suggested_label"].value_counts())

print("\nValidation sample rows:", len(sample))
print("Validation sample distribution:")
print(sample["suggested_label"].value_counts())

print("\nSaved:")
print(WEAK_LABEL_FILE)
print(VALIDATION_FILE)

print("\nIMPORTANT:")
print("Do NOT manually label all articles.")
print("Only the validation sample needs human review.")
print("Allowed labels:")
print("Abuse | Killing | Poisoning | Disease | Rescue | Human-Animal Conflict | Conflict | Other")
