import pandas as pd
import re
from pathlib import Path

INPUT_FILE = "data/v2_master_dataset.csv"
OUTPUT_FILE = "data/v2_relevant_dataset.csv"
REJECT_FILE = "data/v2_rejected_articles.csv"

Path("data").mkdir(exist_ok=True)

df = pd.read_csv(INPUT_FILE)

required = ["article_id", "title", "source"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# -------------------------------------------------
# 1. High-confidence stray/animal terms in headline
# -------------------------------------------------
ANIMAL_TERMS = [
    "dog", "dogs", "puppy", "puppies",
    "cat", "cats", "kitten", "kittens",
    "animal", "animals",
    "stray", "strays",
    "canine", "canines",
    "feline", "felines",
    "street dog", "street dogs",
    "street cat", "street cats",
    "free roaming dog", "free-roaming dog"
]

# Rabies can be relevant even when a headline does not explicitly say dog/cat,
# but we require an animal-conflict context to reduce unrelated health news.
RABIES_CONTEXT = [
    "rabies", "dog bite", "animal bite", "bite victim",
    "anti-rabies", "anti rabies"
]

# Strong animal-welfare phrases
WELFARE_TERMS = [
    "animal welfare", "animal cruelty", "animal abuse",
    "animal rights", "dog rescue", "cat rescue",
    "stray dog", "stray cat", "street dog", "street cat"
]

def contains_phrase(text, terms):
    text = str(text).lower()
    for term in terms:
        if re.search(r"\b" + re.escape(term) + r"\b", text):
            return True
    return False

def relevance_reason(title):
    t = str(title).lower()

    if contains_phrase(t, WELFARE_TERMS):
        return "Strong welfare phrase"

    if contains_phrase(t, ANIMAL_TERMS):
        return "Explicit animal term"

    if contains_phrase(t, RABIES_CONTEXT):
        return "Rabies/bite context"

    return "No clear animal relevance in title"

df["relevance_reason"] = df["title"].apply(relevance_reason)
df["is_relevant"] = (df["relevance_reason"] != "No clear animal relevance in title").astype(int)

relevant = df[df["is_relevant"] == 1].copy()
rejected = df[df["is_relevant"] == 0].copy()

relevant.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
rejected.to_csv(REJECT_FILE, index=False, encoding="utf-8-sig")

print("===== V2 RELEVANCE FILTER COMPLETE =====")
print("Input rows:", len(df))
print("Relevant rows kept:", len(relevant))
print("Rejected rows:", len(rejected))
print(f"Retention rate: {len(relevant)/len(df)*100:.2f}%")
print("\nRelevance reasons:")
print(relevant["relevance_reason"].value_counts())
print("\nSaved relevant data:", OUTPUT_FILE)
print("Saved rejected data:", REJECT_FILE)
