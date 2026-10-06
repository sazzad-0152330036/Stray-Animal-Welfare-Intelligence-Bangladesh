import pandas as pd
import numpy as np
import re
from pathlib import Path

# ============================================================
# V2 EXPANDED DATA PREPROCESSING + RELEVANCE FILTER
# Input : data/raw_news_v2_expanded.csv
# Output:
#   data/v2_expanded_master_dataset.csv
#   data/v2_expanded_rejected_articles.csv
# ============================================================

INPUT_FILE = "data/raw_news_v2_expanded.csv"
OUTPUT_FILE = "data/v2_expanded_master_dataset.csv"
REJECT_FILE = "data/v2_expanded_rejected_articles.csv"

Path("data").mkdir(exist_ok=True)

df = pd.read_csv(INPUT_FILE)

required = ["keyword", "title", "source", "link", "date"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if "description" not in df.columns:
    df["description"] = ""

# ------------------------------------------------------------
# 1. Basic cleaning
# ------------------------------------------------------------
for col in ["keyword", "title", "description", "source", "link", "date"]:
    df[col] = df[col].fillna("").astype(str).str.strip()

df = df[~df["title"].str.lower().isin(["", "nan", "no title"])].copy()

df = df.drop_duplicates(subset=["link"], keep="first")
df = df.drop_duplicates(subset=["title"], keep="first")

# ------------------------------------------------------------
# 2. Text cleaning
# ------------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_title"] = df["title"].apply(clean_text)
df["clean_description"] = df["description"].apply(clean_text)

# Important:
# Relevance uses only actual article content (title + RSS description),
# NOT the search keyword, so search terms do not falsely make a row relevant.
df["article_text"] = (
    df["clean_title"] + " " + df["clean_description"]
).str.replace(r"\s+", " ", regex=True).str.strip()

# For later ML, keyword can still be retained as metadata,
# but article_text is the main NLP text.
df["text_for_nlp"] = df["article_text"]

# ------------------------------------------------------------
# 3. Date features
# ------------------------------------------------------------
df["parsed_date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
df["year"] = df["parsed_date"].dt.year
df["month"] = df["parsed_date"].dt.month
df["month_name"] = df["parsed_date"].dt.month_name()
df["day_of_week"] = df["parsed_date"].dt.day_name()

# ------------------------------------------------------------
# 4. Relevance filtering
# ------------------------------------------------------------
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

RABIES_CONTEXT = [
    "rabies", "anti rabies", "anti-rabies",
    "dog bite", "animal bite", "bite victim"
]

WELFARE_TERMS = [
    "animal welfare", "animal cruelty", "animal abuse",
    "animal rights", "dog rescue", "cat rescue",
    "stray dog", "stray cat", "street dog", "street cat"
]

def contains_term(text, terms):
    text = str(text).lower()
    return any(re.search(r"\b" + re.escape(term) + r"\b", text) for term in terms)

def relevance_reason(text):
    if contains_term(text, WELFARE_TERMS):
        return "Strong welfare phrase"
    if contains_term(text, ANIMAL_TERMS):
        return "Explicit animal term"
    if contains_term(text, RABIES_CONTEXT):
        return "Rabies/bite context"
    return "No clear animal relevance"

df["relevance_reason"] = df["article_text"].apply(relevance_reason)
df["is_relevant"] = (df["relevance_reason"] != "No clear animal relevance").astype(int)

relevant = df[df["is_relevant"] == 1].copy()
rejected = df[df["is_relevant"] == 0].copy()

# ------------------------------------------------------------
# 5. Animal type detection
# ------------------------------------------------------------
DOG_WORDS = [
    "dog", "dogs", "puppy", "puppies", "canine", "canines",
    "street dog", "street dogs", "stray dog", "stray dogs",
    "free roaming dog", "free-roaming dog"
]

CAT_WORDS = [
    "cat", "cats", "kitten", "kittens", "feline", "felines",
    "street cat", "street cats", "stray cat", "stray cats"
]

def detect_animal(text):
    dog = contains_term(text, DOG_WORDS)
    cat = contains_term(text, CAT_WORDS)

    if dog and cat:
        return "Dog & Cat"
    elif dog:
        return "Dog"
    elif cat:
        return "Cat"
    else:
        return "Unknown"

relevant["animal_type_v2"] = relevant["article_text"].apply(detect_animal)

# ------------------------------------------------------------
# 6. Context indicators
# ------------------------------------------------------------
VIOLENCE_WORDS = [
    "kill", "killed", "killing", "dead", "death", "murder",
    "beat", "beaten", "torture", "tortured", "cruelty",
    "abuse", "abused", "poison", "poisoned", "poisoning",
    "toxic", "shoot", "shot", "burn", "burned", "attack"
]

DISEASE_WORDS = [
    "rabies", "disease", "infected", "infection", "sick",
    "ill", "virus", "vaccination", "vaccine", "vaccinated"
]

HUMAN_CONFLICT_WORDS = [
    "bite", "bitten", "attack", "attacked", "aggressive",
    "injury", "injured", "conflict", "nuisance", "fear",
    "relocation", "relocate"
]

RESCUE_WORDS = [
    "rescue", "rescued", "save", "saved", "shelter",
    "adopt", "adoption", "adopted", "treatment",
    "treated", "rehabilitation"
]

POISON_WORDS = [
    "poison", "poisoned", "poisoning", "toxic", "toxicity"
]

def count_terms(text, terms):
    return sum(
        1 for term in terms
        if re.search(r"\b" + re.escape(term) + r"\b", str(text).lower())
    )

feature_sets = {
    "violence": VIOLENCE_WORDS,
    "disease": DISEASE_WORDS,
    "human_conflict": HUMAN_CONFLICT_WORDS,
    "rescue": RESCUE_WORDS,
    "poison": POISON_WORDS,
}

for name, terms in feature_sets.items():
    relevant[f"{name}_keyword_count"] = relevant["article_text"].apply(
        lambda x, t=terms: count_terms(x, t)
    )
    relevant[f"{name}_indicator"] = (
        relevant[f"{name}_keyword_count"] > 0
    ).astype(int)

# ------------------------------------------------------------
# 7. Severity feature
# This is only a feature, NOT the final SARI V2 score.
# ------------------------------------------------------------
relevant["severity_feature"] = (
    2 * relevant["violence_keyword_count"]
    + 2 * relevant["poison_keyword_count"]
    + relevant["disease_keyword_count"]
    + relevant["human_conflict_keyword_count"]
)

# ------------------------------------------------------------
# 8. Text metadata
# ------------------------------------------------------------
relevant["title_word_count"] = relevant["clean_title"].str.split().str.len()
relevant["description_word_count"] = (
    relevant["clean_description"].str.split().str.len()
)
relevant["article_word_count"] = relevant["article_text"].str.split().str.len()

# ------------------------------------------------------------
# 9. Stable new IDs
# ------------------------------------------------------------
relevant = relevant.reset_index(drop=True)
relevant.insert(0, "v2_article_id", range(1, len(relevant) + 1))

rejected = rejected.reset_index(drop=True)

# ------------------------------------------------------------
# 10. Save
# ------------------------------------------------------------
relevant.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
rejected.to_csv(REJECT_FILE, index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 11. Summary
# ------------------------------------------------------------
print("===== EXPANDED V2 PREPROCESSING COMPLETE =====")
print("Input rows:", len(df))
print("Relevant rows kept:", len(relevant))
print("Rejected rows:", len(rejected))
print(f"Retention rate: {len(relevant) / len(df) * 100:.2f}%")

print("\nRows with usable description among relevant data:",
      (relevant["clean_description"].str.len() > 0).sum())

print("\nAnimal type distribution:")
print(relevant["animal_type_v2"].value_counts(dropna=False))

print("\nYear range:")
if relevant["year"].notna().any():
    print(int(relevant["year"].min()), "to", int(relevant["year"].max()))
else:
    print("No valid years")

print("\nUnique sources:", relevant["source"].nunique())

print("\nSaved relevant dataset:", OUTPUT_FILE)
print("Saved rejected dataset:", REJECT_FILE)
