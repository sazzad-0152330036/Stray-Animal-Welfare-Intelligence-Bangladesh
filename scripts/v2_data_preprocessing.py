
import pandas as pd
import numpy as np
import re
from pathlib import Path

# =========================================================
# V2 DATA PREPROCESSING
# Input : data/raw_news.csv
# Output: data/v2_master_dataset.csv
# =========================================================

INPUT_FILE = "data/raw_news.csv"
OUTPUT_FILE = "data/v2_master_dataset.csv"

Path("data").mkdir(exist_ok=True)

# -----------------------------
# 1. Load raw data
# -----------------------------
df = pd.read_csv(INPUT_FILE)

print("Original rows:", len(df))
print("Columns:", list(df.columns))

# -----------------------------
# 2. Basic cleaning
# -----------------------------
required_cols = ["keyword", "title", "source", "link", "date"]

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Remove rows with no usable title
df["title"] = df["title"].astype(str).str.strip()
df = df[df["title"].notna()]
df = df[~df["title"].str.lower().isin(["", "nan", "no title"])]

# Standardize text columns
for col in ["keyword", "title", "source", "link"]:
    df[col] = df[col].fillna("").astype(str).str.strip()

# Remove exact duplicate URLs first, then duplicate titles
df = df.drop_duplicates(subset=["link"], keep="first")
df = df.drop_duplicates(subset=["title"], keep="first")

# -----------------------------
# 3. Clean title text for NLP
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_title"] = df["title"].apply(clean_text)

# Remove empty cleaned titles
df = df[df["clean_title"].str.len() > 0]

# -----------------------------
# 4. Parse date
# -----------------------------
df["parsed_date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

df["year"] = df["parsed_date"].dt.year
df["month"] = df["parsed_date"].dt.month
df["month_name"] = df["parsed_date"].dt.month_name()
df["day_of_week"] = df["parsed_date"].dt.day_name()

# -----------------------------
# 5. Animal type detection V2
# -----------------------------
DOG_WORDS = [
    "dog", "dogs", "puppy", "puppies", "canine", "street dog",
    "stray dog", "free roaming dog", "free-roaming dog"
]

CAT_WORDS = [
    "cat", "cats", "kitten", "kittens", "feline",
    "street cat", "stray cat"
]

def contains_any(text, terms):
    return any(re.search(r"\b" + re.escape(term) + r"\b", text) for term in terms)

def detect_animal_v2(text):
    dog = contains_any(text, DOG_WORDS)
    cat = contains_any(text, CAT_WORDS)

    if dog and cat:
        return "Dog & Cat"
    elif dog:
        return "Dog"
    elif cat:
        return "Cat"
    else:
        return "Unknown"

# Use title + search keyword together
df["combined_text"] = (
    df["clean_title"] + " " + df["keyword"].str.lower()
).str.strip()

df["animal_type_v2"] = df["combined_text"].apply(detect_animal_v2)

# -----------------------------
# 6. Context / incident indicators
# -----------------------------
VIOLENCE_WORDS = [
    "kill", "killed", "killing", "dead", "death", "murder",
    "beat", "beaten", "torture", "tortured", "cruelty",
    "abuse", "abused", "poison", "poisoned", "toxic",
    "shoot", "shot", "burn", "burned", "attack"
]

DISEASE_WORDS = [
    "rabies", "disease", "infected", "infection", "sick",
    "ill", "virus", "vaccination", "vaccine", "vaccinated"
]

HUMAN_CONFLICT_WORDS = [
    "bite", "bitten", "attack", "attacked", "aggressive",
    "injury", "injured", "conflict", "nuisance", "fear",
    "protest", "relocation", "relocate"
]

RESCUE_WORDS = [
    "rescue", "rescued", "save", "saved", "shelter",
    "adopt", "adoption", "treatment", "treated", "rehabilitation"
]

POISON_WORDS = [
    "poison", "poisoned", "poisoning", "toxic", "toxicity"
]

def count_terms(text, terms):
    return sum(
        1 for term in terms
        if re.search(r"\b" + re.escape(term) + r"\b", text)
    )

df["violence_keyword_count"] = df["combined_text"].apply(
    lambda x: count_terms(x, VIOLENCE_WORDS)
)
df["disease_keyword_count"] = df["combined_text"].apply(
    lambda x: count_terms(x, DISEASE_WORDS)
)
df["human_conflict_keyword_count"] = df["combined_text"].apply(
    lambda x: count_terms(x, HUMAN_CONFLICT_WORDS)
)
df["rescue_keyword_count"] = df["combined_text"].apply(
    lambda x: count_terms(x, RESCUE_WORDS)
)
df["poison_keyword_count"] = df["combined_text"].apply(
    lambda x: count_terms(x, POISON_WORDS)
)

# Binary indicators
df["violence_indicator"] = (df["violence_keyword_count"] > 0).astype(int)
df["disease_indicator"] = (df["disease_keyword_count"] > 0).astype(int)
df["human_conflict_indicator"] = (
    df["human_conflict_keyword_count"] > 0
).astype(int)
df["rescue_indicator"] = (df["rescue_keyword_count"] > 0).astype(int)
df["poison_indicator"] = (df["poison_keyword_count"] > 0).astype(int)

# -----------------------------
# 7. Simple severity feature
# NOT the final SARI score
# -----------------------------
df["severity_feature"] = (
    2 * df["violence_keyword_count"]
    + 2 * df["poison_keyword_count"]
    + df["disease_keyword_count"]
    + df["human_conflict_keyword_count"]
)

# -----------------------------
# 8. Text metadata
# -----------------------------
df["title_word_count"] = df["clean_title"].str.split().str.len()
df["title_char_count"] = df["clean_title"].str.len()

# -----------------------------
# 9. Stable unique article ID
# -----------------------------
df = df.reset_index(drop=True)
df.insert(0, "article_id", range(1, len(df) + 1))

# -----------------------------
# 10. Select final column order
# -----------------------------
final_cols = [
    "article_id",
    "keyword",
    "title",
    "clean_title",
    "combined_text",
    "source",
    "link",
    "date",
    "parsed_date",
    "year",
    "month",
    "month_name",
    "day_of_week",
    "animal_type_v2",
    "violence_keyword_count",
    "disease_keyword_count",
    "human_conflict_keyword_count",
    "rescue_keyword_count",
    "poison_keyword_count",
    "violence_indicator",
    "disease_indicator",
    "human_conflict_indicator",
    "rescue_indicator",
    "poison_indicator",
    "severity_feature",
    "title_word_count",
    "title_char_count"
]

df = df[final_cols]

# -----------------------------
# 11. Save V2 master dataset
# -----------------------------
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("\n===== V2 PREPROCESSING COMPLETE =====")
print("Final rows:", len(df))
print("Final columns:", len(df.columns))
print("\nAnimal type distribution:")
print(df["animal_type_v2"].value_counts(dropna=False))

print("\nMissing parsed dates:", df["parsed_date"].isna().sum())
print("\nSaved:", OUTPUT_FILE)
