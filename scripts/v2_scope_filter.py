import pandas as pd
import re
from pathlib import Path

# ============================================================
# STEP 6 — BANGLADESH + STRAY-ANIMAL SCOPE FILTER
#
# Input:
#   data/v2_expanded_master_dataset.csv
#
# Outputs:
#   data/v2_bangladesh_primary.csv
#   data/v2_foreign_background.csv
#   data/v2_out_of_scope.csv
#   data/v2_scope_audit.csv
#
# IMPORTANT:
# - Search keyword is NOT used to decide Bangladesh relevance.
# - We only use article title/description/source.
# ============================================================

INPUT_FILE = "data/v2_expanded_master_dataset.csv"

PRIMARY_FILE = "data/v2_bangladesh_primary.csv"
BACKGROUND_FILE = "data/v2_foreign_background.csv"
OUTSCOPE_FILE = "data/v2_out_of_scope.csv"
AUDIT_FILE = "data/v2_scope_audit.csv"

Path("data").mkdir(exist_ok=True)

df = pd.read_csv(INPUT_FILE)

required = ["v2_article_id", "title", "source"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if "description" not in df.columns:
    df["description"] = ""

for col in ["title", "description", "source"]:
    df[col] = df[col].fillna("").astype(str).str.strip()

df["scope_text"] = (
    df["title"] + " " + df["description"]
).str.lower().str.replace(r"\s+", " ", regex=True)

# ------------------------------------------------------------
# 1. Bangladesh geographic markers
# ------------------------------------------------------------
BANGLADESH_TERMS = [
    "bangladesh", "bangladeshi",
    "dhaka", "dacca",
    "chattogram", "chittagong",
    "gazipur", "narayanganj", "cumilla", "comilla",
    "rajshahi", "khulna", "sylhet", "barishal", "barisal",
    "rangpur", "mymensingh", "cox's bazar", "cox s bazar",
    "tangail", "savar", "keraniganj", "uttara", "mirpur",
    "mohammadpur", "badda", "banani", "gulshan",
    "purbachal", "narshingdi", "narsingdi",
    "bogura", "bogra", "dinajpur", "jashore", "jeshore",
    "kushtia", "feni", "noakhali", "brahmanbaria",
    "habiganj", "moulvibazar", "sunamganj",
    "patuakhali", "bhola", "barguna", "jhalokathi",
    "satkhira", "chuadanga", "jhenaidah", "magura",
    "meherpur", "sirajganj", "pabna", "natore",
    "naogaon", "chapainawabganj", "joypurhat",
    "gaibandha", "kurigram", "lalmonirhat",
    "nilphamari", "thakurgaon", "panchagarh",
    "jamalpur", "netrokona", "sherpur",
    "munshiganj", "manikganj", "faridpur",
    "gopalganj", "madaripur", "rajbari", "shariatpur"
]

# Bangladesh media/source names that can serve as a secondary location signal.
# This is only used when the article is already animal-relevant.
BANGLADESH_SOURCES = [
    "the daily star",
    "dhaka tribune",
    "the business standard",
    "the financial express",
    "bangladesh sangbad sangstha",
    "bss",
    "prothom alo",
    "new age",
    "the daily observer",
    "daily sun",
    "unb",
    "united news of bangladesh",
    "bdnews24",
    "tbs news",
    "the report",
    "views bangladesh"
]

# ------------------------------------------------------------
# 2. Core project scope markers
# ------------------------------------------------------------
STRAY_DOMESTIC_TERMS = [
    "stray dog", "stray dogs",
    "street dog", "street dogs",
    "free roaming dog", "free-roaming dog",
    "dog", "dogs", "puppy", "puppies",
    "stray cat", "stray cats",
    "street cat", "street cats",
    "cat", "cats", "kitten", "kittens",
    "animal welfare", "animal cruelty", "animal abuse",
    "animal rights", "animal bite", "rabies"
]

# Wildlife/zoo stories are not the primary stray-dog/cat incident dataset.
WILDLIFE_TERMS = [
    "tiger", "elephant", "leopard", "deer", "monkey",
    "crocodile", "snake", "python", "dolphin", "whale",
    "shark", "bird", "birds", "eagle", "vulture",
    "bear", "rhino", "rhinoceros", "zebra", "giraffe",
    "lion", "hippo", "wildlife", "zoo", "marine animal"
]

DOG_CAT_TERMS = [
    "dog", "dogs", "puppy", "puppies",
    "cat", "cats", "kitten", "kittens",
    "stray", "street dog", "street cat"
]

# ------------------------------------------------------------
# 3. Helpers
# ------------------------------------------------------------
def contains_any(text, terms):
    text = str(text).lower()
    for term in terms:
        if re.search(r"\b" + re.escape(term) + r"\b", text):
            return True
    return False

def get_geo_signal(row):
    text = row["scope_text"]
    source = row["source"].lower()

    explicit_geo = contains_any(text, BANGLADESH_TERMS)
    bd_source = contains_any(source, BANGLADESH_SOURCES)

    if explicit_geo:
        return "Explicit Bangladesh/location mention"
    elif bd_source:
        return "Bangladesh news source"
    else:
        return "No Bangladesh signal"

def is_project_scope(text):
    return contains_any(text, STRAY_DOMESTIC_TERMS)

def is_wildlife_only(text):
    has_wildlife = contains_any(text, WILDLIFE_TERMS)
    has_dog_cat = contains_any(text, DOG_CAT_TERMS)

    # If article has both wildlife and dog/cat terms, keep it for later review.
    return has_wildlife and not has_dog_cat

# ------------------------------------------------------------
# 4. Classify scope
# ------------------------------------------------------------
df["geo_signal"] = df.apply(get_geo_signal, axis=1)
df["project_scope_match"] = df["scope_text"].apply(is_project_scope)
df["wildlife_only"] = df["scope_text"].apply(is_wildlife_only)

def classify_scope(row):
    if row["wildlife_only"] or not row["project_scope_match"]:
        return "OUT_OF_SCOPE"

    if row["geo_signal"] != "No Bangladesh signal":
        return "BANGLADESH_PRIMARY"

    return "FOREIGN_OR_BACKGROUND"

df["scope_class"] = df.apply(classify_scope, axis=1)

# ------------------------------------------------------------
# 5. Save all groups
# ------------------------------------------------------------
primary = df[df["scope_class"] == "BANGLADESH_PRIMARY"].copy()
background = df[df["scope_class"] == "FOREIGN_OR_BACKGROUND"].copy()
outscope = df[df["scope_class"] == "OUT_OF_SCOPE"].copy()

primary.to_csv(PRIMARY_FILE, index=False, encoding="utf-8-sig")
background.to_csv(BACKGROUND_FILE, index=False, encoding="utf-8-sig")
outscope.to_csv(OUTSCOPE_FILE, index=False, encoding="utf-8-sig")

audit_cols = [
    "v2_article_id",
    "title",
    "source",
    "geo_signal",
    "project_scope_match",
    "wildlife_only",
    "scope_class"
]
df[audit_cols].to_csv(AUDIT_FILE, index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 6. Summary
# ------------------------------------------------------------
print("===== V2 BANGLADESH + SCOPE FILTER COMPLETE =====")
print("Input relevant rows:", len(df))
print("Bangladesh primary:", len(primary))
print("Foreign/background:", len(background))
print("Out of scope:", len(outscope))

if len(df) > 0:
    print(f"\nBangladesh-primary share: {len(primary)/len(df)*100:.2f}%")

print("\nBangladesh evidence:")
print(primary["geo_signal"].value_counts())

if "animal_type_v2" in primary.columns:
    print("\nAnimal type in Bangladesh-primary dataset:")
    print(primary["animal_type_v2"].value_counts(dropna=False))

if "year" in primary.columns and primary["year"].notna().any():
    print("\nBangladesh-primary year range:")
    print(int(primary["year"].min()), "to", int(primary["year"].max()))

print("\nUnique Bangladesh-primary sources:", primary["source"].nunique())

print("\nSaved:")
print(PRIMARY_FILE)
print(BACKGROUND_FILE)
print(OUTSCOPE_FILE)
print(AUDIT_FILE)

print("\nIMPORTANT:")
print("Use v2_bangladesh_primary.csv for Bangladesh incident statistics.")
print("Do not mix foreign/background rows into Bangladesh-specific counts.")
