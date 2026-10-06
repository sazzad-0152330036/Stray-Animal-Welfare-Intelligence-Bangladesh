import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# 1. PATH SETUP
# ==========================================================

INPUT_FILE = "data/survey_raw.csv"

CLEANED_FILE = "data/v2_survey_cleaned.csv"
SUMMARY_FILE = "results/v2_survey_summary.txt"
LOCATION_SUMMARY_FILE = "results/v2_survey_location_summary.csv"

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("visualization", exist_ok=True)


# ==========================================================
# 2. LOAD RAW SURVEY
# ==========================================================

df = pd.read_csv(INPUT_FILE)

print("\n========================================")
print("V2 FIELD SURVEY ANALYSIS")
print("========================================")
print(f"Raw rows    : {len(df)}")
print(f"Raw columns : {len(df.columns)}")


# ==========================================================
# 3. CLEAN COLUMN NAMES
# ==========================================================

# Remove extra spaces/new lines from Google Form column names
df.columns = [
    re.sub(r"\s+", " ", str(col)).strip()
    for col in df.columns
]


# ==========================================================
# 4. RENAME IMPORTANT COLUMNS
# ==========================================================

rename_map = {
    "Timestamp": "response_timestamp",

    "A.1 Animal Type (প্রাণী ধরণ)": "animal_type",
    "A.2 Name / নাম": "animal_name",
    "A.3 Gender (লিঙ্গ)": "gender",
    "A.4 Age (বয়স)": "age_stage",
    "A.5 Color & Fur Type (রঙ ও পশমের ধরন)": "color_fur_type",
    "A.6 Weight in kg (ওজন)": "weight_raw",
    "A.7 Body Marker (শারীরিক চিহ্ন)": "body_marker",

    "Location Marker (এলাকা)": "location",

    "A.8 Health Condition (স্বাস্থ্যের অবস্থা)": "health_condition",
    "A.9 Visible/ Known Health Issues (স্বাস্থ্য সমস্যা)": "health_issue_raw",
    "A.10 Behavior Towards Human (মানুষের প্রতি আচরণ)": "human_behavior",

    "Behavior Towards Other Animals – অন্য প্রাণীর প্রতি আচরণ":
        "other_animal_behavior",

    "Does the animal reside in the area? (প্রাণীটি এলাকায় নিয়মিত থাকে কি না?)":
        "resides_in_area",

    "B.1 Vaccinated? – ভ্যাক্সিন দেয়া হয়েছে?": "vaccinated",
    "B.2 Vaccine Date (ভ্যাক্সিনের তারিখ)": "vaccine_date",
    "B.3 Vaccine Details (ভ্যাক্সিনের বিবরণ)": "vaccine_details",

    "C.1 Sterilized? (বন্ধ্যা করা হয়েছে?)": "sterilized",
    "C.2 Sterilized Date (বন্ধ্যাত্বকরণ তারিখ)": "sterilized_date",

    "D.1 Disease Identified (রোগ শনাক্তকরণ)": "disease_identified",
    "Vet Check Done? – ভেট দেখা হয়েছে?": "vet_check_done",
    "Spot Treatment Given? – স্পট চিকিৎসা দেয়া হয়েছে?":
        "spot_treatment_given",

    "Transported to Shelter? – শেল্টারে পাঠানো হয়েছে?":
        "transported_to_shelter",

    "Medicine Provided – ঔষধের বিবরণ":
        "medicine_provided",

    "Survey Date (সার্ভে তারিখ)": "survey_date",
    "Add Image (ছবি যুক্ত করুন)": "image_url",
}

df = df.rename(columns=rename_map)


# ==========================================================
# 5. KEEP ONLY RESEARCH-SAFE ANALYTICAL COLUMNS
# ==========================================================
#
# We intentionally exclude:
# phone number
# email
# respondent address
# donor contact
# rescuer contact
# surveyor identity
#
# This protects privacy in the public/research dataset.
# ==========================================================

keep_columns = [
    "response_timestamp",
    "animal_type",
    "animal_name",
    "gender",
    "age_stage",
    "color_fur_type",
    "weight_raw",
    "body_marker",
    "location",
    "health_condition",
    "health_issue_raw",
    "human_behavior",
    "other_animal_behavior",
    "resides_in_area",
    "vaccinated",
    "vaccine_date",
    "vaccine_details",
    "sterilized",
    "sterilized_date",
    "disease_identified",
    "vet_check_done",
    "spot_treatment_given",
    "transported_to_shelter",
    "medicine_provided",
    "survey_date",
    "image_url",
]

# Keep only columns that actually exist
keep_columns = [c for c in keep_columns if c in df.columns]

clean = df[keep_columns].copy()


# ==========================================================
# 6. BASIC TEXT CLEANING
# ==========================================================

def clean_text(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    # Remove multiple spaces
    value = re.sub(r"\s+", " ", value)

    if value == "":
        return np.nan

    return value


for col in clean.columns:
    if clean[col].dtype == "object":
        clean[col] = clean[col].apply(clean_text)


# ==========================================================
# 7. NORMALIZE ANIMAL TYPE
# ==========================================================

def normalize_animal(value):
    if pd.isna(value):
        return "Unknown"

    text = str(value).lower()

    if "dog" in text:
        return "Dog"

    if "cat" in text:
        return "Cat"

    return "Other"


if "animal_type" in clean.columns:
    clean["animal_type"] = clean["animal_type"].apply(
        normalize_animal
    )


# ==========================================================
# 8. NORMALIZE GENDER
# ==========================================================

def normalize_gender(value):
    if pd.isna(value):
        return "Unknown"

    text = str(value).lower()

    if "male" in text and "female" in text:
        return "Mixed / Multiple"

    if "female" in text:
        return "Female"

    if "male" in text:
        return "Male"

    return "Unknown"


if "gender" in clean.columns:
    clean["gender"] = clean["gender"].apply(
        normalize_gender
    )


# ==========================================================
# 9. NORMALIZE LOCATION
# ==========================================================

def normalize_location(value):
    if pd.isna(value):
        return "Unknown"

    text = str(value).strip().lower()

    location_map = {
        "uttar badda": "Uttar Badda",
        "dokkhin badda": "Dokkhin Badda",
        "moddho badda": "Moddho Badda",
        "jheelpar": "Jheelpar",
    }

    return location_map.get(text, str(value).strip())


if "location" in clean.columns:
    clean["location"] = clean["location"].apply(
        normalize_location
    )


# ==========================================================
# 10. NORMALIZE YES / NO FIELDS
# ==========================================================

def normalize_yes_no(value):
    if pd.isna(value):
        return "Unknown"

    text = str(value).lower().strip()

    if text == "yes":
        return "Yes"

    if text == "no":
        return "No"

    if "not sure" in text:
        return "Not Sure"

    if "yes" in text and "no" in text:
        return "Mixed"

    return str(value).strip()


for col in ["vaccinated", "sterilized", "resides_in_area"]:
    if col in clean.columns:
        clean[col] = clean[col].apply(
            normalize_yes_no
        )


# ==========================================================
# 11. NORMALIZE HEALTH CONDITION
# ==========================================================

def normalize_health_condition(value):
    if pd.isna(value):
        return "Unknown"

    text = str(value).lower().strip()

    if "healthy" in text:
        return "Healthy"

    if "under" in text or "malnour" in text:
        return "Underweight / Malnourished"

    if "starv" in text:
        return "Starved"

    return str(value).strip()


if "health_condition" in clean.columns:
    clean["health_condition"] = clean[
        "health_condition"
    ].apply(normalize_health_condition)


# ==========================================================
# 12. NORMALIZE HUMAN BEHAVIOR
# ==========================================================

def normalize_behavior(value):
    if pd.isna(value):
        return "Unknown"

    text = str(value).lower()

    friendly = "friendly" in text
    scared = "scared" in text
    aggressive = "aggressive" in text

    if friendly and scared:
        return "Friendly + Scared"

    if aggressive and friendly:
        return "Aggressive + Friendly"

    if aggressive:
        return "Aggressive"

    if scared:
        return "Scared"

    if friendly:
        return "Friendly"

    return str(value).strip()


if "human_behavior" in clean.columns:
    clean["human_behavior"] = clean[
        "human_behavior"
    ].apply(normalize_behavior)


# ==========================================================
# 13. WEIGHT CLEANING
# ==========================================================
#
# Examples in your dataset:
# 25-30
# 20-25 kg
# Approximately 15-20 kg
# 20+
# Around 20 kg
# Under 10 kg
# 3.4 kg
# Bengali digits such as ২০ থেকে ২৫
#
# We create:
# weight_min_kg
# weight_max_kg
# weight_mid_kg
# weight_parse_status
# ==========================================================


BANGLA_DIGITS = str.maketrans(
    "০১২৩৪৫৬৭৮৯",
    "0123456789"
)


def parse_weight(value):

    if pd.isna(value):
        return pd.Series(
            [np.nan, np.nan, np.nan, "Missing"]
        )

    original = str(value)

    text = original.translate(BANGLA_DIGITS)
    text = text.lower().strip()

    text = text.replace("kgs", "")
    text = text.replace("kg", "")
    text = text.replace("kh", "")

    text = text.replace("approximately", "")
    text = text.replace("approx.", "")
    text = text.replace("approx", "")
    text = text.replace("around", "")

    text = text.replace("_", "-")
    text = text.replace("--", "-")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("\t", "")

    text = text.replace(" থেকে ", "-")
    text = text.replace(" to ", "-")

    text = re.sub(r"\s+", "", text)


    # --------------------------
    # Under X
    # --------------------------

    under_match = re.search(
        r"under(\d+(?:\.\d+)?)",
        text
    )

    if under_match:
        maximum = float(under_match.group(1))

        return pd.Series(
            [
                np.nan,
                maximum,
                np.nan,
                "Upper bound only"
            ]
        )


    # --------------------------
    # X+
    # --------------------------

    plus_match = re.search(
        r"(\d+(?:\.\d+)?)\+",
        text
    )

    if plus_match:
        minimum = float(plus_match.group(1))

        return pd.Series(
            [
                minimum,
                np.nan,
                np.nan,
                "Lower bound only"
            ]
        )


    # --------------------------
    # Standard Range X-Y
    # --------------------------

    range_match = re.search(
        r"(\d+(?:\.\d+)?)"
        r"-"
        r"(\d+(?:\.\d+)?)",
        text
    )

    if range_match:

        a = float(range_match.group(1))
        b = float(range_match.group(2))

        # Some survey values were reversed,
        # e.g. 35-30 or 25-20
        minimum = min(a, b)
        maximum = max(a, b)

        midpoint = (
            minimum + maximum
        ) / 2

        return pd.Series(
            [
                minimum,
                maximum,
                midpoint,
                "Range"
            ]
        )


    # --------------------------
    # Single number
    # --------------------------

    number_match = re.search(
        r"(\d+(?:\.\d+)?)",
        text
    )

    if number_match:

        number = float(
            number_match.group(1)
        )

        return pd.Series(
            [
                number,
                number,
                number,
                "Exact / Approximate single value"
            ]
        )


    # --------------------------
    # Failed parsing
    # --------------------------

    return pd.Series(
        [
            np.nan,
            np.nan,
            np.nan,
            "Unparsed"
        ]
    )


if "weight_raw" in clean.columns:

    clean[
        [
            "weight_min_kg",
            "weight_max_kg",
            "weight_mid_kg",
            "weight_parse_status"
        ]
    ] = clean["weight_raw"].apply(
        parse_weight
    )


# ==========================================================
# 14. HEALTH ISSUE NORMALIZATION
# ==========================================================

NO_ISSUE_TERMS = [
    "no",
    "none",
    "no issue",
    "no issues",
    "na",
    "n/a",
    "nothing",
    "healthy"
]


def classify_health_issue(value):

    if pd.isna(value):
        return "Not Reported"

    text = str(value).lower().strip()

    text_clean = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text_clean = re.sub(
        r"\s+",
        " ",
        text_clean
    ).strip()


    if text_clean in NO_ISSUE_TERMS:
        return "No Visible Issue"


    if any(
        word in text
        for word in [
            "skin",
            "fung",
            "mange",
            "rash",
            "itch"
        ]
    ):
        return "Skin Condition"


    if any(
        word in text
        for word in [
            "scratch",
            "wound",
            "injury",
            "injured",
            "cut",
            "bleed"
        ]
    ):
        return "Injury / Wound"


    if any(
        word in text
        for word in [
            "leg",
            "limp",
            "fracture",
            "broken"
        ]
    ):
        return "Mobility / Limb Issue"


    if any(
        word in text
        for word in [
            "eye",
            "eyes"
        ]
    ):
        return "Eye Issue"


    if any(
        word in text
        for word in [
            "ear",
            "ears"
        ]
    ):
        return "Ear Issue"


    return "Other Reported Issue"


if "health_issue_raw" in clean.columns:

    clean["health_issue_group"] = clean[
        "health_issue_raw"
    ].apply(classify_health_issue)


# ==========================================================
# 15. DATE CLEANING
# ==========================================================

for col in [
    "response_timestamp",
    "survey_date",
    "vaccine_date",
    "sterilized_date"
]:

    if col in clean.columns:

        clean[col] = pd.to_datetime(
            clean[col],
            errors="coerce"
        )


# ==========================================================
# 16. REMOVE COMPLETELY EMPTY COLUMNS
# ==========================================================

clean = clean.dropna(
    axis=1,
    how="all"
)


# ==========================================================
# 17. CREATE UNIQUE FIELD ID
# ==========================================================

clean.insert(
    0,
    "field_id",
    [
        f"BP-{i:04d}"
        for i in range(
            1,
            len(clean) + 1
        )
    ]
)


# ==========================================================
# 18. SAVE CLEANED DATA
# ==========================================================

clean.to_csv(
    CLEANED_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nCleaned dataset saved:")
print(CLEANED_FILE)


# ==========================================================
# 19. MAIN SUMMARY STATISTICS
# ==========================================================

total_animals = len(clean)

dog_count = (
    clean["animal_type"]
    .eq("Dog")
    .sum()
)

cat_count = (
    clean["animal_type"]
    .eq("Cat")
    .sum()
)


vaccinated_yes = (
    clean["vaccinated"]
    .eq("Yes")
    .sum()
)

sterilized_yes = (
    clean["sterilized"]
    .eq("Yes")
    .sum()
)


vaccination_rate = (
    vaccinated_yes
    / total_animals
    * 100
)

sterilization_rate = (
    sterilized_yes
    / total_animals
    * 100
)


healthy_count = (
    clean["health_condition"]
    .eq("Healthy")
    .sum()
)

malnourished_count = (
    clean["health_condition"]
    .eq(
        "Underweight / Malnourished"
    )
    .sum()
)

starved_count = (
    clean["health_condition"]
    .eq("Starved")
    .sum()
)


# ==========================================================
# 20. LOCATION SUMMARY
# ==========================================================

location_summary = pd.DataFrame()

location_summary["animals"] = (
    clean.groupby("location")
    .size()
)

location_summary["vaccinated"] = (
    clean[
        clean["vaccinated"] == "Yes"
    ]
    .groupby("location")
    .size()
)

location_summary["sterilized"] = (
    clean[
        clean["sterilized"] == "Yes"
    ]
    .groupby("location")
    .size()
)

location_summary["healthy"] = (
    clean[
        clean["health_condition"]
        == "Healthy"
    ]
    .groupby("location")
    .size()
)

location_summary["malnourished"] = (
    clean[
        clean["health_condition"]
        == "Underweight / Malnourished"
    ]
    .groupby("location")
    .size()
)

location_summary["starved"] = (
    clean[
        clean["health_condition"]
        == "Starved"
    ]
    .groupby("location")
    .size()
)


location_summary = (
    location_summary
    .fillna(0)
    .astype(int)
)


location_summary[
    "vaccination_rate_percent"
] = (
    location_summary["vaccinated"]
    /
    location_summary["animals"]
    * 100
).round(2)


location_summary[
    "sterilization_rate_percent"
] = (
    location_summary["sterilized"]
    /
    location_summary["animals"]
    * 100
).round(2)


location_summary.to_csv(
    LOCATION_SUMMARY_FILE,
    encoding="utf-8-sig"
)


# ==========================================================
# 21. SAVE TEXT SUMMARY
# ==========================================================

summary_text = f"""
V2 FIELD SURVEY SUMMARY
=======================

Total Surveyed Animals : {total_animals}

Animal Type
-----------
Dogs : {dog_count}
Cats : {cat_count}

Vaccination
-----------
Vaccinated : {vaccinated_yes}
Vaccination Rate : {vaccination_rate:.2f}%

Sterilization
-------------
Sterilized : {sterilized_yes}
Sterilization Rate : {sterilization_rate:.2f}%

Health Condition
----------------
Healthy : {healthy_count}
Underweight / Malnourished : {malnourished_count}
Starved : {starved_count}

Locations
---------
{clean["location"].value_counts().to_string()}

Age Distribution
----------------
{clean["age_stage"].value_counts().to_string()}

Human Behaviour
---------------
{clean["human_behavior"].value_counts().to_string()}

Weight Parsing
--------------
{clean["weight_parse_status"].value_counts().to_string()}
"""


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(summary_text)


print("\nSummary saved:")
print(SUMMARY_FILE)


# ==========================================================
# 22. VISUALIZATION FUNCTION
# ==========================================================

def save_figure(filename):

    plt.tight_layout()

    plt.savefig(
        f"visualization/{filename}.png",
        dpi=350,
        bbox_inches="tight"
    )

    plt.savefig(
        f"visualization/{filename}.pdf",
        bbox_inches="tight"
    )

    plt.close()


# ==========================================================
# 23. FIGURE 1 — LOCATION DISTRIBUTION
# ==========================================================

location_counts = (
    clean["location"]
    .value_counts()
)

plt.figure(figsize=(9, 5))

location_counts.plot(
    kind="bar"
)

plt.title(
    "Distribution of Surveyed Stray Animals by Location"
)

plt.xlabel("Location")
plt.ylabel("Number of Animals")

plt.xticks(
    rotation=20,
    ha="right"
)

save_figure(
    "survey_location_distribution"
)


# ==========================================================
# 24. FIGURE 2 — HEALTH CONDITION
# ==========================================================

health_counts = (
    clean["health_condition"]
    .value_counts()
)

plt.figure(figsize=(8, 5))

health_counts.plot(
    kind="bar"
)

plt.title(
    "Observed Health Condition of Surveyed Animals"
)

plt.xlabel(
    "Health Condition"
)

plt.ylabel(
    "Number of Animals"
)

plt.xticks(
    rotation=20,
    ha="right"
)

save_figure(
    "survey_health_condition"
)


# ==========================================================
# 25. FIGURE 3 — VACCINATION STATUS
# ==========================================================

vaccination_counts = (
    clean["vaccinated"]
    .value_counts()
)

plt.figure(figsize=(7, 5))

vaccination_counts.plot(
    kind="bar"
)

plt.title(
    "Vaccination Status of Surveyed Animals"
)

plt.xlabel(
    "Vaccination Status"
)

plt.ylabel(
    "Number of Animals"
)

plt.xticks(
    rotation=0
)

save_figure(
    "survey_vaccination_status"
)


# ==========================================================
# 26. FIGURE 4 — STERILIZATION STATUS
# ==========================================================

sterilization_counts = (
    clean["sterilized"]
    .value_counts()
)

plt.figure(figsize=(7, 5))

sterilization_counts.plot(
    kind="bar"
)

plt.title(
    "Sterilization Status of Surveyed Animals"
)

plt.xlabel(
    "Sterilization Status"
)

plt.ylabel(
    "Number of Animals"
)

plt.xticks(
    rotation=0
)

save_figure(
    "survey_sterilization_status"
)


# ==========================================================
# 27. FIGURE 5 — AGE DISTRIBUTION
# ==========================================================

age_counts = (
    clean["age_stage"]
    .value_counts()
)

plt.figure(figsize=(10, 5))

age_counts.plot(
    kind="bar"
)

plt.title(
    "Age-Stage Distribution of Surveyed Animals"
)

plt.xlabel(
    "Age Stage"
)

plt.ylabel(
    "Number of Animals"
)

plt.xticks(
    rotation=30,
    ha="right"
)

save_figure(
    "survey_age_distribution"
)


# ==========================================================
# 28. FIGURE 6 — HUMAN BEHAVIOUR
# ==========================================================

behavior_counts = (
    clean["human_behavior"]
    .value_counts()
)

plt.figure(figsize=(9, 5))

behavior_counts.plot(
    kind="bar"
)

plt.title(
    "Observed Behaviour Towards Humans"
)

plt.xlabel(
    "Behaviour"
)

plt.ylabel(
    "Number of Animals"
)

plt.xticks(
    rotation=25,
    ha="right"
)

save_figure(
    "survey_human_behavior"
)


# ==========================================================
# 29. FIGURE 7 — VACCINATION RATE BY LOCATION
# ==========================================================

location_rate = (
    location_summary[
        "vaccination_rate_percent"
    ]
    .sort_values(
        ascending=False
    )
)

plt.figure(figsize=(9, 5))

location_rate.plot(
    kind="bar"
)

plt.title(
    "Vaccination Rate by Survey Location"
)

plt.xlabel(
    "Location"
)

plt.ylabel(
    "Vaccinated Animals (%)"
)

plt.xticks(
    rotation=20,
    ha="right"
)

save_figure(
    "survey_vaccination_rate_by_location"
)


# ==========================================================
# 30. FIGURE 8 — STERILIZATION RATE BY LOCATION
# ==========================================================

sterilization_location_rate = (
    location_summary[
        "sterilization_rate_percent"
    ]
    .sort_values(
        ascending=False
    )
)

plt.figure(figsize=(9, 5))

sterilization_location_rate.plot(
    kind="bar"
)

plt.title(
    "Sterilization Rate by Survey Location"
)

plt.xlabel(
    "Location"
)

plt.ylabel(
    "Sterilized Animals (%)"
)

plt.xticks(
    rotation=20,
    ha="right"
)

save_figure(
    "survey_sterilization_rate_by_location"
)


# ==========================================================
# 31. FIGURE 9 — HEALTH ISSUE GROUPS
# ==========================================================

health_issue_counts = (
    clean["health_issue_group"]
    .value_counts()
)

plt.figure(figsize=(10, 5))

health_issue_counts.plot(
    kind="bar"
)

plt.title(
    "Reported Visible Health-Issue Categories"
)

plt.xlabel(
    "Health Issue Group"
)

plt.ylabel(
    "Number of Animals"
)

plt.xticks(
    rotation=30,
    ha="right"
)

save_figure(
    "survey_visible_health_issues"
)


# ==========================================================
# 32. FINAL TERMINAL OUTPUT
# ==========================================================

print("\n========================================")
print("FIELD SURVEY ANALYSIS COMPLETE")
print("========================================")

print(f"""
Total Animals       : {total_animals}
Dogs                : {dog_count}
Cats                : {cat_count}

Vaccinated          : {vaccinated_yes}
Vaccination Rate    : {vaccination_rate:.2f}%

Sterilized          : {sterilized_yes}
Sterilization Rate  : {sterilization_rate:.2f}%

Healthy             : {healthy_count}
Malnourished        : {malnourished_count}
Starved             : {starved_count}
""")

print("Created files:")
print("1. data/v2_survey_cleaned.csv")
print("2. results/v2_survey_summary.txt")
print("3. results/v2_survey_location_summary.csv")

print("\nFigures created in visualization/ folder.")
print("Both PNG and PDF versions were generated.")