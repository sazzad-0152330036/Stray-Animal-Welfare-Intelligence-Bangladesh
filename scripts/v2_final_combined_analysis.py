import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# 1. PATH SETUP
# ==========================================================

SURVEY_FILE = "data/v2_survey_cleaned.csv"
ML_FILE = "data/v2_ml_training_dataset.csv"
SARI_FILE = "data/v2_sari_scored.csv"

MODEL_COMPARISON_FILE = "results/model_comparison.csv"

OUTPUT_KEY_METRICS = "results/v2_combined_key_metrics.csv"
OUTPUT_SUMMARY = "results/v2_combined_project_summary.txt"
OUTPUT_LOCATION = "results/v2_survey_location_intervention_indicators.csv"
OUTPUT_HEALTH_LOCATION = "results/v2_survey_health_by_location.csv"
OUTPUT_AGE_HEALTH = "results/v2_survey_age_health_crosstab.csv"

os.makedirs("results", exist_ok=True)
os.makedirs("visualization", exist_ok=True)


# ==========================================================
# 2. LOAD DATA
# ==========================================================

print("\n==============================================")
print("V2 FINAL COMBINED ANALYSIS")
print("==============================================")

survey = pd.read_csv(SURVEY_FILE)
ml_data = pd.read_csv(ML_FILE)
sari = pd.read_csv(SARI_FILE)
models = pd.read_csv(MODEL_COMPARISON_FILE)

print(f"\nSurvey rows         : {len(survey)}")
print(f"ML corpus rows      : {len(ml_data)}")
print(f"SARI scored rows    : {len(sari)}")
print(f"Models compared     : {len(models)}")


# ==========================================================
# 3. HELPER FUNCTION
# ==========================================================

def safe_percent(numerator, denominator):

    if denominator == 0:
        return 0.0

    return (
        numerator / denominator
    ) * 100


# ==========================================================
# 4. FIELD SURVEY BASIC METRICS
# ==========================================================

total_survey = len(survey)

dogs = (
    survey["animal_type"]
    .eq("Dog")
    .sum()
)

cats = (
    survey["animal_type"]
    .eq("Cat")
    .sum()
)

vaccinated = (
    survey["vaccinated"]
    .eq("Yes")
    .sum()
)

not_vaccinated = (
    survey["vaccinated"]
    .eq("No")
    .sum()
)

sterilized = (
    survey["sterilized"]
    .eq("Yes")
    .sum()
)

not_sterilized = (
    survey["sterilized"]
    .eq("No")
    .sum()
)

healthy = (
    survey["health_condition"]
    .eq("Healthy")
    .sum()
)

malnourished = (
    survey["health_condition"]
    .eq("Underweight / Malnourished")
    .sum()
)

starved = (
    survey["health_condition"]
    .eq("Starved")
    .sum()
)

health_concern = (
    malnourished + starved
)

vaccination_rate = safe_percent(
    vaccinated,
    total_survey
)

sterilization_rate = safe_percent(
    sterilized,
    total_survey
)

health_concern_rate = safe_percent(
    health_concern,
    total_survey
)


# ==========================================================
# 5. BEHAVIOUR METRICS
# ==========================================================

friendly = (
    survey["human_behavior"]
    .eq("Friendly")
    .sum()
)

scared = (
    survey["human_behavior"]
    .str.contains(
        "Scared",
        case=False,
        na=False
    )
    .sum()
)

aggressive = (
    survey["human_behavior"]
    .str.contains(
        "Aggressive",
        case=False,
        na=False
    )
    .sum()
)


# ==========================================================
# 6. RESIDENCY METRICS
# ==========================================================

resident_yes = (
    survey["resides_in_area"]
    .eq("Yes")
    .sum()
)

resident_no = (
    survey["resides_in_area"]
    .eq("No")
    .sum()
)

resident_unsure = (
    survey["resides_in_area"]
    .eq("Not Sure")
    .sum()
)


# ==========================================================
# 7. NEWS / ML METRICS
# ==========================================================

total_ml = len(ml_data)
total_sari = len(sari)

# Find best model using Macro F1
macro_f1_col = None

for col in models.columns:

    normalized = (
        col.lower()
        .replace(" ", "")
        .replace("_", "")
    )

    if normalized in [
        "macrof1",
        "macrof1score"
    ]:
        macro_f1_col = col
        break


if macro_f1_col is None:

    raise ValueError(
        "Macro F1 column could not be found "
        "inside results/model_comparison.csv"
    )


best_index = (
    models[macro_f1_col]
    .astype(float)
    .idxmax()
)

best_model_row = models.loc[best_index]


# Detect model-name column
model_name_col = None

for candidate in [
    "Model",
    "model",
    "Model Name",
    "model_name"
]:

    if candidate in models.columns:
        model_name_col = candidate
        break


if model_name_col is None:
    best_model_name = "Best Model"
else:
    best_model_name = str(
        best_model_row[model_name_col]
    )


best_macro_f1 = float(
    best_model_row[macro_f1_col]
)


# Weighted F1 if available
weighted_f1_col = None

for col in models.columns:

    normalized = (
        col.lower()
        .replace(" ", "")
        .replace("_", "")
    )

    if normalized in [
        "weightedf1",
        "weightedf1score"
    ]:
        weighted_f1_col = col
        break


if weighted_f1_col is not None:

    best_weighted_f1 = float(
        best_model_row[
            weighted_f1_col
        ]
    )

else:
    best_weighted_f1 = np.nan


# ==========================================================
# 8. SARI METRICS
# ==========================================================

SARI_SCORE_COL = "sari_v2_score"
SARI_LEVEL_COL = "sari_v2_level"

if SARI_SCORE_COL not in sari.columns:
    raise ValueError(
        f"Missing expected SARI score column: {SARI_SCORE_COL}"
    )

if SARI_LEVEL_COL not in sari.columns:
    raise ValueError(
        f"Missing expected SARI level column: {SARI_LEVEL_COL}"
    )


# Numeric SARI score
sari[SARI_SCORE_COL] = pd.to_numeric(
    sari[SARI_SCORE_COL],
    errors="coerce"
)

mean_sari = sari[SARI_SCORE_COL].mean()
max_sari = sari[SARI_SCORE_COL].max()


# Risk-level counts
risk_counts = (
    sari[SARI_LEVEL_COL]
    .astype(str)
    .str.strip()
    .value_counts()
)

low_risk = int(
    risk_counts.get("Low", 0)
)

medium_risk = int(
    risk_counts.get("Medium", 0)
)

high_risk = int(
    risk_counts.get("High", 0)
)

# ==========================================================
# 9. LOCATION-WISE FIELD ANALYSIS
# ==========================================================

locations = sorted(
    survey["location"]
    .dropna()
    .unique()
)

location_rows = []

for location in locations:

    temp = survey[
        survey["location"] == location
    ].copy()

    n = len(temp)

    vaccinated_n = (
        temp["vaccinated"]
        .eq("Yes")
        .sum()
    )

    sterilized_n = (
        temp["sterilized"]
        .eq("Yes")
        .sum()
    )

    healthy_n = (
        temp["health_condition"]
        .eq("Healthy")
        .sum()
    )

    malnourished_n = (
        temp["health_condition"]
        .eq(
            "Underweight / Malnourished"
        )
        .sum()
    )

    starved_n = (
        temp["health_condition"]
        .eq("Starved")
        .sum()
    )

    concern_n = (
        malnourished_n
        +
        starved_n
    )

    location_rows.append(
        {
            "location": location,
            "surveyed_animals": n,

            "vaccinated": vaccinated_n,

            "vaccination_rate_percent":
                safe_percent(
                    vaccinated_n,
                    n
                ),

            "sterilized": sterilized_n,

            "sterilization_rate_percent":
                safe_percent(
                    sterilized_n,
                    n
                ),

            "healthy": healthy_n,

            "underweight_malnourished":
                malnourished_n,

            "starved":
                starved_n,

            "health_concern_animals":
                concern_n,

            "health_concern_rate_percent":
                safe_percent(
                    concern_n,
                    n
                )
        }
    )


location_analysis = pd.DataFrame(
    location_rows
)


for col in [
    "vaccination_rate_percent",
    "sterilization_rate_percent",
    "health_concern_rate_percent"
]:

    location_analysis[col] = (
        location_analysis[col]
        .round(2)
    )


location_analysis.to_csv(
    OUTPUT_LOCATION,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================================
# 10. HEALTH CONDITION BY LOCATION
# ==========================================================

health_by_location = pd.crosstab(
    survey["location"],
    survey["health_condition"]
)

health_by_location.to_csv(
    OUTPUT_HEALTH_LOCATION,
    encoding="utf-8-sig"
)


# ==========================================================
# 11. AGE vs HEALTH CONDITION
# ==========================================================

age_health = pd.crosstab(
    survey["age_stage"],
    survey["health_condition"]
)

age_health.to_csv(
    OUTPUT_AGE_HEALTH,
    encoding="utf-8-sig"
)


# ==========================================================
# 12. KEY METRICS TABLE
# ==========================================================

key_metrics = pd.DataFrame(
    [
        {
            "data_source": "Field Survey",
            "metric": "Surveyed animals",
            "value": total_survey
        },

        {
            "data_source": "Field Survey",
            "metric": "Dogs",
            "value": dogs
        },

        {
            "data_source": "Field Survey",
            "metric": "Cats",
            "value": cats
        },

        {
            "data_source": "Field Survey",
            "metric": "Vaccination rate (%)",
            "value": round(
                vaccination_rate,
                2
            )
        },

        {
            "data_source": "Field Survey",
            "metric": "Sterilization rate (%)",
            "value": round(
                sterilization_rate,
                2
            )
        },

        {
            "data_source": "Field Survey",
            "metric": "Health concern rate (%)",
            "value": round(
                health_concern_rate,
                2
            )
        },

        {
            "data_source": "News Intelligence",
            "metric": "Final ML corpus",
            "value": total_ml
        },

        {
            "data_source": "News Intelligence",
            "metric": "Best model Macro F1",
            "value": round(
                best_macro_f1,
                4
            )
        },

        {
            "data_source": "News Intelligence",
            "metric": "Best model Weighted F1",
            "value": round(
                best_weighted_f1,
                4
            )
        },

        {
            "data_source": "News Intelligence",
            "metric": "Mean SARI V2",
            "value": round(
                mean_sari,
                3
            )
        },

        {
            "data_source": "News Intelligence",
            "metric": "Maximum SARI V2",
            "value": round(
                max_sari,
                3
            )
        }
    ]
)


key_metrics.to_csv(
    OUTPUT_KEY_METRICS,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================================
# 13. IDENTIFY DESCRIPTIVE LOCATION EXTREMES
# ==========================================================

highest_health_concern_location = (
    location_analysis
    .sort_values(
        "health_concern_rate_percent",
        ascending=False
    )
    .iloc[0]
)

lowest_vaccination_location = (
    location_analysis
    .sort_values(
        "vaccination_rate_percent",
        ascending=True
    )
    .iloc[0]
)

lowest_sterilization_location = (
    location_analysis
    .sort_values(
        "sterilization_rate_percent",
        ascending=True
    )
    .iloc[0]
)


# ==========================================================
# 14. SAVE FINAL TEXT SUMMARY
# ==========================================================

summary_text = f"""
V2 FINAL COMBINED PROJECT SUMMARY
=================================

IMPORTANT INTERPRETATION
------------------------
The news dataset and field survey dataset represent
different units of observation.

News unit:
One article / incident-related news record.

Survey unit:
One surveyed stray animal.

Therefore, they are NOT merged row-by-row.
They are integrated at the evidence and analytical level.


A. NEWS INTELLIGENCE
====================

Final ML corpus:
{total_ml} articles

Best incident classifier:
{best_model_name}

Macro F1:
{best_macro_f1:.4f}

Weighted F1:
{best_weighted_f1:.4f}

SARI V2 scored records:
{total_sari}

Mean SARI:
{mean_sari:.3f}

Maximum SARI:
{max_sari:.3f}

Risk distribution:
Low    : {low_risk}
Medium : {medium_risk}
High   : {high_risk}


B. FIELD SURVEY INTELLIGENCE
============================

Total surveyed animals:
{total_survey}

Dogs:
{dogs}

Cats:
{cats}

Vaccinated:
{vaccinated}

Vaccination rate:
{vaccination_rate:.2f}%

Sterilized:
{sterilized}

Sterilization rate:
{sterilization_rate:.2f}%

Healthy:
{healthy}

Underweight / Malnourished:
{malnourished}

Starved:
{starved}

Animals with observed nutritional health concern:
{health_concern}

Health concern rate:
{health_concern_rate:.2f}%


C. HUMAN BEHAVIOUR
==================

Friendly:
{friendly}

Scared-containing responses:
{scared}

Aggressive-containing responses:
{aggressive}


D. AREA RESIDENCY
=================

Yes:
{resident_yes}

No:
{resident_no}

Not Sure:
{resident_unsure}


E. LOCATION-WISE DESCRIPTIVE FINDINGS
=====================================

Highest observed health-concern rate:
{highest_health_concern_location["location"]}
({highest_health_concern_location["health_concern_rate_percent"]:.2f}%)

Lowest observed vaccination rate:
{lowest_vaccination_location["location"]}
({lowest_vaccination_location["vaccination_rate_percent"]:.2f}%)

Lowest observed sterilization rate:
{lowest_sterilization_location["location"]}
({lowest_sterilization_location["sterilization_rate_percent"]:.2f}%)


SCIENTIFIC CAUTION
==================

1. Survey findings describe the sampled Badda-area animals only.
2. They must not be interpreted as Bangladesh-wide prevalence.
3. Location comparisons are descriptive and sample sizes differ.
4. SARI V2 is an interpretable heuristic score, not a clinical
   or probabilistic risk estimate.
5. ML performance reflects predictive agreement with the
   project's weakly supervised / assisted-reviewed labels.
6. The field survey does not provide an independent validation
   benchmark for the news incident classifier.


FINAL V2 ARCHITECTURE
=====================

NEWS INTELLIGENCE
News collection
-> relevance filtering
-> Bangladesh scope filtering
-> weak supervision + assisted review
-> incident classification
-> SARI V2
-> explainability

PLUS

FIELD SURVEY INTELLIGENCE
Animal survey
-> privacy-safe cleaning
-> demographics
-> health condition
-> vaccination
-> sterilization
-> behaviour
-> location analysis

RESULT
------
Machine Learning-Driven Stray Animal Welfare
Intelligence Framework for Bangladesh
"""


with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    f.write(summary_text)


# ==========================================================
# 15. FIGURE SAVING FUNCTION
# ==========================================================

def save_plot(name):

    plt.tight_layout()

    plt.savefig(
        f"visualization/{name}.png",
        dpi=350,
        bbox_inches="tight"
    )

    plt.savefig(
        f"visualization/{name}.pdf",
        bbox_inches="tight"
    )

    plt.close()


# ==========================================================
# 16. FIGURE 1
# INTERVENTION COVERAGE BY LOCATION
# ==========================================================

coverage = (
    location_analysis[
        [
            "location",
            "vaccination_rate_percent",
            "sterilization_rate_percent"
        ]
    ]
    .set_index("location")
)

plt.figure(
    figsize=(10, 6)
)

coverage.plot(
    kind="bar"
)
plt.legend(
    ["Vaccination Rate", "Sterilization Rate"],
    title="Intervention Indicator"
)
plt.title(
    "Vaccination and Sterilization Coverage by Survey Location"
)

plt.xlabel(
    "Survey Location"
)

plt.ylabel(
    "Coverage (%)"
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.ylim(
    bottom=0
)

plt.tight_layout()

plt.savefig(
    "visualization/final_survey_intervention_coverage_by_location.png",
    dpi=350,
    bbox_inches="tight"
)

plt.savefig(
    "visualization/final_survey_intervention_coverage_by_location.pdf",
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# 17. FIGURE 2
# HEALTH CONCERN RATE BY LOCATION
# ==========================================================

health_rate = (
    location_analysis
    .sort_values(
        "health_concern_rate_percent",
        ascending=False
    )
)

plt.figure(
    figsize=(9, 5)
)

plt.bar(
    health_rate["location"],
    health_rate[
        "health_concern_rate_percent"
    ]
)

plt.title(
    "Observed Nutritional Health-Concern Rate by Location"
)

plt.xlabel(
    "Survey Location"
)

plt.ylabel(
    "Health-Concern Rate (%)"
)

plt.xticks(
    rotation=20,
    ha="right"
)

save_plot(
    "final_survey_health_concern_by_location"
)


# ==========================================================
# 18. FIGURE 3
# HEALTH STATUS BY LOCATION
# ==========================================================

health_plot = pd.crosstab(
    survey["location"],
    survey["health_condition"]
)

plt.figure(
    figsize=(10, 6)
)

health_plot.plot(
    kind="bar"
)
plt.legend(title="Health Condition")
plt.title(
    "Observed Health Status Across Survey Locations"
)

plt.xlabel(
    "Survey Location"
)

plt.ylabel(
    "Number of Animals"
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "visualization/final_survey_health_status_by_location.png",
    dpi=350,
    bbox_inches="tight"
)

plt.savefig(
    "visualization/final_survey_health_status_by_location.pdf",
    bbox_inches="tight"
)

plt.close()


# ==========================================================
# 19. FIGURE 4
# OVERALL FIELD INTERVENTION GAP
# ==========================================================

intervention_data = pd.Series(
    {
        "Vaccinated":
            vaccination_rate,

        "Not Vaccinated":
            safe_percent(
                not_vaccinated,
                total_survey
            ),

        "Sterilized":
            sterilization_rate,

        "Not Sterilized":
            safe_percent(
                not_sterilized,
                total_survey
            )
    }
)

plt.figure(
    figsize=(9, 5)
)

intervention_data.plot(
    kind="bar"
)

plt.title(
    "Field Survey Intervention Coverage and Gaps"
)

plt.ylabel(
    "Surveyed Animals (%)"
)

plt.xlabel(
    "Indicator"
)

plt.xticks(
    rotation=20,
    ha="right"
)

save_plot(
    "final_survey_intervention_gap"
)


# ==========================================================
# 20. FIGURE 5
# DUAL-SOURCE EVIDENCE OVERVIEW
# ==========================================================

labels = [
    "Final News\nML Corpus",
    "Field Survey\nAnimals"
]

values = [
    total_ml,
    total_survey
]

plt.figure(
    figsize=(8, 5)
)

bars = plt.bar(
    labels,
    values
)

plt.title(
    "Two Evidence Streams Used in the V2 Framework"
)

plt.ylabel(
    "Number of Analytical Records"
)

for bar, value in zip(
    bars,
    values
):

    plt.text(
        bar.get_x()
        +
        bar.get_width() / 2,

        bar.get_height(),

        str(value),

        ha="center",
        va="bottom"
    )

save_plot(
    "final_dual_source_evidence_overview"
)


# ==========================================================
# 21. TERMINAL OUTPUT
# ==========================================================

print("\n==============================================")
print("FINAL COMBINED ANALYSIS COMPLETE")
print("==============================================")

print("\nNEWS INTELLIGENCE")
print("-----------------")
print(f"Final ML Corpus    : {total_ml}")
print(f"Best Model         : {best_model_name}")
print(f"Macro F1           : {best_macro_f1:.4f}")
print(f"Weighted F1        : {best_weighted_f1:.4f}")
print(f"Mean SARI          : {mean_sari:.3f}")
print(f"Maximum SARI       : {max_sari:.3f}")

print("\nFIELD SURVEY")
print("------------")
print(f"Surveyed Animals   : {total_survey}")
print(f"Dogs               : {dogs}")
print(f"Cats               : {cats}")
print(f"Vaccination Rate   : {vaccination_rate:.2f}%")
print(f"Sterilization Rate : {sterilization_rate:.2f}%")
print(f"Health Concern     : {health_concern_rate:.2f}%")

print("\nLOCATION FINDINGS")
print("-----------------")

print(
    "Highest Health Concern :",
    highest_health_concern_location[
        "location"
    ],
    f'({highest_health_concern_location["health_concern_rate_percent"]:.2f}%)'
)

print(
    "Lowest Vaccination     :",
    lowest_vaccination_location[
        "location"
    ],
    f'({lowest_vaccination_location["vaccination_rate_percent"]:.2f}%)'
)

print(
    "Lowest Sterilization   :",
    lowest_sterilization_location[
        "location"
    ],
    f'({lowest_sterilization_location["sterilization_rate_percent"]:.2f}%)'
)

print("\nFiles created:")
print("1. results/v2_combined_key_metrics.csv")
print("2. results/v2_combined_project_summary.txt")
print("3. results/v2_survey_location_intervention_indicators.csv")
print("4. results/v2_survey_health_by_location.csv")
print("5. results/v2_survey_age_health_crosstab.csv")

print("\nNew final figures created:")
print("1. final_survey_intervention_coverage_by_location")
print("2. final_survey_health_concern_by_location")
print("3. final_survey_health_status_by_location")
print("4. final_survey_intervention_gap")
print("5. final_dual_source_evidence_overview")

print("\nBoth PNG and PDF versions were saved.")