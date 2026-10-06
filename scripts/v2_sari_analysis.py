import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# STEP 9 — SARI V2 + TEMPORAL / RISK ANALYSIS
#
# Input:
#   data/v2_ml_predictions.csv
#
# Outputs:
#   data/v2_sari_scored.csv
#   results/sari_v2_summary.csv
#   visualization/sari_v2_distribution.png
#   visualization/yearly_incident_trend.png
#   visualization/incident_distribution_v2.png
#
# SARI V2 is an interpretable heuristic risk index, not an ML target.
# This avoids the V1 circularity problem where risk was derived almost
# directly from incident type.
# ============================================================

INPUT_FILE = "data/v2_ml_predictions.csv"
OUTPUT_FILE = "data/v2_sari_scored.csv"

Path("results").mkdir(exist_ok=True)
Path("visualization").mkdir(exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# ------------------------------------------------------------
# 1. Required/fallback columns
# ------------------------------------------------------------
if "ml_label" not in df.columns:
    raise ValueError("ml_label column not found.")

for col in [
    "violence_indicator",
    "poison_indicator",
    "disease_indicator",
    "human_conflict_indicator"
]:
    if col not in df.columns:
        df[col] = 0

for col in [
    "violence_keyword_count",
    "poison_keyword_count",
    "disease_keyword_count",
    "human_conflict_keyword_count"
]:
    if col not in df.columns:
        df[col] = 0

# ------------------------------------------------------------
# 2. SARI V2 components (0–10 total)
#
# A = abuse/violence severity       max 3.0
# H = human-conflict evidence       max 2.5
# S = sickness/rabies evidence      max 2.0
# P = poisoning evidence            max 1.5
# C = contextual incident severity  max 1.0
#
# The weights are transparent expert-designed heuristic weights.
# They are NOT empirically learned/calibrated clinical probabilities.
# ------------------------------------------------------------

df["sari_abuse_component"] = np.minimum(
    3.0,
    1.5 * df["violence_indicator"]
    + 0.5 * np.minimum(df["violence_keyword_count"], 3)
)

df["sari_human_component"] = np.minimum(
    2.5,
    1.5 * df["human_conflict_indicator"]
    + 0.5 * np.minimum(df["human_conflict_keyword_count"], 2)
)

df["sari_sickness_component"] = np.minimum(
    2.0,
    1.0 * df["disease_indicator"]
    + 0.5 * np.minimum(df["disease_keyword_count"], 2)
)

df["sari_poison_component"] = np.minimum(
    1.5,
    1.0 * df["poison_indicator"]
    + 0.5 * np.minimum(df["poison_keyword_count"], 1)
)

# Context severity from the incident label, intentionally low-weighted
# so SARI is not simply a renamed incident category.
context_weight = {
    "Poisoning": 1.0,
    "Killing": 1.0,
    "Abuse": 0.9,
    "Human-Animal Conflict": 0.8,
    "Disease": 0.6,
    "Conflict": 0.4,
    "Rescue": 0.1,
    "Other": 0.2
}
df["sari_context_component"] = (
    df["ml_label"].map(context_weight).fillna(0.2)
)

df["sari_v2_score"] = (
    df["sari_abuse_component"]
    + df["sari_human_component"]
    + df["sari_sickness_component"]
    + df["sari_poison_component"]
    + df["sari_context_component"]
).clip(0, 10).round(2)

def risk_level(score):
    if score >= 6.0:
        return "High"
    elif score >= 3.0:
        return "Medium"
    return "Low"

df["sari_v2_level"] = df["sari_v2_score"].apply(risk_level)

# ------------------------------------------------------------
# 3. Save scored dataset
# ------------------------------------------------------------
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

summary = (
    df.groupby("sari_v2_level")
      .agg(
          Articles=("v2_article_id", "count"),
          Mean_SARI=("sari_v2_score", "mean"),
          Median_SARI=("sari_v2_score", "median")
      )
      .reset_index()
)

summary["Percentage"] = (
    summary["Articles"] / len(df) * 100
).round(2)

summary.to_csv(
    "results/sari_v2_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# 4. SARI distribution figure
# ------------------------------------------------------------
order = ["Low", "Medium", "High"]
risk_counts = (
    df["sari_v2_level"]
    .value_counts()
    .reindex(order, fill_value=0)
)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(risk_counts.index, risk_counts.values)
ax.set_title("SARI V2 Risk-Level Distribution")
ax.set_xlabel("Risk Level")
ax.set_ylabel("Number of Articles")

for bar, value in zip(bars, risk_counts.values):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        value,
        str(value),
        ha="center",
        va="bottom"
    )

fig.tight_layout()
fig.savefig(
    "visualization/sari_v2_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# ------------------------------------------------------------
# 5. Incident distribution
# ------------------------------------------------------------
incident_counts = df["ml_label"].value_counts().sort_values()

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(incident_counts.index, incident_counts.values)
ax.set_title("Bangladesh V2 Incident-Type Distribution")
ax.set_xlabel("Number of Articles")
ax.set_ylabel("Incident Type")

for bar, value in zip(bars, incident_counts.values):
    ax.text(
        value,
        bar.get_y() + bar.get_height()/2,
        f" {value}",
        va="center"
    )

fig.tight_layout()
fig.savefig(
    "visualization/incident_distribution_v2.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# ------------------------------------------------------------
# 6. Temporal trend
# ------------------------------------------------------------
if "year" in df.columns:
    years = pd.to_numeric(df["year"], errors="coerce")
else:
    years = pd.Series(np.nan, index=df.index)

df["_year_numeric"] = years

yearly = (
    df.dropna(subset=["_year_numeric"])
      .assign(_year_numeric=lambda x: x["_year_numeric"].astype(int))
      .groupby("_year_numeric")
      .size()
)

if len(yearly) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(yearly.index, yearly.values, marker="o")
    ax.set_title("Bangladesh Animal-Welfare News Trend by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Articles")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(
        "visualization/yearly_incident_trend.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

df.drop(columns=["_year_numeric"], inplace=True, errors="ignore")

# ------------------------------------------------------------
# 7. Terminal summary
# ------------------------------------------------------------
print("===== SARI V2 ANALYSIS COMPLETE =====")
print("Articles scored:", len(df))

print("\nSARI V2 risk distribution:")
print(risk_counts)

print("\nSARI V2 descriptive statistics:")
print(df["sari_v2_score"].describe().round(3))

print("\nMean SARI V2 by incident type:")
print(
    df.groupby("ml_label")["sari_v2_score"]
      .mean()
      .sort_values(ascending=False)
      .round(3)
)

print("\nFiles saved:")
print("data/v2_sari_scored.csv")
print("results/sari_v2_summary.csv")
print("visualization/sari_v2_distribution.png")
print("visualization/incident_distribution_v2.png")
print("visualization/yearly_incident_trend.png")

print("\nIMPORTANT FOR PAPER:")
print("Describe SARI V2 as an interpretable heuristic risk index.")
print("Do not describe SARI V2 as a clinically validated or learned probability.")
