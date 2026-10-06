# A Machine Learning-Driven Stray Animal Welfare Intelligence Framework for Bangladesh

## Overview
This project presents a research prototype for Bangladesh-focused stray animal welfare intelligence. It combines news-derived incident intelligence with a localized field survey pilot from Badda, Dhaka.

The framework has two evidence streams:

1. **News Intelligence** — news collection, relevance filtering, Bangladesh scope filtering, weak-label preparation, assisted review, incident classification, and SARI V2 prioritization.
2. **Field Intelligence** — observational survey analysis of stray-animal health condition, vaccination, sterilization, human-animal behavior, and location-level intervention indicators.

A Streamlit dashboard integrates the final outputs for exploration, live incident analysis, heuristic prioritization, evidence-consistency checking, decision support, and prototype Gmail alerts.

> **Research scope:** Bangladesh is the overall framework/news scope. The field survey is a localized Badda, Dhaka pilot and is not nationally representative.

## Project Title
**A Machine Learning-Driven Stray Animal Welfare Intelligence Framework for Bangladesh**

## Core Pipeline
```text
Google News RSS
    ↓
Expanded news collection
    ↓
Relevance filtering
    ↓
Bangladesh scope filtering
    ↓
Weak labels + assisted review
    ↓
Final ML corpus
    ↓
TF-IDF + incident classifiers
    ↓
Linear SVM (selected model)
    ↓
SARI V2 heuristic prioritization
    ↓
Evidence-consistency layer
    ↓
Streamlit dashboard + prototype email alert

Badda field survey
    ↓
Cleaning and descriptive analysis
    ↓
Health / vaccination / sterilization / behavior / location indicators
    ↓
Field intelligence dashboard
```

## Key Data Results
### News pipeline
- Expanded collected news records: **1,094**
- Relevant records after relevance filtering: **690**
- Bangladesh-primary records: **250**
- Final ML corpus after assisted-review exclusions/ambiguities: **226**
- Bangladesh-primary sources: **48**
- Expanded unique sources: **260**

### Assisted review
- Reviewed sample: **64**
- KEEP: **40**
- EXCLUDE: **22**
- AMBIGUOUS: **2**
- Suggested-label agreement among KEEP records: **90.0%**

This 90.0% value is **not model accuracy**. It is agreement with the suggested weak label among reviewed records retained as KEEP.

## Machine Learning
Three incident classifiers were compared using stratified **2-fold cross-validation** because the rarest class had only two examples.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Linear SVM | 0.7611 | **0.6068** | 0.7517 |
| Logistic Regression | 0.7257 | 0.5944 | 0.7206 |
| Complement Naive Bayes | 0.7389 | 0.5328 | 0.7282 |

**Selected model:** Linear SVM, based on the best Macro F1 score.

The evaluation measures predictive agreement with the weakly supervised and assisted-reviewed labels. It is **not an independent expert-annotated benchmark**.

## SARI V2
SARI V2 is a separate, interpretable heuristic prioritization index on a **0–10** scale. It is not the ML target and is not a probability of harm.

Final corpus distribution:
- Low: **191**
- Medium: **35**
- High: **0**
- Mean score: **1.550**
- Maximum score: **5.500**

Thresholds:
- **Low:** score < 3
- **Medium:** 3 ≤ score < 6
- **High:** score ≥ 6

The absence of High cases is an observed result under the current heuristic weights and dataset; it was not manually altered.

## Field Survey Pilot
The field dataset contains **325 observed animals** from Badda, Dhaka.

Key descriptive indicators:
- Dogs: **322**
- Cats: **3**
- Vaccinated: **32 / 325 (9.85%)**
- Sterilized: **34 / 325 (10.46%)**
- Observed nutritional health concern: **93 / 325 (28.62%)**

The survey is observational. These indicators should not be interpreted as veterinary diagnoses or Bangladesh-wide prevalence estimates.

## Dashboard
The Streamlit application includes:
- Overview
- News Intelligence
- Field Intelligence
- Live Incident Analyzer
- SARI Risk Assessment
- Decision Support
- Alert Center
- Methodology

### Live Incident Analyzer
The live analyzer:
1. predicts an incident category with the saved Linear SVM pipeline,
2. calculates SARI V2 independently,
3. extracts rule-based evidence signals,
4. checks selected model–evidence inconsistencies,
5. recommends human review when appropriate.

The consistency layer is a **prototype human-in-the-loop check**, not a validated accuracy-improvement mechanism.

### Email Alerts
The Alert Center can send prototype Gmail notifications using a Google App Password stored locally in a `.env` file.

**Never commit or submit `.env` or any real App Password.**

## Installation
Recommended Python environment: Python 3.x.

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local `.env` file beside `app.py`:

```env
ALERT_EMAIL_SENDER=your_sender@gmail.com
ALERT_EMAIL_APP_PASSWORD=your_google_app_password
ALERT_EMAIL_RECIPIENT=your_demo_recipient@gmail.com
```

Run the dashboard:

```bash
streamlit run app.py
```

## Important Limitations
- News analysis uses titles and available RSS descriptions rather than full article text.
- Labels are weakly supervised with assisted review; there is no independent expert-labelled gold-standard dataset.
- Keyword-derived weak labels may overlap with textual features used by the classifier.
- Rare-class evaluation is unstable, especially Rescue and Poisoning.
- 2026 is a partial year in the collected news timeline.
- Reliable geocoordinates were unavailable, so DBSCAN/HDBSCAN hotspot modelling was not used.
- SARI V2 weights and thresholds are heuristic and not externally validated.
- The Badda survey is a localized observational pilot.
- Survey images were not used as machine-learning labels or clinical validation.
- News and survey evidence are integrated at the summary/intelligence level, not by row-level linkage.

## Research Use
This system is a research prototype. Its outputs should support human review and prioritization; they should not be treated as clinical diagnoses, legal judgments, probabilities of harm, or autonomous intervention decisions.
