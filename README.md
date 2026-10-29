# Stray Animal Welfare Intelligence – Bangladesh (V2)

### A Machine Learning-Driven Stray Animal Welfare Intelligence Framework for Bangladesh

**Project Type:** Machine Learning, Natural Language Processing, Data Analytics & Welfare Intelligence

**Project Version:** 2.0 (Research Prototype)

**Geographic Scope:** Bangladesh (News Intelligence) and Badda, Dhaka (Field Survey Pilot)

---

## 1. Project Overview

**Stray Animal Welfare Intelligence – Bangladesh (V2)** is a machine learning-driven research prototype designed to analyze stray animal welfare incidents, identify patterns, and support evidence-informed welfare prioritization in Bangladesh.

The project extends the initial news-based welfare analysis (V1) by introducing supervised incident classification, expanded news collection, assisted label review, SARI V2 risk prioritization, and localized field survey intelligence.

The framework combines two complementary evidence streams:

**News Intelligence**

Collects and analyzes Bangladesh-related stray animal news through web scraping, data cleaning, relevance filtering, weak-label preparation, assisted review, machine learning classification, and heuristic risk prioritization.

**Field Intelligence**

Analyzes observational survey data collected from Badda, Dhaka, focusing on animal health conditions, vaccination, sterilization, human-animal behavior, and location-level intervention indicators.

Both evidence streams are presented through an interactive Streamlit dashboard.

> **Research Scope:** Bangladesh is the overall news intelligence scope. The field survey is a localized pilot conducted in Badda, Dhaka, and is not nationally representative.

---

## 2. Research Objectives

The main objectives of this project are:

1. Collect and organize stray animal welfare-related news from Bangladesh.
2. Identify relevant incidents using data filtering and preprocessing.
3. Develop a machine learning-based incident classification framework.
4. Compare supervised classification models using appropriate evaluation metrics.
5. Introduce an interpretable welfare prioritization index (SARI V2).
6. Analyze localized field observations related to stray animal welfare.
7. Visualize welfare patterns through an interactive dashboard.
8. Support human review and evidence-informed intervention prioritization.
9. Explore prototype automated alerts for welfare-related incidents.

---

## 3. Project Architecture

The system consists of two primary intelligence pipelines.

### 3.1 News Intelligence Pipeline

```text
Google News RSS
       |
       v
Expanded News Collection
       |
       v
Relevance Filtering
       |
       v
Bangladesh Scope Filtering
       |
       v
Weak Label Generation
       |
       v
Assisted Human Review
       |
       v
Final Machine Learning Corpus
       |
       v
TF-IDF Feature Extraction
       |
       v
Machine Learning Classification
       |
       v
Linear SVM (Selected Model)
       |
       v
SARI V2 Heuristic Prioritization
       |
       v
Evidence Consistency Checking
       |
       v
Streamlit Dashboard
       |
       v
Decision Support & Prototype Alerts
```

### 3.2 Field Intelligence Pipeline

```text
Badda, Dhaka Field Survey
          |
          v
Data Cleaning
          |
          v
Descriptive Analysis
          |
          v
Health Condition Indicators
          |
          v
Vaccination & Sterilization Analysis
          |
          v
Behavior & Location Indicators
          |
          v
Field Intelligence Dashboard
```

The two pipelines are integrated at the summary and intelligence presentation level rather than through direct row-level linkage.

---

## 4. Dataset Description

### 4.1 News Dataset

The news intelligence pipeline uses an expanded collection of stray animal-related news records.

| Processing Stage | Records |
|---|---:|
| Expanded News Collection | 1,094 |
| Relevant Records | 690 |
| Bangladesh-Primary Records | 250 |
| Final Machine Learning Corpus | 226 |
| Bangladesh-Primary News Sources | 48 |
| Expanded Unique News Sources | 260 |

The processing pipeline removes irrelevant records, applies geographic scope filtering, prepares incident labels, and excludes selected ambiguous or unsuitable records during assisted review.

The final corpus contains **226 records** for machine learning analysis.

### 4.2 Field Survey Dataset

The localized field survey contains observations of **325 stray animals** from Badda, Dhaka.

| Animal Type | Count |
|---|---:|
| Dogs | 322 |
| Cats | 3 |
| Total | 325 |

The survey includes observational indicators related to animal health, vaccination, sterilization, behavior, and location.

These observations are not equivalent to clinical veterinary diagnoses.

---

## 5. Data Collection and Preprocessing

### 5.1 News Collection

The news pipeline uses Google News RSS to collect potentially relevant news records.

Collected records are processed through several stages:

- News collection and extraction
- Relevance filtering
- Bangladesh-specific scope filtering
- Data cleaning and preprocessing
- Weak-label preparation
- Assisted review
- Final corpus preparation

The analysis primarily uses news titles and available RSS descriptions rather than complete article text.

### 5.2 Weak Supervision and Assisted Review

Incident labels are initially prepared using weak supervision.

A sample of records is then reviewed to assess the suitability of the suggested labels and determine whether records should be retained or excluded.

**Assisted Review Results**

| Review Category | Count |
|---|---:|
| Total Reviewed | 64 |
| KEEP | 40 |
| EXCLUDE | 22 |
| AMBIGUOUS | 2 |

Suggested-label agreement among KEEP records: **90.0%**

**Important:** This percentage represents agreement with the suggested weak labels among reviewed records retained as KEEP. It is not machine learning model accuracy.

---

## 6. Machine Learning Methodology

The project uses supervised machine learning to classify stray animal-related news incidents.

### 6.1 Feature Extraction

**TF-IDF (Term Frequency–Inverse Document Frequency)** is used to convert textual news information into numerical features suitable for machine learning classifiers.

### 6.2 Classification Models

Three machine learning algorithms were evaluated:

1. Linear Support Vector Machine (Linear SVM)
2. Logistic Regression
3. Complement Naive Bayes

### 6.3 Cross-Validation

The project uses **Stratified 2-Fold Cross-Validation**.

This approach was selected because the rarest incident class contained only two examples, limiting the number of feasible stratified folds.

### 6.4 Evaluation Metrics

Three evaluation metrics were considered:

- Accuracy
- Macro F1 Score
- Weighted F1 Score

**Macro F1** was selected as the primary evaluation metric because it gives equal importance to each class and is more informative than accuracy alone when evaluating imbalanced incident categories.

---

## 7. Machine Learning Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| **Linear SVM** | **0.7611** | **0.6068** | **0.7517** |
| Logistic Regression | 0.7257 | 0.5944 | 0.7206 |
| Complement Naive Bayes | 0.7389 | 0.5328 | 0.7282 |

### Selected Model: Linear SVM

Linear SVM achieved the highest Macro F1 score of **0.6068** among the evaluated models.

Therefore, Linear SVM was selected as the final incident classification model.

The results indicate that Linear SVM performed best under the project's current dataset, labeling process, and evaluation setup.

**Evaluation Note:** These scores measure predictive agreement with weakly supervised and assisted-reviewed labels. They do not represent performance on an independent expert-annotated benchmark.

---

## 8. Stray Animal Risk Index (SARI V2)

SARI V2 is an interpretable heuristic prioritization index developed to support welfare-related incident prioritization.

The index operates on a **0–10 scale**.

SARI V2 is calculated independently of the machine learning classifier.

It is not the prediction target of the classifier and should not be interpreted as a probability of harm.

### Risk Categories

| Risk Level | Score Range |
|---|---|
| Low | Score < 3 |
| Medium | 3 ≤ Score < 6 |
| High | Score ≥ 6 |

### SARI V2 Results

| Indicator | Result |
|---|---:|
| Low Risk Records | 191 |
| Medium Risk Records | 35 |
| High Risk Records | 0 |
| Mean Risk Score | 1.550 |
| Maximum Risk Score | 5.500 |

The absence of High-risk records is an observed result under the current dataset and heuristic weights.

The results were not manually adjusted to produce a particular risk distribution.

**Important:** SARI V2 weights and thresholds are heuristic and have not been externally validated.

---

## 9. Field Survey Intelligence

The field intelligence component analyzes observational survey records from Badda, Dhaka.

### Key Findings

| Indicator | Observation |
|---|---:|
| Total Animals Observed | 325 |
| Dogs | 322 |
| Cats | 3 |
| Vaccinated Animals | 32 |
| Vaccination Percentage | 9.85% |
| Sterilized Animals | 34 |
| Sterilization Percentage | 10.46% |
| Observed Nutritional Health Concern | 93 |
| Nutritional Concern Percentage | 28.62% |

### Analysis Areas

The field intelligence component focuses on:

- Animal health observations
- Vaccination status
- Sterilization status
- Human-animal behavioral indicators
- Location-level welfare indicators
- Potential intervention priorities

### Interpretation

The results provide descriptive insights into the observed animals within the selected survey area.

They should not be generalized to the entire stray animal population of Bangladesh.

---

## 10. Interactive Streamlit Dashboard

The project includes a Streamlit-based dashboard that integrates news intelligence, field observations, incident classification, and decision-support features.

### Dashboard Modules

| Module | Purpose |
|---|---|
| Overview | Summary of the project and major indicators |
| News Intelligence | News-based welfare incident analysis |
| Field Intelligence | Badda survey observations and indicators |
| Live Incident Analyzer | Machine learning-based incident analysis |
| SARI Risk Assessment | Heuristic welfare prioritization |
| Decision Support | Evidence-informed prioritization support |
| Alert Center | Prototype Gmail notifications |
| Methodology | Research workflow and methodological information |

---

## 11. Live Incident Analyzer

The Live Incident Analyzer is a prototype interface for analyzing incident-related text.

The workflow includes:

1. Receiving incident text.
2. Predicting an incident category using the saved Linear SVM pipeline.
3. Calculating the SARI V2 score independently.
4. Extracting rule-based evidence signals.
5. Checking selected inconsistencies between model predictions and evidence signals.
6. Recommending human review when appropriate.

### Evidence-Consistency Layer

The evidence-consistency layer is designed as a prototype human-in-the-loop review mechanism.

It highlights selected inconsistencies between machine learning predictions and rule-based evidence.

It has not been independently validated as a mechanism for improving classification accuracy.

---

## 12. Prototype Email Alert System

The dashboard includes an Alert Center capable of sending prototype Gmail notifications.

The email feature uses a Google App Password stored locally through environment variables.

### Environment Configuration

Create a `.env` file beside `app.py`:

```env
ALERT_EMAIL_SENDER=your_sender@gmail.com
ALERT_EMAIL_APP_PASSWORD=your_google_app_password
ALERT_EMAIL_RECIPIENT=your_demo_recipient@gmail.com
```

**Security Warning:**

Never upload real passwords, API keys, or the `.env` file to GitHub.

The email notification component is a research prototype rather than a validated production alerting service.

---

## 13. Technologies Used

### Programming and Data Analysis

- Python
- Pandas
- Data Cleaning and Wrangling
- Natural Language Processing
- TF-IDF Feature Extraction

### Machine Learning

- Scikit-learn
- Linear Support Vector Machine
- Logistic Regression
- Complement Naive Bayes
- Stratified Cross-Validation

### Application and Visualization

- Streamlit
- Interactive Data Visualization
- News and Field Intelligence Dashboards

### Data Sources and Supporting Tools

- Google News RSS
- Localized Field Survey Data
- CSV-Based Data Processing
- Gmail Prototype Notifications
- Environment Variable Configuration

---

## 14. Installation and Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Stray-Animal-Welfare-Intelligence-V2.git
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 2: Open the Project Directory

```bash
cd Stray-Animal-Welfare-Intelligence-V2
```

### Step 3: Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Configure Environment Variables (Optional)

If using the prototype email notification feature, create a local `.env` file and configure the required email credentials.

### Step 6: Run the Dashboard

```bash
streamlit run app.py
```

Open the local URL provided by Streamlit to access the dashboard.

**Note:** These commands assume `app.py` and `requirements.txt` are present in the project root. Adjust the paths if your repository uses a different structure.

---

## 15. V1 vs V2: Project Development

Version 2 extends the initial news-based welfare intelligence project with additional data processing, machine learning, field observations, and decision-support functionality.

| Feature | Version 1 | Version 2 |
|---|---|---|
| News-Based Analysis | Yes | Expanded |
| Data Cleaning | Initial Pipeline | Extended Pipeline |
| Bangladesh News Filtering | Initial Analysis | Expanded Filtering |
| Risk Prioritization | Initial SARI | SARI V2 |
| Machine Learning Classification | Not Included | Linear SVM |
| Model Comparison | Not Included | Three Models |
| Cross-Validation | Not Included | Stratified 2-Fold |
| Assisted Label Review | Not Included | Included |
| Field Survey | Not Included | 325 Observations |
| Data Visualization | Initial Visualizations | Integrated Streamlit Dashboard |
| Live Incident Analyzer | Not Included | Prototype |
| Evidence-Consistency Checking | Not Included | Prototype |
| Email Alerts | Not Included | Prototype |

### Development Progression

**V1: News-Based Welfare Intelligence**

Focused on news collection, data wrangling, risk scoring, and exploratory analysis of stray animal welfare incidents in Bangladesh.

**V2: Machine Learning-Driven Welfare Intelligence**

Introduces supervised incident classification, expanded data collection, SARI V2, localized field observations, evidence-consistency checking, and integrated decision-support functionality.

---

## 16. Research Limitations

The project has several important methodological and technical limitations:

1. News analysis uses titles and available RSS descriptions rather than full article text.
2. Incident labels are weakly supervised with assisted review.
3. No independent expert-labeled gold-standard dataset is available.
4. Keyword-derived weak labels may overlap with the textual features used by the classifier.
5. Rare incident categories create unstable evaluation conditions, especially Rescue and Poisoning.
6. The year 2026 is only partially represented in the collected news timeline.
7. Reliable geographic coordinates were unavailable, preventing dependable DBSCAN/HDBSCAN hotspot modeling.
8. SARI V2 weights and thresholds are heuristic rather than externally validated.
9. The field survey is restricted to Badda, Dhaka.
10. Observational health indicators do not represent veterinary diagnoses.
11. Survey images were not used as machine learning labels or clinical validation.
12. News and field observations are integrated at the summary level rather than through row-level matching.
13. The evidence-consistency mechanism remains a prototype.
14. Prototype alerts should not be treated as a validated operational emergency notification system.

---

## 17. Future Development

Potential future improvements include:

- Expanding the field survey to additional locations in Bangladesh.
- Developing a larger expert-annotated incident dataset.
- Improving classification performance for rare incident categories.
- Evaluating more advanced NLP and machine learning methods.
- Incorporating additional verified welfare data sources.
- Developing more reliable geographic hotspot analysis when appropriate coordinates become available.
- Externally validating welfare prioritization methods.
- Improving the evidence-consistency and human-review workflow.
- Extending the dashboard for broader research and welfare monitoring applications.

These are proposed future directions, not completed features of the current prototype.

---

## 18. Research Significance

This project explores how machine learning, data wrangling, and observational field intelligence can be combined to support stray animal welfare research in Bangladesh.

Its main contributions include:

- A Bangladesh-focused news intelligence pipeline.
- A weakly supervised incident classification framework.
- A comparative machine learning evaluation.
- An interpretable SARI V2 prioritization mechanism.
- A localized field intelligence component.
- An interactive dashboard for exploratory analysis and decision support.

The system demonstrates a possible framework for organizing fragmented welfare-related information and supporting human interpretation of available evidence.

---

## 19. Research and Ethical Use

This project is intended for academic research, exploratory analysis, and prototype development.

Its outputs should support human review and welfare prioritization.

They should not be treated as:

- Clinical or veterinary diagnoses.
- Legal judgments.
- Independently verified incident classifications.
- Validated probabilities of animal harm.
- Nationally representative estimates of stray animal welfare conditions.

Any real-world intervention decisions should involve appropriate human expertise, verification, and consideration of animal welfare.

---

## 20. Conclusion

**Stray Animal Welfare Intelligence – Bangladesh (V2)** demonstrates a machine learning-driven approach to combining news-derived incident intelligence with localized observational field data.

The project integrates news collection, preprocessing, weak supervision, incident classification, SARI V2 prioritization, and interactive visualization into a unified research prototype.

Among the evaluated classifiers, Linear SVM achieved the highest Macro F1 score of **0.6068**.

The field survey contributes descriptive observations from **325 stray animals** in Badda, Dhaka.

Although the framework has methodological and technical limitations, it provides a foundation for further research into data-driven stray animal welfare intelligence and human-centered decision support in Bangladesh.

---

**Project:** Stray Animal Welfare Intelligence – Bangladesh

**Version:** V2

**Research Title:** A Machine Learning-Driven Stray Animal Welfare Intelligence Framework for Bangladesh

**Status:** Research Prototype

**Primary Focus:** Machine Learning, NLP, Data Analytics, Stray Animal Welfare Intelligence

**Geographic Scope:** Bangladesh, with a localized Badda, Dhaka field survey pilot
