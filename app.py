from pathlib import Path
import warnings
import os
import smtplib
from email.message import EmailMessage

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# Load local secret configuration from .env
load_dotenv()

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Stray Animal Welfare Intelligence",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PROJECT PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
VIS_DIR = BASE_DIR / "visualization"


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(34,197,94,0.08), transparent 24%),
            radial-gradient(circle at 90% 0%, rgba(59,130,246,0.08), transparent 24%),
            #07111f;
        color: #f8fafc;
    }

    /* Top spacing */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #081522 0%, #0c1d2d 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    /* Main hero */
    .hero {
        background:
            linear-gradient(135deg, rgba(16,185,129,0.20), rgba(37,99,235,0.16)),
            #0b1727;
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 24px;
        padding: 28px 30px;
        margin-bottom: 20px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.25);
    }

    .hero h1 {
        color: #ffffff;
        font-size: 2.25rem;
        margin: 0 0 8px 0;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 1.03rem;
        margin: 0;
        line-height: 1.65;
    }

    .hero-badge {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 999px;
        background: rgba(34,197,94,0.16);
        border: 1px solid rgba(34,197,94,0.30);
        color: #86efac;
        font-size: 0.82rem;
        margin-bottom: 12px;
        font-weight: 700;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #0e1c2d, #102236);
        border: 1px solid rgba(148,163,184,0.14);
        border-radius: 18px;
        padding: 18px 20px;
        min-height: 132px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.18);
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .metric-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        margin-top: 7px;
    }

    .metric-sub {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-top: 5px;
    }

    /* Cards */
    .info-card {
        background: #0c1929;
        border: 1px solid rgba(148,163,184,0.14);
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 12px;
    }

    .info-card h3 {
        margin-top: 0;
        color: #f8fafc;
    }

    .soft-green {
        border-left: 4px solid #22c55e;
    }

    .soft-blue {
        border-left: 4px solid #3b82f6;
    }

    .soft-orange {
        border-left: 4px solid #f59e0b;
    }

    .soft-red {
        border-left: 4px solid #ef4444;
    }

    /* Section titles */
    .section-title {
        margin-top: 8px;
        margin-bottom: 12px;
        color: white;
        font-size: 1.45rem;
        font-weight: 800;
    }

    .section-subtitle {
        color: #94a3b8;
        margin-top: -5px;
        margin-bottom: 18px;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(148,163,184,0.14);
        border-radius: 14px;
        overflow: hidden;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        min-height: 44px;
        font-weight: 700;
        border: 1px solid rgba(255,255,255,0.12);
    }

    /* Inputs */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    textarea {
        border-radius: 12px !important;
    }

    /* Hide Streamlit decoration */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def locate_file(folder: Path, *possible_names):
    """
    Finds files even if Windows hides the extension in Explorer.
    """
    for name in possible_names:
        direct = folder / name
        if direct.exists():
            return direct

        if not str(name).lower().endswith(".csv"):
            csv_file = folder / f"{name}.csv"
            if csv_file.exists():
                return csv_file

    return None


@st.cache_data(show_spinner=False)
def load_csv_safe(path):
    if path is None:
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")
    except Exception:
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model_safe(path):
    if path is None or not path.exists():
        return None

    try:
        return joblib.load(path)
    except Exception:
        return None


def find_column(df, possible_names):
    """
    Flexible column finder.
    """
    if df.empty:
        return None

    normalized = {
        str(col).strip().lower().replace("_", " ").replace("-", " "): col
        for col in df.columns
    }

    for candidate in possible_names:
        candidate_norm = candidate.strip().lower().replace("_", " ").replace("-", " ")
        if candidate_norm in normalized:
            return normalized[candidate_norm]

    # partial match
    for candidate in possible_names:
        candidate_norm = candidate.strip().lower().replace("_", " ").replace("-", " ")
        for norm_name, original in normalized.items():
            if candidate_norm in norm_name:
                return original

    return None


def metric_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(badge, title, description):
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-badge">{badge}</div>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plot(fig, height=420):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"),
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.10)",
        zeroline=False,
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.10)",
        zeroline=False,
    )

    return fig


def build_analysis_text(title, description):
    title = "" if pd.isna(title) else str(title)
    description = "" if pd.isna(description) else str(description)

    text = f"{title} {description}".strip()
    return text



# =========================================================
# SECURE EMAIL ALERT HELPERS
# =========================================================
def get_email_config():
    """
    Reads Gmail credentials from environment variables.
    Never hard-code passwords inside app.py.
    """
    sender = os.getenv("ALERT_EMAIL_SENDER", "").strip()
    app_password = os.getenv("ALERT_EMAIL_APP_PASSWORD", "").replace(" ", "").strip()
    default_recipient = os.getenv("ALERT_EMAIL_RECIPIENT", "").strip()

    return sender, app_password, default_recipient


def email_config_ready():
    sender, app_password, _ = get_email_config()
    return bool(sender and app_password)


def send_gmail_alert(recipient, subject, body):
    """
    Sends a plain-text alert through Gmail SMTP over SSL.
    Requires a Google App Password, not the normal Gmail password.
    """
    sender, app_password, _ = get_email_config()

    if not sender:
        raise ValueError("ALERT_EMAIL_SENDER is missing in the .env file.")

    if not app_password:
        raise ValueError("ALERT_EMAIL_APP_PASSWORD is missing in the .env file.")

    if not recipient:
        raise ValueError("Recipient email address is required.")

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as smtp:
        smtp.login(sender, app_password)
        smtp.send_message(msg)


def build_alert_email(incident_text, predicted_label, sari_result, manual_review_required):
    evidence = sari_result.get("evidence", [])
    evidence_text = ", ".join(evidence) if evidence else "No explicit rule-based signal detected"

    disagreement_reasons = sari_result.get("disagreement_reasons", [])

    if manual_review_required:
        disagreement_text = "MODEL–EVIDENCE DISAGREEMENT DETECTED"
        disagreement_details = (
            "\n".join(f"- {reason}" for reason in disagreement_reasons)
            if disagreement_reasons
            else "- A conflicting rule-based evidence signal was detected."
        )
    else:
        disagreement_text = "No predefined critical disagreement detected"
        disagreement_details = "- No predefined critical disagreement was detected."

    recommended_action = (
        "Manual verification and context review are recommended before intervention."
        if manual_review_required
        else "Human review should be applied according to operational context."
    )

    subject = (
        f"Welfare Alert | {sari_result.get('level', 'N/A')} Priority | "
        f"{'Manual Review Required' if manual_review_required else 'Review'}"
    )

    body = f"""STRAY ANIMAL WELFARE INTELLIGENCE FRAMEWORK — V2

Prototype Alert Notification

INCIDENT
{incident_text}

ML CLASSIFICATION
{predicted_label}

DETECTED EVIDENCE
{evidence_text}

SARI V2 SCORE
{sari_result.get('score', 'N/A')} / 10

PRIORITY LEVEL
{sari_result.get('level', 'N/A')}

MODEL–EVIDENCE STATUS
{disagreement_text}

DETAILS
{disagreement_details}

RECOMMENDED ACTION
{recommended_action}

IMPORTANT
This message was generated by a research prototype.
SARI V2 is a heuristic prioritization index and is not a clinical diagnosis,
probability of harm, legal judgment, or automated intervention decision.
"""

    return subject, body


# =========================================================
# SARI V2 PROTOTYPE FUNCTION
# =========================================================
def calculate_sari_v2(text, predicted_incident=None):
    """
    Live implementation aligned with the V2 preprocessing
    and SARI V2 analysis scripts.

    SARI V2 is a heuristic prioritization index,
    not a clinical probability.
    """

    import re

    text = str(text).lower()

    # -----------------------------------------
    # Same keyword groups as V2 preprocessing
    # -----------------------------------------
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

    POISON_WORDS = [
        "poison", "poisoned", "poisoning", "toxic", "toxicity"
    ]

    RESCUE_WORDS = [
        "rescue", "rescued", "save", "saved", "shelter",
        "adopt", "adoption", "treatment", "treated",
        "rehabilitation"
    ]

    def count_terms(text_value, terms):
        return sum(
            1
            for term in terms
            if re.search(r"\b" + re.escape(term) + r"\b", text_value)
        )

    # -----------------------------------------
    # Same keyword counts as preprocessing
    # -----------------------------------------
    violence_count = count_terms(text, VIOLENCE_WORDS)
    disease_count = count_terms(text, DISEASE_WORDS)
    human_count = count_terms(text, HUMAN_CONFLICT_WORDS)
    poison_count = count_terms(text, POISON_WORDS)
    rescue_count = count_terms(text, RESCUE_WORDS)

    # Binary indicators
    violence_indicator = int(violence_count > 0)
    disease_indicator = int(disease_count > 0)
    human_indicator = int(human_count > 0)
    poison_indicator = int(poison_count > 0)
    rescue_indicator = int(rescue_count > 0)

    # -----------------------------------------
    # EXACT SARI V2 COMPONENT LOGIC
    # -----------------------------------------
    abuse_component = min(
        3.0,
        1.5 * violence_indicator
        + 0.5 * min(violence_count, 3)
    )

    human_component = min(
        2.5,
        1.5 * human_indicator
        + 0.5 * min(human_count, 2)
    )

    sickness_component = min(
        2.0,
        1.0 * disease_indicator
        + 0.5 * min(disease_count, 2)
    )

    poison_component = min(
        1.5,
        1.0 * poison_indicator
        + 0.5 * min(poison_count, 1)
    )

    context_weight = {
        "Poisoning": 1.0,
        "Killing": 1.0,
        "Abuse": 0.9,
        "Human-Animal Conflict": 0.8,
        "Disease": 0.6,
        "Conflict": 0.4,
        "Rescue": 0.1,
        "Other": 0.2,
    }

    context_component = context_weight.get(
        str(predicted_incident),
        0.2
    )

    score = (
        abuse_component
        + human_component
        + sickness_component
        + poison_component
        + context_component
    )

    score = round(min(10.0, score), 2)

    # Same thresholds as original SARI V2
    if score >= 6.0:
        level = "High"
    elif score >= 3.0:
        level = "Medium"
    else:
        level = "Low"

    # -----------------------------------------
    # Evidence detection for dashboard
    # -----------------------------------------
    evidence = []

    if poison_indicator:
        evidence.append("Poisoning")

    if violence_indicator:
        evidence.append("Violence / Abuse")

    if human_indicator:
        evidence.append("Human-Animal Conflict")

    if disease_indicator:
        evidence.append("Disease / Rabies")

    # Rescue/treatment evidence needs extra context.
    # Generic words such as "treatment" can refer to a human bite victim,
    # so we only surface Rescue / Treatment when the text clearly refers
    # to rescuing, sheltering, adopting, rehabilitating, or treating the animal.
    rescue_context_patterns = [
        r"\b(rescue|rescued|save|saved|shelter|adopt|adoption|rehabilitation)\b",
        r"\b(dog|dogs|cat|cats|animal|animals|stray|strays|puppy|puppies|kitten|kittens)\b.{0,45}\b(treatment|treated)\b",
        r"\b(treatment|treated)\b.{0,45}\b(dog|dogs|cat|cats|animal|animals|stray|strays|puppy|puppies|kitten|kittens)\b",
    ]

    rescue_context_detected = any(
        re.search(pattern, text)
        for pattern in rescue_context_patterns
    )

    if rescue_context_detected:
        evidence.append("Rescue / Treatment")

    # Detect important ML-rule disagreement.
    # The consistency layer is independent from the SARI priority threshold:
    # a case may still have a Low SARI score but require manual review when
    # strong rule-based evidence conflicts with the classifier output.
    disagreement_reasons = []

    if poison_indicator and predicted_incident != "Poisoning":
        disagreement_reasons.append(
            f"Poisoning evidence detected while classifier predicted {predicted_incident}"
        )

    if violence_indicator and predicted_incident not in ("Abuse", "Killing", "Poisoning"):
        disagreement_reasons.append(
            f"Violence / Abuse evidence detected while classifier predicted {predicted_incident}"
        )

    # Rescue/Treatment evidence is intentionally not treated as a critical
    # disagreement because it is a low-risk supportive signal in this prototype.

    disagreement = len(disagreement_reasons) > 0

    return {
        "score": score,
        "level": level,

        "abuse": abuse_component,
        "human_conflict": human_component,
        "sickness": sickness_component,
        "poisoning": poison_component,
        "context": context_component,

        "violence_count": violence_count,
        "human_count": human_count,
        "disease_count": disease_count,
        "poison_count": poison_count,
        "rescue_count": rescue_count,
        "rescue_context_detected": rescue_context_detected,

        "evidence": evidence,
        "disagreement": disagreement,
        "disagreement_reasons": disagreement_reasons,
    }


def sari_gauge(score):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 10"},
            title={"text": "SARI V2 Priority Score"},
            gauge={
                "axis": {"range": [0, 10]},
                "bar": {"color": "#38bdf8"},
                "bgcolor": "rgba(255,255,255,0.05)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 3], "color": "rgba(34,197,94,0.22)"},
                    {"range": [3, 6], "color": "rgba(245,158,11,0.24)"},
                    {"range": [6, 10], "color": "rgba(239,68,68,0.24)"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 3},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )

    fig.update_layout(
        height=310,
        margin=dict(l=30, r=30, t=70, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )

    return fig


# =========================================================
# LOCATE PROJECT FILES
# =========================================================
training_path = locate_file(
    DATA_DIR,
    "v2_ml_training_dataset.csv",
    "v2_ml_training_dataset",
)

predictions_path = locate_file(
    DATA_DIR,
    "v2_ml_predictions.csv",
    "v2_ml_predictions",
)

sari_path = locate_file(
    DATA_DIR,
    "v2_sari_scored.csv",
    "v2_sari_scored",
)

survey_path = locate_file(
    DATA_DIR,
    "v2_survey_cleaned.csv",
    "v2_survey_cleaned",
)

primary_path = locate_file(
    DATA_DIR,
    "v2_bangladesh_primary.csv",
    "v2_bangladesh_primary",
)

model_path = MODEL_DIR / "v2_incident_classifier.joblib"


# =========================================================
# LOAD DATA
# =========================================================
training_df = load_csv_safe(training_path)
predictions_df = load_csv_safe(predictions_path)
sari_df = load_csv_safe(sari_path)
survey_df = load_csv_safe(survey_path)
primary_df = load_csv_safe(primary_path)

model = load_model_safe(model_path)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div style="
            text-align:center;
            padding:18px 8px 10px 8px;
        ">
            <div style="font-size:3rem;">🐾</div>
            <div style="
                font-size:1.25rem;
                font-weight:800;
                color:white;
                line-height:1.2;
            ">
                Welfare Intelligence
            </div>
            <div style="
                font-size:0.78rem;
                color:#94a3b8;
                margin-top:6px;
            ">
                Bangladesh • V2 Prototype
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "📰 News Intelligence",
            "🐕 Field Intelligence",
            "🧠 Live Incident Analyzer",
            "⚠️ SARI Risk Assessment",
            "🎯 Decision Support",
            "📧 Alert Center",
            "📚 Methodology",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    model_status = "Online" if model is not None else "Not Loaded"

    st.markdown(
        f"""
        <div style="
            background:#0d1d2c;
            border:1px solid rgba(255,255,255,0.08);
            padding:14px;
            border-radius:14px;
        ">
            <div style="font-size:0.75rem;color:#94a3b8;">
                SYSTEM STATUS
            </div>
            <div style="margin-top:8px;font-weight:700;">
                {'🟢' if model is not None else '🔴'} ML Model: {model_status}
            </div>
            <div style="margin-top:5px;font-weight:700;">
                🟢 Survey Data: {'Online' if not survey_df.empty else 'Missing'}
            </div>
            <div style="margin-top:5px;font-weight:700;">
                🟢 SARI Data: {'Online' if not sari_df.empty else 'Missing'}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Proof-of-concept research prototype")


# =========================================================
# PAGE: OVERVIEW
# =========================================================
if page == "🏠 Overview":
    page_header(
        "V2 • DUAL-SOURCE INTELLIGENCE",
        "🐾 Stray Animal Welfare Intelligence Framework",
        "A machine learning-driven research prototype combining Bangladesh-focused "
        "news intelligence with localized field observations from Badda, Dhaka.",
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Candidate News Records",
            "1,094",
            "Expanded news collection",
        )

    with c2:
        metric_card(
            "Final ML Corpus",
            "226",
            "Bangladesh-focused analytical records",
        )

    with c3:
        metric_card(
            "Field Observations",
            "325",
            "Badda pilot survey",
        )

    with c4:
        metric_card(
            "Best Macro F1",
            "0.6068",
            "Linear SVM",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.25, 1])

    with left:
        st.markdown(
            '<div class="section-title">System Architecture</div>',
            unsafe_allow_html=True,
        )

        funnel_df = pd.DataFrame(
            {
                "Stage": [
                    "Expanded Candidate Records",
                    "Animal-Relevant Records",
                    "Bangladesh-Primary",
                    "Final ML Corpus",
                ],
                "Records": [1094, 690, 250, 226],
            }
        )

        fig = px.funnel(
            funnel_df,
            x="Records",
            y="Stage",
            title="News Intelligence Data Refinement",
        )

        fig = style_plot(fig, 430)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown(
            '<div class="section-title">Research Highlights</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="info-card soft-blue">
                <h3>📰 News Intelligence</h3>
                Bangladesh-focused incident analysis using TF–IDF and supervised
                text classification.
            </div>

            <div class="info-card soft-green">
                <h3>🐕 Field Intelligence</h3>
                Localized observations of health, vaccination, sterilization,
                behavior and location across the Badda pilot sample.
            </div>

            <div class="info-card soft-orange">
                <h3>⚠️ SARI V2</h3>
                Transparent heuristic prioritization score maintained separately
                from the supervised incident classifier.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Key Field Indicators</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "Vaccinated",
            "9.85%",
            "Within surveyed sample",
        )

    with c2:
        metric_card(
            "Sterilized",
            "10.46%",
            "Within surveyed sample",
        )

    with c3:
        metric_card(
            "Nutritional Concern",
            "28.62%",
            "Observed underweight/malnourished or starved",
        )

    st.info(
        "The field statistics describe the 325 observed animals in the Badda pilot "
        "survey and should not be interpreted as national prevalence estimates."
    )


# =========================================================
# PAGE: NEWS INTELLIGENCE
# =========================================================
elif page == "📰 News Intelligence":
    page_header(
        "NEWS INTELLIGENCE",
        "📰 Bangladesh-Focused Incident Intelligence",
        "Explore the refined media corpus, incident distribution, temporal patterns "
        "and model-derived analytical results.",
    )

    df = sari_df.copy() if not sari_df.empty else training_df.copy()

    if df.empty:
        st.error("News dataset could not be loaded.")
    else:
        incident_col = find_column(
            df,
            [
                "incident_type",
                "incident label",
                "incident_label",
                "label",
                "predicted_incident",
                "final_label",
            ],
        )

        animal_col = find_column(
            df,
            ["animal_type", "animal", "animal category"],
        )

        source_col = find_column(
            df,
            ["source", "news_source"],
        )

        date_col = find_column(
            df,
            ["date", "published date", "publication_date"],
        )

        title_col = find_column(
            df,
            ["title", "headline"],
        )

        # Filters
        st.markdown(
            '<div class="section-title">Explore Records</div>',
            unsafe_allow_html=True,
        )

        filtered_df = df.copy()

        f1, f2, f3 = st.columns(3)

        if incident_col:
            incidents = sorted(
                filtered_df[incident_col].dropna().astype(str).unique().tolist()
            )

            with f1:
                selected_incidents = st.multiselect(
                    "Incident Category",
                    incidents,
                    default=incidents,
                )

            if selected_incidents:
                filtered_df = filtered_df[
                    filtered_df[incident_col].astype(str).isin(selected_incidents)
                ]

        if animal_col:
            animals = sorted(
                df[animal_col].dropna().astype(str).unique().tolist()
            )

            with f2:
                selected_animals = st.multiselect(
                    "Animal Type",
                    animals,
                    default=animals,
                )

            if selected_animals:
                filtered_df = filtered_df[
                    filtered_df[animal_col].astype(str).isin(selected_animals)
                ]

        if source_col:
            sources = sorted(
                df[source_col].dropna().astype(str).unique().tolist()
            )

            with f3:
                selected_sources = st.multiselect(
                    "News Source",
                    sources,
                    default=sources,
                )

            if selected_sources:
                filtered_df = filtered_df[
                    filtered_df[source_col].astype(str).isin(selected_sources)
                ]

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            metric_card(
                "Displayed Records",
                f"{len(filtered_df):,}",
                "Current filters",
            )

        with m2:
            metric_card(
                "Final Corpus",
                "226",
                "ML analytical dataset",
            )

        with m3:
            metric_card(
                "Linear SVM Accuracy",
                "0.7611",
                "2-fold cross-validation",
            )

        with m4:
            metric_card(
                "Weighted F1",
                "0.7517",
                "Selected model",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            if incident_col:
                counts = (
                    filtered_df[incident_col]
                    .astype(str)
                    .value_counts()
                    .reset_index()
                )
                counts.columns = ["Incident", "Count"]

                fig = px.bar(
                    counts,
                    x="Count",
                    y="Incident",
                    orientation="h",
                    title="Incident Category Distribution",
                    text="Count",
                )

                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                fig = style_plot(fig)
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            sari_level_col = find_column(
                df,
                ["sari_v2_level", "risk_level", "sari level"],
            )

            if sari_level_col:
                risk_counts = (
                    filtered_df[sari_level_col]
                    .astype(str)
                    .value_counts()
                    .reset_index()
                )
                risk_counts.columns = ["Risk Level", "Count"]

                fig = px.pie(
                    risk_counts,
                    names="Risk Level",
                    values="Count",
                    hole=0.58,
                    title="SARI V2 Risk-Level Distribution",
                )

                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig = style_plot(fig)
                st.plotly_chart(fig, use_container_width=True)

        # Timeline
        if date_col:
            temp_df = filtered_df.copy()
            temp_df[date_col] = pd.to_datetime(
                temp_df[date_col],
                errors="coerce",
            )
            temp_df["Year"] = temp_df[date_col].dt.year
            yearly = (
                temp_df.dropna(subset=["Year"])
                .groupby("Year")
                .size()
                .reset_index(name="Records")
            )

            if not yearly.empty:
                fig = px.area(
                    yearly,
                    x="Year",
                    y="Records",
                    markers=True,
                    title="Temporal Distribution of Retained Records",
                )
                fig = style_plot(fig)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="section-title">Record Explorer</div>',
            unsafe_allow_html=True,
        )

        display_columns = []

        for col in [
            title_col,
            incident_col,
            animal_col,
            source_col,
            date_col,
            find_column(df, ["sari_v2_score"]),
            find_column(df, ["sari_v2_level"]),
        ]:
            if col and col not in display_columns:
                display_columns.append(col)

        if display_columns:
            st.dataframe(
                filtered_df[display_columns],
                use_container_width=True,
                hide_index=True,
                height=440,
            )
        else:
            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True,
                height=440,
            )


# =========================================================
# PAGE: FIELD INTELLIGENCE
# =========================================================
elif page == "🐕 Field Intelligence":
    page_header(
        "FIELD INTELLIGENCE",
        "🐕 Badda Pilot Welfare Observation",
        "Explore localized field observations on health, vaccination, sterilization, "
        "behavior and intervention gaps across the Badda pilot sample.",
    )

    if survey_df.empty:
        st.error("v2_survey_cleaned.csv could not be loaded.")
    else:
        df = survey_df.copy()

        location_col = find_column(
            df,
            ["location", "area", "locality"],
        )

        animal_col = find_column(
            df,
            ["animal_type", "animal type", "animal"],
        )

        health_col = find_column(
            df,
            [
                "health_condition",
                "health condition",
                "health_status",
                "health status",
            ],
        )

        vacc_col = find_column(
            df,
            ["vaccinated", "vaccination_status", "vaccination"],
        )

        steril_col = find_column(
            df,
            ["sterilized", "sterilization_status", "sterilization"],
        )

        behavior_col = find_column(
            df,
            ["human_behavior", "behavior", "behaviour"],
        )

        age_col = find_column(
            df,
            ["age_group", "age group", "age"],
        )

        filtered = df.copy()

        f1, f2 = st.columns(2)

        if location_col:
            locations = sorted(
                df[location_col].dropna().astype(str).unique().tolist()
            )

            with f1:
                selected_locations = st.multiselect(
                    "Locality",
                    locations,
                    default=locations,
                )

            if selected_locations:
                filtered = filtered[
                    filtered[location_col].astype(str).isin(selected_locations)
                ]

        if animal_col:
            animals = sorted(
                df[animal_col].dropna().astype(str).unique().tolist()
            )

            with f2:
                selected_animals = st.multiselect(
                    "Animal Type",
                    animals,
                    default=animals,
                )

            if selected_animals:
                filtered = filtered[
                    filtered[animal_col].astype(str).isin(selected_animals)
                ]

        # Dynamic indicator calculation
        def yes_rate(frame, column):
            if not column or frame.empty:
                return np.nan

            values = (
                frame[column]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            yes_values = values.isin(
                ["yes", "y", "1", "true", "vaccinated", "sterilized"]
            )

            return round(100 * yes_values.mean(), 2)

        vacc_rate = yes_rate(filtered, vacc_col)
        steril_rate = yes_rate(filtered, steril_col)

        nutritional_rate = np.nan

        if health_col:
            health_text = filtered[health_col].astype(str).str.lower()

            concern = (
                health_text.str.contains("under", na=False)
                | health_text.str.contains("malnour", na=False)
                | health_text.str.contains("starved", na=False)
                | health_text.str.contains("starv", na=False)
            )

            nutritional_rate = round(
                concern.mean() * 100,
                2,
            )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "Observed Animals",
                f"{len(filtered):,}",
                "Current filtered sample",
            )

        with c2:
            metric_card(
                "Vaccination",
                f"{vacc_rate:.2f}%" if not np.isnan(vacc_rate) else "N/A",
                "Observed sample",
            )

        with c3:
            metric_card(
                "Sterilization",
                f"{steril_rate:.2f}%" if not np.isnan(steril_rate) else "N/A",
                "Observed sample",
            )

        with c4:
            metric_card(
                "Nutritional Concern",
                f"{nutritional_rate:.2f}%"
                if not np.isnan(nutritional_rate)
                else "N/A",
                "Observed field condition",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            if health_col:
                health_counts = (
                    filtered[health_col]
                    .astype(str)
                    .value_counts()
                    .reset_index()
                )

                health_counts.columns = ["Health Condition", "Count"]

                fig = px.bar(
                    health_counts,
                    x="Health Condition",
                    y="Count",
                    text="Count",
                    title="Observed Health Condition",
                )

                fig = style_plot(fig)
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            if age_col:
                age_counts = (
                    filtered[age_col]
                    .astype(str)
                    .value_counts()
                    .reset_index()
                )

                age_counts.columns = ["Age Group", "Count"]

                fig = px.pie(
                    age_counts,
                    names="Age Group",
                    values="Count",
                    hole=0.48,
                    title="Observed Age Distribution",
                )

                fig = style_plot(fig)
                st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            if location_col:
                loc_counts = (
                    filtered[location_col]
                    .astype(str)
                    .value_counts()
                    .reset_index()
                )

                loc_counts.columns = ["Locality", "Observations"]

                fig = px.bar(
                    loc_counts,
                    x="Locality",
                    y="Observations",
                    text="Observations",
                    title="Survey Coverage by Locality",
                )

                fig = style_plot(fig)
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            if behavior_col:
                behavior_counts = (
                    filtered[behavior_col]
                    .astype(str)
                    .value_counts()
                    .reset_index()
                )

                behavior_counts.columns = ["Behavior", "Count"]

                fig = px.bar(
                    behavior_counts,
                    x="Count",
                    y="Behavior",
                    orientation="h",
                    text="Count",
                    title="Observed Human-Directed Behavior",
                )

                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                fig = style_plot(fig)
                st.plotly_chart(fig, use_container_width=True)

        st.warning(
            "These statistics describe the observed Badda pilot sample only. "
            "They are not national prevalence estimates and are not veterinary diagnoses."
        )

        with st.expander("View cleaned field records"):
            st.dataframe(
                filtered,
                use_container_width=True,
                hide_index=True,
                height=430,
            )


# =========================================================
# PAGE: LIVE INCIDENT ANALYZER
# =========================================================
elif page == "🧠 Live Incident Analyzer":
    page_header(
        "LIVE MODEL DEMONSTRATION",
        "🧠 Incident Analyzer",
        "Enter a new incident description to demonstrate how the saved V2 text "
        "classifier can categorize unseen text and pass the result to the separate "
        "SARI V2 prioritization module.",
    )

    st.markdown(
        """
        <div class="info-card soft-blue">
        <b>Demo principle:</b> The incident classifier and SARI V2 are intentionally
        separate. The classifier predicts incident category, while SARI V2 produces
        a transparent heuristic priority score.
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_text = (
        "Several stray dogs were reportedly poisoned in a residential area. "
        "Local residents requested immediate animal welfare intervention."
    )

    incident_text = st.text_area(
        "Incident description",
        value=default_text,
        height=180,
        placeholder="Type or paste a new stray-animal-related incident here...",
    )

    analyze = st.button(
        "🔍 Analyze Incident",
        use_container_width=True,
        type="primary",
    )

    if analyze:
        if not incident_text.strip():
            st.warning("Please enter an incident description.")
        else:
            predicted_label = None
            prediction_error = None

            if model is not None:
                try:
                    prediction = model.predict([incident_text])
                    predicted_label = str(prediction[0])
                except Exception as exc:
                    prediction_error = str(exc)

            if predicted_label is None:
                st.error(
                    "The saved model was loaded but direct prediction could not be completed."
                    if model is not None
                    else "Saved ML model could not be loaded."
                )

                if prediction_error:
                    with st.expander("Technical details"):
                        st.code(prediction_error)

            else:
                sari_result = calculate_sari_v2(
                    incident_text,
                    predicted_label,
                )

                st.session_state["latest_incident_text"] = incident_text
                st.session_state["latest_predicted_label"] = predicted_label
                st.session_state["latest_sari"] = sari_result

                c1, c2, c3 = st.columns(3)

                with c1:
                    metric_card(
                        "Predicted Incident",
                        predicted_label,
                        "Linear SVM prototype",
                    )

                with c2:
                    metric_card(
                        "SARI V2 Score",
                        f"{sari_result['score']:.2f}",
                        "Heuristic priority index",
                    )

                with c3:
                    metric_card(
                        "Priority Level",
                        sari_result["level"],
                        "Low / Medium / High threshold",
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                c1, c2 = st.columns([1, 1.15])

                with c1:
                    st.plotly_chart(
                        sari_gauge(sari_result["score"]),
                        use_container_width=True,
                    )

                with c2:
                    component_df = pd.DataFrame(
                        {
                            "Component": [
                                "Abuse / Violence",
                                "Human Conflict",
                                "Sickness / Rabies",
                                "Poisoning",
                                "Incident Context",
                            ],
                            "Contribution": [
                                sari_result["abuse"],
                                sari_result["human_conflict"],
                                sari_result["sickness"],
                                sari_result["poisoning"],
                                sari_result["context"],
                            ],
                        }
                    )

                    fig = px.bar(
                        component_df,
                        x="Contribution",
                        y="Component",
                        orientation="h",
                        text="Contribution",
                        title="SARI V2 Component Contributions",
                    )

                    fig.update_layout(
                        yaxis={"categoryorder": "total ascending"}
                    )

                    fig = style_plot(fig, 340)
                    st.plotly_chart(fig, use_container_width=True)
                

                # =====================================================
                # MODEL-EVIDENCE CONSISTENCY CHECK
                # =====================================================
                st.markdown(
                    '<div class="section-title">🔎 Evidence Consistency Check</div>',
                    unsafe_allow_html=True,
                )

                evidence = sari_result.get("evidence", [])

                if evidence:
                    evidence_text = " • ".join(evidence)
                    st.markdown(
                        f"""
                        <div class="info-card soft-blue">
                            <h3>Detected Evidence Signals</h3>
                            <p style="font-size:1.05rem; margin-bottom:0;">
                                {evidence_text}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No explicit rule-based welfare evidence signal was detected.")

                if sari_result.get("disagreement", False):
                    st.markdown(
                        f"""
                        <div class="info-card soft-orange">
                            <h3>⚠️ Model–Evidence Disagreement Detected</h3>
                            <p>
                                The trained classifier predicted <b>{predicted_label}</b>,
                                while the independent evidence layer detected an explicit
                                <b>poisoning-related signal</b>.
                            </p>
                            <p style="margin-bottom:0;">
                                <b>Decision:</b> Manual verification is recommended before
                                any operational intervention.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.session_state["manual_review_required"] = True
                else:
                    st.markdown(
                        """
                        <div class="info-card soft-green">
                            <h3>✓ No Critical Model–Evidence Conflict</h3>
                            <p style="margin-bottom:0;">
                                No predefined critical disagreement was detected between
                                the classifier output and the independent evidence layer.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.session_state["manual_review_required"] = False

                if sari_result["level"] == "High":
                    st.error(
                        "High-priority prototype alert: manual verification and "
                        "rapid welfare review recommended."
                    )
                elif sari_result["level"] == "Medium":
                    st.warning(
                        "Medium-priority prototype alert: manual review and "
                        "context verification recommended."
                    )
                else:
                    st.success(
                        "Low heuristic priority under the current SARI V2 rules. "
                        "Human review may still be necessary."
                    )

                st.caption(
                    "SARI V2 is an interpretable heuristic prioritization index. "
                    "It is not a clinical diagnosis, probability of harm, or legal judgment."
                )


# =========================================================
# PAGE: SARI RISK ASSESSMENT
# =========================================================
elif page == "⚠️ SARI Risk Assessment":
    page_header(
        "INTERPRETABLE PRIORITIZATION",
        "⚠️ SARI V2 Risk Assessment",
        "Inspect the separate heuristic risk layer used to summarize abuse, "
        "human-conflict, sickness/rabies, poisoning and contextual severity signals.",
    )

    if sari_df.empty:
        st.error("v2_sari_scored.csv could not be loaded.")
    else:
        score_col = find_column(
            sari_df,
            ["sari_v2_score", "sari score"],
        )

        level_col = find_column(
            sari_df,
            ["sari_v2_level", "risk level"],
        )

        incident_col = find_column(
            sari_df,
            ["incident_type", "incident label", "label", "final_label"],
        )

        if score_col:
            scores = pd.to_numeric(
                sari_df[score_col],
                errors="coerce",
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                metric_card(
                    "Mean SARI",
                    f"{scores.mean():.3f}",
                    "Final analytical corpus",
                )

            with c2:
                metric_card(
                    "Maximum SARI",
                    f"{scores.max():.3f}",
                    "Observed maximum",
                )

            with c3:
                low_count = (
                    (sari_df[level_col].astype(str).str.lower() == "low").sum()
                    if level_col
                    else 0
                )

                metric_card(
                    "Low Priority",
                    str(low_count),
                    "Score < 3",
                )

            with c4:
                medium_count = (
                    (sari_df[level_col].astype(str).str.lower() == "medium").sum()
                    if level_col
                    else 0
                )

                metric_card(
                    "Medium Priority",
                    str(medium_count),
                    "3 ≤ score < 6",
                )

            st.markdown("<br>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)

            with c1:
                fig = px.histogram(
                    sari_df,
                    x=score_col,
                    nbins=20,
                    title="Distribution of SARI V2 Scores",
                )
                fig = style_plot(fig)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                if incident_col:
                    temp = sari_df.copy()
                    temp[score_col] = pd.to_numeric(
                        temp[score_col],
                        errors="coerce",
                    )

                    mean_df = (
                        temp.groupby(incident_col)[score_col]
                        .mean()
                        .sort_values()
                        .reset_index()
                    )

                    fig = px.bar(
                        mean_df,
                        x=score_col,
                        y=incident_col,
                        orientation="h",
                        text_auto=".2f",
                        title="Mean SARI V2 Score by Incident Category",
                    )

                    fig = style_plot(fig)
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                """
                <div class="info-card soft-orange">
                <h3>Important interpretation</h3>
                SARI V2 is a transparent heuristic index with predefined thresholds:
                <b>Low &lt; 3</b>, <b>Medium 3–&lt;6</b>, and <b>High ≥ 6</b>.
                The weights and thresholds have not been externally calibrated
                against clinical or epidemiological outcomes.
                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================================================
# PAGE: DECISION SUPPORT
# =========================================================
elif page == "🎯 Decision Support":
    page_header(
        "EVIDENCE-LEVEL INTEGRATION",
        "🎯 Welfare Decision Support",
        "Combine news-derived incident intelligence with localized field indicators "
        "without incorrectly merging their individual records.",
    )

    st.markdown(
        '<div class="section-title">Current Evidence Snapshot</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="info-card soft-blue">
                <h3>📰 News Evidence</h3>
                <b>Final ML corpus:</b> 226 records<br><br>
                <b>Best classifier:</b> Linear SVM<br><br>
                <b>Macro F1:</b> 0.6068<br><br>
                <b>Weighted F1:</b> 0.7517<br><br>
                <b>Use:</b> Incident monitoring and record prioritization
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="info-card soft-green">
                <h3>🐕 Field Evidence</h3>
                <b>Pilot observations:</b> 325 animals<br><br>
                <b>Vaccinated:</b> 9.85%<br><br>
                <b>Sterilized:</b> 10.46%<br><br>
                <b>Nutritional concern:</b> 28.62%<br><br>
                <b>Use:</b> Local intervention planning and repeat survey targeting
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Suggested Operational Workflow</div>',
        unsafe_allow_html=True,
    )

    workflow_df = pd.DataFrame(
        {
            "Stage": [
                "1. Detect",
                "2. Classify",
                "3. Prioritize",
                "4. Verify",
                "5. Respond",
                "6. Record",
            ],
            "Purpose": [
                "Receive news or field evidence",
                "Identify likely incident category",
                "Calculate transparent SARI priority",
                "Conduct human/manual verification",
                "Plan appropriate welfare intervention",
                "Store outcome for future learning",
            ],
        }
    )

    st.dataframe(
        workflow_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        <div class="info-card soft-orange">
            <h3>⚠️ Human-in-the-loop principle</h3>
            The system supports prioritization and organization of evidence.
            It does not automatically make veterinary diagnoses, legal judgments,
            or intervention decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# PAGE: ALERT CENTER
# =========================================================
elif page == "📧 Alert Center":
    page_header(
        "PROTOTYPE ALERT WORKFLOW",
        "📧 Intervention & Email Alert Center",
        "Review the latest analyzed incident and, during the prototype demo, "
        "send a human-review alert to a configured email inbox.",
    )

    sender_email, _, default_recipient = get_email_config()

    # -----------------------------------------------------
    # EMAIL SYSTEM STATUS
    # -----------------------------------------------------
    if email_config_ready():
        st.markdown(
            """
            <div class="info-card soft-green">
                <h3>✅ Email System Ready</h3>
                <p style="margin-bottom:0;">
                    Gmail SMTP credentials were loaded securely from the local
                    environment configuration. The app password is not displayed.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="info-card soft-orange">
                <h3>⚙️ Email Setup Required</h3>
                <p style="margin-bottom:0;">
                    Add the Gmail sender address and Google App Password to the
                    local <b>.env</b> file, then restart Streamlit.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if "latest_predicted_label" in st.session_state:
        latest_sari = st.session_state.get("latest_sari", {})
        latest_incident = st.session_state.get("latest_incident_text", "")
        latest_prediction = st.session_state.get("latest_predicted_label", "N/A")
        manual_review_required = st.session_state.get("manual_review_required", False)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "Latest Incident",
                latest_prediction,
                "Linear SVM result",
            )

        with c2:
            metric_card(
                "SARI Score",
                latest_sari.get("score", "N/A"),
                "Heuristic priority index",
            )

        with c3:
            metric_card(
                "Priority",
                latest_sari.get("level", "N/A"),
                "Prototype decision support",
            )

        with c4:
            metric_card(
                "Review Status",
                "Required" if manual_review_required else "Standard",
                "Human-in-the-loop",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        st.text_area(
            "Latest analyzed incident",
            value=latest_incident,
            height=140,
            disabled=True,
        )

        evidence = latest_sari.get("evidence", [])
        evidence_text = " • ".join(evidence) if evidence else "No explicit evidence signal"

        st.markdown(
            f"""
            <div class="info-card soft-blue">
                <h3>🔎 Alert Evidence Summary</h3>
                <p><b>Detected evidence:</b> {evidence_text}</p>
                <p><b>Model–evidence status:</b>
                {"⚠️ Disagreement detected" if manual_review_required else "✅ No predefined critical disagreement"}</p>
                <p style="margin-bottom:0;"><b>Recommended action:</b>
                {"Manual verification before operational intervention." if manual_review_required else "Review according to operational context."}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">📨 Send Prototype Email Alert</div>',
            unsafe_allow_html=True,
        )

        recipient = st.text_input(
            "Recipient email",
            value=default_recipient,
            placeholder="example@gmail.com",
            help="For the classroom demo, you can use your own Gmail as the recipient.",
        )

        subject, body = build_alert_email(
            latest_incident,
            latest_prediction,
            latest_sari,
            manual_review_required,
        )

        with st.expander("Preview email before sending"):
            st.text_input("Subject", value=subject, disabled=True)
            st.text_area("Email body", value=body, height=420, disabled=True)

        send_clicked = st.button(
            "📧 Send Alert Email",
            use_container_width=True,
            type="primary",
            disabled=not email_config_ready(),
        )

        if send_clicked:
            try:
                with st.spinner("Sending prototype alert..."):
                    send_gmail_alert(
                        recipient=recipient.strip(),
                        subject=subject,
                        body=body,
                    )

                st.success(
                    f"✅ Alert email sent successfully to {recipient.strip()}."
                )

                st.session_state["last_email_recipient"] = recipient.strip()
                st.session_state["last_email_status"] = "Sent"

            except smtplib.SMTPAuthenticationError:
                st.error(
                    "Gmail authentication failed. Use a Google App Password, "
                    "not your normal Gmail password."
                )

            except Exception as exc:
                st.error(f"Email could not be sent: {exc}")

    else:
        st.info(
            "No live incident has been analyzed yet. Open "
            "'🧠 Live Incident Analyzer', analyze a case, then return here."
        )

        st.button(
            "📧 Send Alert Email",
            disabled=True,
            use_container_width=True,
        )

    st.caption(
        "Prototype alerts support human review only. They do not trigger an "
        "automatic veterinary, legal, or field intervention."
    )


# =========================================================
# PAGE: METHODOLOGY
# =========================================================
elif page == "📚 Methodology":
    page_header(
        "RESEARCH TRANSPARENCY",
        "📚 Methodology & System Scope",
        "A transparent summary of the analytical pipeline, model evaluation, "
        "field component and methodological limitations.",
    )

    st.markdown(
        """
        ### News Intelligence Pipeline

        **Expanded collection → Relevance filtering → Bangladesh scope refinement → Assisted review → TF–IDF → Model comparison → Incident classification**

        **1,094 → 690 → 250 → 226**

        The final corpus is evaluated using Linear SVM, Logistic Regression
        and Complement Naive Bayes. Macro F1 is used as the principal model
        selection criterion because the incident categories are imbalanced.

        ### SARI V2

        SARI V2 is maintained separately from the supervised incident classifier.
        It is a transparent heuristic prioritization index rather than a learned
        probability of harm.

        ### Field Intelligence

        The field component contains **325 localized animal observations**
        collected across selected Badda localities. Health, vaccination,
        sterilization, behavior, age and location indicators are analyzed
        descriptively.

        ### Evidence Integration

        News records and field-survey observations are **not row-level merged**.
        They represent different statistical units and are combined only at the
        evidence and decision-support level.

        ### Important Limitations

        The news labels are weakly supervised and only partially reviewed.
        Rare classes remain very small. SARI V2 has not been externally calibrated.
        The Badda survey is localized and not nationally representative.
        Reliable news coordinates were insufficient for DBSCAN/HDBSCAN hotspot
        analysis, and field photographs were not used as verified computer-vision labels.
        """
    )

    st.success(
        "The interface is designed as a research and demonstration prototype "
        "consistent with the methodology described in the V2 paper."
    )