from pathlib import Path
import sys
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# DATABASE
# ============================================================

from database.connection import SessionLocal
from database.crud import (
    get_patient_profile_by_id,
    get_latest_assessment,
    get_patient_assessments,
    get_assessment_explanations,
)


# ============================================================
# RISK SERVICE
# ============================================================

from services.risk_assessment_service import (
    assess_patient_risk,
)


# ============================================================
# CONSTANTS
# ============================================================

ACCENT = "#5B4BDB"
ACCENT_DARK = "#4F40CA"
BACKGROUND = "#F6F7FB"
WHITE = "#FFFFFF"
TEXT = "#182033"
MUTED = "#66718A"
BORDER = "#E1E5EE"


# ============================================================
# PAGE CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       PAGE
       ======================================================== */

    .stApp {
        background: #F6F7FB !important;
    }

    [data-testid="stAppViewContainer"] {
        background: #F6F7FB !important;
    }

    [data-testid="stMain"] {
        background: #F6F7FB !important;
    }

    [data-testid="stHeader"] {
        background: #F6F7FB !important;
        box-shadow: none !important;
        border-bottom: none !important;
    }

    .block-container {
        max-width: 1050px !important;
        padding-top: 2.8rem !important;
        padding-bottom: 3rem !important;
    }


    /* ========================================================
       TEXT
       ======================================================== */

    section[data-testid="stMain"] h1,
    section[data-testid="stMain"] h2,
    section[data-testid="stMain"] h3,
    section[data-testid="stMain"] h4,
    section[data-testid="stMain"] h5,
    section[data-testid="stMain"] h6,
    section[data-testid="stMain"] p,
    section[data-testid="stMain"] span,
    section[data-testid="stMain"] label {
        color: #182033 !important;
    }


    /* ========================================================
       HEADER
       ======================================================== */

    .page-eyebrow {
        color: #5B4BDB !important;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .page-title {
        color: #182033 !important;
        font-size: 2.15rem;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.04em;
        margin-bottom: 0.45rem;
    }

    .page-subtitle {
        color: #66718A !important;
        font-size: 0.94rem;
        line-height: 1.65;
        max-width: 900px;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        min-height: 42px !important;
        border-radius: 10px !important;
        background: #FFFFFF !important;
        color: #182033 !important;
        border: 1px solid #D8DCE7 !important;
        font-weight: 650 !important;
    }

    .stButton > button:hover {
        border-color: #5B4BDB !important;
        color: #5B4BDB !important;
    }

    .stButton > button[kind="primary"] {
        background: #5B4BDB !important;
        border-color: #5B4BDB !important;
        color: #FFFFFF !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: #4F40CA !important;
        border-color: #4F40CA !important;
        color: #FFFFFF !important;
    }


    /* ========================================================
       RISK HERO
       ======================================================== */

    .risk-hero {
        background: #FFFFFF;
        border: 1px solid #E1E5EE;
        border-radius: 18px;
        padding: 1.5rem;
        margin-top: 1.2rem;
    }

    .risk-label {
        color: #747F97 !important;
        font-size: 0.76rem;
        font-weight: 650;
        margin-bottom: 0.25rem;
    }

    .risk-value {
        color: #182033 !important;
        font-size: 3.2rem;
        font-weight: 850;
        line-height: 1.05;
        letter-spacing: -0.04em;
    }

    .risk-band {
        display: inline-block;
        background: #EFEDFF;
        color: #4D40B6 !important;
        border-radius: 999px;
        padding: 0.38rem 0.75rem;
        font-size: 0.76rem;
        font-weight: 750;
        margin-top: 0.55rem;
    }


    /* ========================================================
       CARDS
       ======================================================== */

    .card {
        background: #FFFFFF;
        border: 1px solid #E1E5EE;
        border-radius: 15px;
        padding: 1.15rem;
        margin-top: 1rem;
    }

    .card-title {
        color: #182033 !important;
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .card-subtitle {
        color: #66718A !important;
        font-size: 0.8rem;
        line-height: 1.55;
    }


    /* ========================================================
       FACTOR CARDS
       ======================================================== */

    .factor-card {
        background: #FFFFFF;
        border: 1px solid #E1E5EE;
        border-radius: 13px;
        padding: 1rem;
        margin-top: 0.7rem;
    }

    .factor-name {
        color: #182033 !important;
        font-size: 0.9rem;
        font-weight: 750;
        margin-bottom: 0.3rem;
    }

    .factor-detail {
        color: #66718A !important;
        font-size: 0.79rem;
        line-height: 1.55;
    }

    .factor-direction {
        color: #4D40B6 !important;
        font-size: 0.77rem;
        font-weight: 700;
        margin-top: 0.4rem;
    }


    /* ========================================================
       INFORMATION BOX
       ======================================================== */

    .info-box {
        background: #FAF9FF;
        border: 1px solid #DED9FF;
        border-radius: 13px;
        padding: 1rem 1.1rem;
        margin-top: 1rem;
    }

    .info-title {
        color: #4D40B6 !important;
        font-size: 0.82rem;
        font-weight: 750;
        margin-bottom: 0.25rem;
    }

    .info-text {
        color: #68738A !important;
        font-size: 0.79rem;
        line-height: 1.6;
    }


    /* ========================================================
       SECTION
       ======================================================== */

    .section-title {
        color: #182033 !important;
        font-size: 1.15rem;
        font-weight: 800;
        margin-top: 1.7rem;
        margin-bottom: 0.35rem;
    }

    .section-description {
        color: #66718A !important;
        font-size: 0.82rem;
        line-height: 1.55;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .privacy-note {
        color: #66718A !important;
        font-size: 0.75rem;
        line-height: 1.6;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #E1E5EE;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_patient_id():
    """
    Resolve the active patient ID from Streamlit session state.
    """

    possible_keys = [
        "current_patient_id",
        "patient_id",
        "health_profile_patient_id",
        "hp_patient_id",
    ]

    for key in possible_keys:

        value = st.session_state.get(key)

        if value is not None:

            try:
                return int(value)

            except (TypeError, ValueError):
                continue

    return None


def get_value(
    item,
    *keys,
    default=None,
):
    """
    Read values from either SQLAlchemy objects or dictionaries.
    """

    if item is None:
        return default

    if isinstance(item, dict):

        for key in keys:

            if key in item:
                return item[key]

        return default


    for key in keys:

        if hasattr(item, key):

            return getattr(
                item,
                key,
            )


    return default


def probability_to_percentage(
    probability
):
    """
    Convert probability in [0, 1] to percentage.
    """

    try:

        return float(probability) * 100.0

    except (
        TypeError,
        ValueError,
    ):

        return 0.0


def format_timestamp(
    timestamp
):
    """
    Format assessment timestamps safely.
    """

    if timestamp is None:
        return "Not available"

    if hasattr(
        timestamp,
        "strftime",
    ):

        return timestamp.strftime(
            "%d %b %Y, %I:%M %p"
        )

    return str(timestamp)


def display_risk_band(
    risk_band
):
    """
    Convert stored risk-band terminology into patient-friendly
    wording.
    """

    if not risk_band:

        return "Not available"

    value = str(
        risk_band
    ).lower()


    if "low" in value:

        return "Lower predicted risk"


    if "medium" in value:

        return "Medium predicted risk"


    if "moderate" in value:

        return "Moderate predicted risk"


    if "high" in value:

        return "Higher predicted risk"


    return str(
        risk_band
    )


def shap_direction(
    shap_value
):
    """
    Describe SHAP direction using safe model-focused wording.
    """

    try:

        value = float(
            shap_value
        )

    except (
        TypeError,
        ValueError,
    ):

        return (
            "Directional contribution could not be determined."
        )


    if value > 0:

        return (
            "Important contributor toward a higher model score"
        )


    if value < 0:

        return (
            "Important contributor toward a lower model score"
        )


    return (
        "Little or no directional contribution"
    )


# ============================================================
# PAGE STATE
# ============================================================

patient_id = get_patient_id()


if patient_id is None:

    st.error(
        "No patient profile is currently selected."
    )

    st.info(
        "Open Health Profile first and select a patient profile."
    )

    st.stop()


# ============================================================
# LOAD PATIENT DATA
# ============================================================

db = SessionLocal()

try:

    profile = get_patient_profile_by_id(
        db,
        patient_id,
    )

    latest = get_latest_assessment(
        db,
        patient_id,
    )

    assessments = get_patient_assessments(
        db,
        patient_id,
    )

finally:

    db.close()


# ============================================================
# PROFILE VALIDATION
# ============================================================

if profile is None:

    st.error(
        "The selected patient profile could not be found."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="page-eyebrow">'
    "OncoGuard AI"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="page-title">'
    "Cancer Risk Assessment"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="page-subtitle">'
    "Review your current model-predicted risk and understand "
    "which recorded factors were important contributors to "
    "the machine-learning assessment."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# ACTIONS
# ============================================================

action_left, action_right = st.columns(
    [1, 1]
)


with action_left:

    run_assessment = st.button(
        "Run New Assessment",
        type="primary",
        use_container_width=True,
        key="run_new_assessment",
    )


with action_right:

    refresh = st.button(
        "Refresh",
        use_container_width=True,
        key="refresh_assessment",
    )


if refresh:

    st.rerun()


# ============================================================
# RUN NEW ASSESSMENT
# ============================================================

if run_assessment:

    db = SessionLocal()

    try:

        with st.spinner(
            "Generating the model-predicted risk assessment..."
        ):

            result = assess_patient_risk(
                db=db,
                patient_id=patient_id,
            )


        st.session_state[
            "last_assessment_result"
        ] = result

        st.success(
            "New model-predicted risk assessment created."
        )

        st.rerun()


    except Exception as exc:

        db.rollback()

        st.error(
            f"Unable to run the assessment: {exc}"
        )


    finally:

        db.close()


# ============================================================
# NO ASSESSMENT YET
# ============================================================

if latest is None:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="card-title">'
        "No assessment available yet"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="card-subtitle">'
        "Complete and save the Health Profile first. "
        "Then select Run New Assessment to generate a "
        "model-predicted cancer risk estimate."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


    st.markdown(
        '<div class="info-box">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="info-title">'
        "About this assessment"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="info-text">'
        "OncoGuard AI provides model-predicted risk estimates "
        "for research and prevention-support purposes. "
        "The result is not a cancer diagnosis."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


    st.stop()


# ============================================================
# CURRENT ASSESSMENT VALUES
# ============================================================

probability = get_value(
    latest,
    "probability",
    "risk_probability",
    default=0.0,
)


risk_percentage = probability_to_percentage(
    probability
)


risk_band = display_risk_band(
    get_value(
        latest,
        "risk_band",
        default="Not available",
    )
)


threshold = get_value(
    latest,
    "threshold",
    "decision_threshold",
)


model_version = get_value(
    latest,
    "model_version",
    default=None,
)


if model_version is None:

    model_version = get_value(
        latest,
        "model_version_id",
        default="Not available",
    )


assessment_timestamp = get_value(
    latest,
    "timestamp",
    "created_at",
    "assessment_date",
)


# ============================================================
# CURRENT RISK HERO
# ============================================================

st.markdown(
    '<div class="section-title">'
    "Current Model-Predicted Risk"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "This percentage represents the output of the trained "
    "research model for the recorded profile."
    "</div>",
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="risk-hero">',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="risk-label">'
    "Model-predicted risk"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="risk-value">'
    f"{risk_percentage:.2f}%"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="risk-band">'
    f"{risk_band}"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# ASSESSMENT DETAILS
# ============================================================

detail1, detail2, detail3 = st.columns(
    3
)


with detail1:

    st.metric(
        "Assessment date",
        format_timestamp(
            assessment_timestamp
        ),
    )


with detail2:

    if threshold is not None:

        try:

            threshold_display = (
                f"{float(threshold):.2f}"
            )

        except (
            TypeError,
            ValueError,
        ):

            threshold_display = str(
                threshold
            )

    else:

        threshold_display = "Not available"


    st.metric(
        "Decision threshold",
        threshold_display,
    )


with detail3:

    st.metric(
        "Model version",
        str(
            model_version
        ),
    )


# ============================================================
# RISK GAUGE
# ============================================================

st.markdown(
    '<div class="section-title">'
    "Risk Overview"
    "</div>",
    unsafe_allow_html=True,
)


figure = go.Figure()


figure.add_trace(
    go.Indicator(
        mode="gauge+number",
        value=risk_percentage,
        number={
            "suffix": "%",
            "font": {
                "size": 34,
                "color": TEXT,
            },
        },
        gauge={
            "axis": {
                "range": [
                    0,
                    100,
                ],
                "tickcolor": "#7A849B",
            },
            "bar": {
                "color": ACCENT,
                "thickness": 0.25,
            },
            "bgcolor": "#EEF0F6",
            "borderwidth": 0,
        },
    )
)


figure.update_layout(
    height=260,
    margin=dict(
        l=25,
        r=25,
        t=20,
        b=10,
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    font={
        "color": TEXT
    },
)


st.plotly_chart(
    figure,
    use_container_width=True,
)


# ============================================================
# MODEL EXPLANATIONS
# ============================================================

st.markdown(
    '<div class="section-title">'
    "What influenced this assessment?"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "These are factors the machine-learning model identified "
    "as important contributors to the current model score."
    "</div>",
    unsafe_allow_html=True,
)


db = SessionLocal()

try:

    explanations = get_assessment_explanations(
        db,
        latest.id,
    )

finally:

    db.close()


if not explanations:

    st.info(
        "No stored explanation is available for this assessment yet."
    )

else:

    for explanation in explanations[:10]:

        feature = get_value(
            explanation,
            "feature",
            default="Feature",
        )

        feature_value = get_value(
            explanation,
            "feature_value",
            default="Not available",
        )

        shap_value = get_value(
            explanation,
            "shap_value",
            default=0.0,
        )

        direction = shap_direction(
            shap_value
        )


        st.markdown(
            '<div class="factor-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="factor-name">'
            f"{feature}"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="factor-detail">'
            f"Profile value: {feature_value}"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="factor-direction">'
            f"{direction}"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


# ============================================================
# HOW TO READ EXPLANATIONS
# ============================================================

st.markdown(
    '<div class="info-box">',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="info-title">'
    "How to read these factors"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="info-text">'
    "These explanations describe model contribution, not "
    "medical causation. A factor being important to the "
    "model does not mean that it caused a person's risk."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# ASSESSMENT HISTORY
# ============================================================

st.markdown(
    '<div class="section-title">'
    "Assessment History"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    "Previous assessments show how the model-predicted risk "
    "has changed across recorded assessments."
    "</div>",
    unsafe_allow_html=True,
)


history_rows = []


for assessment in assessments:

    assessment_probability = get_value(
        assessment,
        "probability",
        "risk_probability",
        default=0.0,
    )


    assessment_band = get_value(
        assessment,
        "risk_band",
        default="Not available",
    )


    assessment_time = get_value(
        assessment,
        "timestamp",
        "created_at",
        "assessment_date",
    )


    history_rows.append(
        {
            "Assessment": format_timestamp(
                assessment_time
            ),
            "Model-predicted risk": (
                f"{probability_to_percentage(assessment_probability):.2f}%"
            ),
            "Risk range": display_risk_band(
                assessment_band
            ),
        }
    )


if history_rows:

    history_dataframe = pd.DataFrame(
        history_rows
    )

    st.dataframe(
        history_dataframe,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No previous assessment history is available."
    )


# ============================================================
# RESULT INFORMATION
# ============================================================

st.markdown(
    '<div class="info-box">',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="info-title">'
    "About your result"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="info-text">'
    "OncoGuard AI provides a machine-learning-based cancer "
    "risk assessment for research and prevention-support "
    "purposes. This result is not a cancer diagnosis, does "
    "not confirm or rule out cancer, and should not replace "
    "evaluation or advice from a qualified healthcare "
    "professional."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# PRIVACY FOOTER
# ============================================================

st.markdown(
    '<div class="privacy-note">'
    "🔒 OncoGuard AI uses recorded health-profile information "
    "to generate model-based risk estimates. Results are "
    "intended for research and decision-support purposes."
    "</div>",
    unsafe_allow_html=True,
)