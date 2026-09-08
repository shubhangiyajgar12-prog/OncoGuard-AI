from pathlib import Path
import sys
import math

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# DATABASE IMPORTS
# ============================================================

from database.connection import SessionLocal
from database.models import PatientProfile
from database.crud import (
    get_patient_profile_by_id,
    update_patient_profile,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Health Profile | OncoGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

ACCENT = "#5B4BDB"
BG = "#F6F7FB"
CARD = "#FFFFFF"
TEXT = "#182033"
MUTED = "#667085"
BORDER = "#E1E5EE"

STEPS = [
    "Basic Information",
    "Lifestyle",
    "Medical History",
    "Body & Prevention",
    "Review",
]


GENDER_TO_MODEL = {
    "Female": 0,
    "Male": 1,
}

SMOKING_TO_MODEL = {
    "No": 0,
    "Yes, I currently smoke": 1,
}

GENETIC_TO_MODEL = {
    "No known inherited risk": 0,
    "Some increased inherited risk": 1,
    "Higher inherited risk": 2,
}

CANCER_HISTORY_TO_MODEL = {
    "No previous cancer diagnosis": 0,
    "Yes, I have had cancer before": 1,
}


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       GLOBAL
    -------------------------------------------------------- */

    .stApp {
        background: #F6F7FB !important;
    }

    [data-testid="stAppViewContainer"] {
        background: #F6F7FB !important;
    }

    [data-testid="stMain"] {
        background: #F6F7FB !important;
    }

    section[data-testid="stMain"] {
        color: #182033 !important;
    }

    .block-container {
        max-width: 1120px !important;
        padding-top: 3.2rem !important;
        padding-bottom: 3rem !important;
    }


    /* --------------------------------------------------------
       STREAMLIT HEADER
       -------------------------------------------------------- */

    [data-testid="stHeader"] {
        background: #F6F7FB !important;
        border-bottom: none !important;
        box-shadow: none !important;
    }

    [data-testid="stHeader"] * {
        color: #182033 !important;
    }


    /* --------------------------------------------------------
       MAIN TEXT
       -------------------------------------------------------- */

    section[data-testid="stMain"] p,
    section[data-testid="stMain"] span,
    section[data-testid="stMain"] label,
    section[data-testid="stMain"] h1,
    section[data-testid="stMain"] h2,
    section[data-testid="stMain"] h3,
    section[data-testid="stMain"] h4,
    section[data-testid="stMain"] h5,
    section[data-testid="stMain"] h6 {
        color: #182033 !important;
    }

    section[data-testid="stMain"] .stCaption,
    section[data-testid="stMain"] [data-testid="stCaptionContainer"] {
        color: #667085 !important;
    }


    /* --------------------------------------------------------
       SIDEBAR
       -------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background: #20212B !important;
        border-right: 1px solid #30313D !important;
    }

    section[data-testid="stSidebar"] * {
        color: #F4F5F7 !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #373946 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #F4F5F7 !important;
    }

    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #FFFFFF !important;
        color: #182033 !important;
        border: 1px solid #D9DDE7 !important;
        border-radius: 10px !important;
    }

    section[data-testid="stSidebar"] [data-baseweb="select"] span {
        color: #182033 !important;
    }


    /* --------------------------------------------------------
       HEADINGS
       -------------------------------------------------------- */

    .page-title {
        font-size: 2.05rem;
        font-weight: 750;
        letter-spacing: -0.035em;
        color: #182033 !important;
        margin-bottom: 0.25rem;
    }

    .page-subtitle {
        font-size: 0.98rem;
        line-height: 1.55;
        color: #667085 !important;
        margin-bottom: 1.15rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #182033 !important;
        margin-bottom: 0.2rem;
    }

    .section-subtitle {
        font-size: 0.9rem;
        color: #667085 !important;
        margin-bottom: 1.3rem;
    }


    /* --------------------------------------------------------
       PROGRESS
       -------------------------------------------------------- */

    [data-testid="stProgress"] {
        margin-top: 0.25rem !important;
        margin-bottom: 0.65rem !important;
    }

    [data-testid="stProgress"] > div {
        background: #E5E7EF !important;
        border-radius: 999px !important;
        height: 7px !important;
    }

    [data-testid="stProgress"] > div > div {
        background: #5B4BDB !important;
        border-radius: 999px !important;
    }


    /* --------------------------------------------------------
       QUESTIONS
       -------------------------------------------------------- */

    .question-label {
        font-size: 0.98rem;
        font-weight: 650;
        color: #182033 !important;
        margin-top: 0.55rem;
        margin-bottom: 0.45rem;
    }

    .question-help {
        font-size: 0.84rem;
        color: #667085 !important;
        line-height: 1.45;
        margin-top: 0.35rem;
        margin-bottom: 0.9rem;
    }


    /* --------------------------------------------------------
       PILLS / SEGMENTED CONTROLS
       -------------------------------------------------------- */

    [data-testid="stPills"],
    [data-testid="stSegmentedControl"] {
        width: 100% !important;
    }

    [data-testid="stPills"] button,
    [data-testid="stSegmentedControl"] button {
        background: #FFFFFF !important;
        color: #344054 !important;
        border: 1px solid #D9DDE7 !important;
        border-radius: 10px !important;
        min-height: 46px !important;
        font-weight: 500 !important;
        box-shadow: none !important;
    }

    /* Current Streamlit DOM: selected */
    [data-testid="stPills"] button[data-testid="stBaseButton-primary"],
    [data-testid="stSegmentedControl"] button[data-testid="stBaseButton-primary"] {
        background: #5B4BDB !important;
        color: #FFFFFF !important;
        border-color: #5B4BDB !important;
        box-shadow: 0 3px 10px rgba(91, 75, 219, 0.16) !important;
    }

    /* Current Streamlit DOM: unselected */
    [data-testid="stPills"] button[data-testid="stBaseButton-secondary"],
    [data-testid="stSegmentedControl"] button[data-testid="stBaseButton-secondary"] {
        background: #FFFFFF !important;
        color: #344054 !important;
        border-color: #D9DDE7 !important;
    }

    /* Older DOM fallback */
    [data-testid="stPills"] button[aria-pressed="true"],
    [data-testid="stPills"] button[aria-selected="true"],
    [data-testid="stSegmentedControl"] button[aria-pressed="true"],
    [data-testid="stSegmentedControl"] button[aria-selected="true"] {
        background: #5B4BDB !important;
        color: #FFFFFF !important;
        border-color: #5B4BDB !important;
    }

    [data-testid="stPills"] button[aria-pressed="false"],
    [data-testid="stPills"] button[aria-selected="false"],
    [data-testid="stSegmentedControl"] button[aria-pressed="false"],
    [data-testid="stSegmentedControl"] button[aria-selected="false"] {
        background: #FFFFFF !important;
        color: #344054 !important;
        border-color: #D9DDE7 !important;
    }


    /* --------------------------------------------------------
       SLIDERS
       -------------------------------------------------------- */

    [data-testid="stSlider"] {
        padding-top: 0.25rem !important;
        padding-bottom: 0.15rem !important;
    }

    [data-testid="stSlider"] [role="slider"] {
        box-shadow: 0 0 0 2px #FFFFFF, 0 0 0 4px rgba(91, 75, 219, 0.25) !important;
    }


    /* --------------------------------------------------------
       BUTTONS
       -------------------------------------------------------- */

    .stButton > button {
        min-height: 46px !important;
        border-radius: 10px !important;
        border: 1px solid #D9DDE7 !important;
        background: #FFFFFF !important;
        color: #182033 !important;
        font-weight: 600 !important;
        transition: all 0.15s ease !important;
    }

    .stButton > button:hover {
        border-color: #5B4BDB !important;
        color: #5B4BDB !important;
        background: #FAF9FF !important;
    }

    .stButton > button[kind="primary"] {
        background: #5B4BDB !important;
        border-color: #5B4BDB !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(91, 75, 219, 0.18) !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: #493BC5 !important;
        border-color: #493BC5 !important;
        color: #FFFFFF !important;
    }

    .stButton > button:disabled {
        opacity: 0.5 !important;
        cursor: not-allowed !important;
        background: #E7E9EF !important;
        color: #7B8190 !important;
        border-color: #D9DDE7 !important;
        box-shadow: none !important;
    }

    .stButton > button:focus,
    .stButton > button:focus-visible,
    input:focus,
    input:focus-visible {
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(91, 75, 219, 0.25) !important;
    }


    /* --------------------------------------------------------
       REVIEW CARDS
       -------------------------------------------------------- */

    .review-card {
        background: #FFFFFF !important;
        border: 1px solid #DDE2EC !important;
        border-radius: 14px !important;
        padding: 1.15rem 1.2rem !important;
        min-height: 245px !important;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.025) !important;
    }

    .review-card-title {
        font-size: 1rem;
        font-weight: 700;
        color: #182033 !important;
        margin-bottom: 1rem;
    }

    .review-label {
        font-size: 0.7rem;
        font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #7A8499 !important;
        margin-top: 0.55rem;
        margin-bottom: 0.15rem;
    }

    .review-value {
        font-size: 0.92rem;
        font-weight: 600;
        color: #182033 !important;
        margin-bottom: 0.55rem;
    }


    /* --------------------------------------------------------
       BMI RESULT
       -------------------------------------------------------- */

    .bmi-card {
        background: #F7F6FF !important;
        border: 1px solid #DDD9FF !important;
        border-radius: 12px !important;
        padding: 0.95rem 1rem !important;
        margin-top: 0.65rem !important;
    }

    .bmi-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #6F659F !important;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    .bmi-value {
        font-size: 1.7rem;
        font-weight: 750;
        color: #30257F !important;
        line-height: 1.2;
    }


    /* --------------------------------------------------------
       PREVENTION CARD
       -------------------------------------------------------- */

    .prevention-card {
        background: #FFFFFF !important;
        border: 1px solid #DDE2EC !important;
        border-radius: 14px !important;
        padding: 1.15rem 1.2rem !important;
        margin-top: 1rem !important;
    }

    .prevention-title {
        font-size: 1rem;
        font-weight: 700;
        color: #182033 !important;
    }

    .prevention-text {
        font-size: 0.88rem;
        line-height: 1.5;
        color: #667085 !important;
    }


    /* --------------------------------------------------------
       FOOTER NOTE
       -------------------------------------------------------- */

    .privacy-note {
        border-top: 1px solid #E1E5EE;
        margin-top: 1.6rem;
        padding-top: 0.85rem;
        font-size: 0.78rem;
        color: #7A8499 !important;
        text-align: center;
    }


    /* --------------------------------------------------------
       REMOVE EXTRA VERTICAL SPACE
       -------------------------------------------------------- */

    [data-testid="stVerticalBlock"] {
        gap: 0.55rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "health_profile_step": 1,
    "health_profile_patient_id": None,

    "hp_age": None,
    "hp_gender": None,
    "hp_smoking": None,
    "hp_physical_activity": None,
    "hp_alcohol": None,
    "hp_genetic_risk": None,
    "hp_cancer_history": None,

    "hp_height": 170,
    "hp_weight": 65,

    # Personalized prevention questions
    "hp_smoking_support": None,
    "hp_activity_goal": None,
    "hp_alcohol_tracking": None,
    "hp_family_discussion": None,
    "hp_followup_care": None,

    "hp_profile_saved": False,
}


for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# DATASET HELPERS
# ============================================================

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "cancer_prediction_dataset.csv"


@st.cache_data
def load_dataset_ranges():
    """
    Load only the information required for friendly UI mapping.

    The model itself still receives the original dataset-compatible
    representations. These ranges are not clinical thresholds.
    """

    try:
        df = pd.read_csv(DATASET_PATH)

        age_min = int(math.floor(df["Age"].min()))
        age_max = int(math.ceil(df["Age"].max()))

        physical = df["PhysicalActivity"].dropna()
        alcohol = df["AlcoholIntake"].dropna()

        physical_values = [
            float(physical.quantile(0.125)),
            float(physical.quantile(0.375)),
            float(physical.quantile(0.625)),
            float(physical.quantile(0.875)),
        ]

        alcohol_values = [
            float(alcohol.quantile(0.125)),
            float(alcohol.quantile(0.375)),
            float(alcohol.quantile(0.625)),
            float(alcohol.quantile(0.875)),
        ]

        return {
            "age_min": age_min,
            "age_max": age_max,
            "physical_values": physical_values,
            "alcohol_values": alcohol_values,
        }

    except Exception:
        return {
            "age_min": 18,
            "age_max": 100,
            "physical_values": [1.0, 3.5, 6.0, 8.5],
            "alcohol_values": [0.0, 1.5, 2.8, 4.2],
        }


RANGES = load_dataset_ranges()


PHYSICAL_LABELS = [
    "Less active",
    "Lightly active",
    "Moderately active",
    "Very active",
]

ALCOHOL_LABELS = [
    "I don't drink alcohol",
    "Occasionally",
    "Regularly",
    "Frequently",
]


def nearest_category(value, values, labels):
    if value is None:
        return None

    try:
        distances = [abs(float(value) - float(v)) for v in values]
        return labels[distances.index(min(distances))]
    except Exception:
        return None


def category_to_value(category, labels, values):
    if category is None:
        return None

    try:
        index = labels.index(category)
        return float(values[index])
    except Exception:
        return None


# ============================================================
# DATABASE HELPERS
# ============================================================

def get_patient_ids():
    db = SessionLocal()

    try:
        rows = (
            db.query(PatientProfile.id)
            .order_by(PatientProfile.id.asc())
            .all()
        )

        return [row[0] for row in rows]

    finally:
        db.close()


def load_patient_profile(patient_id):
    db = SessionLocal()

    try:
        return get_patient_profile_by_id(db, patient_id)

    finally:
        db.close()


# ============================================================
# FORM INITIALIZATION
# ============================================================

def initialize_patient_form(patient_id):
    profile = load_patient_profile(patient_id)

    if not profile:
        return None

    st.session_state.hp_age = (
        int(profile.age)
        if profile.age is not None
        else int((RANGES["age_min"] + RANGES["age_max"]) / 2)
    )

    if profile.gender is not None:
        st.session_state.hp_gender = (
            "Female" if int(profile.gender) == 0 else "Male"
        )
    else:
        st.session_state.hp_gender = None

    if profile.smoking is not None:
        st.session_state.hp_smoking = (
            "Yes, I currently smoke"
            if int(profile.smoking) == 1
            else "No"
        )
    else:
        st.session_state.hp_smoking = None

    if profile.physical_activity is not None:
        st.session_state.hp_physical_activity = nearest_category(
            profile.physical_activity,
            RANGES["physical_values"],
            PHYSICAL_LABELS,
        )

    if profile.alcohol_intake is not None:
        st.session_state.hp_alcohol = nearest_category(
            profile.alcohol_intake,
            RANGES["alcohol_values"],
            ALCOHOL_LABELS,
        )

    if profile.genetic_risk is not None:
        genetic_value = int(profile.genetic_risk)

        mapping = {
            0: "No known inherited risk",
            1: "Some increased inherited risk",
            2: "Higher inherited risk",
        }

        st.session_state.hp_genetic_risk = mapping.get(genetic_value)

    if profile.cancer_history is not None:
        history_value = int(profile.cancer_history)

        st.session_state.hp_cancer_history = (
            "Yes, I have had cancer before"
            if history_value == 1
            else "No previous cancer diagnosis"
        )

    # Height and weight are not currently stored in the DB.
    # BMI is stored, so user is asked to confirm current measurements.
    st.session_state.hp_height = 170
    st.session_state.hp_weight = 65

    # Reset prevention questions when changing patient.
    st.session_state.hp_smoking_support = None
    st.session_state.hp_activity_goal = None
    st.session_state.hp_alcohol_tracking = None
    st.session_state.hp_family_discussion = None
    st.session_state.hp_followup_care = None

    st.session_state.health_profile_step = 1
    st.session_state.hp_profile_saved = False

    return profile


# ============================================================
# PATIENT SELECTION
# ============================================================

patient_ids = get_patient_ids()

if not patient_ids:
    st.error(
        "No patient profile is available. Create a patient profile before using Health Profile."
    )
    st.stop()


current_patient_id = st.session_state.health_profile_patient_id

if current_patient_id not in patient_ids:
    current_patient_id = patient_ids[0]

selected_patient_id = st.sidebar.selectbox(
    "Patient profile",
    patient_ids,
    index=patient_ids.index(current_patient_id),
    key="health_profile_patient_selector",
)

if selected_patient_id != st.session_state.health_profile_patient_id:
    st.session_state.health_profile_patient_id = selected_patient_id
    initialize_patient_form(selected_patient_id)


# First page load
if st.session_state.health_profile_patient_id is None:
    st.session_state.health_profile_patient_id = selected_patient_id
    initialize_patient_form(selected_patient_id)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🛡️ OncoGuard AI")

st.sidebar.caption(
    "AI-assisted cancer risk assessment and prevention support."
)

st.sidebar.divider()

st.sidebar.markdown("**PATIENT PROFILE**")

st.sidebar.selectbox(
    "Select patient",
    patient_ids,
    index=patient_ids.index(st.session_state.health_profile_patient_id),
    key="health_profile_sidebar_patient",
    label_visibility="collapsed",
)


st.sidebar.divider()

st.sidebar.markdown("**PROFILE JOURNEY**")

current_step = st.session_state.health_profile_step

for index, step_name in enumerate(STEPS, start=1):

    if index < current_step:
        st.sidebar.markdown(f"✓  **{index}. {step_name}**")

    elif index == current_step:
        st.sidebar.markdown(
            f"●  **{index}. {step_name}**"
        )

    else:
        st.sidebar.caption(
            f"○  {index}. {step_name}"
        )


st.sidebar.divider()

st.sidebar.caption(
    "Your health information is used to support the risk assessment workflow. "
    "The model provides risk estimates and does not diagnose cancer."
)


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="page-title">Health Profile</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="page-subtitle">'
    "Tell us about your health and everyday habits. "
    "Your answers are converted into the input format used by the current research model."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# SINGLE PROGRESS BAR
# ============================================================

progress_value = (current_step - 1) / (len(STEPS) - 1)

st.progress(progress_value)

progress_left, progress_right = st.columns([1, 1])

with progress_left:
    st.caption(f"Step {current_step} of {len(STEPS)}")

with progress_right:
    st.caption(
        STEPS[current_step - 1],
        text_alignment="right",
    )


# ============================================================
# HELPER: CHOICE CONTROL
# ============================================================

def choice_control(label, options, key, help_text=None):

    current_value = st.session_state.get(key)

    if hasattr(st, "pills"):

        value = st.pills(
            label,
            options,
            selection_mode="single",
            default=current_value,
            key=key,
            width="stretch",
        )

    else:

        current_index = (
            options.index(current_value)
            if current_value in options
            else 0
        )

        value = st.selectbox(
            label,
            options,
            index=current_index,
            key=key,
        )

    if help_text:
        st.caption(help_text)

    return value


# ============================================================
# NAVIGATION
# ============================================================

def go_next():
    st.session_state.health_profile_step += 1
    st.rerun()


def go_back():
    st.session_state.health_profile_step -= 1
    st.rerun()


# ============================================================
# STEP 1 — BASIC INFORMATION
# ============================================================

if current_step == 1:

    st.markdown(
        '<div class="section-title">Basic Information</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        "Let's start with a few basic details."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="question-label">How old are you?</div>',
        unsafe_allow_html=True,
    )

    st.slider(
        "Age",
        min_value=RANGES["age_min"],
        max_value=RANGES["age_max"],
        value=int(st.session_state.hp_age),
        step=1,
        key="hp_age",
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="question-help">'
        "Age is one of the features used by the current research model."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="question-label">How do you identify your gender?</div>',
        unsafe_allow_html=True,
    )

    choice_control(
        "",
        ["Female", "Male"],
        "hp_gender",
        "Choose the option that best matches the information you want to use for this assessment.",
    )

    st.write("")

    _, button_col = st.columns([1.7, 1])

    with button_col:
        if st.button(
            "Continue →",
            type="primary",
            use_container_width=True,
        ):
            if st.session_state.hp_gender is None:
                st.warning("Please select an option before continuing.")
            else:
                go_next()


# ============================================================
# STEP 2 — LIFESTYLE
# ============================================================

elif current_step == 2:

    st.markdown(
        '<div class="section-title">Lifestyle</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        "A few simple questions about everyday habits."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="question-label">Do you currently smoke?</div>',
        unsafe_allow_html=True,
    )

    choice_control(
        "",
        ["No", "Yes, I currently smoke"],
        "hp_smoking",
        "Choose the option that best describes your current smoking status.",
    )

    st.markdown(
        '<div class="question-label">Which best describes your usual physical activity?</div>',
        unsafe_allow_html=True,
    )

    choice_control(
        "",
        PHYSICAL_LABELS,
        "hp_physical_activity",
        "These descriptions are used to create a model-compatible activity representation.",
    )

    st.markdown(
        '<div class="question-label">How would you describe your alcohol intake?</div>',
        unsafe_allow_html=True,
    )

    choice_control(
        "",
        ALCOHOL_LABELS,
        "hp_alcohol",
        "These categories are used only to create a model-compatible representation and are not clinical drinking thresholds.",
    )

    st.write("")

    back_col, _, next_col = st.columns([1, 0.15, 1])

    with back_col:
        if st.button(
            "← Back",
            use_container_width=True,
        ):
            go_back()

    with next_col:
        if st.button(
            "Continue →",
            type="primary",
            use_container_width=True,
        ):
            required = [
                st.session_state.hp_smoking,
                st.session_state.hp_physical_activity,
                st.session_state.hp_alcohol,
            ]

            if any(value is None for value in required):
                st.warning(
                    "Please answer all lifestyle questions before continuing."
                )
            else:
                go_next()


# ============================================================
# STEP 3 — MEDICAL HISTORY
# ============================================================

elif current_step == 3:

    st.markdown(
        '<div class="section-title">Medical History</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        "This helps the model understand relevant health-history factors."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="question-label">'
        "Has a healthcare professional ever told you about an inherited cancer risk?"
        "</div>",
        unsafe_allow_html=True,
    )

    choice_control(
        "",
        list(GENETIC_TO_MODEL.keys()) + ["I'm not sure"],
        "hp_genetic_risk",
        "If you're unsure, choose “I'm not sure”. Uncertainty is not silently treated as increased risk.",
    )

    st.markdown(
        '<div class="question-label">'
        "Have you ever been diagnosed with cancer?"
        "</div>",
        unsafe_allow_html=True,
    )

    choice_control(
        "",
        list(CANCER_HISTORY_TO_MODEL.keys()) + ["I'm not sure"],
        "hp_cancer_history",
        "This information is used as a model input and does not determine or confirm a cancer diagnosis.",
    )

    st.write("")

    back_col, _, next_col = st.columns([1, 0.15, 1])

    with back_col:
        if st.button(
            "← Back",
            use_container_width=True,
        ):
            go_back()

    with next_col:
        if st.button(
            "Continue →",
            type="primary",
            use_container_width=True,
        ):

            if st.session_state.hp_genetic_risk is None:
                st.warning("Please answer the inherited-risk question.")

            elif st.session_state.hp_cancer_history is None:
                st.warning("Please answer the previous-cancer question.")

            else:
                go_next()


# ============================================================
# STEP 4 — BODY + PERSONALIZED PREVENTION
# ============================================================

elif current_step == 4:

    st.markdown(
        '<div class="section-title">Body & Prevention</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        "Confirm your measurements and answer a few personalized prevention questions."
        "</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # BODY MEASUREMENTS
    # --------------------------------------------------------

    height_col, weight_col = st.columns(2)

    with height_col:

        st.markdown(
            '<div class="question-label">Height</div>',
            unsafe_allow_html=True,
        )

        st.slider(
            "Height",
            min_value=130,
            max_value=220,
            value=int(st.session_state.hp_height),
            step=1,
            key="hp_height",
            label_visibility="collapsed",
        )

        st.caption(f"{st.session_state.hp_height} cm")

    with weight_col:

        st.markdown(
            '<div class="question-label">Weight</div>',
            unsafe_allow_html=True,
        )

        st.slider(
            "Weight",
            min_value=35,
            max_value=180,
            value=int(st.session_state.hp_weight),
            step=1,
            key="hp_weight",
            label_visibility="collapsed",
        )

        st.caption(f"{st.session_state.hp_weight} kg")

    height_m = st.session_state.hp_height / 100
    bmi = st.session_state.hp_weight / (height_m * height_m)

    st.markdown(
        f"""
        <div class="bmi-card">
            <div class="bmi-label">Calculated BMI</div>
            <div class="bmi-value">{bmi:.1f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "BMI is used as a model input. It is not a diagnosis or a complete measure of health."
    )

    # --------------------------------------------------------
    # PERSONALIZED PREVENTION QUESTIONS
    # --------------------------------------------------------

    st.markdown(
        '<div class="prevention-card">'
        '<div class="prevention-title">Personalized Prevention Check</div>'
        '<div class="prevention-text">'
        "These questions are selected based on your previous answers. "
        "They help personalize prevention guidance and do not change the trained model inputs."
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Smoking follow-up
    if st.session_state.hp_smoking == "Yes, I currently smoke":

        st.markdown(
            '<div class="question-label">'
            "Would you like your prevention plan to include support for reducing or stopping smoking?"
            "</div>",
            unsafe_allow_html=True,
        )

        choice_control(
            "",
            ["Yes", "Not right now"],
            "hp_smoking_support",
        )

    # Activity follow-up
    if st.session_state.hp_physical_activity in [
        "Less active",
        "Lightly active",
    ]:

        st.markdown(
            '<div class="question-label">'
            "Would you like to set a simple physical-activity goal for your prevention plan?"
            "</div>",
            unsafe_allow_html=True,
        )

        choice_control(
            "",
            ["Yes", "Not right now"],
            "hp_activity_goal",
        )

    # Alcohol follow-up
    if st.session_state.hp_alcohol in [
        "Occasionally",
        "Regularly",
        "Frequently",
    ]:

        st.markdown(
            '<div class="question-label">'
            "Would you like to track your alcohol-related habits as part of your prevention plan?"
            "</div>",
            unsafe_allow_html=True,
        )

        choice_control(
            "",
            ["Yes", "Not right now"],
            "hp_alcohol_tracking",
        )

    # Genetic-risk follow-up
    if st.session_state.hp_genetic_risk in [
        "Some increased inherited risk",
        "Higher inherited risk",
    ]:

        st.markdown(
            '<div class="question-label">'
            "Have you discussed this inherited-risk information with a healthcare professional?"
            "</div>",
            unsafe_allow_html=True,
        )

        choice_control(
            "",
            ["Yes", "Not yet", "I'm not sure"],
            "hp_family_discussion",
        )

    # Previous cancer follow-up
    if st.session_state.hp_cancer_history == "Yes, I have had cancer before":

        st.markdown(
            '<div class="question-label">'
            "Are you currently following a healthcare professional's recommended follow-up plan?"
            "</div>",
            unsafe_allow_html=True,
        )

        choice_control(
            "",
            ["Yes", "No", "I'm not sure"],
            "hp_followup_care",
        )

    st.write("")

    back_col, _, next_col = st.columns([1, 0.15, 1])

    with back_col:
        if st.button(
            "← Back",
            use_container_width=True,
        ):
            go_back()

    with next_col:
        if st.button(
            "Review Profile →",
            type="primary",
            use_container_width=True,
        ):
            go_next()


# ============================================================
# STEP 5 — REVIEW
# ============================================================

elif current_step == 5:

    st.markdown(
        '<div class="section-title">Review Your Profile</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        "Review the information before saving your health profile."
        "</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # REQUIRED MODEL INPUT VALIDATION
    # --------------------------------------------------------

    model_ready = all(
        value is not None
        for value in [
            st.session_state.hp_age,
            st.session_state.hp_gender,
            st.session_state.hp_smoking,
            st.session_state.hp_physical_activity,
            st.session_state.hp_alcohol,
            st.session_state.hp_genetic_risk
            if st.session_state.hp_genetic_risk != "I'm not sure"
            else None,
            st.session_state.hp_cancer_history
            if st.session_state.hp_cancer_history != "I'm not sure"
            else None,
            st.session_state.hp_height,
            st.session_state.hp_weight,
        ]
    )

    if model_ready:
        st.success(
            "Your required model inputs are complete. You can save this profile."
        )
    else:
        st.warning(
            "Some required model inputs are missing or uncertain. "
            "Please go back and provide the information required by the current model."
        )

    st.write("")

    # --------------------------------------------------------
    # VALUES
    # --------------------------------------------------------

    age = st.session_state.hp_age
    gender = st.session_state.hp_gender
    smoking = st.session_state.hp_smoking
    physical = st.session_state.hp_physical_activity
    alcohol = st.session_state.hp_alcohol
    genetic = st.session_state.hp_genetic_risk
    cancer_history = st.session_state.hp_cancer_history
    height = st.session_state.hp_height
    weight = st.session_state.hp_weight

    bmi = weight / ((height / 100) ** 2)

    # --------------------------------------------------------
    # REVIEW GRID — ROW 1
    # --------------------------------------------------------

    left, right = st.columns(2, gap="medium")

    with left:

        with st.container(border=True):

            st.markdown(
                '<div class="review-card-title">Basic Information</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="review-label">Age</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="review-value">{age} years</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="review-label">Gender</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="review-value">{gender or "Not provided"}</div>',
                unsafe_allow_html=True,
            )

            if st.button(
                "Edit Basic Information",
                key="edit_basic",
                use_container_width=True,
            ):
                st.session_state.health_profile_step = 1
                st.rerun()

    with right:

        with st.container(border=True):

            st.markdown(
                '<div class="review-card-title">Lifestyle</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="review-label">Smoking</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="review-value">{smoking or "Not provided"}</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="review-label">Physical Activity</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="review-value">{physical or "Not provided"}</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="review-label">Alcohol Intake</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="review-value">{alcohol or "Not provided"}</div>',
                unsafe_allow_html=True,
            )

            if st.button(
                "Edit Lifestyle",
                key="edit_lifestyle",
                use_container_width=True,
            ):
                st.session_state.health_profile_step = 2
                st.rerun()

    st.write("")

    # --------------------------------------------------------
    # REVIEW GRID — ROW 2
    # --------------------------------------------------------

    left, right = st.columns(2, gap="medium")

    with left:

        with st.container(border=True):

            st.markdown(
                '<div class="review-card-title">Medical History</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="review-label">Inherited Risk</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="review-value">{genetic or "Not provided"}</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="review-label">Previous Cancer History</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="review-value">{cancer_history or "Not provided"}</div>',
                unsafe_allow_html=True,
            )

            if st.button(
                "Edit Medical History",
                key="edit_medical",
                use_container_width=True,
            ):
                st.session_state.health_profile_step = 3
                st.rerun()

    with right:

        with st.container(border=True):

            st.markdown(
                '<div class="review-card-title">Body Measurements</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="review-label">Height</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="review-value">{height} cm</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="review-label">Weight</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="review-value">{weight} kg</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="review-label">Calculated BMI</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="review-value">{bmi:.1f}</div>',
                unsafe_allow_html=True,
            )

            if st.button(
                "Edit Measurements & Prevention",
                key="edit_body",
                use_container_width=True,
            ):
                st.session_state.health_profile_step = 4
                st.rerun()

    # --------------------------------------------------------
    # PREVENTION SUMMARY
    # --------------------------------------------------------

    st.write("")

    with st.container(border=True):

        st.markdown(
            '<div class="review-card-title">Personalized Prevention Focus</div>',
            unsafe_allow_html=True,
        )

        prevention_items = []

        if smoking == "Yes, I currently smoke":
            if st.session_state.hp_smoking_support == "Yes":
                prevention_items.append(
                    "Smoking reduction or cessation support"
                )
            else:
                prevention_items.append(
                    "Smoking status identified as a prevention focus"
                )

        if physical in ["Less active", "Lightly active"]:
            if st.session_state.hp_activity_goal == "Yes":
                prevention_items.append(
                    "Physical-activity goal"
                )
            else:
                prevention_items.append(
                    "Physical activity identified as a prevention focus"
                )

        if alcohol in ["Occasionally", "Regularly", "Frequently"]:
            if st.session_state.hp_alcohol_tracking == "Yes":
                prevention_items.append(
                    "Alcohol-habit tracking"
                )
            else:
                prevention_items.append(
                    "Alcohol intake identified as a prevention focus"
                )

        if genetic in [
            "Some increased inherited risk",
            "Higher inherited risk",
        ]:
            prevention_items.append(
                "Inherited-risk discussion with a qualified healthcare professional"
            )

        if cancer_history == "Yes, I have had cancer before":
            prevention_items.append(
                "Appropriate follow-up care and healthcare discussion"
            )

        if prevention_items:

            for item in prevention_items:
                st.markdown(f"• {item}")

        else:

            st.caption(
                "No additional prevention follow-up was triggered by the current answers."
            )

    st.caption(
        "These prevention prompts are personalized from your answers. "
        "They are not diagnoses and do not imply that a specific behavior causes or prevents cancer."
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    st.write("")

    back_col, _, save_col = st.columns([1, 0.15, 1])

    with back_col:

        if st.button(
            "← Back",
            use_container_width=True,
        ):
            go_back()

    with save_col:

        if st.button(
            "Save Health Profile",
            type="primary",
            use_container_width=True,
            disabled=not model_ready,
        ):

            patient_id = st.session_state.health_profile_patient_id

            genetic_value = GENETIC_TO_MODEL.get(
                st.session_state.hp_genetic_risk
            )

            cancer_value = CANCER_HISTORY_TO_MODEL.get(
                st.session_state.hp_cancer_history
            )

            physical_value = category_to_value(
                st.session_state.hp_physical_activity,
                PHYSICAL_LABELS,
                RANGES["physical_values"],
            )

            alcohol_value = category_to_value(
                st.session_state.hp_alcohol,
                ALCOHOL_LABELS,
                RANGES["alcohol_values"],
            )

            gender_value = GENDER_TO_MODEL.get(
                st.session_state.hp_gender
            )

            smoking_value = SMOKING_TO_MODEL.get(
                st.session_state.hp_smoking
            )

            db = SessionLocal()

            try:

                profile = get_patient_profile_by_id(
                    db,
                    patient_id,
                )

                if profile is None:

                    st.error(
                        "Patient profile could not be found."
                    )

                else:

                    updated = update_patient_profile(
                        db=db,
                        user_id=profile.user_id,
                        age=int(st.session_state.hp_age),
                        gender=gender_value,
                        bmi=float(bmi),
                        smoking=smoking_value,
                        genetic_risk=genetic_value,
                        physical_activity=float(physical_value),
                        alcohol_intake=float(alcohol_value),
                        cancer_history=cancer_value,
                    )

                    if updated:

                        st.session_state.hp_profile_saved = True

                        st.success(
                            "Health profile saved successfully."
                        )

                        st.info(
                            "Your next step is Risk Assessment. "
                            "The assessment will use the trained machine-learning model to calculate a model-predicted cancer risk."
                        )

                    else:

                        st.error(
                            "The health profile could not be updated."
                        )

            except Exception as exc:

                db.rollback()

                st.error(
                    f"Unable to save the health profile: {exc}"
                )

            finally:

                db.close()


# ============================================================
# PRIVACY / MEDICAL DISCLAIMER
# ============================================================

st.markdown(
    '<div class="privacy-note">'
    "🔒 OncoGuard AI provides model-based risk assessment and prevention support. "
    "It does not diagnose cancer or replace professional medical advice."
    "</div>",
    unsafe_allow_html=True,
)