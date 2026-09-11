from pathlib import Path
import sys
import math

import pandas as pd
import streamlit as st


# ================================================================
# PROJECT ROOT
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ================================================================
# DATABASE IMPORTS
# ================================================================

from database.connection import SessionLocal
from database.models import PatientProfile
from database.crud import (
    get_patient_profile_by_id,
    update_patient_profile,
)


# ================================================================
# PAGE CONSTANTS
# ================================================================

ACCENT = "#5B4BDB"
ACCENT_DARK = "#4939C6"
BACKGROUND = "#F6F7FB"
CARD = "#FFFFFF"
TEXT = "#182033"
MUTED = "#667085"
BORDER = "#E2E5EC"
SOFT_PURPLE = "#F1EFFF"

STEPS = [
    "Basic Information",
    "Lifestyle",
    "Medical History",
    "Body & Prevention",
    "Review",
]


# ================================================================
# MODEL MAPPINGS
# ================================================================

# IMPORTANT:
# These values are the numeric representation expected by the
# current research dataset/model layer.
#
# The UI uses patient-friendly wording and converts the answer
# before saving it.

GENDER_TO_MODEL = {
    "Male": 0,
    "Female": 1,
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


# ================================================================
# PATIENT-FRIENDLY OPTIONS
# ================================================================

GENDER_OPTIONS = [
    "Male",
    "Female",
]

SMOKING_OPTIONS = [
    "No",
    "Yes, I currently smoke",
]

SMOKING_DURATION_OPTIONS = [
    "Less than 1 year",
    "1–5 years",
    "6–10 years",
    "More than 10 years",
    "Prefer not to say",
]

SMOKING_FREQUENCY_OPTIONS = [
    "Occasionally",
    "Daily",
    "Multiple times a day",
    "Prefer not to say",
]

ALCOHOL_OPTIONS = [
    "I don't drink alcohol",
    "I drink alcohol",
]

ALCOHOL_FREQUENCY_OPTIONS = [
    "Occasionally",
    "1–2 days per week",
    "3–5 days per week",
    "Most days",
    "Prefer not to say",
]

ACTIVITY_LEVEL_OPTIONS = [
    "Mostly inactive",
    "Lightly active",
    "Moderately active",
    "Very active",
]

GENETIC_OPTIONS = [
    "No known inherited risk",
    "Some increased inherited risk",
    "Higher inherited risk",
    "I'm not sure",
]

RISK_SOURCE_OPTIONS = [
    "Family history",
    "Genetic testing",
    "Healthcare professional",
    "I'm not sure",
]

CANCER_HISTORY_OPTIONS = [
    "No previous cancer diagnosis",
    "Yes, I have had cancer before",
    "I'm not sure",
]

FOLLOWUP_OPTIONS = [
    "Yes",
    "No",
    "I'm not sure",
]

PREVENTION_OPTIONS = [
    "Yes",
    "Not right now",
]


# ================================================================
# DATASET RANGE
# ================================================================

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cancer_prediction_dataset.csv"
)


@st.cache_data
def load_dataset_limits():
    """
    Read the actual numeric limits used by the research dataset.

    The model uses:
        Age             -> integer
        BMI             -> continuous
        PhysicalActivity-> continuous hours/week
        AlcoholIntake   -> continuous units/week
        GeneticRisk     -> 0/1/2
        Smoking         -> 0/1
        CancerHistory   -> 0/1
        Gender          -> 0/1
    """

    try:
        df = pd.read_csv(DATASET_PATH)

        return {
            "age_min": int(math.floor(df["Age"].min())),
            "age_max": int(math.ceil(df["Age"].max())),
            "bmi_min": float(df["BMI"].min()),
            "bmi_max": float(df["BMI"].max()),
            "physical_min": float(df["PhysicalActivity"].min()),
            "physical_max": float(df["PhysicalActivity"].max()),
            "alcohol_min": float(df["AlcoholIntake"].min()),
            "alcohol_max": float(df["AlcoholIntake"].max()),
        }

    except Exception:

        # Safe fallback matching the documented research ranges.
        return {
            "age_min": 20,
            "age_max": 80,
            "bmi_min": 15.0,
            "bmi_max": 40.0,
            "physical_min": 0.0,
            "physical_max": 10.0,
            "alcohol_min": 0.0,
            "alcohol_max": 5.0,
        }


DATA_LIMITS = load_dataset_limits()


# ================================================================
# GLOBAL CSS
# ================================================================

st.markdown(
    """
    <style>

    /* =========================================================
       PAGE
    ========================================================= */

    .stApp {
        background: #F6F7FB !important;
    }

    [data-testid="stAppViewContainer"] {
        background: #F6F7FB !important;
    }

    [data-testid="stMain"] {
        background: #F6F7FB !important;
    }

    .block-container {
        max-width: 1080px !important;
        padding-top: 2.6rem !important;
        padding-bottom: 3.5rem !important;
    }


    /* =========================================================
       HEADER
    ========================================================= */

    [data-testid="stHeader"] {
        background: #F6F7FB !important;
        border: none !important;
        box-shadow: none !important;
    }

    [data-testid="stHeader"] * {
        color: #182033 !important;
    }


    /* =========================================================
       TEXT
    ========================================================= */

    h1,
    h2,
    h3,
    h4,
    h5,
    h6 {
        color: #182033 !important;
    }

    p,
    label,
    span {
        color: #182033;
    }

    .page-eyebrow {
        color: #5B4BDB !important;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .page-title {
        color: #182033 !important;
        font-size: 2.25rem;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 0.35rem;
    }

    .page-description {
        color: #667085 !important;
        font-size: 1rem;
        line-height: 1.65;
        margin-bottom: 1.2rem;
    }

    .section-title {
        color: #182033 !important;
        font-size: 1.35rem;
        font-weight: 750;
        margin-top: 0.8rem;
        margin-bottom: 0.2rem;
    }

    .section-description {
        color: #667085 !important;
        font-size: 0.95rem;
        line-height: 1.55;
        margin-bottom: 1.5rem;
    }

    .question-label {
        color: #182033 !important;
        font-size: 1rem;
        font-weight: 700;
        margin-top: 1.35rem;
        margin-bottom: 0.55rem;
    }

    .helper-text {
        color: #667085 !important;
        font-size: 0.84rem;
        line-height: 1.5;
        margin-top: 0.35rem;
        margin-bottom: 0.75rem;
    }


    /* =========================================================
       PROGRESS
    ========================================================= */

    .progress-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 0.6rem;
        margin-bottom: 0.45rem;
    }

    .progress-text {
        color: #667085 !important;
        font-size: 0.82rem;
        font-weight: 600;
    }

    [data-testid="stProgress"] {
        margin-bottom: 1.8rem;
    }

    [data-testid="stProgress"] > div {
        background: #E6E2FF !important;
        border-radius: 999px !important;
        height: 9px !important;
    }

    [data-testid="stProgress"] > div > div {
        background: #5B4BDB !important;
        border-radius: 999px !important;
    }


    /* =========================================================
       SEGMENTED CONTROLS
    ========================================================= */

    [data-testid="stSegmentedControl"] {
        width: 100% !important;
    }

    [data-testid="stSegmentedControl"] > div {
        gap: 0.55rem !important;
    }

    [data-testid="stSegmentedControl"] button {
        min-height: 48px !important;
        border-radius: 12px !important;
        border: 1px solid #D9DCE5 !important;
        background: #FFFFFF !important;
        color: #182033 !important;
        font-weight: 600 !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
    }

    [data-testid="stSegmentedControl"] button:hover {
        border-color: #A9A1F2 !important;
        background: #FAF9FF !important;
        color: #182033 !important;
    }

    [data-testid="stSegmentedControl"]
    button[aria-pressed="true"],
    [data-testid="stSegmentedControl"]
    button[aria-selected="true"] {
        background: #5B4BDB !important;
        border-color: #5B4BDB !important;
        color: #FFFFFF !important;
    }

    [data-testid="stSegmentedControl"]
    button[aria-pressed="true"] *,
    [data-testid="stSegmentedControl"]
    button[aria-selected="true"] * {
        color: #FFFFFF !important;
    }

    [data-testid="stSegmentedControl"] button:focus,
    [data-testid="stSegmentedControl"] button:focus-visible {
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(91, 75, 219, 0.18) !important;
    }


    /* =========================================================
       INPUTS
    ========================================================= */

    [data-testid="stNumberInput"] input {
        background: #FFFFFF !important;
        color: #182033 !important;
        border: 1px solid #D9DCE5 !important;
        border-radius: 11px !important;
        min-height: 46px !important;
    }

    [data-testid="stNumberInput"] input:focus {
        border-color: #5B4BDB !important;
        box-shadow: 0 0 0 3px rgba(91, 75, 219, 0.12) !important;
    }

    [data-testid="stNumberInput"] button {
        background: #FFFFFF !important;
        color: #5B4BDB !important;
        border: none !important;
    }


    /* =========================================================
       BUTTONS
    ========================================================= */

    .stButton > button {
        min-height: 46px !important;
        border-radius: 11px !important;
        border: 1px solid #D9DCE5 !important;
        background: #FFFFFF !important;
        color: #182033 !important;
        font-weight: 650 !important;
        box-shadow: none !important;
    }

    .stButton > button:hover {
        border-color: #A9A1F2 !important;
        color: #182033 !important;
        background: #FAF9FF !important;
    }

    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: #5B4BDB !important;
        border-color: #5B4BDB !important;
        color: #FFFFFF !important;
    }

    .stButton > button[kind="primary"] *,
    .stButton > button[data-testid="stBaseButton-primary"] * {
        color: #FFFFFF !important;
    }

    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background: #4939C6 !important;
        border-color: #4939C6 !important;
        color: #FFFFFF !important;
    }

    .stButton > button:focus,
    .stButton > button:focus-visible {
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(91, 75, 219, 0.18) !important;
    }


    /* =========================================================
       INFORMATION CARDS
    ========================================================= */

    .info-card {
        background: #FFFFFF;
        border: 1px solid #E2E5EC;
        border-radius: 16px;
        padding: 1.1rem 1.25rem;
        margin: 1.2rem 0;
    }

    .info-card-title {
        color: #182033 !important;
        font-size: 0.95rem;
        font-weight: 750;
        margin-bottom: 0.25rem;
    }

    .info-card-text {
        color: #667085 !important;
        font-size: 0.86rem;
        line-height: 1.55;
    }

    .followup-card {
        background: #F7F5FF;
        border: 1px solid #DDD8FF;
        border-left: 4px solid #5B4BDB;
        border-radius: 13px;
        padding: 1rem 1.15rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .followup-title {
        color: #4939C6 !important;
        font-weight: 750;
        font-size: 0.92rem;
        margin-bottom: 0.2rem;
    }

    .followup-text {
        color: #667085 !important;
        font-size: 0.83rem;
        line-height: 1.5;
    }


    /* =========================================================
       BMI CARD
    ========================================================= */

    .bmi-card {
        background: #FFFFFF;
        border: 1px solid #DDD8FF;
        border-radius: 15px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
    }

    .bmi-label {
        color: #667085 !important;
        font-size: 0.82rem;
        font-weight: 650;
    }

    .bmi-value {
        color: #5B4BDB !important;
        font-size: 2rem;
        font-weight: 800;
        margin-top: 0.1rem;
    }

    .bmi-note {
        color: #667085 !important;
        font-size: 0.8rem;
        margin-top: 0.1rem;
    }


    /* =========================================================
       REVIEW CARDS
    ========================================================= */

    .review-card {
        background: #FFFFFF;
        border: 1px solid #E2E5EC;
        border-radius: 16px;
        padding: 1.2rem;
        min-height: 245px;
        margin-bottom: 1rem;
    }

    .review-card-title {
        color: #182033 !important;
        font-size: 1rem;
        font-weight: 750;
        margin-bottom: 1rem;
        padding-bottom: 0.65rem;
        border-bottom: 1px solid #ECEEF3;
    }

    .review-item {
        margin-bottom: 0.8rem;
    }

    .review-label {
        color: #667085 !important;
        font-size: 0.76rem;
        font-weight: 600;
        margin-bottom: 0.12rem;
    }

    .review-value {
        color: #182033 !important;
        font-size: 0.92rem;
        font-weight: 650;
    }


    /* =========================================================
       MODEL INPUT SUMMARY
    ========================================================= */

    .model-summary {
        background: #F7F5FF;
        border: 1px solid #DDD8FF;
        border-radius: 15px;
        padding: 1.1rem 1.2rem;
        margin-top: 1rem;
    }

    .model-summary-title {
        color: #4939C6 !important;
        font-size: 0.92rem;
        font-weight: 750;
        margin-bottom: 0.35rem;
    }

    .model-summary-text {
        color: #667085 !important;
        font-size: 0.82rem;
        line-height: 1.55;
    }


    /* =========================================================
       PRIVACY
    ========================================================= */

    .privacy-footer {
        border-top: 1px solid #E2E5EC;
        margin-top: 2rem;
        padding-top: 1rem;
        color: #667085 !important;
        font-size: 0.78rem;
        line-height: 1.5;
    }


    /* =========================================================
       SIDEBAR
    ========================================================= */

    section[data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid #E2E5EC !important;
    }

    section[data-testid="stSidebar"] > div {
        background: #FFFFFF !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div {
        color: #182033 !important;
    }

    .sidebar-brand {
        color: #182033 !important;
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }

    .sidebar-subtitle {
        color: #667085 !important;
        font-size: 0.78rem;
        line-height: 1.45;
    }

    .sidebar-heading {
        color: #182033 !important;
        font-size: 0.78rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.8rem;
        margin-bottom: 0.5rem;
    }

    .sidebar-step {
        padding: 0.55rem 0.7rem;
        border-radius: 9px;
        margin-bottom: 0.25rem;
        font-size: 0.82rem;
        color: #667085 !important;
    }

    .sidebar-step.active {
        background: #F1EFFF;
        color: #4939C6 !important;
        font-weight: 750;
    }

    .sidebar-step.completed {
        color: #344054 !important;
    }


    /* =========================================================
       MOBILE
    ========================================================= */

    @media (max-width: 800px) {

        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        .page-title {
            font-size: 1.75rem;
        }

        [data-testid="stSegmentedControl"] > div {
            flex-wrap: wrap !important;
        }

        [data-testid="stSegmentedControl"] button {
            min-width: 100% !important;
        }

        .review-card {
            min-height: auto;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ================================================================
# SESSION STATE HELPERS
# ================================================================

def set_default(key, value):
    """
    Set a session-state value only if it does not exist.
    """
    if key not in st.session_state:
        st.session_state[key] = value


def clear_followup_values():
    """
    Remove contextual follow-up answers when the parent answer
    no longer requires them.
    """

    if st.session_state.get("hp_smoking") != "Yes, I currently smoke":
        st.session_state.hp_smoking_duration = None
        st.session_state.hp_smoking_frequency = None
        st.session_state.hp_smoking_support = None

    if st.session_state.get("hp_alcohol") != "I drink alcohol":
        st.session_state.hp_alcohol_frequency = None
        st.session_state.hp_alcohol_days = 0
        st.session_state.hp_alcohol_tracking = None

    if st.session_state.get("hp_activity") not in [
        "Mostly inactive",
        "Lightly active",
    ]:
        st.session_state.hp_activity_days = None
        st.session_state.hp_activity_goal = None

    if st.session_state.get("hp_genetic") not in [
        "Some increased inherited risk",
        "Higher inherited risk",
    ]:
        st.session_state.hp_risk_source = None
        st.session_state.hp_family_discussion = None

    if st.session_state.get("hp_cancer_history") != (
        "Yes, I have had cancer before"
    ):
        st.session_state.hp_followup_care = None


# ================================================================
# CHOICE CONTROL
# ================================================================

def choice_control(
    label,
    options,
    key,
    current_value=None,
    help_text=None,
):
    """
    Clean patient-friendly selection control.

    Uses Streamlit segmented controls when available.
    Falls back to selectbox for compatibility.
    """

    if hasattr(st, "segmented_control"):

        kwargs = {
            "label": label,
            "options": options,
            "selection_mode": "single",
            "key": key,
            "width": "stretch",
        }

        if key not in st.session_state and current_value in options:
            kwargs["default"] = current_value

        value = st.segmented_control(**kwargs)

    else:

        index = (
            options.index(current_value)
            if current_value in options
            else 0
        )

        value = st.selectbox(
            label,
            options,
            index=index,
            key=key,
        )

    if help_text:
        st.markdown(
            f'<div class="helper-text">{help_text}</div>',
            unsafe_allow_html=True,
        )

    return value


# ================================================================
# BMI
# ================================================================

def calculate_bmi(height_cm, weight_kg):
    """
    Calculate BMI from height and weight.

    BMI is stored as the model input.
    Height and weight are currently questionnaire values and
    are not separately persisted by the current database schema.
    """

    if height_cm is None or weight_kg is None:
        return None

    try:

        height_cm = float(height_cm)
        weight_kg = float(weight_kg)

        if height_cm <= 0 or weight_kg <= 0:
            return None

        height_m = height_cm / 100.0

        bmi = weight_kg / (height_m ** 2)

        return round(bmi, 1)

    except (TypeError, ValueError, ZeroDivisionError):

        return None


def bmi_description(bmi):
    if bmi is None:
        return "BMI will be calculated after height and weight are entered."

    if bmi < 18.5:
        return "Below the usual adult BMI reference range."

    if bmi < 25:
        return "Within the usual adult BMI reference range."

    if bmi < 30:
        return "Above the usual adult BMI reference range."

    return "Higher than the usual adult BMI reference range."


# ================================================================
# PHYSICAL ACTIVITY MAPPING
# ================================================================

def activity_choice_to_hours(choice):
    """
    Convert the patient-friendly activity description into a
    continuous weekly-hours value.

    IMPORTANT:
    We deliberately do not use quantile representatives anymore.

    The patient provides an exact estimate of weekly activity
    hours after selecting the broad activity level.

    This avoids mismatches between labels and numeric arrays.
    """

    if choice == "Mostly inactive":
        return 0.0

    if choice == "Lightly active":
        return 2.5

    if choice == "Moderately active":
        return 5.0

    if choice == "Very active":
        return 8.0

    return None


def activity_hours_from_patient_input():
    """
    Return the exact activity-hours value entered by the patient.

    If the patient supplied the optional exact value, use it.
    Otherwise use the category representative.
    """

    choice = st.session_state.get("hp_activity")

    if choice is None:
        return None

    exact_hours = st.session_state.get(
        "hp_activity_hours"
    )

    if exact_hours is not None:

        try:

            value = float(exact_hours)

            if (
                DATA_LIMITS["physical_min"]
                <= value
                <= DATA_LIMITS["physical_max"]
            ):
                return round(value, 2)

        except (TypeError, ValueError):
            pass

    return activity_choice_to_hours(choice)


# ================================================================
# ALCOHOL MAPPING
# ================================================================

def alcohol_frequency_to_days(choice):
    """
    Contextual frequency answer.

    This does not directly redefine the model feature.
    It helps the patient describe the pattern before entering
    the model-compatible weekly-unit estimate.
    """

    if choice == "Occasionally":
        return 1

    if choice == "1–2 days per week":
        return 2

    if choice == "3–5 days per week":
        return 4

    if choice == "Most days":
        return 6

    return None


# ================================================================
# VALUE CONVERSIONS
# ================================================================

def gender_value(choice):
    return GENDER_TO_MODEL.get(choice)


def smoking_value(choice):
    return SMOKING_TO_MODEL.get(choice)


def genetic_value(choice):
    return GENETIC_TO_MODEL.get(choice)


def cancer_history_value(choice):
    return CANCER_HISTORY_TO_MODEL.get(choice)


# ================================================================
# PATIENT DATABASE HELPERS
# ================================================================

def get_patient_ids():
    """
    Load available patient profile IDs.
    """

    db = SessionLocal()

    try:

        rows = (
            db.query(PatientProfile.id)
            .order_by(PatientProfile.id.asc())
            .all()
        )

        return [
            int(row[0])
            for row in rows
        ]

    finally:

        db.close()


def load_patient_profile(patient_id):
    """
    Load a patient profile and immediately return it while the
    session is active.
    """

    db = SessionLocal()

    try:

        profile = get_patient_profile_by_id(
            db,
            patient_id,
        )

        if profile is None:
            return None

        # Copy all fields needed by this page into a plain dict.
        # This avoids detached SQLAlchemy objects being retained
        # in Streamlit session state.
        return {
            "id": profile.id,
            "user_id": profile.user_id,
            "age": profile.age,
            "gender": profile.gender,
            "bmi": profile.bmi,
            "smoking": profile.smoking,
            "genetic_risk": profile.genetic_risk,
            "physical_activity": profile.physical_activity,
            "alcohol_intake": profile.alcohol_intake,
            "cancer_history": profile.cancer_history,
        }

    finally:

        db.close()


# ================================================================
# INITIALIZE FORM
# ================================================================

def initialize_patient_form(patient_id):
    """
    Initialize the complete questionnaire for a patient.

    Existing DB values are loaded into patient-friendly answers.
    """

    profile = load_patient_profile(patient_id)

    if profile is None:
        return None

    st.session_state.health_profile_patient_id = patient_id

    st.session_state.hp_age = (
        int(profile["age"])
        if profile["age"] is not None
        else 40
    )

    if profile["gender"] is not None:

        gender_numeric = int(profile["gender"])

        # Current project model mapping:
        # 0 = Male
        # 1 = Female
        st.session_state.hp_gender = (
            "Male"
            if gender_numeric == 0
            else "Female"
        )

    else:

        st.session_state.hp_gender = None

    if profile["smoking"] is not None:

        st.session_state.hp_smoking = (
            "Yes, I currently smoke"
            if int(profile["smoking"]) == 1
            else "No"
        )

    else:

        st.session_state.hp_smoking = None

    if profile["genetic_risk"] is not None:

        genetic_numeric = int(
            profile["genetic_risk"]
        )

        st.session_state.hp_genetic = {
            0: "No known inherited risk",
            1: "Some increased inherited risk",
            2: "Higher inherited risk",
        }.get(genetic_numeric)

    else:

        st.session_state.hp_genetic = None

    if profile["cancer_history"] is not None:

        st.session_state.hp_cancer_history = (
            "Yes, I have had cancer before"
            if int(profile["cancer_history"]) == 1
            else "No previous cancer diagnosis"
        )

    else:

        st.session_state.hp_cancer_history = None

    # ------------------------------------------------------------
    # Physical activity
    # ------------------------------------------------------------

    physical = profile["physical_activity"]

    if physical is not None:

        physical = float(physical)

        if physical < 2:
            activity = "Mostly inactive"

        elif physical < 4:
            activity = "Lightly active"

        elif physical < 7:
            activity = "Moderately active"

        else:
            activity = "Very active"

        st.session_state.hp_activity = activity
        st.session_state.hp_activity_hours = round(
            physical,
            2,
        )

    else:

        st.session_state.hp_activity = None
        st.session_state.hp_activity_hours = None

    # ------------------------------------------------------------
    # Alcohol
    # ------------------------------------------------------------

    alcohol = profile["alcohol_intake"]

    if alcohol is not None:

        alcohol = float(alcohol)

        if alcohol <= 0.05:

            st.session_state.hp_alcohol = (
                "I don't drink alcohol"
            )

            st.session_state.hp_alcohol_units = 0.0

        else:

            st.session_state.hp_alcohol = (
                "I drink alcohol"
            )

            st.session_state.hp_alcohol_units = round(
                alcohol,
                2,
            )

    else:

        st.session_state.hp_alcohol = None
        st.session_state.hp_alcohol_units = None

    # ------------------------------------------------------------
    # Height / weight
    # ------------------------------------------------------------

    # Current database stores BMI, not height and weight.
    # We therefore use a reasonable editable starting point.
    # The patient can change both before saving.

    default_height = 170.0

    if profile["bmi"] is not None:

        try:

            existing_bmi = float(profile["bmi"])

            estimated_weight = (
                existing_bmi
                * (default_height / 100.0) ** 2
            )

            estimated_weight = max(
                35.0,
                min(180.0, estimated_weight),
            )

        except (TypeError, ValueError):

            estimated_weight = 65.0

    else:

        estimated_weight = 65.0

    st.session_state.hp_height = round(
        default_height,
        1,
    )

    st.session_state.hp_weight = round(
        estimated_weight,
        1,
    )

    # ------------------------------------------------------------
    # Contextual follow-ups
    # ------------------------------------------------------------

    st.session_state.hp_smoking_duration = None
    st.session_state.hp_smoking_frequency = None
    st.session_state.hp_smoking_support = None

    st.session_state.hp_activity_days = None
    st.session_state.hp_activity_goal = None

    st.session_state.hp_alcohol_frequency = None
    st.session_state.hp_alcohol_days = None
    st.session_state.hp_alcohol_tracking = None

    st.session_state.hp_risk_source = None
    st.session_state.hp_family_discussion = None

    st.session_state.hp_followup_care = None

    st.session_state.hp_saved = False

    st.session_state.health_profile_step = 1

    return profile


# ================================================================
# SESSION INITIALIZATION
# ================================================================

set_default(
    "health_profile_patient_id",
    None,
)

set_default(
    "health_profile_step",
    1,
)

set_default(
    "hp_saved",
    False,
)


# ================================================================
# PATIENT LIST
# ================================================================

patient_ids = get_patient_ids()

if not patient_ids:

    st.error(
        "No patient profile is available in the database."
    )

    st.stop()


# ================================================================
# FIRST LOAD
# ================================================================

current_patient_id = (
    st.session_state.health_profile_patient_id
)

if (
    current_patient_id is None
    or current_patient_id not in patient_ids
):

    current_patient_id = patient_ids[0]

    initialize_patient_form(
        current_patient_id
    )


# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">🛡️ OncoGuard AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        "Cancer risk assessment and prevention support."
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div class="sidebar-heading">Patient profile</div>',
        unsafe_allow_html=True,
    )

    selected_patient_id = st.selectbox(
        "Patient profile",
        options=patient_ids,
        index=patient_ids.index(
            current_patient_id
        ),
        key="health_profile_patient_selector",
        label_visibility="collapsed",
    )

    if selected_patient_id != current_patient_id:

        initialize_patient_form(
            selected_patient_id
        )

        st.rerun()

    st.divider()

    st.markdown(
        '<div class="sidebar-heading">Profile journey</div>',
        unsafe_allow_html=True,
    )

    current_step = int(
        st.session_state.health_profile_step
    )

    for index, step_name in enumerate(
        STEPS,
        start=1,
    ):

        if index < current_step:

            st.markdown(
                f'<div class="sidebar-step completed">'
                f'✓ {index}. {step_name}'
                f'</div>',
                unsafe_allow_html=True,
            )

        elif index == current_step:

            st.markdown(
                f'<div class="sidebar-step active">'
                f'● {index}. {step_name}'
                f'</div>',
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                f'<div class="sidebar-step">'
                f'{index}. {step_name}'
                f'</div>',
                unsafe_allow_html=True,
            )


# ================================================================
# PAGE HEADER
# ================================================================

current_step = int(
    st.session_state.health_profile_step
)

st.markdown(
    '<div class="page-eyebrow">ONCOGUARD AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="page-title">Health Profile</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="page-description">'
    "Tell us about your health and everyday habits. "
    "Your answers are converted into the input format "
    "used by the current research model."
    "</div>",
    unsafe_allow_html=True,
)


# ================================================================
# PROGRESS
# ================================================================

progress_value = (
    current_step
    / len(STEPS)
)

progress_col1, progress_col2 = st.columns(
    [1, 1]
)

with progress_col1:

    st.markdown(
        f'<div class="progress-text">'
        f'Step {current_step} of {len(STEPS)}'
        f'</div>',
        unsafe_allow_html=True,
    )

with progress_col2:

    st.markdown(
        f'<div class="progress-text" '
        f'style="text-align:right;">'
        f'{STEPS[current_step - 1]}'
        f'</div>',
        unsafe_allow_html=True,
    )

st.progress(
    progress_value,
)


# ================================================================
# STEP 1 — BASIC INFORMATION
# ================================================================

if current_step == 1:

    st.markdown(
        '<div class="section-title">'
        "Basic Information"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        "Let’s begin with a few basic details."
        "</div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # AGE
    # ------------------------------------------------------------

    st.markdown(
        '<div class="question-label">'
        "How old are you?"
        "</div>",
        unsafe_allow_html=True,
    )

    age = st.number_input(
        "Age",
        min_value=DATA_LIMITS["age_min"],
        max_value=DATA_LIMITS["age_max"],
        value=int(
            st.session_state.hp_age
        ),
        step=1,
        key="hp_age_input",
        label_visibility="collapsed",
    )

    st.session_state.hp_age = int(age)

    st.markdown(
        '<div class="helper-text">'
        "Age is one of the inputs used by the current "
        "research model."
        "</div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # GENDER
    # ------------------------------------------------------------

    st.markdown(
        '<div class="question-label">'
        "How do you identify your gender?"
        "</div>",
        unsafe_allow_html=True,
    )

    gender = choice_control(
        "",
        GENDER_OPTIONS,
        "hp_gender_control",
        st.session_state.get("hp_gender"),
    )

    if gender is not None:

        st.session_state.hp_gender = gender

    st.markdown(
        '<div class="helper-text">'
        "Choose the option that best matches the information "
        "you want to use for this assessment."
        "</div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # INFORMATION
    # ------------------------------------------------------------

    st.markdown(
        '<div class="info-card">'
        '<div class="info-card-title">'
        "Why we ask"
        "</div>"
        '<div class="info-card-text">'
        "These basic details are combined with lifestyle and "
        "health-history information to create the structured "
        "input used by the research model."
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


# ================================================================
# STEP 2 — LIFESTYLE
# ================================================================

elif current_step == 2:

    st.markdown(
        '<div class="section-title">'
        "Lifestyle"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        "A few questions about everyday habits. "
        "Some follow-up questions appear only when relevant."
        "</div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # SMOKING
    # ------------------------------------------------------------

    st.markdown(
        '<div class="question-label">'
        "Do you currently smoke?"
        "</div>",
        unsafe_allow_html=True,
    )

    smoking = choice_control(
        "",
        SMOKING_OPTIONS,
        "hp_smoking_control",
        st.session_state.get("hp_smoking"),
    )

    if smoking is not None:

        st.session_state.hp_smoking = smoking

    # ------------------------------------------------------------
    # CONDITIONAL SMOKING QUESTIONS
    # ------------------------------------------------------------

    if (
        st.session_state.hp_smoking
        == "Yes, I currently smoke"
    ):

        st.markdown(
            '<div class="followup-card">'
            '<div class="followup-title">'
            "A little more context"
            "</div>"
            '<div class="followup-text">'
            "Because you indicated current smoking, "
            "we’ll ask a couple of additional questions. "
            "These contextual answers are not additional "
            "features in the current trained model."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="question-label">'
            "About how long have you been smoking?"
            "</div>",
            unsafe_allow_html=True,
        )

        smoking_duration = choice_control(
            "",
            SMOKING_DURATION_OPTIONS,
            "hp_smoking_duration_control",
            st.session_state.get(
                "hp_smoking_duration"
            ),
        )

        if smoking_duration is not None:

            st.session_state.hp_smoking_duration = (
                smoking_duration
            )

        st.markdown(
            '<div class="question-label">'
            "How frequently do you currently smoke?"
            "</div>",
            unsafe_allow_html=True,
        )

        smoking_frequency = choice_control(
            "",
            SMOKING_FREQUENCY_OPTIONS,
            "hp_smoking_frequency_control",
            st.session_state.get(
                "hp_smoking_frequency"
            ),
        )

        if smoking_frequency is not None:

            st.session_state.hp_smoking_frequency = (
                smoking_frequency
            )

    else:

        st.session_state.hp_smoking_duration = None
        st.session_state.hp_smoking_frequency = None
        st.session_state.hp_smoking_support = None

    # ------------------------------------------------------------
    # PHYSICAL ACTIVITY
    # ------------------------------------------------------------

    st.markdown(
        '<div class="question-label">'
        "How would you describe your usual physical activity?"
        "</div>",
        unsafe_allow_html=True,
    )

    activity = choice_control(
        "",
        ACTIVITY_LEVEL_OPTIONS,
        "hp_activity_control",
        st.session_state.get("hp_activity"),
    )

    if activity is not None:

        st.session_state.hp_activity = activity

    st.markdown(
        '<div class="helper-text">'
        "The research dataset represents physical activity "
        "as weekly hours, so we will ask for an estimate "
        "that can be passed directly to the model."
        "</div>",
        unsafe_allow_html=True,
    )

    # Exact model-compatible activity value.
    if st.session_state.get("hp_activity") is not None:

        default_hours = activity_choice_to_hours(
            st.session_state.hp_activity
        )

        existing_hours = (
            st.session_state.get(
                "hp_activity_hours"
            )
        )

        if existing_hours is None:

            existing_hours = default_hours

        st.markdown(
            '<div class="question-label">'
            "Approximately how many hours per week are you "
            "physically active?"
            "</div>",
            unsafe_allow_html=True,
        )

        activity_hours = st.number_input(
            "Weekly activity hours",
            min_value=float(
                DATA_LIMITS["physical_min"]
            ),
            max_value=float(
                DATA_LIMITS["physical_max"]
            ),
            value=float(
                existing_hours
            ),
            step=0.5,
            key="hp_activity_hours_input",
            label_visibility="collapsed",
        )

        st.session_state.hp_activity_hours = (
            float(activity_hours)
        )

    # ------------------------------------------------------------
    # CONDITIONAL ACTIVITY FOLLOW-UP
    # ------------------------------------------------------------

    if st.session_state.get("hp_activity") in [
        "Mostly inactive",
        "Lightly active",
    ]:

        st.markdown(
            '<div class="followup-card">'
            '<div class="followup-title">'
            "One additional activity question"
            "</div>"
            '<div class="followup-text">'
            "This helps describe your weekly activity pattern "
            "for prevention-focused guidance."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="question-label">'
            "On how many days per week are you physically active?"
            "</div>",
            unsafe_allow_html=True,
        )

        activity_days = st.number_input(
            "Active days per week",
            min_value=0,
            max_value=7,
            value=int(
                st.session_state.get(
                    "hp_activity_days"
                )
                or 0
            ),
            step=1,
            key="hp_activity_days_input",
            label_visibility="collapsed",
        )

        st.session_state.hp_activity_days = int(
            activity_days
        )

        st.markdown(
            '<div class="question-label">'
            "Would you like a simple activity goal included "
            "in your prevention plan?"
            "</div>",
            unsafe_allow_html=True,
        )

        activity_goal = choice_control(
            "",
            PREVENTION_OPTIONS,
            "hp_activity_goal_control",
            st.session_state.get(
                "hp_activity_goal"
            ),
        )

        if activity_goal is not None:

            st.session_state.hp_activity_goal = (
                activity_goal
            )

    else:

        st.session_state.hp_activity_days = None
        st.session_state.hp_activity_goal = None

    # ------------------------------------------------------------
    # ALCOHOL
    # ------------------------------------------------------------

    st.markdown(
        '<div class="question-label">'
        "Do you drink alcohol?"
        "</div>",
        unsafe_allow_html=True,
    )

    alcohol = choice_control(
        "",
        ALCOHOL_OPTIONS,
        "hp_alcohol_control",
        st.session_state.get("hp_alcohol"),
    )

    if alcohol is not None:

        st.session_state.hp_alcohol = alcohol

    # ------------------------------------------------------------
    # CONDITIONAL ALCOHOL QUESTIONS
    # ------------------------------------------------------------

    if (
        st.session_state.get("hp_alcohol")
        == "I drink alcohol"
    ):

        st.markdown(
            '<div class="followup-card">'
            '<div class="followup-title">'
            "Let’s describe your alcohol pattern"
            "</div>"
            '<div class="followup-text">'
            "The current research model represents alcohol "
            "intake as units per week, so the final numeric "
            "value below is the model-compatible input."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="question-label">'
            "How often do you usually drink?"
            "</div>",
            unsafe_allow_html=True,
        )

        alcohol_frequency = choice_control(
            "",
            ALCOHOL_FREQUENCY_OPTIONS,
            "hp_alcohol_frequency_control",
            st.session_state.get(
                "hp_alcohol_frequency"
            ),
        )

        if alcohol_frequency is not None:

            st.session_state.hp_alcohol_frequency = (
                alcohol_frequency
            )

        days_default = (
            alcohol_frequency_to_days(
                st.session_state.get(
                    "hp_alcohol_frequency"
                )
            )
            or st.session_state.get(
                "hp_alcohol_days"
            )
            or 1
        )

        st.markdown(
            '<div class="question-label">'
            "Approximately how many days per week?"
            "</div>",
            unsafe_allow_html=True,
        )

        alcohol_days = st.number_input(
            "Alcohol days per week",
            min_value=0,
            max_value=7,
            value=int(days_default),
            step=1,
            key="hp_alcohol_days_input",
            label_visibility="collapsed",
        )

        st.session_state.hp_alcohol_days = int(
            alcohol_days
        )

        current_units = (
            st.session_state.get(
                "hp_alcohol_units"
            )
        )

        if current_units is None:
            current_units = 1.0

        st.markdown(
            '<div class="question-label">'
            "Approximately how many alcohol units do you "
            "consume per week?"
            "</div>",
            unsafe_allow_html=True,
        )

        alcohol_units = st.number_input(
            "Alcohol units per week",
            min_value=float(
                DATA_LIMITS["alcohol_min"]
            ),
            max_value=float(
                DATA_LIMITS["alcohol_max"]
            ),
            value=float(current_units),
            step=0.1,
            key="hp_alcohol_units_input",
            label_visibility="collapsed",
        )

        st.session_state.hp_alcohol_units = (
            float(alcohol_units)
        )

        st.markdown(
            '<div class="question-label">'
            "Would you like alcohol-habit tracking included "
            "in your prevention plan?"
            "</div>",
            unsafe_allow_html=True,
        )

        alcohol_tracking = choice_control(
            "",
            PREVENTION_OPTIONS,
            "hp_alcohol_tracking_control",
            st.session_state.get(
                "hp_alcohol_tracking"
            ),
        )

        if alcohol_tracking is not None:

            st.session_state.hp_alcohol_tracking = (
                alcohol_tracking
            )

    else:

        st.session_state.hp_alcohol_days = 0
        st.session_state.hp_alcohol_units = 0.0
        st.session_state.hp_alcohol_frequency = None
        st.session_state.hp_alcohol_tracking = None


# ================================================================
# STEP 3 — MEDICAL HISTORY
# ================================================================

elif current_step == 3:

    st.markdown(
        '<div class="section-title">'
        "Medical History"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        "These questions help capture relevant health-history "
        "information used by the current assessment workflow."
        "</div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # GENETIC RISK
    # ------------------------------------------------------------

    st.markdown(
        '<div class="question-label">'
        "Has a healthcare professional or genetic information "
        "ever indicated an inherited cancer risk?"
        "</div>",
        unsafe_allow_html=True,
    )

    genetic = choice_control(
        "",
        GENETIC_OPTIONS,
        "hp_genetic_control",
        st.session_state.get("hp_genetic"),
    )

    if genetic is not None:

        st.session_state.hp_genetic = genetic

    st.markdown(
        '<div class="helper-text">'
        "The current research dataset represents inherited "
        "risk using three levels: low, medium, and high."
        "</div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # CONDITIONAL GENETIC FOLLOW-UP
    # ------------------------------------------------------------

    if st.session_state.get("hp_genetic") in [
        "Some increased inherited risk",
        "Higher inherited risk",
    ]:

        st.markdown(
            '<div class="followup-card">'
            '<div class="followup-title">'
            "Inherited-risk context"
            "</div>"
            '<div class="followup-text">'
            "This additional information is kept as contextual "
            "questionnaire information. It does not create a "
            "new feature for the trained model."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="question-label">'
            "Where does this information mainly come from?"
            "</div>",
            unsafe_allow_html=True,
        )

        risk_source = choice_control(
            "",
            RISK_SOURCE_OPTIONS,
            "hp_risk_source_control",
            st.session_state.get(
                "hp_risk_source"
            ),
        )

        if risk_source is not None:

            st.session_state.hp_risk_source = (
                risk_source
            )

        st.markdown(
            '<div class="question-label">'
            "Have you discussed this information with a "
            "healthcare professional?"
            "</div>",
            unsafe_allow_html=True,
        )

        family_discussion = choice_control(
            "",
            FOLLOWUP_OPTIONS,
            "hp_family_discussion_control",
            st.session_state.get(
                "hp_family_discussion"
            ),
        )

        if family_discussion is not None:

            st.session_state.hp_family_discussion = (
                family_discussion
            )

    else:

        st.session_state.hp_risk_source = None
        st.session_state.hp_family_discussion = None

    # ------------------------------------------------------------
    # PREVIOUS CANCER HISTORY
    # ------------------------------------------------------------

    st.markdown(
        '<div class="question-label">'
        "Have you ever had a previous cancer diagnosis?"
        "</div>",
        unsafe_allow_html=True,
    )

    cancer_history = choice_control(
        "",
        CANCER_HISTORY_OPTIONS,
        "hp_cancer_history_control",
        st.session_state.get(
            "hp_cancer_history"
        ),
    )

    if cancer_history is not None:

        st.session_state.hp_cancer_history = (
            cancer_history
        )

    st.markdown(
        '<div class="helper-text">'
        "This is a personal-history field used by the "
        "research model. It does not mean that the current "
        "assessment is diagnosing cancer."
        "</div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # CONDITIONAL CANCER FOLLOW-UP
    # ------------------------------------------------------------

    if (
        st.session_state.get("hp_cancer_history")
        == "Yes, I have had cancer before"
    ):

        st.markdown(
            '<div class="followup-card">'
            '<div class="followup-title">'
            "Follow-up context"
            "</div>"
            '<div class="followup-text">'
            "This question helps the prevention workflow "
            "understand whether follow-up care is already "
            "part of your current healthcare plan."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="question-label">'
            "Are you currently following a healthcare "
            "professional’s recommended follow-up plan?"
            "</div>",
            unsafe_allow_html=True,
        )

        followup_care = choice_control(
            "",
            FOLLOWUP_OPTIONS,
            "hp_followup_care_control",
            st.session_state.get(
                "hp_followup_care"
            ),
        )

        if followup_care is not None:

            st.session_state.hp_followup_care = (
                followup_care
            )

    else:

        st.session_state.hp_followup_care = None


# ================================================================
# STEP 4 — BODY & PREVENTION
# ================================================================

elif current_step == 4:

    st.markdown(
        '<div class="section-title">'
        "Body & Prevention"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        "Enter your current body measurements and review "
        "the prevention questions selected from your answers."
        "</div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # HEIGHT / WEIGHT
    # ------------------------------------------------------------

    height_col, weight_col = st.columns(
        2,
        gap="medium",
    )

    with height_col:

        st.markdown(
            '<div class="question-label">'
            "Height"
            "</div>",
            unsafe_allow_html=True,
        )

        height = st.number_input(
            "Height in centimetres",
            min_value=120.0,
            max_value=230.0,
            value=float(
                st.session_state.hp_height
            ),
            step=0.5,
            key="hp_height_input",
            label_visibility="collapsed",
        )

        st.session_state.hp_height = float(
            height
        )

    with weight_col:

        st.markdown(
            '<div class="question-label">'
            "Weight"
            "</div>",
            unsafe_allow_html=True,
        )

        weight = st.number_input(
            "Weight in kilograms",
            min_value=30.0,
            max_value=200.0,
            value=float(
                st.session_state.hp_weight
            ),
            step=0.5,
            key="hp_weight_input",
            label_visibility="collapsed",
        )

        st.session_state.hp_weight = float(
            weight
        )

    bmi = calculate_bmi(
        st.session_state.hp_height,
        st.session_state.hp_weight,
    )

    # ------------------------------------------------------------
    # BMI
    # ------------------------------------------------------------

    if bmi is not None:

        st.markdown(
            f"""
            <div class="bmi-card">
                <div class="bmi-label">
                    Calculated BMI
                </div>
                <div class="bmi-value">
                    {bmi:.1f}
                </div>
                <div class="bmi-note">
                    {bmi_description(bmi)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="helper-text">'
        "BMI is one of the model inputs. It is a calculated "
        "measure and is not a diagnosis or complete description "
        "of your health."
        "</div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # PREVENTION CHECK
    # ------------------------------------------------------------

    st.markdown(
        '<div class="section-title" '
        'style="margin-top:2rem;">'
        "Personalized Prevention Check"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        "These follow-up questions are selected from your "
        "earlier answers. They support prevention planning "
        "and do not redefine the trained model."
        "</div>",
        unsafe_allow_html=True,
    )

    prevention_question_count = 0

    # ------------------------------------------------------------
    # SMOKING SUPPORT
    # ------------------------------------------------------------

    if (
        st.session_state.get("hp_smoking")
        == "Yes, I currently smoke"
    ):

        prevention_question_count += 1

        st.markdown(
            '<div class="question-label">'
            "Would you like your prevention plan to include "
            "support for reducing or stopping smoking?"
            "</div>",
            unsafe_allow_html=True,
        )

        smoking_support = choice_control(
            "",
            PREVENTION_OPTIONS,
            "hp_smoking_support_control",
            st.session_state.get(
                "hp_smoking_support"
            ),
        )

        if smoking_support is not None:

            st.session_state.hp_smoking_support = (
                smoking_support
            )

    # ------------------------------------------------------------
    # ACTIVITY GOAL
    # ------------------------------------------------------------

    if st.session_state.get("hp_activity") in [
        "Mostly inactive",
        "Lightly active",
    ]:

        prevention_question_count += 1

        st.markdown(
            '<div class="question-label">'
            "Would you like to set a simple physical-activity "
            "goal for your prevention plan?"
            "</div>",
            unsafe_allow_html=True,
        )

        activity_goal = choice_control(
            "",
            PREVENTION_OPTIONS,
            "hp_activity_goal_body_control",
            st.session_state.get(
                "hp_activity_goal"
            ),
        )

        if activity_goal is not None:

            st.session_state.hp_activity_goal = (
                activity_goal
            )

    # ------------------------------------------------------------
    # ALCOHOL TRACKING
    # ------------------------------------------------------------

    if (
        st.session_state.get("hp_alcohol")
        == "I drink alcohol"
    ):

        prevention_question_count += 1

        st.markdown(
            '<div class="question-label">'
            "Would you like to track alcohol-related habits "
            "as part of your prevention plan?"
            "</div>",
            unsafe_allow_html=True,
        )

        alcohol_tracking = choice_control(
            "",
            PREVENTION_OPTIONS,
            "hp_alcohol_tracking_body_control",
            st.session_state.get(
                "hp_alcohol_tracking"
            ),
        )

        if alcohol_tracking is not None:

            st.session_state.hp_alcohol_tracking = (
                alcohol_tracking
            )

    # ------------------------------------------------------------
    # GENETIC DISCUSSION
    # ------------------------------------------------------------

    if st.session_state.get("hp_genetic") in [
        "Some increased inherited risk",
        "Higher inherited risk",
    ]:

        prevention_question_count += 1

        st.markdown(
            '<div class="question-label">'
            "Would you like to keep inherited-risk discussion "
            "as a prevention focus?"
            "</div>",
            unsafe_allow_html=True,
        )

        inherited_focus = choice_control(
            "",
            PREVENTION_OPTIONS,
            "hp_inherited_focus_control",
            st.session_state.get(
                "hp_inherited_focus"
            ),
        )

        if inherited_focus is not None:

            st.session_state.hp_inherited_focus = (
                inherited_focus
            )

    # ------------------------------------------------------------
    # CANCER FOLLOW-UP
    # ------------------------------------------------------------

    if (
        st.session_state.get("hp_cancer_history")
        == "Yes, I have had cancer before"
    ):

        prevention_question_count += 1

        st.markdown(
            '<div class="question-label">'
            "Would you like follow-up care to remain a focus "
            "of your prevention plan?"
            "</div>",
            unsafe_allow_html=True,
        )

        followup_focus = choice_control(
            "",
            PREVENTION_OPTIONS,
            "hp_followup_focus_control",
            st.session_state.get(
                "hp_followup_focus"
            ),
        )

        if followup_focus is not None:

            st.session_state.hp_followup_focus = (
                followup_focus
            )

    if prevention_question_count == 0:

        st.markdown(
            '<div class="info-card">'
            '<div class="info-card-title">'
            "No additional prevention questions"
            "</div>"
            '<div class="info-card-text">'
            "No additional contextual questions were triggered "
            "by your current answers."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )


# ================================================================
# VALIDATION
# ================================================================

def validate_current_step():
    """
    Validate the currently visible step.
    """

    step = int(
        st.session_state.health_profile_step
    )

    # ------------------------------------------------------------
    # STEP 1
    # ------------------------------------------------------------

    if step == 1:

        age = st.session_state.get("hp_age")
        gender = st.session_state.get("hp_gender")

        if age is None:

            st.warning(
                "Please enter your age."
            )

            return False

        if not (
            DATA_LIMITS["age_min"]
            <= int(age)
            <= DATA_LIMITS["age_max"]
        ):

            st.warning(
                "Please enter an age within the "
                "supported research-data range."
            )

            return False

        if gender not in GENDER_OPTIONS:

            st.warning(
                "Please select your gender before continuing."
            )

            return False

        return True

    # ------------------------------------------------------------
    # STEP 2
    # ------------------------------------------------------------

    if step == 2:

        smoking = st.session_state.get(
            "hp_smoking"
        )

        activity = st.session_state.get(
            "hp_activity"
        )

        alcohol = st.session_state.get(
            "hp_alcohol"
        )

        if smoking not in SMOKING_OPTIONS:

            st.warning(
                "Please select your current smoking status."
            )

            return False

        if activity not in ACTIVITY_LEVEL_OPTIONS:

            st.warning(
                "Please select your usual physical activity."
            )

            return False

        activity_hours = (
            st.session_state.get(
                "hp_activity_hours"
            )
        )

        if activity_hours is None:

            st.warning(
                "Please provide your approximate weekly "
                "physical activity hours."
            )

            return False

        if not (
            DATA_LIMITS["physical_min"]
            <= float(activity_hours)
            <= DATA_LIMITS["physical_max"]
        ):

            st.warning(
                "Physical activity must be within the "
                "supported research-data range."
            )

            return False

        if alcohol not in ALCOHOL_OPTIONS:

            st.warning(
                "Please indicate whether you drink alcohol."
            )

            return False

        if alcohol == "I drink alcohol":

            units = st.session_state.get(
                "hp_alcohol_units"
            )

            if units is None:

                st.warning(
                    "Please enter your approximate "
                    "alcohol units per week."
                )

                return False

            if not (
                DATA_LIMITS["alcohol_min"]
                <= float(units)
                <= DATA_LIMITS["alcohol_max"]
            ):

                st.warning(
                    "Alcohol intake must be within the "
                    "supported research-data range."
                )

                return False

        return True

    # ------------------------------------------------------------
    # STEP 3
    # ------------------------------------------------------------

    if step == 3:

        genetic = st.session_state.get(
            "hp_genetic"
        )

        cancer_history = st.session_state.get(
            "hp_cancer_history"
        )

        if genetic not in GENETIC_OPTIONS:

            st.warning(
                "Please select an inherited-risk option."
            )

            return False

        if genetic == "I'm not sure":

            st.warning(
                "The current research model requires a "
                "three-level inherited-risk value. "
                "Please choose the level you want to use "
                "for this assessment."
            )

            return False

        if cancer_history not in CANCER_HISTORY_OPTIONS:

            st.warning(
                "Please select an option for previous "
                "cancer history."
            )

            return False

        if cancer_history == "I'm not sure":

            st.warning(
                "The current research model requires a "
                "yes/no previous-cancer-history value. "
                "Please choose the option you want to use."
            )

            return False

        return True

    # ------------------------------------------------------------
    # STEP 4
    # ------------------------------------------------------------

    if step == 4:

        height = st.session_state.get(
            "hp_height"
        )

        weight = st.session_state.get(
            "hp_weight"
        )

        bmi = calculate_bmi(
            height,
            weight,
        )

        if bmi is None:

            st.warning(
                "Please enter valid height and weight values."
            )

            return False

        if not (
            DATA_LIMITS["bmi_min"]
            <= bmi
            <= DATA_LIMITS["bmi_max"]
        ):

            st.warning(
                f"The calculated BMI ({bmi:.1f}) is outside "
                "the range represented in the research dataset. "
                "Please review your measurements."
            )

            return False

        return True

    return True


# ================================================================
# MODEL INPUT PREPARATION
# ================================================================

def get_model_inputs():
    """
    Prepare the exact eight original model inputs.

    Engineered features are generated later by ml/prediction.py,
    keeping the feature-engineering logic in one place.
    """

    age = st.session_state.get(
        "hp_age"
    )

    gender = gender_value(
        st.session_state.get(
            "hp_gender"
        )
    )

    smoking = smoking_value(
        st.session_state.get(
            "hp_smoking"
        )
    )

    genetic = genetic_value(
        st.session_state.get(
            "hp_genetic"
        )
    )

    cancer_history = cancer_history_value(
        st.session_state.get(
            "hp_cancer_history"
        )
    )

    physical_activity = (
        activity_hours_from_patient_input()
    )

    alcohol_intake = (
        st.session_state.get(
            "hp_alcohol_units"
        )
    )

    bmi = calculate_bmi(
        st.session_state.get(
            "hp_height"
        ),
        st.session_state.get(
            "hp_weight"
        ),
    )

    return {
        "Age": int(age)
        if age is not None
        else None,

        "Gender": gender,

        "BMI": bmi,

        "Smoking": smoking,

        "GeneticRisk": genetic,

        "PhysicalActivity": (
            float(physical_activity)
            if physical_activity is not None
            else None
        ),

        "AlcoholIntake": (
            float(alcohol_intake)
            if alcohol_intake is not None
            else None
        ),

        "CancerHistory": cancer_history,
    }


def model_inputs_complete():
    inputs = get_model_inputs()

    return all(
        value is not None
        for value in inputs.values()
    )


# ================================================================
# SAVE PROFILE
# ================================================================

def save_profile(patient_id):
    """
    Persist the eight original model-compatible inputs.

    Height and weight are used to calculate BMI because the
    current database schema stores BMI rather than height/weight.
    """

    inputs = get_model_inputs()

    if not model_inputs_complete():

        st.error(
            "Some required model inputs are still missing. "
            "Please review the questionnaire."
        )

        return False

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

            return False

        updated = update_patient_profile(
            db=db,
            user_id=profile.user_id,

            age=inputs["Age"],

            gender=inputs["Gender"],

            bmi=float(
                inputs["BMI"]
            ),

            smoking=inputs["Smoking"],

            genetic_risk=inputs["GeneticRisk"],

            physical_activity=float(
                inputs["PhysicalActivity"]
            ),

            alcohol_intake=float(
                inputs["AlcoholIntake"]
            ),

            cancer_history=inputs[
                "CancerHistory"
            ],
        )

        if updated is None:

            st.error(
                "The health profile could not be updated."
            )

            return False

        st.session_state.hp_saved = True

        return True

    except Exception as exc:

        db.rollback()

        st.error(
            f"Unable to save the health profile: {exc}"
        )

        return False

    finally:

        db.close()


# ================================================================
# STEP NAVIGATION
# ================================================================

def go_next():
    """
    Validate and move to the next step.
    """

    if not validate_current_step():

        return

    clear_followup_values()

    if (
        st.session_state.health_profile_step
        < len(STEPS)
    ):

        st.session_state.health_profile_step += 1

        st.rerun()


def go_back():
    """
    Move one step backward.
    """

    if (
        st.session_state.health_profile_step
        > 1
    ):

        st.session_state.health_profile_step -= 1

        st.rerun()


# ================================================================
# STEP 5 — REVIEW
# ================================================================

if current_step == 5:

    st.markdown(
        '<div class="section-title">'
        "Review Your Profile"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        "Review your answers before saving the health profile."
        "</div>",
        unsafe_allow_html=True,
    )

    model_inputs = get_model_inputs()

    ready = model_inputs_complete()

    if ready:

        st.success(
            "Your required model inputs are complete."
        )

    else:

        st.warning(
            "Some required model inputs are missing. "
            "Use the Edit buttons below to complete them."
        )

    # ------------------------------------------------------------
    # ROW 1
    # ------------------------------------------------------------

    left, right = st.columns(
        2,
        gap="medium",
    )

    with left:

        st.markdown(
            '<div class="review-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="review-card-title">'
            "Basic Information"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="review-item">'
            '<div class="review-label">Age</div>'
            f'<div class="review-value">'
            f'{model_inputs["Age"]} years'
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        gender_display = (
            st.session_state.get(
                "hp_gender"
            )
            or "Not provided"
        )

        st.markdown(
            '<div class="review-item">'
            '<div class="review-label">Gender</div>'
            f'<div class="review-value">'
            f'{gender_display}'
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        if st.button(
            "Edit Basic Information",
            key="edit_basic_information",
            use_container_width=True,
        ):

            st.session_state.health_profile_step = 1
            st.rerun()

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with right:

        st.markdown(
            '<div class="review-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="review-card-title">'
            "Lifestyle"
            "</div>",
            unsafe_allow_html=True,
        )

        smoking_display = (
            st.session_state.get(
                "hp_smoking"
            )
            or "Not provided"
        )

        activity_display = (
            st.session_state.get(
                "hp_activity"
            )
            or "Not provided"
        )

        activity_hours_display = (
            st.session_state.get(
                "hp_activity_hours"
            )
        )

        alcohol_display = (
            st.session_state.get(
                "hp_alcohol"
            )
            or "Not provided"
        )

        alcohol_units_display = (
            st.session_state.get(
                "hp_alcohol_units"
            )
        )

        st.markdown(
            '<div class="review-item">'
            '<div class="review-label">Smoking</div>'
            f'<div class="review-value">'
            f'{smoking_display}'
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="review-item">'
            '<div class="review-label">Physical activity</div>'
            f'<div class="review-value">'
            f'{activity_display}'
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        if activity_hours_display is not None:

            st.markdown(
                '<div class="review-item">'
                '<div class="review-label">'
                "Activity estimate"
                "</div>"
                f'<div class="review-value">'
                f'{float(activity_hours_display):.1f} '
                "hours/week"
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="review-item">'
            '<div class="review-label">Alcohol</div>'
            f'<div class="review-value">'
            f'{alcohol_display}'
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        if alcohol_units_display is not None:

            st.markdown(
                '<div class="review-item">'
                '<div class="review-label">'
                "Alcohol estimate"
                "</div>"
                f'<div class="review-value">'
                f'{float(alcohol_units_display):.1f} '
                "units/week"
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        if st.button(
            "Edit Lifestyle",
            key="edit_lifestyle",
            use_container_width=True,
        ):

            st.session_state.health_profile_step = 2
            st.rerun()

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------
    # ROW 2
    # ------------------------------------------------------------

    left, right = st.columns(
        2,
        gap="medium",
    )

    with left:

        st.markdown(
            '<div class="review-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="review-card-title">'
            "Medical History"
            "</div>",
            unsafe_allow_html=True,
        )

        genetic_display = (
            st.session_state.get(
                "hp_genetic"
            )
            or "Not provided"
        )

        cancer_history_display = (
            st.session_state.get(
                "hp_cancer_history"
            )
            or "Not provided"
        )

        st.markdown(
            '<div class="review-item">'
            '<div class="review-label">'
            "Inherited risk"
            "</div>"
            f'<div class="review-value">'
            f'{genetic_display}'
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="review-item">'
            '<div class="review-label">'
            "Previous cancer history"
            "</div>"
            f'<div class="review-value">'
            f'{cancer_history_display}'
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        if st.session_state.get(
            "hp_risk_source"
        ):

            st.markdown(
                '<div class="review-item">'
                '<div class="review-label">'
                "Inherited-risk source"
                "</div>"
                f'<div class="review-value">'
                f'{st.session_state.hp_risk_source}'
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        if st.button(
            "Edit Medical History",
            key="edit_medical_history",
            use_container_width=True,
        ):

            st.session_state.health_profile_step = 3
            st.rerun()

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with right:

        st.markdown(
            '<div class="review-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="review-card-title">'
            "Body Measurements"
            "</div>",
            unsafe_allow_html=True,
        )

        height_display = (
            st.session_state.get(
                "hp_height"
            )
        )

        weight_display = (
            st.session_state.get(
                "hp_weight"
            )
        )

        bmi_display = model_inputs.get(
            "BMI"
        )

        st.markdown(
            '<div class="review-item">'
            '<div class="review-label">Height</div>'
            f'<div class="review-value">'
            f'{float(height_display):.1f} cm'
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="review-item">'
            '<div class="review-label">Weight</div>'
            f'<div class="review-value">'
            f'{float(weight_display):.1f} kg'
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="review-item">'
            '<div class="review-label">'
            "Calculated BMI"
            "</div>"
            f'<div class="review-value">'
            f'{float(bmi_display):.1f}'
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        if st.button(
            "Edit Body & Prevention",
            key="edit_body_prevention",
            use_container_width=True,
        ):

            st.session_state.health_profile_step = 4
            st.rerun()

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------
    # PREVENTION SUMMARY
    # ------------------------------------------------------------

    prevention_items = []

    if (
        st.session_state.get("hp_smoking")
        == "Yes, I currently smoke"
    ):

        support = st.session_state.get(
            "hp_smoking_support"
        )

        if support == "Yes":

            prevention_items.append(
                "Smoking reduction or cessation support"
            )

        else:

            prevention_items.append(
                "Smoking status identified as a prevention focus"
            )

    if st.session_state.get("hp_activity") in [
        "Mostly inactive",
        "Lightly active",
    ]:

        prevention_items.append(
            "Physical-activity habits"
        )

    if (
        st.session_state.get("hp_alcohol")
        == "I drink alcohol"
    ):

        prevention_items.append(
            "Alcohol-habit awareness and tracking"
        )

    if st.session_state.get("hp_genetic") in [
        "Some increased inherited risk",
        "Higher inherited risk",
    ]:

        prevention_items.append(
            "Inherited-risk discussion with a qualified "
            "healthcare professional"
        )

    if (
        st.session_state.get("hp_cancer_history")
        == "Yes, I have had cancer before"
    ):

        prevention_items.append(
            "Appropriate follow-up care and healthcare discussion"
        )

    st.markdown(
        '<div class="model-summary">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="model-summary-title">'
        "Personalized Prevention Focus"
        "</div>",
        unsafe_allow_html=True,
    )

    if prevention_items:

        for item in prevention_items:

            st.markdown(
                f"• {item}"
            )

    else:

        st.markdown(
            '<div class="model-summary-text">'
            "No additional prevention focus was triggered "
            "by the current answers."
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # MODEL INPUT SUMMARY
    # ------------------------------------------------------------

    st.markdown(
        '<div class="model-summary">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="model-summary-title">'
        "Assessment input check"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="model-summary-text">'
        "Your patient-friendly answers are converted into "
        "the numeric representation required by the current "
        "research model. The trained model uses the eight "
        "original health and lifestyle inputs; engineered "
        "features are created by the prediction pipeline."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# ================================================================
# NAVIGATION BUTTONS
# ================================================================

st.write("")

if current_step < 5:

    back_col, spacer_col, next_col = st.columns(
        [1, 0.08, 1]
    )

    with back_col:

        if current_step > 1:

            if st.button(
                "← Back",
                key=f"back_step_{current_step}",
                use_container_width=True,
            ):

                go_back()

    with next_col:

        next_label = (
            "Review Profile →"
            if current_step == 4
            else "Continue →"
        )

        if st.button(
            next_label,
            key=f"next_step_{current_step}",
            type="primary",
            use_container_width=True,
        ):

            go_next()

else:

    back_col, spacer_col, save_col = st.columns(
        [1, 0.08, 1]
    )

    with back_col:

        if st.button(
            "← Back",
            key="review_back_button",
            use_container_width=True,
        ):

            st.session_state.health_profile_step = 4
            st.rerun()

    with save_col:

        if st.button(
            "Save Health Profile",
            key="save_health_profile_button",
            type="primary",
            use_container_width=True,
            disabled=not model_inputs_complete(),
        ):

            if save_profile(
                st.session_state.health_profile_patient_id
            ):

                st.success(
                    "Health profile saved successfully."
                )

                st.session_state.hp_saved = True

                st.info(
                    "Your profile is ready for Cancer Risk Assessment. "
                    "Open the Cancer Risk Assessment page to run the "
                    "current machine-learning assessment."
                )


# ================================================================
# PRIVACY FOOTER
# ================================================================

st.markdown(
    '<div class="privacy-footer">'
    "🔒 OncoGuard AI provides model-based cancer risk assessment "
    "and prevention support for research and decision-support "
    "purposes. It does not diagnose cancer, confirm or rule out "
    "cancer, or replace evaluation by a qualified healthcare "
    "professional."
    "</div>",
    unsafe_allow_html=True,
)