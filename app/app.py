from pathlib import Path
import sys

import streamlit as st


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="OncoGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# APPLICATION CONSTANTS
# ============================================================

ACCENT = "#5B4BDB"
ACCENT_DARK = "#4F40CA"
BACKGROUND = "#F6F7FB"
WHITE = "#FFFFFF"
TEXT = "#182033"
MUTED = "#66718A"
BORDER = "#E1E5EE"


# ============================================================
# GLOBAL APPLICATION CSS
# ============================================================

st.markdown(
    f"""
    <style>

    /* ========================================================
       GLOBAL APPLICATION
       ======================================================== */

    .stApp {{
        background: {BACKGROUND} !important;
    }}

    [data-testid="stAppViewContainer"] {{
        background: {BACKGROUND} !important;
    }}

    [data-testid="stMain"] {{
        background: {BACKGROUND} !important;
    }}

    section[data-testid="stMain"] {{
        background: {BACKGROUND} !important;
        color: {TEXT} !important;
    }}

    .block-container {{
        max-width: 1120px !important;
        padding-top: 2.8rem !important;
        padding-bottom: 3rem !important;
    }}


    /* ========================================================
       HEADER
       ======================================================== */

    [data-testid="stHeader"] {{
        background: {BACKGROUND} !important;
        border-bottom: none !important;
        box-shadow: none !important;
    }}

    [data-testid="stHeader"] * {{
        color: {TEXT} !important;
    }}


    /* ========================================================
       MAIN TYPOGRAPHY
       ======================================================== */

    section[data-testid="stMain"] h1,
    section[data-testid="stMain"] h2,
    section[data-testid="stMain"] h3,
    section[data-testid="stMain"] h4,
    section[data-testid="stMain"] h5,
    section[data-testid="stMain"] h6 {{
        color: {TEXT} !important;
    }}

    section[data-testid="stMain"] p,
    section[data-testid="stMain"] span,
    section[data-testid="stMain"] label {{
        color: {TEXT} !important;
    }}


    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {{
        background: {WHITE} !important;
        border-right: 1px solid {BORDER} !important;
    }}

    [data-testid="stSidebar"] > div {{
        background: {WHITE} !important;
    }}

    [data-testid="stSidebar"] * {{
        color: {TEXT} !important;
    }}

    [data-testid="stSidebarNav"] {{
        padding-top: 1rem !important;
    }}

    [data-testid="stSidebarNav"] a {{
        border-radius: 10px !important;
    }}

    [data-testid="stSidebarNav"] a[aria-current="page"] {{
        background: #EFEDFF !important;
        border-radius: 10px !important;
    }}

    [data-testid="stSidebarNav"] a[aria-current="page"] span {{
        color: {ACCENT} !important;
        font-weight: 700 !important;
    }}


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {{
        min-height: 42px !important;
        border-radius: 10px !important;
        border: 1px solid #D8DCE7 !important;
        background: {WHITE} !important;
        color: {TEXT} !important;
        font-weight: 650 !important;
        transition: all 0.15s ease !important;
    }}

    .stButton > button:hover {{
        border-color: {ACCENT} !important;
        color: {ACCENT} !important;
    }}

    .stButton > button:focus {{
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(91, 75, 219, 0.18) !important;
    }}

    .stButton > button[kind="primary"] {{
        background: {ACCENT} !important;
        border-color: {ACCENT} !important;
        color: #FFFFFF !important;
    }}

    .stButton > button[kind="primary"]:hover {{
        background: {ACCENT_DARK} !important;
        border-color: {ACCENT_DARK} !important;
        color: #FFFFFF !important;
    }}


    /* ========================================================
       HOME PAGE
       ======================================================== */

    .home-eyebrow {{
        color: {ACCENT} !important;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }}

    .home-title {{
        color: {TEXT} !important;
        font-size: 2.45rem;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.04em;
        margin-bottom: 0.7rem;
    }}

    .home-subtitle {{
        color: {MUTED} !important;
        font-size: 1rem;
        line-height: 1.7;
        max-width: 900px;
    }}

    .home-card {{
        background: {WHITE};
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 1.5rem;
        margin-top: 1.25rem;
    }}

    .home-card-title {{
        color: {TEXT} !important;
        font-size: 1.25rem;
        font-weight: 750;
        margin-bottom: 0.45rem;
    }}

    .home-card-text {{
        color: {MUTED} !important;
        line-height: 1.65;
        font-size: 0.92rem;
    }}

    .feature-card {{
        background: {WHITE};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 1.25rem;
        min-height: 175px;
    }}

    .feature-icon {{
        font-size: 1.45rem;
        margin-bottom: 0.55rem;
    }}

    .feature-title {{
        color: {TEXT} !important;
        font-weight: 750;
        margin-bottom: 0.4rem;
    }}

    .feature-text {{
        color: {MUTED} !important;
        font-size: 0.83rem;
        line-height: 1.6;
    }}


    /* ========================================================
       INFO BOX
       ======================================================== */

    .app-info {{
        background: #FAF9FF;
        border: 1px solid #DED9FF;
        border-radius: 13px;
        padding: 1rem 1.1rem;
        margin-top: 1rem;
    }}

    .app-info-title {{
        color: #4E42B5 !important;
        font-weight: 750;
        font-size: 0.84rem;
        margin-bottom: 0.25rem;
    }}

    .app-info-text {{
        color: #66718A !important;
        font-size: 0.82rem;
        line-height: 1.6;
    }}


    /* ========================================================
       FOOTER
       ======================================================== */

    .app-footer {{
        color: {MUTED} !important;
        font-size: 0.75rem;
        line-height: 1.6;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid {BORDER};
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DEVELOPMENT SESSION
# ============================================================
#
# This is temporary until the real authentication system is
# connected to the Streamlit navigation.
#
# Patient ID 1 is used for development/testing.
# ============================================================

st.session_state.setdefault("authenticated", True)
st.session_state.setdefault("role", "patient")
st.session_state.setdefault("patient_id", 1)
st.session_state.setdefault("current_patient_id", 1)


# ============================================================
# SIDEBAR BRAND
# ============================================================

with st.sidebar:

    st.markdown(
        "### 🛡️ OncoGuard AI"
    )

    st.caption(
        "AI-assisted cancer risk assessment "
        "and prevention support."
    )


# ============================================================
# HOME PAGE
# ============================================================

def show_home():
    """
    Render the OncoGuard AI landing page.

    This function intentionally lives inside app.py.
    Therefore a separate home.py file is NOT required.
    """

    st.markdown(
        '<div class="home-eyebrow">OncoGuard AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="home-title">Cancer Risk Assessment</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="home-subtitle">'
        "A machine-learning-based platform for assessing "
        "model-predicted cancer risk from health and lifestyle "
        "information and presenting understandable, "
        "prevention-focused insights."
        "</div>",
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # MAIN INTRODUCTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="home-card">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="home-card-title">'
        "Understand your risk factors"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="home-card-text">'
        "Start by completing your health profile. "
        "Your patient-friendly answers are converted into "
        "the numerical representation required by the trained "
        "machine-learning model. The resulting assessment "
        "shows model-predicted risk and explains which recorded "
        "factors were important contributors to that result."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # FEATURE CARDS
    # --------------------------------------------------------

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            '<div class="feature-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="feature-icon">🧾</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="feature-title">'
            "Guided Health Profile"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="feature-text">'
            "Answer simple health and lifestyle questions "
            "instead of entering raw machine-learning "
            "feature values."
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


    with col2:

        st.markdown(
            '<div class="feature-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="feature-icon">📊</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="feature-title">'
            "Cancer Risk Assessment"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="feature-text">'
            "View the current model-predicted risk, risk "
            "category, model version, threshold and previous "
            "assessment history."
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


    with col3:

        st.markdown(
            '<div class="feature-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="feature-icon">🔎</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="feature-title">'
            "Explainable Assessment"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="feature-text">'
            "Review important model contributors using "
            "SHAP-based explanations without presenting "
            "them as medical causation."
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


    # --------------------------------------------------------
    # SAFETY INFORMATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="app-info">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="app-info-title">'
        "Important"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="app-info-text">'
        "OncoGuard AI provides model-predicted risk estimates "
        "for research and prevention-support purposes. "
        "The result is not a cancer diagnosis and should not "
        "replace evaluation or advice from a qualified "
        "healthcare professional."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# STREAMLIT NAVIGATION
# ============================================================

home_page = st.Page(
    show_home,
    title="Home",
    icon="🏠",
    default=True,
)

health_profile_page = st.Page(
    "pages/patient/health_profile.py",
    title="Health Profile",
    icon="👤",
)

risk_assessment_page = st.Page(
    "pages/patient/risk_assessment.py",
    title="Cancer Risk Assessment",
    icon="🛡️",
)


# ============================================================
# NAVIGATION
# ============================================================

navigation = st.navigation(
    {
        "OncoGuard AI": [
            home_page,
        ],
        "Patient": [
            health_profile_page,
            risk_assessment_page,
        ],
    }
)


# ============================================================
# RUN SELECTED PAGE
# ============================================================

navigation.run()