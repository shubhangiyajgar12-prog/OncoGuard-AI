import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from database.connection import SessionLocal

from services.risk_history_service import (
    get_current_risk,
    get_risk_history,
    get_risk_history_summary,
    get_risk_trend,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Risk Journey | OncoGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #f5f7fb;
    }

    .main .block-container {
        max-width: 1450px;
        padding: 35px 45px 60px 45px;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    /* Global text */
    h1, h2, h3, h4, h5, h6 {
        color: #172033 !important;
    }

    p, label, .stMarkdown {
        color: #5f6b7d !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e6eaf0;
    }

    section[data-testid="stSidebar"] * {
        color: #263248 !important;
    }

    /* Sidebar number input */
    section[data-testid="stSidebar"] input {
        color: #20283a !important;
        background: #ffffff !important;
    }

    /* Page title */
    .page-kicker {
        color: #5b5ce2 !important;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }

    .page-title {
        color: #172033 !important;
        font-size: 38px;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 8px;
    }

    .page-subtitle {
        color: #788398 !important;
        font-size: 15px;
        margin-bottom: 28px;
    }

    /* Notice */
    .notice {
        background: #eef2ff;
        border: 1px solid #dce3ff;
        border-radius: 14px;
        padding: 15px 18px;
        color: #56627a !important;
        font-size: 13px;
        line-height: 1.55;
        margin-bottom: 28px;
    }

    .notice strong {
        color: #4947ba !important;
    }

    /* Section heading */
    .section-title {
        color: #20283a !important;
        font-size: 20px;
        font-weight: 750;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .section-caption {
        color: #8b95a7 !important;
        font-size: 12px;
        margin-bottom: 12px;
    }

    /* Cards */
    .card {
        background: #ffffff;
        border: 1px solid #e6eaf0;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 5px 18px rgba(30, 41, 59, 0.04);
        min-height: 145px;
    }

    .card-label {
        color: #7c8798 !important;
        font-size: 11px;
        font-weight: 750;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .card-value {
        color: #182033 !important;
        font-size: 30px;
        font-weight: 800;
        line-height: 1.2;
    }

    .card-description {
        color: #8993a5 !important;
        font-size: 12px;
        margin-top: 9px;
        line-height: 1.5;
    }

    .risk-low {
        color: #168866 !important;
    }

    .risk-medium {
        color: #bd7d0f !important;
    }

    .risk-high {
        color: #d4495b !important;
    }

    /* Status pills */
    .pill {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 750;
        margin-top: 10px;
    }

    .pill-low {
        background: #e7f8f0;
        color: #16845f !important;
    }

    .pill-medium {
        background: #fff5dc;
        color: #a8730d !important;
    }

    .pill-high {
        background: #fdebed;
        color: #c43e50 !important;
    }

    .pill-neutral {
        background: #eef1f5;
        color: #697487 !important;
    }

    /* Detail card */
    .detail-card {
        background: #ffffff;
        border: 1px solid #e6eaf0;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 5px 18px rgba(30, 41, 59, 0.04);
    }

    .detail-label {
        color: #7c8798 !important;
        font-size: 11px;
        font-weight: 750;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 14px;
    }

    .detail-main {
        color: #20283a !important;
        font-size: 25px;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .detail-row {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        padding: 10px 0;
        border-bottom: 1px solid #eef1f5;
        font-size: 13px;
    }

    .detail-row:last-child {
        border-bottom: none;
    }

    .detail-row span {
        color: #687487 !important;
    }

    .detail-row strong {
        color: #20283a !important;
        text-align: right;
    }

    /* Trend information */
    .trend-increasing {
        background: #fff1f3;
        border: 1px solid #f2d9dd;
        color: #a04753 !important;
    }

    .trend-decreasing {
        background: #edf9f4;
        border: 1px solid #d4eee3;
        color: #28765c !important;
    }

    .trend-stable {
        background: #f0f3f7;
        border: 1px solid #e0e5eb;
        color: #657082 !important;
    }

    .trend-box {
        padding: 14px 17px;
        border-radius: 13px;
        margin: 18px 0;
        font-size: 13px;
    }

    /* Table */
    [data-testid="stDataFrame"] {
        border: 1px solid #e6eaf0;
        border-radius: 14px;
        overflow: hidden;
    }

    /* Disclaimer */
    .disclaimer {
        background: #ffffff;
        border: 1px solid #e6eaf0;
        border-radius: 14px;
        padding: 16px 18px;
        color: #737e90 !important;
        font-size: 12px;
        line-height: 1.6;
        margin-top: 28px;
    }

    .disclaimer strong {
        color: #4c5668 !important;
    }

    /* Sidebar brand */
    .brand-name {
        color: #20283a !important;
        font-size: 22px;
        font-weight: 800;
    }

    .brand-subtitle {
        color: #8b95a7 !important;
        font-size: 12px;
        margin-top: 3px;
        margin-bottom: 25px;
    }

    .side-heading {
        color: #9aa3b2 !important;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin: 15px 0 8px 0;
    }

    .side-item {
        color: #596579 !important;
        font-size: 13px;
        font-weight: 550;
        padding: 10px 12px;
        border-radius: 10px;
        margin: 3px 0;
    }

    .side-active {
        background: #eef0ff;
        color: #5148d9 !important;
        font-weight: 750;
    }

    .side-divider {
        height: 1px;
        background: #edf0f5;
        margin: 18px 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_value(item, *keys, default=None):

    if item is None:
        return default

    if isinstance(item, dict):

        for key in keys:
            if key in item:
                return item[key]

        return default

    for key in keys:

        if hasattr(item, key):
            return getattr(item, key)

    return default


def format_timestamp(timestamp):

    if timestamp is None:
        return "N/A"

    if hasattr(timestamp, "strftime"):
        return timestamp.strftime("%d %b %Y, %I:%M %p")

    return str(timestamp)


def risk_percentage(item):

    value = get_value(
        item,
        "risk_percentage",
        default=None,
    )

    if value is not None:
        return float(value)

    probability = get_value(
        item,
        "probability",
        "risk_probability",
        default=0,
    )

    return float(probability) * 100


def display_band(band):

    if not band:
        return "N/A"

    value = str(band).lower()

    if "low" in value or "lower" in value:
        return "Lower predicted risk"

    if "medium" in value or "moderate" in value:
        return "Moderate predicted risk"

    if "high" in value or "higher" in value:
        return "Higher predicted risk"

    return str(band)


def risk_class(band):

    value = str(band).lower()

    if "low" in value or "lower" in value:
        return "low"

    if "medium" in value or "moderate" in value:
        return "medium"

    if "high" in value or "higher" in value:
        return "high"

    return "neutral"


def trend_label(trend):

    value = str(trend).lower()

    if value == "increasing":
        return "Increasing"

    if value == "decreasing":
        return "Decreasing"

    return "Stable"


def trend_icon(trend):

    value = str(trend).lower()

    if value == "increasing":
        return "↗"

    if value == "decreasing":
        return "↘"

    return "→"


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    with st.sidebar:

        st.markdown(
            '<div class="brand-name">🛡️ OncoGuard AI</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="brand-subtitle">Patient Portal</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="side-heading">Navigation</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="side-item">🏠 &nbsp; Dashboard</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="side-item">👤 &nbsp; Health Profile</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="side-item">🧪 &nbsp; Risk Assessment</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="side-item side-active">📈 &nbsp; Risk Journey</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="side-item">🧠 &nbsp; Risk Explanation</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="side-item">💡 &nbsp; Recommendations</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="side-item">🔬 &nbsp; What-If Simulation</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="side-item">📄 &nbsp; Reports</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="side-divider"></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="side-heading">Account</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="side-item">👤 &nbsp; My Account</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="side-item">⚙️ &nbsp; Settings</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="side-item">🚪 &nbsp; Logout</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        patient_id = st.number_input(
            "Development Patient ID",
            min_value=1,
            value=1,
            step=1,
        )

        return int(patient_id)


# ============================================================
# MAIN PAGE
# ============================================================

def render_risk_history(patient_id):

    db = SessionLocal()

    try:

        # --------------------------------------------------------
        # DATABASE
        # --------------------------------------------------------

        history = get_risk_history(
            db,
            patient_id,
        )

        current = get_current_risk(
            db,
            patient_id,
        )

        summary = get_risk_history_summary(
            db,
            patient_id,
        )

        trend = get_risk_trend(
            db,
            patient_id,
        )

        # --------------------------------------------------------
        # HEADER
        # --------------------------------------------------------

        st.markdown(
            '<div class="page-kicker">ONCOGUARD AI</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="page-title">Risk Journey</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="page-subtitle">'
            'Track how your model-predicted risk has changed '
            'across previous assessments.'
            '</div>',
            unsafe_allow_html=True,
        )

        # --------------------------------------------------------
        # SAFETY NOTICE
        # --------------------------------------------------------

        st.markdown(
            '<div class="notice">'
            '<strong>About your Risk Journey:</strong> '
            'This page shows changes in model-predicted risk '
            'across assessments. It does not represent cancer '
            'progression, cancer stage, or a medical diagnosis.'
            '</div>',
            unsafe_allow_html=True,
        )

        # --------------------------------------------------------
        # EMPTY STATE
        # --------------------------------------------------------

        if not history:

            st.markdown(
                '<div class="detail-card">'
                '<div class="detail-label">Risk Journey</div>'
                '<div class="detail-main">'
                'No assessments yet'
                '</div>'
                '<div class="card-description">'
                'Complete a risk assessment to begin tracking '
                'model-predicted risk over time.'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            return

        # --------------------------------------------------------
        # CURRENT DATA
        # --------------------------------------------------------

        current_probability = get_value(
            current,
            "probability",
            "risk_probability",
            default=0,
        )

        current_percentage = get_value(
            current,
            "risk_percentage",
            default=None,
        )

        if current_percentage is None:
            current_percentage = float(current_probability) * 100

        current_band = get_value(
            current,
            "risk_band",
            default="N/A",
        )

        current_class = risk_class(
            current_band
        )

        # --------------------------------------------------------
        # PREVIOUS
        # --------------------------------------------------------

        previous_percentage = None

        if len(history) >= 2:

            previous_percentage = risk_percentage(
                history[-2]
            )

        if previous_percentage is None:

            comparison = "First recorded assessment"

        else:

            change = (
                float(current_percentage)
                - previous_percentage
            )

            if change > 0:
                comparison = (
                    f"+{change:.2f} percentage points "
                    "from previous"
                )

            elif change < 0:
                comparison = (
                    f"{change:.2f} percentage points "
                    "from previous"
                )

            else:
                comparison = (
                    "No change from previous assessment"
                )

        # --------------------------------------------------------
        # SECTION
        # --------------------------------------------------------

        st.markdown(
            '<div class="section-title">Current Overview</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-caption">'
            'Latest available assessment'
            '</div>',
            unsafe_allow_html=True,
        )

        # --------------------------------------------------------
        # STAT CARDS
        # --------------------------------------------------------

        c1, c2, c3 = st.columns(
            3,
            gap="medium",
        )

        # Current risk
        with c1:

            st.markdown(
                f'<div class="card">'
                f'<div class="card-label">'
                f'MODEL-PREDICTED RISK'
                f'</div>'
                f'<div class="card-value risk-{current_class}">'
                f'{float(current_percentage):.2f}%'
                f'</div>'
                f'<span class="pill pill-{current_class}">'
                f'{display_band(current_band)}'
                f'</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Trend
        with c2:

            clean_trend = trend_label(trend)

            if str(trend).lower() == "increasing":
                trend_css = "risk-high"

            elif str(trend).lower() == "decreasing":
                trend_css = "risk-low"

            else:
                trend_css = ""

            st.markdown(
                f'<div class="card">'
                f'<div class="card-label">RISK TREND</div>'
                f'<div class="card-value {trend_css}">'
                f'{trend_icon(trend)} {clean_trend}'
                f'</div>'
                f'<div class="card-description">'
                f'{comparison}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Assessment count
        with c3:

            count = get_value(
                summary,
                "assessment_count",
                "assessments",
                "count",
                default=len(history),
            )

            last_timestamp = get_value(
                history[-1],
                "timestamp",
                "created_at",
                "assessment_date",
                default=None,
            )

            st.markdown(
                f'<div class="card">'
                f'<div class="card-label">'
                f'ASSESSMENT ACTIVITY'
                f'</div>'
                f'<div class="card-value">'
                f'{count}'
                f'</div>'
                f'<div class="card-description">'
                f'Last assessed: '
                f'{format_timestamp(last_timestamp)}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # --------------------------------------------------------
        # CHART
        # --------------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Model-Predicted Risk Journey'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-caption">'
            'Historical assessment trend'
            '</div>',
            unsafe_allow_html=True,
        )

        chart_data = []

        for index, item in enumerate(
            history,
            start=1,
        ):

            timestamp = get_value(
                item,
                "timestamp",
                "created_at",
                "assessment_date",
                default=None,
            )

            band = get_value(
                item,
                "risk_band",
                default="N/A",
            )

            chart_data.append(
                {
                    "Assessment": index,
                    "Risk": risk_percentage(item),
                    "Date": format_timestamp(timestamp),
                    "Band": display_band(band),
                }
            )

        chart_df = pd.DataFrame(
            chart_data
        )

        if len(chart_df) >= 2:

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=chart_df["Assessment"],
                    y=chart_df["Risk"],
                    mode="lines+markers",
                    customdata=chart_df[
                        ["Date", "Band"]
                    ].values,
                    hovertemplate=(
                        "<b>Assessment %{x}</b>"
                        "<br>Risk: %{y:.2f}%"
                        "<br>Date: %{customdata[0]}"
                        "<br>Status: %{customdata[1]}"
                        "<extra></extra>"
                    ),
                    line=dict(
                        width=3,
                        color="#6366f1",
                    ),
                    marker=dict(
                        size=9,
                        color="#6366f1",
                    ),
                )
            )

            fig.update_layout(
                height=390,
                margin=dict(
                    l=20,
                    r=20,
                    t=25,
                    b=20,
                ),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                showlegend=False,
                xaxis=dict(
                    title="Assessment",
                    dtick=1,
                    showgrid=False,
                    zeroline=False,
                ),
                yaxis=dict(
                    title="Model-Predicted Risk (%)",
                    showgrid=True,
                    gridcolor="#edf0f5",
                    zeroline=False,
                    rangemode="tozero",
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                },
            )

        else:

            st.markdown(
                '<div class="detail-card">'
                '<div class="detail-label">'
                'Risk Journey'
                '</div>'
                '<div class="detail-main">'
                'One assessment recorded'
                '</div>'
                '<div class="card-description">'
                'The trend chart will become available '
                'after another assessment.'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        # --------------------------------------------------------
        # TREND MESSAGE
        # --------------------------------------------------------

        if str(trend).lower() == "increasing":

            trend_message = (
                "The model-predicted risk has increased "
                "across the available assessments."
            )

            trend_style = "trend-increasing"

        elif str(trend).lower() == "decreasing":

            trend_message = (
                "The model-predicted risk has decreased "
                "across the available assessments."
            )

            trend_style = "trend-decreasing"

        else:

            trend_message = (
                "The model-predicted risk has remained "
                "relatively stable across the available "
                "assessments."
            )

            trend_style = "trend-stable"

        st.markdown(
            f'<div class="trend-box {trend_style}">'
            f'<strong>{trend_icon(trend)} '
            f'{trend_label(trend)}:</strong> '
            f'{trend_message}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # --------------------------------------------------------
        # JOURNEY SUMMARY
        # --------------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Journey Summary'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-caption">'
            'Latest assessment details'
            '</div>',
            unsafe_allow_html=True,
        )

        d1, d2 = st.columns(
            2,
            gap="medium",
        )

        # --------------------------------------------------------
        # LATEST ASSESSMENT
        # --------------------------------------------------------

        with d1:

            latest = history[-1]

            latest_timestamp = get_value(
                latest,
                "timestamp",
                "created_at",
                "assessment_date",
                default=None,
            )

            latest_model = get_value(
                latest,
                "model_version",
                default="N/A",
            )

            threshold = get_value(
                latest,
                "threshold",
                "decision_threshold",
                default=None,
            )

            if threshold is None:
                threshold_text = "N/A"
            else:
                threshold_text = f"{float(threshold):.2f}"

            st.markdown(
                f'<div class="detail-card">'
                f'<div class="detail-label">'
                f'Latest Assessment'
                f'</div>'
                f'<div class="detail-main">'
                f'{risk_percentage(latest):.2f}%'
                f'</div>'
                f'<div class="detail-row">'
                f'<span>Date</span>'
                f'<strong>'
                f'{format_timestamp(latest_timestamp)}'
                f'</strong>'
                f'</div>'
                f'<div class="detail-row">'
                f'<span>Model</span>'
                f'<strong>{latest_model}</strong>'
                f'</div>'
                f'<div class="detail-row">'
                f'<span>Decision threshold</span>'
                f'<strong>{threshold_text}</strong>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # --------------------------------------------------------
        # SUMMARY
        # --------------------------------------------------------

        with d2:

            percentages = [
                risk_percentage(item)
                for item in history
            ]

            highest = max(percentages)
            lowest = min(percentages)

            total_change = (
                percentages[-1]
                - percentages[0]
            )

            if total_change > 0:
                total_change_text = (
                    f"+{total_change:.2f} pp"
                )
            else:
                total_change_text = (
                    f"{total_change:.2f} pp"
                )

            st.markdown(
                f'<div class="detail-card">'
                f'<div class="detail-label">'
                f'Risk Journey Summary'
                f'</div>'
                f'<div class="detail-row">'
                f'<span>Highest recorded risk</span>'
                f'<strong>{highest:.2f}%</strong>'
                f'</div>'
                f'<div class="detail-row">'
                f'<span>Lowest recorded risk</span>'
                f'<strong>{lowest:.2f}%</strong>'
                f'</div>'
                f'<div class="detail-row">'
                f'<span>Change since first assessment</span>'
                f'<strong>{total_change_text}</strong>'
                f'</div>'
                f'<div class="detail-row">'
                f'<span>Total assessments</span>'
                f'<strong>{count}</strong>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # --------------------------------------------------------
        # HISTORY TABLE
        # --------------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Assessment History'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-caption">'
            'Each row represents a separate model assessment.'
            '</div>',
            unsafe_allow_html=True,
        )

        rows = []

        for index, item in enumerate(
            history,
            start=1,
        ):

            timestamp = get_value(
                item,
                "timestamp",
                "created_at",
                "assessment_date",
                default=None,
            )

            band = get_value(
                item,
                "risk_band",
                default="N/A",
            )

            change = get_value(
                item,
                "change_percentage_points",
                "change",
                default=None,
            )

            prediction = get_value(
                item,
                "prediction",
                default=0,
            )

            model = get_value(
                item,
                "model_version",
                default="N/A",
            )

            rows.append(
                {
                    "Assessment": index,
                    "Date": format_timestamp(timestamp),
                    "Risk": f"{risk_percentage(item):.2f}%",
                    "Risk Band": display_band(band),
                    "Change": (
                        f"{float(change):+.2f} pp"
                        if change is not None
                        else "—"
                    ),
                    "Prediction": (
                        "Positive"
                        if prediction == 1
                        else "Negative"
                    ),
                    "Model": model,
                }
            )

        table_df = pd.DataFrame(rows)

        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
            height=min(
                450,
                90 + len(table_df) * 38,
            ),
        )

        # --------------------------------------------------------
        # DISCLAIMER
        # --------------------------------------------------------

        st.markdown(
            '<div class="disclaimer">'
            '<strong>⚕️ Important:</strong> '
            'OncoGuard AI provides model-predicted risk '
            'estimates for research and decision-support '
            'purposes. These estimates are not a diagnosis '
            'and should not replace evaluation by a qualified '
            'healthcare professional.'
            '</div>',
            unsafe_allow_html=True,
        )

    except Exception as exc:

        st.error(
            "Unable to load the Risk Journey."
        )

        st.exception(exc)

    finally:

        db.close()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    patient_id = render_sidebar()

    render_risk_history(
        patient_id
    )