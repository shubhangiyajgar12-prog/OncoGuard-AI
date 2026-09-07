import sys
from pathlib import Path

# ================================================================
# PROJECT ROOT
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ================================================================
# IMPORTS
# ================================================================

import streamlit as st
import pandas as pd
import plotly.express as px

from database.connection import SessionLocal
from services.risk_history_service import (
    get_current_risk,
    get_risk_history,
    get_risk_history_summary,
    get_risk_trend,
)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def format_timestamp(timestamp):
    """Safely format a timestamp."""

    if timestamp is None:
        return "N/A"

    if hasattr(timestamp, "strftime"):
        return timestamp.strftime("%d %b %Y, %I:%M %p")

    return str(timestamp)


def display_band(band):
    """Convert stored risk-band values into patient-friendly labels."""

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


def band_class(band):
    """Return CSS class for a risk band."""

    value = str(band).lower()

    if "low" in value or "lower" in value:
        return "low"

    if "medium" in value or "moderate" in value:
        return "medium"

    if "high" in value or "higher" in value:
        return "high"

    return "unknown"


def get_value(item, *keys, default=None):
    """
    Read a value from either a dictionary or an object.
    """

    if isinstance(item, dict):

        for key in keys:

            if key in item:
                return item[key]

        return default

    for key in keys:

        if hasattr(item, key):
            return getattr(item, key)

    return default


# ================================================================
# RISK HISTORY PAGE
# ================================================================

def render_risk_history(patient_id):
    """
    Render the patient's risk history.
    """

    st.title("📈 Risk Journey")

    st.caption(
        "Review how your model-predicted risk has changed across "
        "previous assessments."
    )

    st.info(
        "This page shows changes in model-predicted risk across "
        "assessments. It does not represent cancer progression, "
        "cancer stage, or a medical diagnosis."
    )

    db = SessionLocal()

    try:

        # ========================================================
        # LOAD DATA
        # ========================================================

        history = get_risk_history(
            db,
            patient_id
        )

        current = get_current_risk(
            db,
            patient_id
        )

        summary = get_risk_history_summary(
            db,
            patient_id
        )

        trend = get_risk_trend(
            db,
            patient_id
        )

        # ========================================================
        # NO HISTORY
        # ========================================================

        if not history:

            st.warning(
                "No risk assessments are available yet."
            )

            st.write(
                "Complete a risk assessment to begin your "
                "risk journey."
            )

            return

        # ========================================================
        # CURRENT RISK
        # ========================================================

        st.subheader(
            "Current Model-Predicted Risk"
        )

        current_probability = get_value(
            current,
            "probability",
            "risk_probability",
            default=0.0,
        )

        current_percentage = get_value(
            current,
            "risk_percentage",
            default=None,
        )

        if current_percentage is None:

            current_percentage = (
                float(current_probability) * 100
            )

        current_band = get_value(
            current,
            "risk_band",
            default="N/A",
        )

        current_prediction = get_value(
            current,
            "prediction",
            default=0,
        )

        css_class = band_class(
            current_band
        )

        # ========================================================
        # RISK CARD
        # ========================================================
        #
        # IMPORTANT:
        # Keep the HTML on a single line.
        # This prevents Streamlit Markdown from treating
        # the inner divs as a code block.
        # ========================================================

        risk_card_html = (
            f'<div class="risk-card {css_class}">'
            f'<div class="risk-card-title">Current Model-Predicted Risk</div>'
            f'<div class="risk-card-value">{float(current_percentage):.2f}%</div>'
            f'<div class="risk-card-band">{display_band(current_band)}</div>'
            f'</div>'
        )

        st.markdown(
            risk_card_html,
            unsafe_allow_html=True,
        )

        # ========================================================
        # SUMMARY METRICS
        # ========================================================

        col1, col2, col3 = st.columns(3)

        assessment_count = get_value(
            summary,
            "assessment_count",
            "assessments",
            "count",
            default=len(history),
        )

        with col1:

            st.metric(
                "Total Assessments",
                assessment_count,
            )

        with col2:

            st.metric(
                "Prediction",
                (
                    "Positive"
                    if current_prediction == 1
                    else "Negative"
                ),
            )

        with col3:

            st.metric(
                "Overall Trend",
                str(trend).capitalize(),
            )

        # ========================================================
        # RISK JOURNEY GRAPH
        # ========================================================

        st.divider()

        st.subheader(
            "Risk Journey"
        )

        chart_rows = []

        for index, item in enumerate(
            history,
            start=1
        ):

            probability = get_value(
                item,
                "probability",
                "risk_probability",
                default=0.0,
            )

            risk_percentage = get_value(
                item,
                "risk_percentage",
                default=None,
            )

            if risk_percentage is None:

                risk_percentage = (
                    float(probability) * 100
                )

            timestamp = get_value(
                item,
                "timestamp",
                "created_at",
                "assessment_date",
                default=None,
            )

            risk_band = get_value(
                item,
                "risk_band",
                default="N/A",
            )

            chart_rows.append(
                {
                    "Assessment": index,

                    "Risk Percentage":
                        float(risk_percentage),

                    "Risk Band":
                        display_band(risk_band),

                    "Date":
                        format_timestamp(timestamp),
                }
            )

        chart_df = pd.DataFrame(
            chart_rows
        )

        if len(chart_df) >= 2:

            fig = px.line(
                chart_df,
                x="Assessment",
                y="Risk Percentage",
                markers=True,

                hover_data=[
                    "Date",
                    "Risk Band",
                ],

                labels={
                    "Assessment":
                        "Assessment",

                    "Risk Percentage":
                        "Model-Predicted Risk (%)",
                },

                title=
                    "Model-Predicted Risk Over Time",
            )

            fig.update_yaxes(
                range=[
                    0,
                    max(
                        100,
                        float(
                            chart_df[
                                "Risk Percentage"
                            ].max()
                        ) + 5,
                    ),
                ]
            )

            fig.update_layout(
                height=450,
                hovermode="x unified",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:

            st.info(
                "A risk journey chart will appear "
                "after additional assessments are recorded."
            )

        # ========================================================
        # TREND MESSAGE
        # ========================================================

        trend_value = str(
            trend
        ).lower()

        if trend_value == "increasing":

            st.warning(
                "The model-predicted risk has increased "
                "across the available assessments."
            )

        elif trend_value == "decreasing":

            st.success(
                "The model-predicted risk has decreased "
                "across the available assessments."
            )

        else:

            st.info(
                "The model-predicted risk has remained "
                "relatively stable across the available "
                "assessments."
            )

        # ========================================================
        # HISTORY TABLE
        # ========================================================

        st.divider()

        st.subheader(
            "Assessment History"
        )

        table_rows = []

        for index, item in enumerate(
            history,
            start=1
        ):

            probability = get_value(
                item,
                "probability",
                "risk_probability",
                default=0.0,
            )

            risk_percentage = get_value(
                item,
                "risk_percentage",
                default=None,
            )

            if risk_percentage is None:

                risk_percentage = (
                    float(probability) * 100
                )

            timestamp = get_value(
                item,
                "timestamp",
                "created_at",
                "assessment_date",
                default=None,
            )

            risk_band = get_value(
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

            if change is None:

                change_display = "—"

            else:

                change_display = (
                    f"{float(change):+.2f} pp"
                )

            prediction = get_value(
                item,
                "prediction",
                default=0,
            )

            model_version = get_value(
                item,
                "model_version",
                default="N/A",
            )

            table_rows.append(
                {
                    "Assessment":
                        index,

                    "Date":
                        format_timestamp(timestamp),

                    "Risk":
                        f"{float(risk_percentage):.2f}%",

                    "Risk Band":
                        display_band(risk_band),

                    "Change":
                        change_display,

                    "Prediction":
                        (
                            "Positive"
                            if prediction == 1
                            else "Negative"
                        ),

                    "Model Version":
                        model_version,
                }
            )

        history_df = pd.DataFrame(
            table_rows
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        # ========================================================
        # LATEST ASSESSMENT DETAILS
        # ========================================================

        st.divider()

        st.subheader(
            "Latest Assessment Details"
        )

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

        latest_threshold = get_value(
            latest,
            "threshold",
            "decision_threshold",
            default=None,
        )

        detail_col1, detail_col2, detail_col3 = (
            st.columns(3)
        )

        with detail_col1:

            st.write(
                "**Assessment Date**"
            )

            st.write(
                format_timestamp(
                    latest_timestamp
                )
            )

        with detail_col2:

            st.write(
                "**Model Version**"
            )

            st.write(
                latest_model
            )

        with detail_col3:

            st.write(
                "**Decision Threshold**"
            )

            if latest_threshold is not None:

                st.write(
                    f"{float(latest_threshold):.2f}"
                )

            else:

                st.write("N/A")

        # ========================================================
        # DISCLAIMER
        # ========================================================

        st.divider()

        st.caption(
            "⚕️ OncoGuard AI provides model-predicted risk "
            "estimates for research and decision-support "
            "purposes. These estimates are not a diagnosis "
            "and should not replace evaluation by a qualified "
            "healthcare professional."
        )

    except Exception as exc:

        st.error(
            "Unable to load the risk history."
        )

        st.exception(exc)

    finally:

        db.close()


# ================================================================
# STANDALONE DEVELOPMENT TEST
# ================================================================

if __name__ == "__main__":

    st.set_page_config(
        page_title=
            "Risk Journey - OncoGuard AI",

        page_icon="📈",

        layout="wide",
    )

    # ============================================================
    # CUSTOM CSS
    # ============================================================

    st.markdown(
        """
<style>

.risk-card {
    padding: 28px;
    border-radius: 16px;
    border: 1px solid rgba(128, 128, 128, 0.25);
    margin: 10px 0 25px 0;
    box-sizing: border-box;
}

.risk-card-title {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 8px;
    color: #222222 !important;
}

.risk-card-value {
    font-size: 46px;
    font-weight: 700;
    line-height: 1.1;
    color: #222222 !important;
}

.risk-card-band {
    font-size: 19px;
    font-weight: 600;
    margin-top: 8px;
    color: #222222 !important;
}

.risk-card.low {
    background-color: #eef8ee;
}

.risk-card.medium {
    background-color: #fff8e6;
}

.risk-card.high {
    background-color: #fdecec;
}

.risk-card.unknown {
    background-color: #eeeeee;
}

</style>
""",
        unsafe_allow_html=True,
    )

    # ============================================================
    # DEVELOPMENT SIDEBAR
    # ============================================================

    st.sidebar.header(
        "Development Test"
    )

    patient_id = st.sidebar.number_input(
        "Patient ID",
        min_value=1,
        value=1,
        step=1,
    )

    # ============================================================
    # RENDER
    # ============================================================

    render_risk_history(
        int(patient_id)
    )