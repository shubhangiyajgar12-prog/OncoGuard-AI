"""
OncoGuard AI - Risk History Service

Provides longitudinal model-predicted risk history for patients.

Important:
This represents a history of ML-based risk assessments.
It does NOT represent cancer progression, cancer stage,
or a clinical diagnosis.
"""

from datetime import datetime

from database.crud import get_patient_assessments


# ============================================================
# TIMESTAMP HELPER
# ============================================================

def _get_assessment_timestamp(assessment):
    """
    Get the timestamp from a RiskAssessment object.

    The project database may use different timestamp field
    names depending on the current schema. Prefer the actual
    timestamp field available on the object.
    """

    # Current / preferred field
    timestamp = getattr(
        assessment,
        "timestamp",
        None,
    )

    if timestamp is not None:
        return timestamp

    # Backward compatibility
    timestamp = getattr(
        assessment,
        "created_at",
        None,
    )

    if timestamp is not None:
        return timestamp

    # Additional compatibility
    timestamp = getattr(
        assessment,
        "assessment_date",
        None,
    )

    if timestamp is not None:
        return timestamp

    return None


# ============================================================
# FORMAT HELPERS
# ============================================================

def _format_timestamp(timestamp):
    """Return a user-friendly timestamp string."""

    if timestamp is None:
        return "Unknown"

    if isinstance(timestamp, datetime):
        return timestamp.strftime("%Y-%m-%d %H:%M")

    return str(timestamp)


def _risk_percentage(probability):
    """Convert probability into percentage."""

    if probability is None:
        return None

    return float(probability) * 100.0


# ============================================================
# RISK CHANGE
# ============================================================

def _calculate_change(
    current_probability,
    previous_probability,
):
    """
    Calculate change in model-predicted risk.

    Returns percentage-point change, NOT percentage change.
    """

    if (
        current_probability is None
        or previous_probability is None
    ):
        return None

    return (
        float(current_probability)
        - float(previous_probability)
    ) * 100.0


# ============================================================
# MODEL VERSION HELPER
# ============================================================

def _get_model_version(assessment):
    """
    Safely retrieve the model version associated with
    a risk assessment.
    """

    model_version = getattr(
        assessment,
        "model_version",
        None,
    )

    if model_version is None:
        return None

    version = getattr(
        model_version,
        "version",
        None,
    )

    if version is not None:
        return version

    return str(model_version)


# ============================================================
# PROBABILITY HELPER
# ============================================================

def _get_probability(assessment):
    """
    Safely retrieve risk probability from RiskAssessment.

    Current database model uses the Python attribute
    'probability', which maps to the database column
    'risk_probability'.

    Backward compatibility is retained for risk_probability.
    """

    probability = getattr(
        assessment,
        "probability",
        None,
    )

    if probability is not None:
        return float(probability)

    probability = getattr(
        assessment,
        "risk_probability",
        None,
    )

    if probability is not None:
        return float(probability)

    return None


# ============================================================
# GET PATIENT RISK HISTORY
# ============================================================

def get_risk_history(
    db,
    patient_id,
):
    """
    Return chronological risk assessment history
    for a patient.

    Each record contains:

    - assessment ID
    - timestamp
    - risk probability
    - risk percentage
    - prediction
    - risk band
    - decision threshold
    - model version
    - previous risk
    - change from previous assessment
    """

    assessments = get_patient_assessments(
        db=db,
        patient_id=patient_id,
    )

    if not assessments:
        return []

    # --------------------------------------------------------
    # Sort oldest -> newest
    # --------------------------------------------------------

    assessments = sorted(
        assessments,
        key=lambda assessment: (
            _get_assessment_timestamp(
                assessment
            )
            if _get_assessment_timestamp(
                assessment
            ) is not None
            else datetime.min
        ),
    )

    history = []

    previous_probability = None

    for assessment in assessments:

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        timestamp = _get_assessment_timestamp(
            assessment
        )

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probability = _get_probability(
            assessment
        )

        # ----------------------------------------------------
        # Model version
        # ----------------------------------------------------

        model_version = _get_model_version(
            assessment
        )

        # ----------------------------------------------------
        # Calculate change
        # ----------------------------------------------------

        change = _calculate_change(
            probability,
            previous_probability,
        )

        # ----------------------------------------------------
        # Build history record
        # ----------------------------------------------------

        record = {
            "assessment_id": assessment.id,

            "timestamp": timestamp,

            "timestamp_display": _format_timestamp(
                timestamp
            ),

            "risk_probability": probability,

            "risk_percentage": _risk_percentage(
                probability
            ),

            "threshold": (
                float(assessment.threshold)
                if assessment.threshold is not None
                else None
            ),

            "prediction": (
                int(assessment.prediction)
                if assessment.prediction is not None
                else None
            ),

            "risk_band": assessment.risk_band,

            "model_version": model_version,

            "previous_risk_percentage": (
                _risk_percentage(
                    previous_probability
                )
                if previous_probability is not None
                else None
            ),

            "change_percentage_points": change,
        }

        history.append(record)

        previous_probability = probability

    return history


# ============================================================
# CURRENT RISK
# ============================================================

def get_current_risk(
    db,
    patient_id,
):
    """
    Return the latest model-predicted risk assessment.
    """

    history = get_risk_history(
        db=db,
        patient_id=patient_id,
    )

    if not history:
        return None

    return history[-1]


# ============================================================
# RISK TREND
# ============================================================

def get_risk_trend(
    db,
    patient_id,
):
    """
    Determine the overall direction of the patient's
    model-predicted risk.

    Returns:

        increasing
        decreasing
        stable
        insufficient_data
    """

    history = get_risk_history(
        db=db,
        patient_id=patient_id,
    )

    if len(history) < 2:
        return "insufficient_data"

    first = history[0]["risk_percentage"]
    latest = history[-1]["risk_percentage"]

    if first is None or latest is None:
        return "insufficient_data"

    difference = latest - first

    # Small changes are considered stable.
    if abs(difference) < 1.0:
        return "stable"

    if difference > 0:
        return "increasing"

    return "decreasing"


# ============================================================
# HISTORY SUMMARY
# ============================================================

def get_risk_history_summary(
    db,
    patient_id,
):
    """
    Return a compact summary suitable for dashboards.
    """

    history = get_risk_history(
        db=db,
        patient_id=patient_id,
    )

    if not history:
        return {
            "assessment_count": 0,
            "current_risk_percentage": None,
            "current_risk_band": None,
            "previous_risk_percentage": None,
            "change_percentage_points": None,
            "trend": "insufficient_data",
        }

    latest = history[-1]

    previous = (
        history[-2]
        if len(history) >= 2
        else None
    )

    return {
        "assessment_count": len(history),

        "current_risk_percentage": (
            latest["risk_percentage"]
        ),

        "current_risk_band": (
            latest["risk_band"]
        ),

        "previous_risk_percentage": (
            previous["risk_percentage"]
            if previous is not None
            else None
        ),

        "change_percentage_points": (
            latest["change_percentage_points"]
        ),

        "trend": get_risk_trend(
            db=db,
            patient_id=patient_id,
        ),
    }


# ============================================================
# SERVICE TEST
# ============================================================

def main():

    from database.connection import SessionLocal

    print("=" * 70)
    print("ONCOGUARD AI - RISK HISTORY SERVICE TEST")
    print("=" * 70)

    db = SessionLocal()

    try:

        patient_id = 1

        print(
            f"✓ Testing patient ID: {patient_id}"
        )

        # ----------------------------------------------------
        # Get history
        # ----------------------------------------------------

        history = get_risk_history(
            db=db,
            patient_id=patient_id,
        )

        print(
            f"\n✓ Risk assessments found: "
            f"{len(history)}"
        )

        if not history:

            print(
                "\nNo risk assessment history "
                "available for this patient."
            )

            print(
                "\n" + "=" * 70
            )

            print(
                "RISK HISTORY SERVICE TEST PASSED"
            )

            print(
                "=" * 70
            )

            return

        # ----------------------------------------------------
        # Display history
        # ----------------------------------------------------

        print("\nRisk History:")

        for index, record in enumerate(
            history,
            start=1,
        ):

            print(
                f"\n  Assessment #{index}"
            )

            print(
                f"    ID             : "
                f"{record['assessment_id']}"
            )

            print(
                f"    Timestamp      : "
                f"{record['timestamp_display']}"
            )

            # Risk
            if record["risk_percentage"] is not None:

                print(
                    f"    Risk           : "
                    f"{record['risk_percentage']:.2f}%"
                )

            else:

                print(
                    "    Risk           : N/A"
                )

            print(
                f"    Risk Band      : "
                f"{record['risk_band']}"
            )

            print(
                f"    Prediction     : "
                f"{record['prediction']}"
            )

            # Threshold
            if record["threshold"] is not None:

                print(
                    f"    Threshold      : "
                    f"{record['threshold']:.4f}"
                )

            else:

                print(
                    "    Threshold      : N/A"
                )

            print(
                f"    Model Version  : "
                f"{record['model_version']}"
            )

            # Change
            if (
                record[
                    "change_percentage_points"
                ]
                is not None
            ):

                change = record[
                    "change_percentage_points"
                ]

                print(
                    f"    Change         : "
                    f"{change:+.2f} percentage points"
                )

        # ----------------------------------------------------
        # Current risk
        # ----------------------------------------------------

        current = get_current_risk(
            db=db,
            patient_id=patient_id,
        )

        print("\nCurrent Risk:")

        if current is not None:

            if current["risk_percentage"] is not None:

                print(
                    f"  Risk       : "
                    f"{current['risk_percentage']:.2f}%"
                )

            else:

                print(
                    "  Risk       : N/A"
                )

            print(
                f"  Risk Band  : "
                f"{current['risk_band']}"
            )

        # ----------------------------------------------------
        # Trend
        # ----------------------------------------------------

        trend = get_risk_trend(
            db=db,
            patient_id=patient_id,
        )

        print(
            f"\n✓ Overall trend: {trend}"
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        summary = get_risk_history_summary(
            db=db,
            patient_id=patient_id,
        )

        print("\nSummary:")

        print(
            f"  Assessments : "
            f"{summary['assessment_count']}"
        )

        if (
            summary["current_risk_percentage"]
            is not None
        ):

            print(
                f"  Current     : "
                f"{summary['current_risk_percentage']:.2f}%"
            )

        else:

            print(
                "  Current     : N/A"
            )

        print(
            f"  Band        : "
            f"{summary['current_risk_band']}"
        )

        print(
            f"  Trend       : "
            f"{summary['trend']}"
        )

        print("\n" + "=" * 70)
        print("RISK HISTORY SERVICE TEST PASSED")
        print("=" * 70)

    finally:

        db.close()


if __name__ == "__main__":
    main()