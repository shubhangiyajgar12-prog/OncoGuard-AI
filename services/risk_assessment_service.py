"""
OncoGuard AI - Risk Assessment Service

Connects:
    Patient Profile
        ↓
    ML Prediction Engine
        ↓
    Risk Assessment
        ↓
    Database

This service does NOT perform diagnosis.
It stores model-predicted risk assessments for the application.
"""

from typing import Optional

from sqlalchemy.orm import Session

from ml.prediction import predict_risk
from database.crud import (
    create_risk_assessment,
    get_active_model,
)


# ============================================================
# CREATE RISK ASSESSMENT
# ============================================================

def assess_patient_risk(
    db: Session,
    patient_id: int,
    age: int,
    gender: int,
    bmi: float,
    smoking: int,
    genetic_risk: int,
    physical_activity: float,
    alcohol_intake: float,
    cancer_history: int,
):
    """
    Run the ML model and save the resulting assessment
    into the database.

    Returns:
        {
            "assessment": RiskAssessment,
            "prediction": dict
        }
    """

    # --------------------------------------------------------
    # 1. Run ML prediction
    # --------------------------------------------------------

    prediction_result = predict_risk(
        age=age,
        gender=gender,
        bmi=bmi,
        smoking=smoking,
        genetic_risk=genetic_risk,
        physical_activity=physical_activity,
        alcohol_intake=alcohol_intake,
        cancer_history=cancer_history,
    )

    # --------------------------------------------------------
    # 2. Find active model version
    # --------------------------------------------------------

    active_model = get_active_model(db)

    if active_model is None:
        raise ValueError(
            "No active model version found in the database. "
            "Create and activate a model version before "
            "creating patient assessments."
        )

    # --------------------------------------------------------
    # 3. Store exact model inputs
    #
    # This snapshot is important for longitudinal tracking
    # and reproducibility.
    # --------------------------------------------------------

    input_snapshot = prediction_result["input_features"]

    # --------------------------------------------------------
    # 4. Save assessment
    # --------------------------------------------------------

    assessment = create_risk_assessment(
        db=db,
        patient_id=patient_id,
        model_version_id=active_model.id,
        risk_probability=prediction_result["probability"],
        decision_threshold=prediction_result["threshold"],
        prediction=prediction_result["prediction"],
        risk_band=prediction_result["risk_band"],
        input_snapshot=input_snapshot,
    )

    return {
        "assessment": assessment,
        "prediction": prediction_result,
    }


# ============================================================
# GET SIMPLE ASSESSMENT RESULT
# ============================================================

def assess_patient(
    db: Session,
    patient_id: int,
    age: int,
    gender: int,
    bmi: float,
    smoking: int,
    genetic_risk: int,
    physical_activity: float,
    alcohol_intake: float,
    cancer_history: int,
):
    """
    Convenience wrapper returning a clean dictionary
    suitable for Streamlit/API responses.
    """

    result = assess_patient_risk(
        db=db,
        patient_id=patient_id,
        age=age,
        gender=gender,
        bmi=bmi,
        smoking=smoking,
        genetic_risk=genetic_risk,
        physical_activity=physical_activity,
        alcohol_intake=alcohol_intake,
        cancer_history=cancer_history,
    )

    prediction = result["prediction"]
    assessment = result["assessment"]

    return {
        "assessment_id": assessment.id,
        "patient_id": assessment.patient_id,
        "model_version_id": assessment.model_version_id,
        "probability": prediction["probability"],
        "risk_percentage": prediction["risk_percentage"],
        "threshold": prediction["threshold"],
        "prediction": prediction["prediction"],
        "risk_band": prediction["risk_band"],
        "input_features": prediction["input_features"],
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    from database.connection import SessionLocal
    from database.schema import create_tables
    from database.crud import (
        get_user_by_username,
        get_patient_profile,
    )

    print("=" * 60)
    print("ONCOGUARD AI - RISK ASSESSMENT SERVICE TEST")
    print("=" * 60)

    # Ensure database tables exist.
    create_tables()

    db = SessionLocal()

    try:

        # ----------------------------------------------------
        # Find existing demo user
        # ----------------------------------------------------

        user = get_user_by_username(
            db,
            "demo_patient",
        )

        if user is None:
            raise ValueError(
                "Demo patient not found. "
                "Run database.test_database first."
            )

        profile = get_patient_profile(
            db,
            user.id,
        )

        if profile is None:
            raise ValueError(
                "Demo patient profile not found."
            )

        print("\n✓ Demo patient found")
        print(f"  User ID:    {user.id}")
        print(f"  Profile ID: {profile.id}")

        # ----------------------------------------------------
        # Make sure an active model exists
        # ----------------------------------------------------

        active_model = get_active_model(db)

        if active_model is None:
            raise ValueError(
                "No active model version found.\n"
                "Run the database/model-version setup first."
            )

        print("\n✓ Active model found")
        print(f"  Model ID: {active_model.id}")
        print(f"  Version:  {active_model.version}")

        # ----------------------------------------------------
        # Run complete assessment
        # ----------------------------------------------------

        result = assess_patient(
            db=db,
            patient_id=profile.id,
            age=profile.age,
            gender=profile.gender,
            bmi=profile.bmi,
            smoking=profile.smoking,
            genetic_risk=profile.genetic_risk,
            physical_activity=profile.physical_activity,
            alcohol_intake=profile.alcohol_intake,
            cancer_history=profile.cancer_history,
        )

        print("\n✓ Risk assessment created")

        print("\nAssessment Result")
        print("-" * 60)

        print(
            f"Assessment ID    : "
            f"{result['assessment_id']}"
        )

        print(
            f"Probability      : "
            f"{result['probability']:.4f}"
        )

        print(
            f"Risk Percentage  : "
            f"{result['risk_percentage']:.2f}%"
        )

        print(
            f"Threshold        : "
            f"{result['threshold']:.4f}"
        )

        print(
            f"Prediction       : "
            f"{result['prediction']}"
        )

        print(
            f"Risk Band        : "
            f"{result['risk_band']}"
        )

        print("\n" + "=" * 60)
        print("RISK ASSESSMENT SERVICE TEST PASSED")
        print("=" * 60)

    finally:
        db.close()