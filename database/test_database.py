from database.connection import SessionLocal
from database.schema import create_tables
from database.models import (
    User,
    PatientProfile,
    ModelVersion,
)
from database.crud import (
    create_user,
    create_patient_profile,
    create_model_version,
    create_risk_assessment,
    create_explanation,
    create_recommendation,
)


# ============================================================
# GET OR CREATE DEMO USER
# ============================================================

def get_or_create_user(db):

    user = (
        db.query(User)
        .filter(User.username == "demo_patient")
        .first()
    )

    if user:
        print(
            f"✓ Existing demo user reused: ID={user.id}"
        )
        return user

    user = create_user(
        db=db,
        username="demo_patient",
        email="demo_patient@oncoguard.local",
        password_hash="DEMO_HASH",
        role="patient",
    )

    print(
        f"✓ Demo user created: ID={user.id}"
    )

    return user


# ============================================================
# GET OR CREATE PATIENT PROFILE
# ============================================================

def get_or_create_profile(db, user):

    profile = (
        db.query(PatientProfile)
        .filter(
            PatientProfile.user_id == user.id
        )
        .first()
    )

    if profile:
        print(
            f"✓ Existing patient profile reused: ID={profile.id}"
        )
        return profile

    profile = create_patient_profile(
        db=db,
        user_id=user.id,

        age=61,
        gender=1,
        bmi=21.16697,

        smoking=0,
        genetic_risk=1,
        physical_activity=7.42636,
        alcohol_intake=1.97042,
        cancer_history=0,
    )

    print(
        f"✓ Patient profile created: ID={profile.id}"
    )

    return profile


# ============================================================
# GET OR CREATE MODEL VERSION
# ============================================================

def get_or_create_model(db):

    model = (
        db.query(ModelVersion)
        .filter(
            ModelVersion.version == "oncoguard-v1.0"
        )
        .first()
    )

    if model:
        print(
            f"✓ Existing model version reused: ID={model.id}"
        )
        return model

    model = create_model_version(
        db=db,

        version="oncoguard-v1.0",

        model_path=(
            "models/"
            "oncoguard_catboost_calibrated.joblib"
        ),

        threshold=0.39,

        metrics={
            "accuracy": 0.9467,
            "precision": 0.9390,
            "recall": 0.9167,
            "specificity": 0.9645,
            "f1": 0.9277,
            "roc_auc": 0.9601,
            "pr_auc": 0.9579,
            "brier_score": 0.0446,
        },

        calibration_method="sigmoid",

        active=True,
    )

    print(
        f"✓ Model version created: ID={model.id}"
    )

    return model


# ============================================================
# MAIN DATABASE TEST
# ============================================================

def main():

    print()
    print("========================================")
    print("   ONCOGUARD AI DATABASE TEST")
    print("========================================")
    print()

    # --------------------------------------------------------
    # CREATE TABLES
    # --------------------------------------------------------

    create_tables()

    print("✓ Database tables verified")

    db = SessionLocal()

    try:

        # ----------------------------------------------------
        # 1. USER
        # ----------------------------------------------------

        user = get_or_create_user(db)

        # ----------------------------------------------------
        # 2. PATIENT PROFILE
        # ----------------------------------------------------

        profile = get_or_create_profile(
            db=db,
            user=user,
        )

        # ----------------------------------------------------
        # 3. MODEL VERSION
        # ----------------------------------------------------

        model = get_or_create_model(db)

        # ----------------------------------------------------
        # 4. RISK ASSESSMENT
        # ----------------------------------------------------

        assessment = create_risk_assessment(

            db=db,

            patient_id=profile.id,

            model_version_id=model.id,

            risk_probability=0.0308,

            decision_threshold=0.39,

            prediction=0,

            risk_band="Lower predicted risk",

            input_snapshot={

                "Age": 61,

                "Gender": 1,

                "BMI": 21.16697,

                "Smoking": 0,

                "GeneticRisk": 1,

                "PhysicalActivity": 7.42636,

                "AlcoholIntake": 1.97042,

                "CancerHistory": 0,

                "Is_Obese": 0,

                "Risk_Factor_Count": 1,

                "Age_Group": 3,

                "BMI_Category": 1,
            },
        )

        print(
            f"✓ Risk assessment created: ID={assessment.id}"
        )

        # ----------------------------------------------------
        # 5. SHAP EXPLANATION
        # ----------------------------------------------------

        explanation = create_explanation(

            db=db,

            assessment_id=assessment.id,

            feature="GeneticRisk",

            feature_value=1,

            shap_value=0.0842,

            direction="Increases predicted risk",

            rank=1,
        )

        print(
            f"✓ Risk explanation created: ID={explanation.id}"
        )

        # ----------------------------------------------------
        # 6. RECOMMENDATION
        # ----------------------------------------------------

        recommendation = create_recommendation(

            db=db,

            patient_id=profile.id,

            assessment_id=assessment.id,

            feature="GeneticRisk",

            text=(
                "Consider discussing your family and "
                "genetic risk history with a qualified "
                "healthcare professional."
            ),

            reason=(
                "The model identified GeneticRisk as "
                "an important contributor to the "
                "predicted risk."
            ),

            status="Not Started",
        )

        print(
            f"✓ Recommendation created: "
            f"ID={recommendation.id}"
        )

        # ----------------------------------------------------
        # FINAL COMMIT
        # ----------------------------------------------------

        db.commit()

        print()
        print("========================================")
        print("   DATABASE TEST PASSED")
        print("========================================")

        print(
            f"User ID:           {user.id}"
        )

        print(
            f"Profile ID:        {profile.id}"
        )

        print(
            f"Model ID:          {model.id}"
        )

        print(
            f"Assessment ID:     {assessment.id}"
        )

        print(
            f"Explanation ID:    {explanation.id}"
        )

        print(
            f"Recommendation ID: {recommendation.id}"
        )

        print("========================================")
        print()

    except Exception as e:

        db.rollback()

        print()
        print("========================================")
        print("   DATABASE TEST FAILED")
        print("========================================")

        print(
            f"Error: {e}"
        )

        print("========================================")
        print()

        raise

    finally:

        db.close()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()