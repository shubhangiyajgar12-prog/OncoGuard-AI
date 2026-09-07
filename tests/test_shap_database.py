import pandas as pd

from database.connection import SessionLocal
from database.crud import (
    get_patient_profile,
    get_latest_assessment,
    get_assessment_explanations,
)
from explainability.shap_engine import explain_and_store


def create_patient_features(profile):
    """
    Build the complete 12-feature input required by the SHAP engine.

    The engineered features match the feature engineering rules
    used by the prediction pipeline.
    """

    age = profile.age
    gender = profile.gender
    bmi = profile.bmi
    smoking = profile.smoking
    genetic_risk = profile.genetic_risk
    physical_activity = profile.physical_activity
    alcohol_intake = profile.alcohol_intake
    cancer_history = profile.cancer_history

    # ---------------------------------------------------------
    # Engineered feature 1: Is_Obese
    # ---------------------------------------------------------
    is_obese = int(bmi >= 30)

    # ---------------------------------------------------------
    # Engineered feature 2: Risk_Factor_Count
    # ---------------------------------------------------------
    risk_factor_count = (
        smoking
        + genetic_risk
        + cancer_history
        + is_obese
    )

    # ---------------------------------------------------------
    # Engineered feature 3: Age_Group
    # ---------------------------------------------------------
    if age < 30:
        age_group = 0
    elif age < 45:
        age_group = 1
    elif age < 60:
        age_group = 2
    else:
        age_group = 3

    # ---------------------------------------------------------
    # Engineered feature 4: BMI_Category
    # ---------------------------------------------------------
    if bmi < 18.5:
        bmi_category = 0
    elif bmi < 25:
        bmi_category = 1
    elif bmi < 30:
        bmi_category = 2
    else:
        bmi_category = 3

    # ---------------------------------------------------------
    # Complete 12-feature DataFrame
    # ---------------------------------------------------------
    return pd.DataFrame(
        [{
            "Age": age,
            "Gender": gender,
            "BMI": bmi,
            "Smoking": smoking,
            "GeneticRisk": genetic_risk,
            "PhysicalActivity": physical_activity,
            "AlcoholIntake": alcohol_intake,
            "CancerHistory": cancer_history,
            "Is_Obese": is_obese,
            "Risk_Factor_Count": risk_factor_count,
            "Age_Group": age_group,
            "BMI_Category": bmi_category,
        }]
    )


def main():
    print("=" * 70)
    print("ONCOGUARD AI - SHAP DATABASE INTEGRATION TEST")
    print("=" * 70)

    db = SessionLocal()

    try:
        # ---------------------------------------------------------
        # 1. Get demo patient
        # ---------------------------------------------------------
        patient_id = 1

        profile = get_patient_profile(db, patient_id)

        if profile is None:
            raise RuntimeError("Patient profile not found.")

        print(f"✓ Patient profile loaded: ID={patient_id}")

        # ---------------------------------------------------------
        # 2. Get latest risk assessment
        # ---------------------------------------------------------
        assessment = get_latest_assessment(db, patient_id)

        if assessment is None:
            raise RuntimeError(
                "No risk assessment found. "
                "Run the risk assessment service test first."
            )

        print(f"✓ Risk assessment loaded: ID={assessment.id}")
        print(f"  Risk probability: {assessment.probability:.4f}")
        print(f"  Risk band: {assessment.risk_band}")

        # ---------------------------------------------------------
        # 3. Build complete patient features
        # ---------------------------------------------------------
        patient_features = create_patient_features(profile)

        print("✓ Patient features prepared as DataFrame")
        print(f"  Feature count: {len(patient_features.columns)}")

        # Verify all 12 features are present
        expected_features = [
            "Age",
            "Gender",
            "BMI",
            "Smoking",
            "GeneticRisk",
            "PhysicalActivity",
            "AlcoholIntake",
            "CancerHistory",
            "Is_Obese",
            "Risk_Factor_Count",
            "Age_Group",
            "BMI_Category",
        ]

        missing_features = [
            feature
            for feature in expected_features
            if feature not in patient_features.columns
        ]

        if missing_features:
            raise RuntimeError(
                f"Missing features: {missing_features}"
            )

        print("✓ All 12 SHAP features verified")

        # ---------------------------------------------------------
        # 4. Generate SHAP explanation and store in database
        # ---------------------------------------------------------
        explanation = explain_and_store(
            db=db,
            assessment_id=assessment.id,
            patient_features=patient_features,
            patient_id=patient_id,
            top_n=10,
        )

        print(
            f"✓ SHAP explanation generated: "
            f"{len(explanation)} features"
        )

        # ---------------------------------------------------------
        # 5. Verify database records
        # ---------------------------------------------------------
        stored = get_assessment_explanations(
            db,
            assessment.id,
        )

        if not stored:
            raise RuntimeError(
                "SHAP explanation was not stored in the database."
            )

        print(
            f"✓ Database contains "
            f"{len(stored)} SHAP explanation records"
        )

        # ---------------------------------------------------------
        # 6. Display stored explanations
        # ---------------------------------------------------------
        print("\nStored SHAP explanations:")

        for item in stored:
            print(
                f"  {item.feature:25s} "
                f"value={str(item.feature_value):10s} "
                f"SHAP={item.shap_value:+.6f} "
                f"direction={item.direction}"
            )

        # ---------------------------------------------------------
        # 7. Final result
        # ---------------------------------------------------------
        print("\n" + "=" * 70)
        print("SHAP DATABASE INTEGRATION TEST PASSED")
        print("=" * 70)

    finally:
        db.close()


if __name__ == "__main__":
    main()