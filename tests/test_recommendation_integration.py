"""
OncoGuard AI - Recommendation Integration Test

End-to-end test:

Prediction
    ->
Risk Assessment
    ->
SHAP Explanation
    ->
Recommendation Generation
    ->
Recommendation Status Lifecycle
    ->
Cleanup
"""

import pandas as pd

from database.connection import SessionLocal

from database.crud import (
    get_patient_profile,
    get_active_model,
    create_risk_assessment,
    get_assessment_explanations,
    get_patient_recommendations,
)

from ml.prediction import predict_risk

from explainability.shap_engine import explain_and_store

from recommendations.recommendation_engine import (
    generate_recommendations,
    update_recommendation_status,
)


# ============================================================
# EXPECTED FEATURES
# ============================================================

FEATURE_NAMES = [
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


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_engineered_features(profile):
    """
    Build the exact 12-feature DataFrame expected
    by the trained CatBoost model and SHAP engine.
    """

    age = float(profile["Age"])

    bmi = float(profile["BMI"])

    smoking = int(profile["Smoking"])

    genetic_risk = int(
        profile["GeneticRisk"]
    )

    cancer_history = int(
        profile["CancerHistory"]
    )

    # --------------------------------------------------------
    # Is Obese
    # --------------------------------------------------------

    is_obese = int(
        bmi >= 30
    )

    # --------------------------------------------------------
    # Risk Factor Count
    # --------------------------------------------------------

    risk_factor_count = (
        smoking
        + genetic_risk
        + cancer_history
        + is_obese
    )

    # --------------------------------------------------------
    # Age Group
    # --------------------------------------------------------

    if age < 30:

        age_group = 0

    elif age < 45:

        age_group = 1

    elif age < 60:

        age_group = 2

    else:

        age_group = 3

    # --------------------------------------------------------
    # BMI Category
    # --------------------------------------------------------

    if bmi < 18.5:

        bmi_category = 0

    elif bmi < 25:

        bmi_category = 1

    elif bmi < 30:

        bmi_category = 2

    else:

        bmi_category = 3

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    data = {

        "Age": age,

        "Gender": int(
            profile["Gender"]
        ),

        "BMI": bmi,

        "Smoking": smoking,

        "GeneticRisk": genetic_risk,

        "PhysicalActivity": float(
            profile["PhysicalActivity"]
        ),

        "AlcoholIntake": float(
            profile["AlcoholIntake"]
        ),

        "CancerHistory": cancer_history,

        "Is_Obese": is_obese,

        "Risk_Factor_Count": (
            risk_factor_count
        ),

        "Age_Group": age_group,

        "BMI_Category": bmi_category,
    }

    df = pd.DataFrame(
        [data]
    )

    # Exact feature order.
    df = df[
        FEATURE_NAMES
    ]

    return df


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 70)

    print(
        "ONCOGUARD AI - "
        "RECOMMENDATION INTEGRATION TEST"
    )

    print("=" * 70)

    db = SessionLocal()

    test_assessment_id = None

    test_recommendation_ids = []

    try:

        # ====================================================
        # 1. LOAD PATIENT
        # ====================================================

        patient_id = 1

        patient_profile = (
            get_patient_profile(
                db,
                patient_id,
            )
        )

        if patient_profile is None:

            raise RuntimeError(
                f"Patient profile with ID="
                f"{patient_id} was not found."
            )

        print(
            f"✓ Test patient ID: "
            f"{patient_id}"
        )

        # ====================================================
        # 2. LOAD ACTIVE MODEL
        # ====================================================

        active_model = (
            get_active_model(db)
        )

        if active_model is None:

            raise RuntimeError(
                "No active model version found."
            )

        print(
            f"✓ Active model loaded: "
            f"{active_model.version}"
        )

        # ====================================================
        # 3. CONTROLLED HIGH-RISK PROFILE
        # ====================================================

        test_profile = {

            "Age": 61,

            "Gender": 1,

            "BMI": 32.0,

            "Smoking": 1,

            "GeneticRisk": 2,

            "PhysicalActivity": 1.0,

            "AlcoholIntake": 4.0,

            "CancerHistory": 1,
        }

        print(
            "\nControlled test profile:"
        )

        for key, value in (
            test_profile.items()
        ):

            print(
                f"  {key:<20}: "
                f"{value}"
            )

        # ====================================================
        # 4. RUN PREDICTION
        # ====================================================

        result = predict_risk(

            age=test_profile["Age"],

            gender=test_profile["Gender"],

            bmi=test_profile["BMI"],

            smoking=test_profile["Smoking"],

            genetic_risk=(
                test_profile["GeneticRisk"]
            ),

            physical_activity=(
                test_profile[
                    "PhysicalActivity"
                ]
            ),

            alcohol_intake=(
                test_profile[
                    "AlcoholIntake"
                ]
            ),

            cancer_history=(
                test_profile[
                    "CancerHistory"
                ]
            ),
        )

        # predict_risk() returns a dictionary.

        probability = float(
            result["probability"]
        )

        prediction = int(
            result["prediction"]
        )

        risk_band = result[
            "risk_band"
        ]

        threshold = float(
            result["threshold"]
        )

        # ====================================================
        # 5. CREATE RISK ASSESSMENT
        # ====================================================

        assessment = (
            create_risk_assessment(

                db=db,

                patient_id=patient_id,

                model_version_id=(
                    active_model.id
                ),

                risk_probability=(
                    probability
                ),

                decision_threshold=(
                    threshold
                ),

                prediction=(
                    prediction
                ),

                risk_band=(
                    risk_band
                ),

                input_snapshot=(
                    test_profile
                ),
            )
        )

        test_assessment_id = (
            assessment.id
        )

        print(
            f"\n✓ Test risk assessment "
            f"created: ID="
            f"{test_assessment_id}"
        )

        print(
            f"  Risk probability: "
            f"{probability:.4f}"
        )

        print(
            f"  Risk percentage : "
            f"{probability * 100:.2f}%"
        )

        print(
            f"  Threshold       : "
            f"{threshold:.4f}"
        )

        print(
            f"  Prediction      : "
            f"{prediction}"
        )

        print(
            f"  Risk band       : "
            f"{risk_band}"
        )

        # ====================================================
        # 6. ENGINEER FEATURES
        # ====================================================

        patient_features = (
            build_engineered_features(
                test_profile
            )
        )

        print(
            "\n✓ Engineered 12-feature "
            "DataFrame created"
        )

        print(
            f"  Feature count: "
            f"{len(patient_features.columns)}"
        )

        if (
            list(
                patient_features.columns
            )
            != FEATURE_NAMES
        ):

            raise AssertionError(
                "Feature names/order do not "
                "match the expected 12 features."
            )

        print(
            "✓ All 12 features verified"
        )

        # ====================================================
        # 7. SHAP
        # ====================================================

        print(
            "\nGenerating SHAP explanation..."
        )

        explanation = (
            explain_and_store(

                db=db,

                assessment_id=(
                    test_assessment_id
                ),

                patient_features=(
                    patient_features
                ),

                patient_id=(
                    patient_id
                ),

                top_n=10,
            )
        )

        print(
            "✓ SHAP explanation generated"
        )

        print(
            f"  Returned features: "
            f"{len(explanation)}"
        )

        explanations = (
            get_assessment_explanations(
                db,
                test_assessment_id,
            )
        )

        print(
            f"✓ Database contains "
            f"{len(explanations)} "
            f"SHAP explanation record(s)"
        )

        if not explanations:

            raise AssertionError(
                "No SHAP explanations were "
                "stored in the database."
            )

        print(
            "\nStored SHAP values:"
        )

        for item in explanations:

            print(
                f"  {item.feature:<20} "
                f"value={item.feature_value} "
                f"SHAP="
                f"{item.shap_value:+.6f}"
            )

        # ====================================================
        # 8. RECOMMENDATIONS
        # ====================================================

        print(
            "\nGenerating recommendations..."
        )

        # IMPORTANT:
        # generate_recommendations() only takes
        # db, patient_id and assessment_id.
        #
        # It reads the stored assessment snapshot
        # and SHAP explanations itself.

        recommendations = (
            generate_recommendations(

                db=db,

                patient_id=patient_id,

                assessment_id=(
                    test_assessment_id
                ),
            )
        )

        print(
            f"\n✓ Recommendation engine "
            f"generated "
            f"{len(recommendations)} "
            f"recommendation(s)"
        )

        if not recommendations:

            raise AssertionError(
                "Expected recommendations for "
                "the controlled high-risk profile, "
                "but none were generated."
            )

        test_recommendation_ids = [

            recommendation.id

            for recommendation
            in recommendations
        ]

        print(
            "\nGenerated recommendations:"
        )

        for recommendation in (
            recommendations
        ):

            print(
                f"  ID "
                f"{recommendation.id} "
                f"{recommendation.feature} "
                f"{recommendation.status}"
            )

            print(
                f"     "
                f"{recommendation.text}"
            )

            print(
                f"     Reason: "
                f"{recommendation.reason}"
            )

        # ====================================================
        # 9. INITIAL STATUS
        # ====================================================

        for recommendation in (
            recommendations
        ):

            if (
                recommendation.status
                != "Not Started"
            ):

                raise AssertionError(
                    f"Recommendation "
                    f"{recommendation.id} "
                    f"has unexpected initial "
                    f"status: "
                    f"{recommendation.status}"
                )

        print(
            "\n✓ Initial status verified: "
            "Not Started"
        )

        # ====================================================
        # 10. IN PROGRESS
        # ====================================================

        first_recommendation_id = (
            test_recommendation_ids[0]
        )

        updated = (
            update_recommendation_status(

                db=db,

                recommendation_id=(
                    first_recommendation_id
                ),

                status="In Progress",
            )
        )

        if (
            updated.status
            != "In Progress"
        ):

            raise AssertionError(
                "Recommendation status did "
                "not update to In Progress."
            )

        print(
            "✓ Status update verified: "
            "In Progress"
        )

        # ====================================================
        # 11. COMPLETED
        # ====================================================

        updated = (
            update_recommendation_status(

                db=db,

                recommendation_id=(
                    first_recommendation_id
                ),

                status="Completed",
            )
        )

        if (
            updated.status
            != "Completed"
        ):

            raise AssertionError(
                "Recommendation status did "
                "not update to Completed."
            )

        print(
            "✓ Status update verified: "
            "Completed"
        )

        # ====================================================
        # 12. VERIFY DATABASE
        # ====================================================

        stored_recommendations = (
            get_patient_recommendations(
                db,
                patient_id,
            )
        )

        stored_ids = {

            recommendation.id

            for recommendation
            in stored_recommendations
        }

        for recommendation_id in (
            test_recommendation_ids
        ):

            if (
                recommendation_id
                not in stored_ids
            ):

                raise AssertionError(
                    f"Recommendation "
                    f"{recommendation_id} "
                    f"was not found in database."
                )

        print(
            "\n✓ Recommendation integration "
            "workflow completed"
        )

        print(
            "=" * 70
        )

        print(
            "RECOMMENDATION INTEGRATION "
            "TEST PASSED"
        )

        print(
            "=" * 70
        )

    finally:

        # ====================================================
        # CLEANUP
        # ====================================================

        try:

            if test_recommendation_ids:

                from database.models import (
                    Recommendation
                )

                db.query(
                    Recommendation
                ).filter(
                    Recommendation.id.in_(
                        test_recommendation_ids
                    )
                ).delete(
                    synchronize_session=False
                )

            if test_assessment_id is not None:

                from database.models import (
                    RiskExplanation,
                    RiskAssessment,
                )

                db.query(
                    RiskExplanation
                ).filter(
                    RiskExplanation.assessment_id
                    == test_assessment_id
                ).delete(
                    synchronize_session=False
                )

                db.query(
                    RiskAssessment
                ).filter(
                    RiskAssessment.id
                    == test_assessment_id
                ).delete(
                    synchronize_session=False
                )

            db.commit()

            print(
                "\n✓ Test data cleaned "
                "from database"
            )

        except Exception as cleanup_error:

            db.rollback()

            print(
                f"\n⚠ Cleanup warning: "
                f"{cleanup_error}"
            )

        db.close()


if __name__ == "__main__":
    main()