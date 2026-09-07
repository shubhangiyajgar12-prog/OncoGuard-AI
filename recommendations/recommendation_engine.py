"""
OncoGuard AI - Recommendation Engine

Generates safe, model-informed recommendations from:
1. Stored SHAP explanations
2. Feature-specific recommendation rules

Important:
- Recommendations are informational.
- They are NOT medical diagnoses.
- They are NOT treatment instructions.
- SHAP contribution is interpreted as model contribution,
  not causal effect.
"""

from database.crud import (
    create_recommendation,
    get_assessment_explanations,
    update_recommendation_status as crud_update_recommendation_status,
)


# ============================================================
# RECOMMENDATION RULES
# ============================================================

RECOMMENDATION_RULES = {

    "Smoking": {
        "condition": lambda value: value == 1,
        "text": (
            "Consider discussing smoking cessation support "
            "with a qualified healthcare professional."
        ),
        "reason": (
            "Smoking is identified by the model as an important "
            "contributor to the current risk assessment."
        ),
    },

    "AlcoholIntake": {
        "condition": lambda value: value >= 3,
        "text": (
            "Consider discussing alcohol-use patterns and "
            "appropriate guidance with a qualified healthcare professional."
        ),
        "reason": (
            "The model identifies alcohol intake as an important "
            "contributor to the current risk assessment."
        ),
    },

    "BMI": {
        "condition": lambda value: value >= 25,
        "text": (
            "Consider discussing healthy weight-management strategies "
            "with a qualified healthcare professional."
        ),
        "reason": (
            "The model identifies BMI as an important contributor "
            "to the current risk assessment."
        ),
    },

    "PhysicalActivity": {
        "condition": lambda value: value < 3,
        "text": (
            "Consider discussing an appropriate level of physical "
            "activity with a qualified healthcare professional."
        ),
        "reason": (
            "The model identifies physical activity as an important "
            "contributor to the current risk assessment."
        ),
    },

    "CancerHistory": {
        "condition": lambda value: value == 1,
        "text": (
            "Consider discussing your personal and family cancer "
            "history with a qualified healthcare professional."
        ),
        "reason": (
            "Cancer history is identified by the model as an "
            "important contributor to the current risk assessment."
        ),
    },

    "GeneticRisk": {
        "condition": lambda value: value >= 2,
        "text": (
            "Consider discussing your genetic and family history "
            "with a qualified healthcare professional."
        ),
        "reason": (
            "The model identifies genetic risk as an important "
            "contributor to the current risk assessment."
        ),
    },
}


# ============================================================
# ALLOWED STATUSES
# ============================================================

ALLOWED_STATUSES = {
    "Not Started",
    "In Progress",
    "Completed",
}


# ============================================================
# VALUE CONVERSION
# ============================================================

def convert_feature_value(value):
    """
    Convert database feature values into numeric values.

    SHAP explanation values may be stored as strings in the
    database, while recommendation rules require numeric
    comparisons.
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        return float(value)

    try:
        return float(str(value).strip())

    except (ValueError, TypeError):
        return None


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    db,
    patient_id,
    assessment_id,
):
    """
    Generate safe recommendations for a risk assessment.

    A recommendation is generated only when ALL conditions
    below are satisfied:

    1. A recommendation rule exists for the feature.
    2. The feature value satisfies that rule.
    3. The SHAP value is positive.

    Positive SHAP means the feature contributes toward the
    model's predicted positive class for this assessment.

    This does NOT imply causation.
    """

    # --------------------------------------------------------
    # Get stored SHAP explanations
    # --------------------------------------------------------

    explanations = get_assessment_explanations(
        db=db,
        assessment_id=assessment_id,
    )

    if not explanations:
        return []

    recommendations = []

    # --------------------------------------------------------
    # Process each SHAP explanation
    # --------------------------------------------------------

    for explanation in explanations:

        feature = explanation.feature

        # ----------------------------------------------------
        # Check if feature has a recommendation rule
        # ----------------------------------------------------

        if feature not in RECOMMENDATION_RULES:
            continue

        # ----------------------------------------------------
        # Convert feature value
        # ----------------------------------------------------

        value = convert_feature_value(
            explanation.feature_value
        )

        if value is None:
            continue

        # ----------------------------------------------------
        # Convert SHAP value
        # ----------------------------------------------------

        try:
            shap_value = float(
                explanation.shap_value
            )

        except (ValueError, TypeError):
            continue

        # ----------------------------------------------------
        # Apply feature-specific condition
        # ----------------------------------------------------

        rule = RECOMMENDATION_RULES[feature]

        if not rule["condition"](value):
            continue

        # ----------------------------------------------------
        # Only recommend features that contribute positively
        # ----------------------------------------------------

        if shap_value <= 0:
            continue

        # ----------------------------------------------------
        # Create database recommendation
        #
        # IMPORTANT:
        # database.crud.create_recommendation() expects
        # "text", NOT "recommendation_text".
        # ----------------------------------------------------

        recommendation = create_recommendation(
            db=db,
            patient_id=patient_id,
            assessment_id=assessment_id,
            feature=feature,
            text=rule["text"],
            reason=rule["reason"],
            status="Not Started",
        )

        recommendations.append(
            recommendation
        )

    return recommendations


# ============================================================
# RECOMMENDATION SUMMARY
# ============================================================

def recommendation_summary(recommendations):
    """
    Convert recommendation database objects into
    UI-friendly dictionaries.
    """

    return [
        {
            "id": recommendation.id,
            "feature": recommendation.feature,
            "text": recommendation.text,
            "reason": recommendation.reason,
            "status": recommendation.status,
        }
        for recommendation in recommendations
    ]


# ============================================================
# UPDATE RECOMMENDATION STATUS
# ============================================================

def update_recommendation_status(
    db,
    recommendation_id,
    status,
):
    """
    Update recommendation tracking status.

    Allowed statuses:

        Not Started
        In Progress
        Completed
    """

    if status not in ALLOWED_STATUSES:
        raise ValueError(
            f"Invalid status '{status}'. "
            f"Allowed values: {sorted(ALLOWED_STATUSES)}"
        )

    # IMPORTANT:
    # database.crud already provides
    # update_recommendation_status().
    #
    # Do NOT call update_recommendation().
    return crud_update_recommendation_status(
        db=db,
        recommendation_id=recommendation_id,
        status=status,
    )


# ============================================================
# ENGINE TEST
# ============================================================

def main():

    from database.connection import SessionLocal

    print("=" * 70)
    print("ONCOGUARD AI - RECOMMENDATION ENGINE TEST")
    print("=" * 70)

    db = SessionLocal()

    try:

        patient_id = 1
        assessment_id = 2

        print(
            f"✓ Patient ID: {patient_id}"
        )

        print(
            f"✓ Assessment ID: {assessment_id}"
        )

        # ----------------------------------------------------
        # Generate recommendations
        # ----------------------------------------------------

        recommendations = generate_recommendations(
            db=db,
            patient_id=patient_id,
            assessment_id=assessment_id,
        )

        print(
            f"\n✓ Recommendation engine generated "
            f"{len(recommendations)} recommendation(s)"
        )

        # ----------------------------------------------------
        # Display recommendations
        # ----------------------------------------------------

        if recommendations:

            print("\nRecommendations:")

            for recommendation in recommendations:

                print(
                    f"\n  Feature : "
                    f"{recommendation.feature}"
                )

                print(
                    f"  Status  : "
                    f"{recommendation.status}"
                )

                print(
                    f"  Text    : "
                    f"{recommendation.text}"
                )

                print(
                    f"  Reason  : "
                    f"{recommendation.reason}"
                )

        else:

            print(
                "\nNo recommendations generated for "
                "this assessment based on the current "
                "rules and SHAP values."
            )

        # ----------------------------------------------------
        # Test status lifecycle
        # ----------------------------------------------------

        if recommendations:

            recommendation = recommendations[0]

            # Not Started
            print(
                f"\n✓ Initial status: "
                f"{recommendation.status}"
            )

            # In Progress
            updated = update_recommendation_status(
                db=db,
                recommendation_id=recommendation.id,
                status="In Progress",
            )

            if updated is None:
                raise RuntimeError(
                    "Recommendation could not be updated."
                )

            print(
                f"✓ Status updated: "
                f"{updated.status}"
            )

            # Completed
            updated = update_recommendation_status(
                db=db,
                recommendation_id=recommendation.id,
                status="Completed",
            )

            if updated is None:
                raise RuntimeError(
                    "Recommendation could not be updated."
                )

            print(
                f"✓ Status updated: "
                f"{updated.status}"
            )

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("RECOMMENDATION ENGINE TEST PASSED")
        print("=" * 70)

    finally:

        db.close()


if __name__ == "__main__":
    main()