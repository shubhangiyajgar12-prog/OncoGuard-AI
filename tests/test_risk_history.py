"""
OncoGuard AI - Risk History Integration Test

Tests the complete longitudinal risk-history workflow:

Risk Assessments
        ↓
Risk History Service
        ↓
Chronological History
        ↓
Risk Percentage
        ↓
Risk Changes
        ↓
Current Risk
        ↓
Trend
        ↓
Summary

Important:
This test validates model-predicted risk history only.
It does NOT represent cancer progression, cancer stage,
or clinical diagnosis.
"""

from database.connection import SessionLocal
from database.crud import (
    get_active_model,
    create_risk_assessment,
)
from database.models import RiskAssessment

from services.risk_history_service import (
    get_risk_history,
    get_current_risk,
    get_risk_trend,
    get_risk_history_summary,
)


# ============================================================
# TEST CONFIGURATION
# ============================================================

TEST_PATIENT_ID = 1
TEST_THRESHOLD = 0.3900


# ============================================================
# ASSERTION HELPER
# ============================================================

def check(condition, message):
    """Raise an error if a test condition fails."""

    if not condition:
        raise AssertionError(
            f"✗ {message}"
        )

    print(
        f"✓ {message}"
    )


# ============================================================
# CLEANUP HELPER
# ============================================================

def cleanup_assessments(
    db,
    assessment_ids,
):
    """Remove assessments created by this test."""

    if not assessment_ids:
        return

    for assessment_id in assessment_ids:

        assessment = (
            db.query(RiskAssessment)
            .filter(
                RiskAssessment.id
                == assessment_id
            )
            .first()
        )

        if assessment is not None:
            db.delete(assessment)

    db.commit()


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 70)
    print("ONCOGUARD AI - RISK HISTORY INTEGRATION TEST")
    print("=" * 70)

    db = SessionLocal()

    created_assessment_ids = []

    try:

        # ----------------------------------------------------
        # Get active model
        # ----------------------------------------------------

        model = get_active_model(db)

        check(
            model is not None,
            "Active model found",
        )

        print(
            f"  Model version: {model.version}"
        )

        # ----------------------------------------------------
        # Test patient
        # ----------------------------------------------------

        patient_id = TEST_PATIENT_ID

        print(
            f"\n✓ Test patient ID: {patient_id}"
        )

        # ----------------------------------------------------
        # Get history before test
        # ----------------------------------------------------

        history_before = get_risk_history(
            db=db,
            patient_id=patient_id,
        )

        print(
            f"  Existing assessments: "
            f"{len(history_before)}"
        )

        # ----------------------------------------------------
        # Create controlled assessments
        # ----------------------------------------------------

        print(
            "\nCreating controlled test assessments..."
        )

        test_predictions = [
            {
                "probability": 0.05,
                "prediction": 0,
                "risk_band": "Low",
            },
            {
                "probability": 0.15,
                "prediction": 0,
                "risk_band": "Medium",
            },
            {
                "probability": 0.45,
                "prediction": 1,
                "risk_band": "High",
            },
        ]

        for index, item in enumerate(
            test_predictions,
            start=1,
        ):

            assessment = create_risk_assessment(
                db=db,
                patient_id=patient_id,
                model_version_id=model.id,
                risk_probability=item["probability"],
                decision_threshold=TEST_THRESHOLD,
                prediction=item["prediction"],
                risk_band=item["risk_band"],
                input_snapshot={
                    "test_assessment": True,
                    "sequence": index,
                    "test_name": (
                        "risk_history_integration_test"
                    ),
                },
            )

            created_assessment_ids.append(
                assessment.id
            )

            print(
                f"  ✓ Test assessment {index} "
                f"created: ID={assessment.id}"
            )

        # ----------------------------------------------------
        # Retrieve complete history
        # ----------------------------------------------------

        history = get_risk_history(
            db=db,
            patient_id=patient_id,
        )

        check(
            len(history)
            == len(history_before) + 3,
            "Three new assessments appear in risk history",
        )

        # ----------------------------------------------------
        # Verify history structure
        # ----------------------------------------------------

        required_fields = {
            "assessment_id",
            "timestamp",
            "timestamp_display",
            "risk_probability",
            "risk_percentage",
            "threshold",
            "prediction",
            "risk_band",
            "model_version",
            "previous_risk_percentage",
            "change_percentage_points",
        }

        for record in history:

            check(
                required_fields.issubset(
                    record.keys()
                ),
                "History record contains all required fields",
            )

            break

        # ----------------------------------------------------
        # Verify chronological ordering
        # ----------------------------------------------------

        timestamps = [
            record["timestamp"]
            for record in history
            if record["timestamp"] is not None
        ]

        check(
            timestamps == sorted(timestamps),
            "Risk history is chronologically ordered",
        )

        # ----------------------------------------------------
        # Get only our three test assessments
        #
        # Their IDs were generated during this test.
        # Sorting by their position in history avoids
        # assumptions about database IDs.
        # ----------------------------------------------------

        test_records = [
            record
            for record in history
            if record["assessment_id"]
            in created_assessment_ids
        ]

        test_records = sorted(
            test_records,
            key=lambda record: record["timestamp"],
        )

        check(
            len(test_records) == 3,
            "All three controlled assessments retrieved",
        )

        # ----------------------------------------------------
        # Verify risk percentages
        # ----------------------------------------------------

        expected_percentages = [
            5.0,
            15.0,
            45.0,
        ]

        actual_percentages = [
            round(
                record["risk_percentage"],
                2,
            )
            for record in test_records
        ]

        check(
            actual_percentages
            == expected_percentages,
            "Risk percentages are correctly converted",
        )

        # ----------------------------------------------------
        # Verify probability values
        # ----------------------------------------------------

        expected_probabilities = [
            0.05,
            0.15,
            0.45,
        ]

        actual_probabilities = [
            round(
                record["risk_probability"],
                2,
            )
            for record in test_records
        ]

        check(
            actual_probabilities
            == expected_probabilities,
            "Risk probabilities are preserved correctly",
        )

        # ----------------------------------------------------
        # Verify risk bands
        # ----------------------------------------------------

        actual_bands = [
            record["risk_band"]
            for record in test_records
        ]

        check(
            actual_bands
            == [
                "Low",
                "Medium",
                "High",
            ],
            "Risk bands are preserved correctly",
        )

        # ----------------------------------------------------
        # Verify predictions
        # ----------------------------------------------------

        actual_predictions = [
            record["prediction"]
            for record in test_records
        ]

        check(
            actual_predictions
            == [0, 0, 1],
            "Predictions are preserved correctly",
        )

        # ----------------------------------------------------
        # Verify threshold
        # ----------------------------------------------------

        actual_thresholds = [
            round(
                record["threshold"],
                4,
            )
            for record in test_records
        ]

        check(
            actual_thresholds
            == [
                TEST_THRESHOLD,
                TEST_THRESHOLD,
                TEST_THRESHOLD,
            ],
            "Decision thresholds are preserved correctly",
        )

        # ----------------------------------------------------
        # Verify model version
        # ----------------------------------------------------

        actual_model_versions = [
            record["model_version"]
            for record in test_records
        ]

        check(
            all(
                version == model.version
                for version in actual_model_versions
            ),
            "Model version is preserved correctly",
        )

        # ----------------------------------------------------
        # Verify risk changes
        # ----------------------------------------------------

        # First test assessment can have a previous
        # assessment from the patient's existing history.
        #
        # Therefore we only verify the changes between
        # controlled test assessments.

        second_change = test_records[1][
            "change_percentage_points"
        ]

        third_change = test_records[2][
            "change_percentage_points"
        ]

        check(
            round(second_change, 2) == 10.0,
            "Second assessment change is +10.00 percentage points",
        )

        check(
            round(third_change, 2) == 30.0,
            "Third assessment change is +30.00 percentage points",
        )

        # ----------------------------------------------------
        # Verify current/latest risk
        # ----------------------------------------------------

        current = get_current_risk(
            db=db,
            patient_id=patient_id,
        )

        check(
            current is not None,
            "Current risk assessment found",
        )

        # Because our final controlled assessment is
        # created last, it should be the current record.

        check(
            current["assessment_id"]
            == created_assessment_ids[-1],
            "Latest assessment is correctly identified",
        )

        check(
            round(
                current["risk_percentage"],
                2,
            )
            == 45.0,
            "Current risk is 45.00%",
        )

        check(
            current["risk_band"] == "High",
            "Current risk band is High",
        )

        check(
            current["prediction"] == 1,
            "Current prediction is 1",
        )

        # ----------------------------------------------------
        # Verify overall trend
        # ----------------------------------------------------

        trend = get_risk_trend(
            db=db,
            patient_id=patient_id,
        )

        check(
            trend == "increasing",
            "Overall risk trend is increasing",
        )

        # ----------------------------------------------------
        # Verify summary
        # ----------------------------------------------------

        summary = get_risk_history_summary(
            db=db,
            patient_id=patient_id,
        )

        check(
            summary["assessment_count"]
            == len(history),
            "Summary assessment count matches history",
        )

        check(
            round(
                summary["current_risk_percentage"],
                2,
            )
            == 45.0,
            "Summary current risk is 45.00%",
        )

        check(
            summary["current_risk_band"]
            == "High",
            "Summary current risk band is High",
        )

        check(
            summary["trend"]
            == "increasing",
            "Summary trend is increasing",
        )

        # ----------------------------------------------------
        # Display controlled history
        # ----------------------------------------------------

        print(
            "\nControlled Test History:"
        )

        for record in test_records:

            change = record[
                "change_percentage_points"
            ]

            if change is None:
                change_text = "N/A"
            else:
                change_text = (
                    f"{change:+.2f} pp"
                )

            print(
                f"  ID={record['assessment_id']} | "
                f"{record['risk_percentage']:.2f}% | "
                f"{record['risk_band']} | "
                f"Change={change_text}"
            )

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        cleanup_assessments(
            db=db,
            assessment_ids=created_assessment_ids,
        )

        print(
            "\n✓ Test data cleaned from database"
        )

        print("\n" + "=" * 70)
        print(
            "RISK HISTORY INTEGRATION TEST PASSED"
        )
        print("=" * 70)

    except Exception:

        # ----------------------------------------------------
        # Cleanup after failure
        # ----------------------------------------------------

        try:

            cleanup_assessments(
                db=db,
                assessment_ids=created_assessment_ids,
            )

            print(
                "\n✓ Test data cleaned after failure"
            )

        except Exception:

            db.rollback()

        raise

    finally:

        db.close()


if __name__ == "__main__":
    main()