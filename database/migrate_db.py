from sqlalchemy import inspect, text

from database.connection import engine
from database.schema import create_tables


def add_column_if_missing(
    connection,
    table_name,
    column_name,
    column_definition,
):
    """
    Add a column only if it does not already exist.
    """

    inspector = inspect(connection)

    existing_columns = {
        column["name"]
        for column in inspector.get_columns(table_name)
    }

    if column_name in existing_columns:
        print(
            f"✓ {table_name}.{column_name} already exists"
        )
        return

    connection.execute(
        text(
            f"ALTER TABLE {table_name} "
            f"ADD COLUMN {column_name} {column_definition}"
        )
    )

    print(
        f"✓ Added {table_name}.{column_name}"
    )


def migrate_database():
    print()
    print("========================================")
    print("   ONCOGUARD AI DATABASE MIGRATION")
    print("========================================")
    print()

    # Make sure all current tables exist first.
    create_tables()

    with engine.begin() as connection:

        # ====================================================
        # RISK ASSESSMENTS
        # ====================================================

        add_column_if_missing(
            connection,
            "risk_assessments",
            "timestamp",
            "DATETIME",
        )

        add_column_if_missing(
            connection,
            "risk_assessments",
            "probability",
            "FLOAT",
        )

        add_column_if_missing(
            connection,
            "risk_assessments",
            "threshold",
            "FLOAT",
        )

        add_column_if_missing(
            connection,
            "risk_assessments",
            "prediction",
            "INTEGER",
        )

        add_column_if_missing(
            connection,
            "risk_assessments",
            "risk_band",
            "VARCHAR(100)",
        )

        add_column_if_missing(
            connection,
            "risk_assessments",
            "input_snapshot",
            "JSON",
        )

        # ====================================================
        # RISK EXPLANATIONS
        # ====================================================

        add_column_if_missing(
            connection,
            "risk_explanations",
            "feature_value",
            "VARCHAR(255)",
        )

        add_column_if_missing(
            connection,
            "risk_explanations",
            "shap_value",
            "FLOAT",
        )

        add_column_if_missing(
            connection,
            "risk_explanations",
            "direction",
            "VARCHAR(100)",
        )

        add_column_if_missing(
            connection,
            "risk_explanations",
            "rank",
            "INTEGER",
        )

        # ====================================================
        # RECOMMENDATIONS
        # ====================================================

        add_column_if_missing(
            connection,
            "recommendations",
            "assessment_id",
            "INTEGER",
        )

        add_column_if_missing(
            connection,
            "recommendations",
            "feature",
            "VARCHAR(100)",
        )

        add_column_if_missing(
            connection,
            "recommendations",
            "text",
            "TEXT",
        )

        add_column_if_missing(
            connection,
            "recommendations",
            "reason",
            "TEXT",
        )

        add_column_if_missing(
            connection,
            "recommendations",
            "status",
            "VARCHAR(50)",
        )

        add_column_if_missing(
            connection,
            "recommendations",
            "created_at",
            "DATETIME",
        )

        add_column_if_missing(
            connection,
            "recommendations",
            "updated_at",
            "DATETIME",
        )

        # ====================================================
        # DOCTOR NOTES
        # ====================================================

        add_column_if_missing(
            connection,
            "doctor_notes",
            "doctor_id",
            "INTEGER",
        )

        add_column_if_missing(
            connection,
            "doctor_notes",
            "note",
            "TEXT",
        )

        add_column_if_missing(
            connection,
            "doctor_notes",
            "created_at",
            "DATETIME",
        )

        # ====================================================
        # FOLLOWUPS
        # ====================================================

        add_column_if_missing(
            connection,
            "followups",
            "followup_date",
            "DATETIME",
        )

        add_column_if_missing(
            connection,
            "followups",
            "status",
            "VARCHAR(50)",
        )

        add_column_if_missing(
            connection,
            "followups",
            "notes",
            "TEXT",
        )

        add_column_if_missing(
            connection,
            "followups",
            "created_at",
            "DATETIME",
        )

        # ====================================================
        # ALERTS
        # ====================================================

        add_column_if_missing(
            connection,
            "alerts",
            "alert_type",
            "VARCHAR(100)",
        )

        add_column_if_missing(
            connection,
            "alerts",
            "message",
            "TEXT",
        )

        add_column_if_missing(
            connection,
            "alerts",
            "severity",
            "VARCHAR(50)",
        )

        add_column_if_missing(
            connection,
            "alerts",
            "acknowledged",
            "BOOLEAN",
        )

        add_column_if_missing(
            connection,
            "alerts",
            "created_at",
            "DATETIME",
        )

    print()
    print("========================================")
    print("   DATABASE MIGRATION COMPLETED")
    print("========================================")
    print()


if __name__ == "__main__":
    migrate_database()