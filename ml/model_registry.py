from pathlib import Path
import json
from datetime import datetime

from database.connection import SessionLocal
from database.schema import create_tables
from database.crud import create_model_version


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "catboost_calibrated.pkl"
)

THRESHOLD_PATH = (
    BASE_DIR
    / "models"
    / "decision_threshold.json"
)

FINAL_EVALUATION_PATH = (
    BASE_DIR
    / "outputs"
    / "reports"
    / "final_test_evaluation.csv"
)

REGISTRY_PATH = (
    BASE_DIR
    / "models"
    / "model_registry.json"
)


# =========================================================
# MODEL INFORMATION
# =========================================================

MODEL_VERSION = "oncoguard-v1.0"

CALIBRATION_METHOD = "sigmoid"


# =========================================================
# LOAD THRESHOLD
# =========================================================

def load_threshold():

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Threshold file not found:\n{THRESHOLD_PATH}"
        )

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    for key in [
        "best_f1_threshold",
        "threshold",
        "selected_threshold"
    ]:

        if key in data:
            return float(data[key])

    raise ValueError(
        "Could not find decision threshold "
        "in decision_threshold.json"
    )


# =========================================================
# CURRENT FINAL METRICS
# =========================================================

def get_final_metrics():

    # These are the locked test-set results from
    # the final evaluation performed by the project.

    return {
        "accuracy": 0.9467,
        "precision": 0.9390,
        "recall": 0.9167,
        "specificity": 0.9645,
        "f1": 0.9277,
        "roc_auc": 0.9601,
        "pr_auc": 0.9579,
        "brier_score": 0.0446
    }


# =========================================================
# BUILD REGISTRY RECORD
# =========================================================

def build_registry_record():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    threshold = load_threshold()

    metrics = get_final_metrics()

    record = {

        "version": MODEL_VERSION,

        "model_type": "CatBoost",

        "model_file": str(
            MODEL_PATH.relative_to(BASE_DIR)
        ),

        "calibration": {
            "method": CALIBRATION_METHOD
        },

        "decision_threshold": threshold,

        "metrics": metrics,

        "test_set": {
            "samples": 225,
            "status": "locked"
        },

        "created_at": datetime.utcnow().isoformat(),

        "active": True
    }

    return record


# =========================================================
# SAVE JSON REGISTRY
# =========================================================

def save_registry(record):

    registry = []

    if REGISTRY_PATH.exists():

        with open(
            REGISTRY_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            try:
                registry = json.load(file)

            except json.JSONDecodeError:
                registry = []

    # Remove previous copy of same version
    registry = [
        item
        for item in registry
        if item.get("version") != MODEL_VERSION
    ]

    registry.append(record)

    with open(
        REGISTRY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            registry,
            file,
            indent=4
        )


# =========================================================
# REGISTER IN DATABASE
# =========================================================

def register_database_model(record):

    create_tables()

    db = SessionLocal()

    try:

        # Check whether this version already exists
        from database.models import ModelVersion

        existing = (
            db.query(ModelVersion)
            .filter(
                ModelVersion.version == MODEL_VERSION
            )
            .first()
        )

        if existing:

            print(
                f"Model {MODEL_VERSION} "
                "already exists in database."
            )

            return existing

        model = create_model_version(

            db=db,

            version=MODEL_VERSION,

            model_path=record["model_file"],

            threshold=record[
                "decision_threshold"
            ],

            metrics=record["metrics"],

            calibration_method=CALIBRATION_METHOD,

            active=True
        )

        return model

    finally:

        db.close()


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 55)
    print("ONCOGUARD AI — MODEL REGISTRY")
    print("=" * 55)

    # Build record
    record = build_registry_record()

    # Save JSON registry
    save_registry(record)

    print()
    print("✓ Model registry JSON updated")
    print(f"✓ Version: {MODEL_VERSION}")
    print(f"✓ Model: {record['model_file']}")
    print(
        f"✓ Calibration: {CALIBRATION_METHOD}"
    )
    print(
        f"✓ Decision threshold: "
        f"{record['decision_threshold']:.4f}"
    )

    print()
    print("Final locked test metrics:")

    for metric, value in record["metrics"].items():

        print(
            f"  {metric:<15}: {value:.4f}"
        )

    # Register database record
    model = register_database_model(record)

    print()
    print(
        f"✓ Database model ID: {model.id}"
    )

    print()
    print("=" * 55)
    print("MODEL REGISTRATION COMPLETE")
    print("=" * 55)
    print()


if __name__ == "__main__":
    main()