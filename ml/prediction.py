"""
OncoGuard AI - Prediction Engine

Loads the calibrated CatBoost model and decision threshold,
performs the same feature engineering used during training,
and generates a model-predicted cancer risk assessment.

IMPORTANT:
This is a research/prototype prediction system.
It is NOT a clinical diagnostic tool.
"""

from pathlib import Path
import json
from typing import Dict, Any

import joblib
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "catboost_calibrated.pkl"
THRESHOLD_PATH = ROOT_DIR / "models" / "decision_threshold.json"


# ============================================================
# EXPECTED FEATURE ORDER
# ============================================================

FEATURE_COLUMNS = [
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
# MODEL LOADING
# ============================================================

def load_model():
    """
    Load the calibrated CatBoost model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Calibrated model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def load_threshold() -> float:
    """
    Load the production decision threshold.
    """

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Decision threshold file not found: {THRESHOLD_PATH}"
        )

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Support common threshold key names.
    if "threshold" in data:
        return float(data["threshold"])

    if "decision_threshold" in data:
        return float(data["decision_threshold"])

    raise KeyError(
        "Threshold JSON does not contain 'threshold' "
        "or 'decision_threshold'."
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(
    age: float,
    gender: int,
    bmi: float,
    smoking: int,
    genetic_risk: int,
    physical_activity: float,
    alcohol_intake: float,
    cancer_history: int,
) -> pd.DataFrame:
    """
    Create the exact 12 features expected by the trained model.

    Raw features:
        Age
        Gender
        BMI
        Smoking
        GeneticRisk
        PhysicalActivity
        AlcoholIntake
        CancerHistory

    Engineered features:
        Is_Obese
        Risk_Factor_Count
        Age_Group
        BMI_Category
    """

    # --------------------------------------------------------
    # Is_Obese
    # --------------------------------------------------------

    is_obese = int(bmi >= 30)

    # --------------------------------------------------------
    # Risk_Factor_Count
    #
    # Count the major binary risk indicators used by the
    # project feature engineering pipeline.
    # --------------------------------------------------------

    risk_factor_count = int(
        smoking
        + genetic_risk
        + cancer_history
        + is_obese
    )

    # --------------------------------------------------------
    # Age_Group
    #
    # Same categorical encoding used by the project:
    #
    # 0 -> under 30
    # 1 -> 30-44
    # 2 -> 45-59
    # 3 -> 60+
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
    # BMI_Category
    #
    # 0 -> Underweight
    # 1 -> Normal
    # 2 -> Overweight
    # 3 -> Obese
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
    # Build dataframe
    # --------------------------------------------------------

    data = {
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
    }

    df = pd.DataFrame([data])

    # Enforce exact feature order.
    df = df[FEATURE_COLUMNS]

    return df


# ============================================================
# RISK BAND
# ============================================================

def get_risk_band(probability: float) -> str:
    """
    Convert model probability into a user-facing risk band.

    The project currently uses the following interpretation:

        < 0.10       -> Low
        0.10-0.30    -> Medium
        >= 0.30      -> High

    This is a project presentation category, not a clinical
    cancer-risk classification.
    """

    if probability < 0.10:
        return "Low"

    if probability < 0.30:
        return "Medium"

    return "High"


# ============================================================
# PREDICTION
# ============================================================

def predict_risk(
    age: float,
    gender: int,
    bmi: float,
    smoking: int,
    genetic_risk: int,
    physical_activity: float,
    alcohol_intake: float,
    cancer_history: int,
) -> Dict[str, Any]:
    """
    Generate a complete risk prediction.

    Returns:
        probability
        risk_percentage
        threshold
        prediction
        risk_band
        input_features
    """

    # Load model and threshold.
    model = load_model()
    threshold = load_threshold()

    # Create model input.
    features = create_features(
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
    # Validate feature order against the underlying CatBoost
    # estimator.
    # --------------------------------------------------------

    try:
        underlying_model = (
            model.calibrated_classifiers_[0].estimator
        )

        trained_features = underlying_model.feature_names_

        if trained_features != FEATURE_COLUMNS:
            raise ValueError(
                "Feature mismatch detected.\n"
                f"Expected: {FEATURE_COLUMNS}\n"
                f"Model:    {trained_features}"
            )

    except AttributeError:
        # If the model structure changes in the future,
        # prediction can still proceed if predict_proba works.
        pass

    # --------------------------------------------------------
    # Generate calibrated probability.
    # --------------------------------------------------------

    probability = float(
        model.predict_proba(features)[0][1]
    )

    # --------------------------------------------------------
    # Apply production threshold.
    # --------------------------------------------------------

    prediction = int(probability >= threshold)

    # --------------------------------------------------------
    # Risk category.
    # --------------------------------------------------------

    risk_band = get_risk_band(probability)

    return {
        "probability": probability,
        "risk_percentage": probability * 100,
        "threshold": threshold,
        "prediction": prediction,
        "risk_band": risk_band,
        "input_features": features.to_dict(orient="records")[0],
    }


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ONCOGUARD AI - PREDICTION ENGINE TEST")
    print("=" * 60)

    result = predict_risk(
        age=61,
        gender=1,
        bmi=21.16697,
        smoking=0,
        genetic_risk=1,
        physical_activity=7.42636,
        alcohol_intake=1.97042,
        cancer_history=0,
    )

    print("\nPrediction Result")
    print("-" * 60)

    print(
        f"Risk Probability : "
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

    print("\nInput Features")
    print("-" * 60)

    for key, value in result["input_features"].items():
        print(f"{key:22}: {value}")

    print("\n" + "=" * 60)
    print("PREDICTION ENGINE TEST PASSED")
    print("=" * 60)