"""
OncoGuard AI - SHAP Explainability Engine

Provides:
1. Global SHAP feature importance
2. Offline patient explanations
3. Application-time explanations for a new patient
4. Database storage of RiskExplanation records

IMPORTANT:
SHAP explanations describe how the underlying CatBoost model
contributes to its prediction. They do not establish medical
causation.
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from database.crud import create_risk_explanation


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "catboost_calibrated.pkl"
)

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "train.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "shap"
)

PLOT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "plots"
)

TARGET = "Diagnosis"


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# DATA LOADING
# ============================================================

def load_data():

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(
            f"Training data not found:\n{TRAIN_PATH}"
        )

    train_df = pd.read_csv(TRAIN_PATH)

    X_train = train_df.drop(
        columns=[TARGET]
    )

    return X_train


# ============================================================
# SHAP VALUES
# ============================================================

def get_shap_values(model, X):

    print("\nCalculating SHAP values...")

    # --------------------------------------------------------
    # CalibratedClassifierCV
    # --------------------------------------------------------

    if hasattr(model, "calibrated_classifiers_"):

        all_values = []

        for calibrated_classifier in (
            model.calibrated_classifiers_
        ):

            estimator = calibrated_classifier.estimator

            explainer = shap.TreeExplainer(
                estimator
            )

            values = explainer.shap_values(X)

            if isinstance(values, list):
                values = values[1]

            values = np.asarray(values)

            # Handle newer SHAP output format:
            # (rows, features, classes)
            if values.ndim == 3:
                values = values[:, :, 1]

            all_values.append(values)

        # Average explanations across the
        # calibrated CatBoost ensemble.
        shap_values = np.mean(
            all_values,
            axis=0
        )

        return shap_values

    # --------------------------------------------------------
    # Normal tree model fallback
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(
        model
    )

    values = explainer.shap_values(X)

    if isinstance(values, list):
        values = values[1]

    values = np.asarray(values)

    if values.ndim == 3:
        values = values[:, :, 1]

    return values


# ============================================================
# FEATURE VALIDATION
# ============================================================

def validate_features(X):

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

    missing = [
        feature
        for feature in expected_features
        if feature not in X.columns
    ]

    if missing:
        raise ValueError(
            f"Missing SHAP features: {missing}"
        )

    return X[expected_features]


# ============================================================
# GLOBAL EXPLANATION
# ============================================================

def global_explanation(
    shap_values,
    X
):

    print(
        "\nGenerating global feature importance..."
    )

    mean_abs_shap = np.mean(
        np.abs(shap_values),
        axis=0
    )

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Mean_Absolute_SHAP": mean_abs_shap
    })

    importance = importance.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    importance.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "global_feature_importance.csv"
        ),
        index=False
    )

    print(
        "\nGlobal Feature Importance:"
    )

    print(
        importance.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Summary plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 7)
    )

    shap.summary_plot(
        shap_values,
        X,
        show=False
    )

    plt.tight_layout()

    os.makedirs(
        PLOT_DIR,
        exist_ok=True
    )

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "shap_summary.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Bar plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 7)
    )

    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "shap_feature_importance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return importance


# ============================================================
# APPLICATION-TIME PATIENT EXPLANATION
# ============================================================

def explain_prediction(
    model,
    patient_features: pd.DataFrame,
    top_n: int = 10,
):
    """
    Generate a local SHAP explanation for a new patient.

    patient_features must contain exactly the model's
    12 feature columns.

    Returns a DataFrame containing:
        Feature
        Feature_Value
        SHAP_Value
        Absolute_SHAP
        Direction
        Rank
    """

    patient_features = validate_features(
        patient_features
    )

    if len(patient_features) != 1:
        raise ValueError(
            "explain_prediction expects exactly "
            "one patient row."
        )

    print(
        "\nGenerating SHAP explanation "
        "for current patient..."
    )

    shap_values = get_shap_values(
        model,
        patient_features
    )

    values = shap_values[0]

    explanation = pd.DataFrame({
        "Feature": patient_features.columns,
        "Feature_Value": patient_features.iloc[0].values,
        "SHAP_Value": values,
    })

    explanation["Absolute_SHAP"] = (
        explanation["SHAP_Value"].abs()
    )

    explanation["Direction"] = np.where(
        explanation["SHAP_Value"] >= 0,
        "Increases predicted risk",
        "Decreases predicted risk",
    )

    explanation = explanation.sort_values(
        "Absolute_SHAP",
        ascending=False
    ).reset_index(drop=True)

    explanation["Rank"] = (
        explanation.index + 1
    )

    return explanation.head(top_n)


# ============================================================
# SAVE PATIENT EXPLANATION
# ============================================================

def save_patient_explanation(
    explanation,
    patient_id=None,
    assessment_id=None,
):
    """
    Save an application-time explanation to CSV.

    This is useful for debugging/research output.
    Database persistence is handled separately.
    """

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    if patient_id is not None:
        filename = (
            f"patient_{patient_id}_"
            f"assessment_{assessment_id}_"
            f"explanation.csv"
        )
    else:
        filename = "patient_explanation.csv"

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    explanation.to_csv(
        path,
        index=False
    )

    return path


# ============================================================
# SAVE EXPLANATION TO DATABASE
# ============================================================

def store_explanation_in_database(
    db,
    assessment_id: int,
    explanation: pd.DataFrame,
):
    """
    Store SHAP explanation rows in the RiskExplanation table.
    """

    created_records = []

    for _, row in explanation.iterrows():

        record = create_risk_explanation(
            db=db,
            assessment_id=assessment_id,
            feature=str(row["Feature"]),
            feature_value=row["Feature_Value"],
            shap_value=float(row["SHAP_Value"]),
            direction=str(row["Direction"]),
            rank=int(row["Rank"]),
        )

        created_records.append(record)

    return created_records


# ============================================================
# COMPLETE DATABASE EXPLANATION PIPELINE
# ============================================================

def explain_and_store(
    db,
    assessment_id: int,
    patient_features: pd.DataFrame,
    patient_id=None,
    top_n: int = 10,
):
    """
    Generate SHAP explanation and save it to the database.
    """

    model = load_model()

    explanation = explain_prediction(
        model=model,
        patient_features=patient_features,
        top_n=top_n,
    )

    records = store_explanation_in_database(
        db=db,
        assessment_id=assessment_id,
        explanation=explanation,
    )

    csv_path = save_patient_explanation(
        explanation=explanation,
        patient_id=patient_id,
        assessment_id=assessment_id,
    )

    return {
        "explanation": explanation,
        "records": records,
        "csv_path": csv_path,
    }


# ============================================================
# OFFLINE PATIENT EXPLANATION
# ============================================================

def explain_patient(
    model,
    X,
    patient_index=0
):

    X = validate_features(X)

    print(
        f"\nGenerating explanation for patient "
        f"index {patient_index}..."
    )

    patient = X.iloc[
        [patient_index]
    ]

    explanation = explain_prediction(
        model=model,
        patient_features=patient,
        top_n=len(X.columns),
    )

    print(
        "\nPatient Explanation:"
    )

    print(
        explanation[
            [
                "Feature",
                "Feature_Value",
                "SHAP_Value",
                "Direction",
            ]
        ].to_string(
            index=False
        )
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    explanation.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"patient_{patient_index}_explanation.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Local SHAP chart
    # --------------------------------------------------------

    top = explanation.head(10)

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        top["Feature"][::-1],
        top["SHAP_Value"][::-1]
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel(
        "SHAP Contribution"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"Patient {patient_index} - SHAP Explanation"
    )

    plt.tight_layout()

    os.makedirs(
        PLOT_DIR,
        exist_ok=True
    )

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            f"patient_{patient_index}_shap.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return explanation


# ============================================================
# MAIN RESEARCH / OFFLINE TEST
# ============================================================

def main():

    print("=" * 70)
    print(
        "ONCOGUARD AI - SHAP EXPLAINABILITY ENGINE"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    print(
        "\n✓ Calibrated CatBoost model loaded."
    )

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    X_train = load_data()

    X_train = validate_features(
        X_train
    )

    print(
        f"✓ Training dataset loaded: "
        f"{X_train.shape[0]} rows x "
        f"{X_train.shape[1]} features"
    )

    # --------------------------------------------------------
    # SHAP values
    # --------------------------------------------------------

    shap_values = get_shap_values(
        model,
        X_train
    )

    print(
        f"✓ SHAP matrix shape: "
        f"{shap_values.shape}"
    )

    # --------------------------------------------------------
    # Global explanation
    # --------------------------------------------------------

    global_explanation(
        shap_values,
        X_train
    )

    # --------------------------------------------------------
    # Local explanation
    # --------------------------------------------------------

    explain_patient(
        model,
        X_train,
        patient_index=0
    )

    print("\n" + "=" * 70)
    print("SHAP EXPLAINABILITY TEST PASSED")
    print("=" * 70)

    print("\nGenerated:")
    print(
        "✓ outputs/shap/global_feature_importance.csv"
    )
    print(
        "✓ outputs/shap/patient_0_explanation.csv"
    )
    print(
        "✓ outputs/plots/shap_summary.png"
    )
    print(
        "✓ outputs/plots/shap_feature_importance.png"
    )
    print(
        "✓ outputs/plots/patient_0_shap.png"
    )


if __name__ == "__main__":
    main()