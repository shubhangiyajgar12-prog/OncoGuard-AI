import os
import json
import joblib
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "catboost_calibrated.pkl"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "models",
    "decision_threshold.json"
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
    "simulation"
)

TARGET = "Diagnosis"


def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}\n"
            "Run ml/calibration.py first."
        )

    return joblib.load(
        MODEL_PATH
    )


def load_threshold():

    if not os.path.exists(THRESHOLD_PATH):

        raise FileNotFoundError(
            f"Threshold configuration not found:\n"
            f"{THRESHOLD_PATH}"
        )

    with open(
        THRESHOLD_PATH,
        "r"
    ) as file:

        config = json.load(file)

    return float(
        config["decision_threshold"]
    )


def predict_risk(
    model,
    profile
):

    df = pd.DataFrame(
        [profile]
    )

    probability = model.predict_proba(
        df
    )[0, 1]

    return float(
        probability
    )


def risk_category(
    probability,
    threshold
):

    if probability >= threshold:

        return "Higher predicted risk"

    return "Lower predicted risk"


def run_simulation(
    model,
    threshold,
    original_profile,
    changes
):

    # ---------------------------------------------------------
    # Current profile
    # ---------------------------------------------------------

    current_profile = (
        original_profile.copy()
    )

    current_probability = predict_risk(
        model,
        current_profile
    )

    # ---------------------------------------------------------
    # Hypothetical profile
    # ---------------------------------------------------------

    hypothetical_profile = (
        original_profile.copy()
    )

    for feature, value in changes.items():

        if feature not in hypothetical_profile:

            raise ValueError(
                f"Unknown feature: {feature}"
            )

        hypothetical_profile[
            feature
        ] = value

    hypothetical_probability = predict_risk(
        model,
        hypothetical_profile
    )

    # ---------------------------------------------------------
    # Difference
    # ---------------------------------------------------------

    difference = (
        hypothetical_probability
        - current_probability
    )

    result = {

        "current_probability": (
            current_probability
        ),

        "hypothetical_probability": (
            hypothetical_probability
        ),

        "difference": difference,

        "current_percentage": (
            current_probability * 100
        ),

        "hypothetical_percentage": (
            hypothetical_probability * 100
        ),

        "difference_percentage_points": (
            difference * 100
        ),

        "current_category": risk_category(
            current_probability,
            threshold
        ),

        "hypothetical_category": risk_category(
            hypothetical_probability,
            threshold
        ),

        "changes": changes
    }

    return result


def print_result(result):

    print("\n" + "=" * 70)
    print("WHAT-IF SIMULATION RESULT")
    print("=" * 70)

    print(
        f"\nCurrent predicted probability:"
        f" {result['current_percentage']:.2f}%"
    )

    print(
        f"Current category:"
        f" {result['current_category']}"
    )

    print(
        f"\nHypothetical predicted probability:"
        f" {result['hypothetical_percentage']:.2f}%"
    )

    print(
        f"Hypothetical category:"
        f" {result['hypothetical_category']}"
    )

    print(
        f"\nChange in model prediction:"
        f" {result['difference_percentage_points']:+.2f}"
        f" percentage points"
    )

    print("\nHypothetical changes:")

    for feature, value in result[
        "changes"
    ].items():

        print(
            f"  {feature}: {value}"
        )

    print("\nIMPORTANT:")

    print(
        "This is a hypothetical model simulation."
    )

    print(
        "It does not establish causation and should "
        "not be interpreted as a guaranteed change "
        "in an individual's actual cancer risk."
    )


def main():

    print("=" * 70)
    print("ONCOGUARD AI — WHAT-IF SIMULATION ENGINE")
    print("=" * 70)

    model = load_model()

    threshold = load_threshold()

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    X_train = train_df.drop(
        columns=[TARGET]
    )

    # ---------------------------------------------------------
    # Select an example profile
    # ---------------------------------------------------------

    patient_index = 0

    original_profile = (
        X_train.iloc[
            patient_index
        ].to_dict()
    )

    print(
        f"\nUsing example profile "
        f"from training data: patient index "
        f"{patient_index}"
    )

    print("\nOriginal profile:")

    for feature, value in (
        original_profile.items()
    ):

        print(
            f"  {feature}: {value}"
        )

    # ---------------------------------------------------------
    # Example hypothetical scenario
    # ---------------------------------------------------------
    #
    # We intentionally modify only one feature.
    # This demonstrates the simulation engine.
    #
    # The exact interpretation of encoded values
    # will be handled in the application layer later.
    # ---------------------------------------------------------

    current_smoking = (
        original_profile["Smoking"]
    )

    if current_smoking == 1:

        hypothetical_smoking = 0

    else:

        hypothetical_smoking = 1

    changes = {

        "Smoking":
        hypothetical_smoking

    }

    # ---------------------------------------------------------
    # Run simulation
    # ---------------------------------------------------------

    result = run_simulation(
        model,
        threshold,
        original_profile,
        changes
    )

    print_result(
        result
    )

    # ---------------------------------------------------------
    # Save result
    # ---------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "example_what_if_result.json"
    )

    with open(
        output_path,
        "w"
    ) as file:

        json.dump(
            result,
            file,
            indent=4
        )

    print(
        f"\n✓ Saved: {output_path}"
    )

    print("\n" + "=" * 70)
    print("STEP 15 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()