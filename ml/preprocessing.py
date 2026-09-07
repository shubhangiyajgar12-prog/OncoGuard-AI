from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cancer_prediction_dataset.csv"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "Diagnosis"

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # BMI-related feature
    # --------------------------------------------------------

    df["Is_Obese"] = (
        df["BMI"] >= 30
    ).astype(int)

    # --------------------------------------------------------
    # Count major binary risk factors
    # --------------------------------------------------------

    df["Risk_Factor_Count"] = (
        df["Smoking"]
        + df["CancerHistory"]
        + (df["GeneticRisk"] > 0).astype(int)
        + (df["AlcoholIntake"] >= 3).astype(int)
        + (df["PhysicalActivity"] <= 3).astype(int)
        + (df["BMI"] >= 30).astype(int)
    )

    # --------------------------------------------------------
    # Age groups
    # --------------------------------------------------------

    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[0, 30, 45, 60, 75, 120],
        labels=False,
        include_lowest=True
    )

    # --------------------------------------------------------
    # BMI category
    # --------------------------------------------------------

    df["BMI_Category"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=False,
        include_lowest=True
    )

    return df


# ============================================================
# SPLIT FEATURES / TARGET
# ============================================================

def split_features_target(df):

    X = df.drop(
        columns=[TARGET]
    )

    y = df[TARGET]

    return X, y


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def create_splits(X, y):

    # First split:
    # 85% temporary training data
    # 15% final untouched test data

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Convert validation proportion to proportion
    # of the temporary training set.

    validation_ratio = (
        VALIDATION_SIZE
        / (1 - TEST_SIZE)
    )

    # Second split:
    # training / validation

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=validation_ratio,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


# ============================================================
# SAVE DATASETS
# ============================================================

def save_split(
    X,
    y,
    name
):

    output = X.copy()

    output[TARGET] = y.values

    path = (
        PROCESSED_DIR
        / f"{name}.csv"
    )

    output.to_csv(
        path,
        index=False
    )

    print(
        f"Saved: {path}"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_preprocessing():

    print("\n" + "=" * 70)
    print("ONCOGUARD AI — PREPROCESSING PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    print(
        f"\nOriginal dataset: {df.shape}"
    )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    df = create_features(df)

    print(
        f"After feature engineering: {df.shape}"
    )

    print("\nCreated features:")

    print(" - Is_Obese")
    print(" - Risk_Factor_Count")
    print(" - Age_Group")
    print(" - BMI_Category")

    # --------------------------------------------------------
    # Separate X and y
    # --------------------------------------------------------

    X, y = split_features_target(df)

    print(
        f"\nFeature matrix: {X.shape}"
    )

    print(
        f"Target vector : {y.shape}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = create_splits(X, y)

    print("\nDATA SPLIT")
    print("-" * 40)

    print(
        f"Training   : {len(X_train)} "
        f"({len(X_train) / len(df) * 100:.1f}%)"
    )

    print(
        f"Validation : {len(X_val)} "
        f"({len(X_val) / len(df) * 100:.1f}%)"
    )

    print(
        f"Test       : {len(X_test)} "
        f"({len(X_test) / len(df) * 100:.1f}%)"
    )

    # --------------------------------------------------------
    # Class distributions
    # --------------------------------------------------------

    print("\nCLASS DISTRIBUTION")

    print(
        "\nTraining:"
    )

    print(
        y_train.value_counts(
            normalize=True
        ).mul(100).round(2)
    )

    print(
        "\nValidation:"
    )

    print(
        y_val.value_counts(
            normalize=True
        ).mul(100).round(2)
    )

    print(
        "\nTest:"
    )

    print(
        y_test.value_counts(
            normalize=True
        ).mul(100).round(2)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_split(
        X_train,
        y_train,
        "train"
    )

    save_split(
        X_val,
        y_val,
        "validation"
    )

    save_split(
        X_test,
        y_test,
        "test"
    )

    print("\n" + "=" * 70)
    print("✅ PREPROCESSING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    run_preprocessing()