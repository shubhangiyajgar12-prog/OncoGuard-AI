from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIG
# ============================================================

TARGET = "Diagnosis"

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    train = pd.read_csv(
        PROCESSED_DIR / "train.csv"
    )

    validation = pd.read_csv(
        PROCESSED_DIR / "validation.csv"
    )

    test = pd.read_csv(
        PROCESSED_DIR / "test.csv"
    )

    X_train = train.drop(
        columns=[TARGET]
    )

    y_train = train[TARGET]

    X_val = validation.drop(
        columns=[TARGET]
    )

    y_val = validation[TARGET]

    X_test = test.drop(
        columns=[TARGET]
    )

    y_test = test[TARGET]

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


# ============================================================
# MODELS
# ============================================================

def create_models():

    models = {}

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    models["Logistic Regression"] = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE
                )
            ),
        ]
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    models["Random Forest"] = (
        RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1
        )
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    models["XGBoost"] = (
        XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1
        )
    )

    # --------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------

    models["LightGBM"] = (
        LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            verbosity=-1,
            n_jobs=-1
        )
    )

    # --------------------------------------------------------
    # CatBoost
    # --------------------------------------------------------

    models["CatBoost"] = (
        CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=RANDOM_STATE,
            verbose=False,
            thread_count=-1
        )
    )

    return models


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    predictions,
    probabilities
):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
    }


# ============================================================
# TRAIN + EVALUATE
# ============================================================

def train_models():

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = load_data()

    models = create_models()

    results = []

    print("\n" + "=" * 80)
    print("ONCOGUARD AI — BASELINE MODEL TRAINING")
    print("=" * 80)

    print(
        f"\nTraining samples   : {len(X_train)}"
    )

    print(
        f"Validation samples : {len(X_val)}"
    )

    print(
        f"Test samples       : {len(X_test)}"
    )

    # --------------------------------------------------------
    # Train each model
    # --------------------------------------------------------

    for name, model in models.items():

        print("\n" + "-" * 80)

        print(
            f"Training: {name}"
        )

        # Fit
        model.fit(
            X_train,
            y_train
        )

        # Validation predictions
        val_predictions = (
            model.predict(X_val)
        )

        val_probabilities = (
            model.predict_proba(X_val)[:, 1]
        )

        val_metrics = calculate_metrics(
            y_val,
            val_predictions,
            val_probabilities
        )

        # Test predictions
        test_predictions = (
            model.predict(X_test)
        )

        test_probabilities = (
            model.predict_proba(X_test)[:, 1]
        )

        test_metrics = calculate_metrics(
            y_test,
            test_predictions,
            test_probabilities
        )

        # Print validation
        print("\nValidation Metrics:")

        for metric, value in val_metrics.items():

            print(
                f"{metric:<15}: "
                f"{value:.4f}"
            )

        # Print test
        print("\nTest Metrics:")

        for metric, value in test_metrics.items():

            print(
                f"{metric:<15}: "
                f"{value:.4f}"
            )

        # Store results
        results.append(
            {
                "Model": name,

                "Validation_Accuracy":
                    val_metrics["Accuracy"],

                "Validation_Precision":
                    val_metrics["Precision"],

                "Validation_Recall":
                    val_metrics["Recall"],

                "Validation_Specificity":
                    val_metrics["Specificity"],

                "Validation_F1":
                    val_metrics["F1"],

                "Validation_ROC_AUC":
                    val_metrics["ROC_AUC"],

                "Validation_PR_AUC":
                    val_metrics["PR_AUC"],

                "Test_Accuracy":
                    test_metrics["Accuracy"],

                "Test_Precision":
                    test_metrics["Precision"],

                "Test_Recall":
                    test_metrics["Recall"],

                "Test_Specificity":
                    test_metrics["Specificity"],

                "Test_F1":
                    test_metrics["F1"],

                "Test_ROC_AUC":
                    test_metrics["ROC_AUC"],

                "Test_PR_AUC":
                    test_metrics["PR_AUC"],
            }
        )

        # Save individual model
        filename = (
            name.lower()
            .replace(" ", "_")
            + ".pkl"
        )

        model_path = (
            MODEL_DIR / filename
        )

        joblib.dump(
            model,
            model_path
        )

        print(
            f"\nSaved model: {model_path}"
        )

    # --------------------------------------------------------
    # Save comparison
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by="Validation_ROC_AUC",
        ascending=False
    )

    comparison_path = (
        OUTPUT_DIR
        / "baseline_model_comparison.csv"
    )

    results_df.to_csv(
        comparison_path,
        index=False
    )

    print("\n" + "=" * 80)

    print(
        "MODEL COMPARISON"
    )

    print("=" * 80)

    print(
        results_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)

    print(
        f"Comparison saved to:\n"
        f"{comparison_path}"
    )

    print("=" * 80)


if __name__ == "__main__":

    train_models()