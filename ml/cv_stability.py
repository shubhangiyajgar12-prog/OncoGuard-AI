from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from sklearn.model_selection import StratifiedKFold, cross_validate


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "reports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "Diagnosis"

RANDOM_STATE = 42

N_SPLITS = 5


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data():

    train_path = (
        PROCESSED_DIR / "train.csv"
    )

    df = pd.read_csv(train_path)

    X = df.drop(
        columns=[TARGET]
    )

    y = df[TARGET]

    return X, y


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    return CatBoostClassifier(

        iterations=400,

        depth=6,

        learning_rate=0.05,

        loss_function="Logloss",

        eval_metric="AUC",

        random_seed=RANDOM_STATE,

        verbose=False,

        thread_count=-1
    )


# ============================================================
# CROSS VALIDATION
# ============================================================

def run_cross_validation():

    X, y = load_training_data()

    model = create_model()

    cv = StratifiedKFold(

        n_splits=N_SPLITS,

        shuffle=True,

        random_state=RANDOM_STATE
    )

    scoring = {

        "accuracy": "accuracy",

        "precision": "precision",

        "recall": "recall",

        "f1": "f1",

        "roc_auc": "roc_auc",

        "pr_auc": "average_precision"
    }

    print("\n" + "=" * 80)

    print(
        "ONCOGUARD AI — CATBOOST "
        "CROSS-VALIDATION STABILITY"
    )

    print("=" * 80)

    print(
        f"\nDataset size: {len(X)}"
    )

    print(
        f"Number of folds: {N_SPLITS}"
    )

    results = cross_validate(

        model,

        X,

        y,

        cv=cv,

        scoring=scoring,

        n_jobs=1,

        return_train_score=False
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    rows = []

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc"
    ]

    for metric in metrics:

        values = results[
            f"test_{metric}"
        ]

        rows.append({

            "Metric": metric.upper(),

            "Mean": np.mean(values),

            "Std": np.std(values),

            "Min": np.min(values),

            "Max": np.max(values)
        })

    results_df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Print fold-level values
    # --------------------------------------------------------

    print("\nFOLD RESULTS")
    print("-" * 80)

    for metric in metrics:

        values = results[
            f"test_{metric}"
        ]

        print(
            f"\n{metric.upper()}"
        )

        for i, value in enumerate(
            values,
            start=1
        ):

            print(
                f"Fold {i}: {value:.4f}"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)

    print(
        "CROSS-VALIDATION SUMMARY"
    )

    print("=" * 80)

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / "catboost_cv_stability.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 80)

    print(
        f"Results saved to:\n{output_path}"
    )

    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_cross_validation()