from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


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
# LOAD DATA
# ============================================================

def load_data():

    path = (
        PROCESSED_DIR / "train.csv"
    )

    df = pd.read_csv(path)

    X = df.drop(
        columns=[TARGET]
    )

    y = df[TARGET]

    return X, y


# ============================================================
# CATBOOST PARAMETERS
# ============================================================

def get_base_parameters():

    return {

        "iterations": 400,

        "depth": 6,

        "learning_rate": 0.05,

        "loss_function": "Logloss",

        "random_seed": RANDOM_STATE,

        "verbose": False,

        "thread_count": -1
    }


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
# CREATE MODEL
# ============================================================

def create_model(
    strategy,
    y_train
):

    params = get_base_parameters()

    if strategy == "Class Weighted":

        negative = (
            y_train == 0
        ).sum()

        positive = (
            y_train == 1
        ).sum()

        weight = (
            negative / positive
        )

        params["class_weights"] = [
            1.0,
            weight
        ]

    return CatBoostClassifier(
        **params
    )


# ============================================================
# CROSS-VALIDATION
# ============================================================

def evaluate_strategy(
    strategy,
    X,
    y
):

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fold_results = []

    for fold, (
        train_idx,
        valid_idx
    ) in enumerate(
        cv.split(X, y),
        start=1
    ):

        X_train = X.iloc[
            train_idx
        ].copy()

        X_valid = X.iloc[
            valid_idx
        ].copy()

        y_train = y.iloc[
            train_idx
        ].copy()

        y_valid = y.iloc[
            valid_idx
        ].copy()

        # ----------------------------------------------------
        # SMOTE ONLY ON TRAINING FOLD
        # ----------------------------------------------------

        if strategy == "SMOTE":

            smote = SMOTE(
                random_state=RANDOM_STATE
            )

            X_train, y_train = (
                smote.fit_resample(
                    X_train,
                    y_train
                )
            )

        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        model = create_model(
            strategy,
            y_train
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # Validation prediction
        # ----------------------------------------------------

        probabilities = (
            model.predict_proba(
                X_valid
            )[:, 1]
        )

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        metrics = calculate_metrics(
            y_valid,
            predictions,
            probabilities
        )

        metrics["Fold"] = fold

        fold_results.append(
            metrics
        )

    return pd.DataFrame(
        fold_results
    )


# ============================================================
# MAIN
# ============================================================

def run_analysis():

    print("\n" + "=" * 80)

    print(
        "ONCOGUARD AI — IMBALANCE STRATEGY ANALYSIS"
    )

    print("=" * 80)

    X, y = load_data()

    print(
        f"\nTraining samples: {len(X)}"
    )

    print(
        "\nOriginal class distribution:"
    )

    print(
        y.value_counts()
    )

    print(
        y.value_counts(
            normalize=True
        ).mul(100).round(2)
    )

    strategies = [
        "Baseline",
        "Class Weighted",
        "SMOTE"
    ]

    all_results = []

    # --------------------------------------------------------
    # Evaluate each strategy
    # --------------------------------------------------------

    for strategy in strategies:

        print("\n" + "-" * 80)

        print(
            f"Evaluating: {strategy}"
        )

        results = evaluate_strategy(
            strategy,
            X,
            y
        )

        results["Strategy"] = strategy

        all_results.append(
            results
        )

        print("\nFold Results:")

        print(
            results[
                [
                    "Fold",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "Specificity",
                    "F1",
                    "ROC_AUC",
                    "PR_AUC"
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------

    combined = pd.concat(
        all_results,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1",
        "ROC_AUC",
        "PR_AUC"
    ]

    summary_rows = []

    for strategy in strategies:

        subset = combined[
            combined["Strategy"]
            == strategy
        ]

        row = {
            "Strategy": strategy
        }

        for metric in metrics:

            row[
                f"{metric}_Mean"
            ] = subset[metric].mean()

            row[
                f"{metric}_Std"
            ] = subset[metric].std()

        summary_rows.append(
            row
        )

    summary = pd.DataFrame(
        summary_rows
    )

    # Sort by ROC-AUC
    summary = summary.sort_values(
        by="ROC_AUC_Mean",
        ascending=False
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)

    print(
        "IMBALANCE STRATEGY SUMMARY"
    )

    print("=" * 80)

    display_columns = [
        "Strategy",
        "Recall_Mean",
        "Specificity_Mean",
        "F1_Mean",
        "ROC_AUC_Mean",
        "PR_AUC_Mean"
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    fold_path = (
        OUTPUT_DIR
        / "imbalance_fold_results.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "imbalance_strategy_comparison.csv"
    )

    combined.to_csv(
        fold_path,
        index=False
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print("\n" + "=" * 80)

    print(
        f"Fold results saved:\n{fold_path}"
    )

    print(
        f"\nSummary saved:\n{summary_path}"
    )

    print("=" * 80)

    print(
        "\n✅ IMBALANCE ANALYSIS COMPLETE"
    )

    print("=" * 80)


if __name__ == "__main__":

    run_analysis()