from pathlib import Path
import json

import numpy as np
import pandas as pd
import optuna

from catboost import CatBoostClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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

MODEL_DIR = (
    PROJECT_ROOT / "models"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "reports"
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
# CONFIGURATION
# ============================================================

TARGET = "Diagnosis"

RANDOM_STATE = 42

N_SPLITS = 5

N_TRIALS = 30


# ============================================================
# LOAD DATA
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
# OPTUNA OBJECTIVE
# ============================================================

def objective(
    trial,
    X,
    y
):

    params = {

        "iterations": trial.suggest_int(
            "iterations",
            200,
            800
        ),

        "depth": trial.suggest_int(
            "depth",
            4,
            10
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.15,
            log=True
        ),

        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg",
            1.0,
            10.0,
            log=True
        ),

        "random_strength": trial.suggest_float(
            "random_strength",
            0.0,
            2.0
        ),

        "bagging_temperature": trial.suggest_float(
            "bagging_temperature",
            0.0,
            5.0
        ),

        "border_count": trial.suggest_int(
            "border_count",
            32,
            255
        ),

        "loss_function": "Logloss",

        "eval_metric": "AUC",

        "random_seed": RANDOM_STATE,

        "verbose": False,

        "thread_count": -1
    }

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fold_scores = []

    for train_index, valid_index in cv.split(
        X,
        y
    ):

        X_train = X.iloc[
            train_index
        ]

        X_valid = X.iloc[
            valid_index
        ]

        y_train = y.iloc[
            train_index
        ]

        y_valid = y.iloc[
            valid_index
        ]

        model = CatBoostClassifier(
            **params
        )

        model.fit(
            X_train,
            y_train
        )

        probabilities = (
            model.predict_proba(
                X_valid
            )[:, 1]
        )

        score = roc_auc_score(
            y_valid,
            probabilities
        )

        fold_scores.append(
            score
        )

        # Allow Optuna to stop poor trials early.
        trial.report(
            np.mean(fold_scores),
            step=len(fold_scores)
        )

        if trial.should_prune():

            raise optuna.TrialPruned()

    return np.mean(
        fold_scores
    )


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

        "PR_AUC": pr_auc
    }


# ============================================================
# TRAIN OPTIMIZED MODEL
# ============================================================

def train_optimized_model(
    X_train,
    y_train,
    X_validation,
    y_validation,
    best_params
):

    model = CatBoostClassifier(

        **best_params,

        loss_function="Logloss",

        eval_metric="AUC",

        random_seed=RANDOM_STATE,

        verbose=False,

        thread_count=-1
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_validation
    )

    probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    metrics = calculate_metrics(
        y_validation,
        predictions,
        probabilities
    )

    return model, metrics


# ============================================================
# MAIN
# ============================================================

def run_optimization():

    print("\n" + "=" * 80)

    print(
        "ONCOGUARD AI — CATBOOST OPTUNA OPTIMIZATION"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    X_train, y_train = (
        load_training_data()
    )

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Features: {X_train.shape[1]}"
    )

    print(
        f"CV folds: {N_SPLITS}"
    )

    print(
        f"Optuna trials: {N_TRIALS}"
    )

    # --------------------------------------------------------
    # Create study
    # --------------------------------------------------------

    study = optuna.create_study(

        direction="maximize",

        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_STATE
        ),

        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5
        )
    )

    print("\nStarting optimization...\n")

    study.optimize(

        lambda trial:
            objective(
                trial,
                X_train,
                y_train
            ),

        n_trials=N_TRIALS
    )

    # --------------------------------------------------------
    # Best result
    # --------------------------------------------------------

    print("\n" + "=" * 80)

    print("OPTIMIZATION COMPLETE")

    print("=" * 80)

    print(
        f"\nBest CV ROC-AUC: "
        f"{study.best_value:.6f}"
    )

    print("\nBest Parameters:")

    for key, value in (
        study.best_params.items()
    ):

        print(
            f"  {key}: {value}"
        )

    # --------------------------------------------------------
    # Save parameters
    # --------------------------------------------------------

    params_path = (
        MODEL_DIR
        / "best_catboost_params.json"
    )

    with open(
        params_path,
        "w"
    ) as file:

        json.dump(
            study.best_params,
            file,
            indent=4
        )

    print(
        f"\nParameters saved to:\n"
        f"{params_path}"
    )

    # --------------------------------------------------------
    # Save Optuna results
    # --------------------------------------------------------

    trials_df = study.trials_dataframe()

    trials_path = (
        OUTPUT_DIR
        / "optuna_trials.csv"
    )

    trials_df.to_csv(
        trials_path,
        index=False
    )

    print(
        f"Optuna trials saved to:\n"
        f"{trials_path}"
    )

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    validation = pd.read_csv(
        PROCESSED_DIR
        / "validation.csv"
    )

    X_validation = validation.drop(
        columns=[TARGET]
    )

    y_validation = validation[TARGET]

    # --------------------------------------------------------
    # Train optimized model
    # --------------------------------------------------------

    print(
        "\nTraining optimized CatBoost..."
    )

    model, metrics = (
        train_optimized_model(
            X_train,
            y_train,
            X_validation,
            y_validation,
            study.best_params
        )
    )

    # --------------------------------------------------------
    # Print validation results
    # --------------------------------------------------------

    print("\n" + "=" * 80)

    print(
        "OPTIMIZED MODEL — VALIDATION RESULTS"
    )

    print("=" * 80)

    for metric, value in (
        metrics.items()
    ):

        print(
            f"{metric:<15}: "
            f"{value:.4f}"
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "catboost_optimized.pkl"
    )

    import joblib

    joblib.dump(
        model,
        model_path
    )

    print(
        f"\nOptimized model saved to:\n"
        f"{model_path}"
    )

    # --------------------------------------------------------
    # Save validation results
    # --------------------------------------------------------

    validation_results = pd.DataFrame(
        [
            {
                "Model": "Optimized CatBoost",

                **metrics,

                "CV_ROC_AUC":
                    study.best_value
            }
        ]
    )

    results_path = (
        OUTPUT_DIR
        / "optimized_model_validation.csv"
    )

    validation_results.to_csv(
        results_path,
        index=False
    )

    print(
        f"Validation results saved to:\n"
        f"{results_path}"
    )

    print("\n" + "=" * 80)

    print(
        "✅ CATBOOST OPTIMIZATION COMPLETE"
    )

    print("=" * 80)


if __name__ == "__main__":

    run_optimization()