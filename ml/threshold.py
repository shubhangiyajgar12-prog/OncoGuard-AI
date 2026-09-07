import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VAL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "validation.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "catboost_calibrated.pkl"
)

REPORT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "reports",
    "threshold_analysis.csv"
)

PLOT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "plots",
    "threshold_analysis.png"
)

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "models",
    "decision_threshold.json"
)

TARGET = "Diagnosis"


def calculate_metrics(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

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

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    return {
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }


def main():

    print("=" * 70)
    print("ONCOGUARD AI — DECISION THRESHOLD OPTIMIZATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load validation data
    # ---------------------------------------------------------

    if not os.path.exists(VAL_PATH):
        raise FileNotFoundError(
            f"Validation dataset not found:\n{VAL_PATH}"
        )

    val_df = pd.read_csv(VAL_PATH)

    X_val = val_df.drop(
        columns=[TARGET]
    )

    y_val = val_df[TARGET]

    print(f"\nValidation samples: {len(val_df)}")

    # ---------------------------------------------------------
    # Load calibrated model
    # ---------------------------------------------------------

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Calibrated model not found:\n{MODEL_PATH}\n"
            "Run ml/calibration.py first."
        )

    model = joblib.load(
        MODEL_PATH
    )

    # ---------------------------------------------------------
    # Generate probabilities
    # ---------------------------------------------------------

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    auc = roc_auc_score(
        y_val,
        probabilities
    )

    print(f"\nValidation ROC-AUC: {auc:.6f}")

    # ---------------------------------------------------------
    # Evaluate thresholds
    # ---------------------------------------------------------

    thresholds = np.arange(
        0.10,
        0.91,
        0.01
    )

    results = []

    for threshold in thresholds:

        metrics = calculate_metrics(
            y_val,
            probabilities,
            threshold
        )

        results.append(metrics)

    results_df = pd.DataFrame(
        results
    )

    # ---------------------------------------------------------
    # Find best F1 threshold
    # ---------------------------------------------------------

    best_f1_row = results_df.loc[
        results_df["F1"].idxmax()
    ]

    best_f1_threshold = float(
        best_f1_row["Threshold"]
    )

    # ---------------------------------------------------------
    # Youden's J statistic
    # ---------------------------------------------------------

    fpr, tpr, roc_thresholds = roc_curve(
        y_val,
        probabilities
    )

    youden_scores = tpr - fpr

    best_youden_index = np.argmax(
        youden_scores
    )

    best_youden_threshold = float(
        roc_thresholds[best_youden_index]
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("THRESHOLD RESULTS")
    print("=" * 70)

    print("\nDefault threshold = 0.50")

    default_row = results_df.iloc[
        (results_df["Threshold"] - 0.50)
        .abs()
        .argmin()
    ]

    print(
        f"Accuracy    : {default_row['Accuracy']:.4f}"
    )
    print(
        f"Precision   : {default_row['Precision']:.4f}"
    )
    print(
        f"Recall      : {default_row['Recall']:.4f}"
    )
    print(
        f"Specificity : {default_row['Specificity']:.4f}"
    )
    print(
        f"F1          : {default_row['F1']:.4f}"
    )

    print("\nBest F1 threshold:")
    print(
        f"Threshold   : {best_f1_threshold:.2f}"
    )
    print(
        f"Accuracy    : {best_f1_row['Accuracy']:.4f}"
    )
    print(
        f"Precision   : {best_f1_row['Precision']:.4f}"
    )
    print(
        f"Recall      : {best_f1_row['Recall']:.4f}"
    )
    print(
        f"Specificity : {best_f1_row['Specificity']:.4f}"
    )
    print(
        f"F1          : {best_f1_row['F1']:.4f}"
    )

    print("\nBest Youden J threshold:")
    print(
        f"Threshold   : {best_youden_threshold:.4f}"
    )

    # ---------------------------------------------------------
    # Save threshold analysis
    # ---------------------------------------------------------

    os.makedirs(
        os.path.dirname(REPORT_PATH),
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(PLOT_PATH),
        exist_ok=True
    )

    results_df.to_csv(
        REPORT_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        results_df["Threshold"],
        results_df["Recall"],
        label="Recall"
    )

    plt.plot(
        results_df["Threshold"],
        results_df["Specificity"],
        label="Specificity"
    )

    plt.plot(
        results_df["Threshold"],
        results_df["F1"],
        label="F1 Score"
    )

    plt.axvline(
        0.50,
        linestyle="--",
        label="Default 0.50"
    )

    plt.axvline(
        best_f1_threshold,
        linestyle=":",
        label=f"Best F1 = {best_f1_threshold:.2f}"
    )

    plt.xlabel(
        "Decision Threshold"
    )

    plt.ylabel(
        "Metric Value"
    )

    plt.title(
        "OncoGuard AI — Decision Threshold Analysis"
    )

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        PLOT_PATH,
        dpi=300
    )

    plt.close()

    # ---------------------------------------------------------
    # Save selected configuration
    # ---------------------------------------------------------

    config = {
        "model": "CatBoost Calibrated",
        "threshold_selection_method": "Maximum F1 on validation set",
        "decision_threshold": best_f1_threshold,
        "validation_roc_auc": float(auc)
    }

    with open(
        CONFIG_PATH,
        "w"
    ) as f:

        json.dump(
            config,
            f,
            indent=4
        )

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    print("\nSaved:")
    print(f"✓ {REPORT_PATH}")
    print(f"✓ {PLOT_PATH}")
    print(f"✓ {CONFIG_PATH}")

    print("\n" + "=" * 70)
    print("STEP 11 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()