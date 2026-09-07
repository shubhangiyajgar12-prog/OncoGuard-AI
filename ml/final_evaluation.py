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
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    brier_score_loss
)


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TEST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "test.csv"
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

REPORT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "reports",
    "final_test_evaluation.csv"
)

CONFUSION_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "plots",
    "final_confusion_matrix.png"
)

ROC_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "plots",
    "final_roc_curve.png"
)

PR_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "plots",
    "final_precision_recall_curve.png"
)

TARGET = "Diagnosis"


def main():

    print("=" * 70)
    print("ONCOGUARD AI — FINAL TEST EVALUATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load test data
    # ---------------------------------------------------------

    test_df = pd.read_csv(TEST_PATH)

    X_test = test_df.drop(
        columns=[TARGET]
    )

    y_test = test_df[TARGET]

    print(f"\nTest samples: {len(test_df)}")

    # ---------------------------------------------------------
    # Load calibrated model
    # ---------------------------------------------------------

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Calibrated model not found. "
            "Run ml/calibration.py first."
        )

    model = joblib.load(
        MODEL_PATH
    )

    # ---------------------------------------------------------
    # Load selected threshold
    # ---------------------------------------------------------

    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(
            "Decision threshold not found. "
            "Run ml/threshold.py first."
        )

    with open(
        THRESHOLD_PATH,
        "r"
    ) as f:

        threshold_config = json.load(f)

    threshold = float(
        threshold_config["decision_threshold"]
    )

    print(
        f"Decision threshold: {threshold:.4f}"
    )

    # ---------------------------------------------------------
    # Generate test probabilities
    # ---------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    brier = brier_score_loss(
        y_test,
        probabilities
    )

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    # ---------------------------------------------------------
    # Print final results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"\nAccuracy       : {accuracy:.4f}"
    )

    print(
        f"Precision      : {precision:.4f}"
    )

    print(
        f"Recall         : {recall:.4f}"
    )

    print(
        f"Specificity    : {specificity:.4f}"
    )

    print(
        f"F1 Score       : {f1:.4f}"
    )

    print(
        f"ROC-AUC        : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC         : {pr_auc:.4f}"
    )

    print(
        f"Brier Score    : {brier:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        f"TN = {tn}"
    )

    print(
        f"FP = {fp}"
    )

    print(
        f"FN = {fn}"
    )

    print(
        f"TP = {tp}"
    )

    # ---------------------------------------------------------
    # Save report
    # ---------------------------------------------------------

    results = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "Specificity",
            "F1",
            "ROC_AUC",
            "PR_AUC",
            "Brier_Score",
            "True_Negative",
            "False_Positive",
            "False_Negative",
            "True_Positive",
            "Decision_Threshold"
        ],

        "Value": [
            accuracy,
            precision,
            recall,
            specificity,
            f1,
            roc_auc,
            pr_auc,
            brier,
            tn,
            fp,
            fn,
            tp,
            threshold
        ]
    })

    os.makedirs(
        os.path.dirname(REPORT_PATH),
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(CONFUSION_PATH),
        exist_ok=True
    )

    results.to_csv(
        REPORT_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # Confusion matrix plot
    # ---------------------------------------------------------

    cm = np.array([
        [tn, fp],
        [fn, tp]
    ])

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(cm)

    plt.title(
        "OncoGuard AI — Final Test Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.xticks(
        [0, 1],
        ["Negative", "Positive"]
    )

    plt.yticks(
        [0, 1],
        ["Negative", "Positive"]
    )

    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_PATH,
        dpi=300
    )

    plt.close()

    # ---------------------------------------------------------
    # ROC Curve
    # ---------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {roc_auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "OncoGuard AI — Final ROC Curve"
    )

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        ROC_PATH,
        dpi=300
    )

    plt.close()

    # ---------------------------------------------------------
    # Precision-Recall Curve
    # ---------------------------------------------------------

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            y_test,
            probabilities
        )
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        recall_curve,
        precision_curve,
        label=f"PR-AUC = {pr_auc:.4f}"
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "OncoGuard AI — Final Precision-Recall Curve"
    )

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        PR_PATH,
        dpi=300
    )

    plt.close()

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    print("\nSaved:")

    print(
        f"✓ {REPORT_PATH}"
    )

    print(
        f"✓ {CONFUSION_PATH}"
    )

    print(
        f"✓ {ROC_PATH}"
    )

    print(
        f"✓ {PR_PATH}"
    )

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION COMPLETE")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "The test set has now been used for final evaluation."
    )

    print(
        "Do not tune the model based on these test results."
    )


if __name__ == "__main__":
    main()