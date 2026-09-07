import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "train.csv")
VAL_PATH = os.path.join(BASE_DIR, "data", "processed", "validation.csv")

MODEL_PATH = os.path.join(BASE_DIR, "models", "catboost_optimized.pkl")

CALIBRATED_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "catboost_calibrated.pkl"
)

REPORT_PATH = os.path.join(
    BASE_DIR, "outputs", "reports", "calibration_results.csv"
)

PLOT_PATH = os.path.join(
    BASE_DIR, "outputs", "plots", "calibration_curve.png"
)


TARGET = "Diagnosis"


def main():

    print("=" * 70)
    print("ONCOGUARD AI — PROBABILITY CALIBRATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_val = val_df.drop(columns=[TARGET])
    y_val = val_df[TARGET]

    print(f"Training samples   : {len(X_train)}")
    print(f"Validation samples : {len(X_val)}")

    # ---------------------------------------------------------
    # Load optimized CatBoost model
    # ---------------------------------------------------------

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Optimized model not found:\n{MODEL_PATH}\n"
            "Run ml/tune_model.py first."
        )

    base_model = joblib.load(MODEL_PATH)

    print("\nBase model loaded:")
    print("CatBoost Optimized")

    # ---------------------------------------------------------
    # Uncalibrated predictions
    # ---------------------------------------------------------

    print("\nGenerating uncalibrated probabilities...")

    uncalibrated_probs = base_model.predict_proba(X_val)[:, 1]

    uncalibrated_brier = brier_score_loss(
        y_val,
        uncalibrated_probs
    )

    uncalibrated_auc = roc_auc_score(
        y_val,
        uncalibrated_probs
    )

    # ---------------------------------------------------------
    # Calibration
    # ---------------------------------------------------------

    print("\nApplying sigmoid probability calibration...")

    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=5
    )

    calibrated_model.fit(
        X_train,
        y_train
    )

    calibrated_probs = calibrated_model.predict_proba(X_val)[:, 1]

    calibrated_brier = brier_score_loss(
        y_val,
        calibrated_probs
    )

    calibrated_auc = roc_auc_score(
        y_val,
        calibrated_probs
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)

    print("\nUncalibrated:")
    print(f"Brier Score : {uncalibrated_brier:.6f}")
    print(f"ROC-AUC     : {uncalibrated_auc:.6f}")

    print("\nCalibrated:")
    print(f"Brier Score : {calibrated_brier:.6f}")
    print(f"ROC-AUC     : {calibrated_auc:.6f}")

    improvement = uncalibrated_brier - calibrated_brier

    print("\nBrier improvement:")
    print(f"{improvement:.6f}")

    if calibrated_brier < uncalibrated_brier:
        print("\n✓ Calibration improved probability quality.")
    else:
        print("\n⚠ Calibration did not improve Brier score.")

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    results = pd.DataFrame({
        "Model": [
            "CatBoost Uncalibrated",
            "CatBoost Calibrated"
        ],
        "Brier_Score": [
            uncalibrated_brier,
            calibrated_brier
        ],
        "ROC_AUC": [
            uncalibrated_auc,
            calibrated_auc
        ]
    })

    os.makedirs(
        os.path.dirname(REPORT_PATH),
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(PLOT_PATH),
        exist_ok=True
    )

    results.to_csv(
        REPORT_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # Calibration curve
    # ---------------------------------------------------------

    prob_true_uncal, prob_pred_uncal = calibration_curve(
        y_val,
        uncalibrated_probs,
        n_bins=10,
        strategy="uniform"
    )

    prob_true_cal, prob_pred_cal = calibration_curve(
        y_val,
        calibrated_probs,
        n_bins=10,
        strategy="uniform"
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect Calibration"
    )

    plt.plot(
        prob_pred_uncal,
        prob_true_uncal,
        marker="o",
        label="Uncalibrated CatBoost"
    )

    plt.plot(
        prob_pred_cal,
        prob_true_cal,
        marker="o",
        label="Calibrated CatBoost"
    )

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")

    plt.title("OncoGuard AI — Probability Calibration")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        PLOT_PATH,
        dpi=300
    )

    plt.close()

    # ---------------------------------------------------------
    # Save calibrated model
    # ---------------------------------------------------------

    joblib.dump(
        calibrated_model,
        CALIBRATED_MODEL_PATH
    )

    print("\nSaved:")
    print(f"✓ {CALIBRATED_MODEL_PATH}")
    print(f"✓ {REPORT_PATH}")
    print(f"✓ {PLOT_PATH}")

    print("\n" + "=" * 70)
    print("STEP 10 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()