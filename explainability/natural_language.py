import os
import pandas as pd
import numpy as np


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SHAP_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "shap",
    "patient_0_explanation.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "shap",
    "patient_0_natural_language_explanation.txt"
)


FEATURE_LABELS = {
    "Age": "age",
    "Gender": "gender",
    "BMI": "body mass index",
    "Smoking": "smoking status",
    "GeneticRisk": "genetic risk",
    "PhysicalActivity": "physical activity",
    "AlcoholIntake": "alcohol intake",
    "CancerHistory": "cancer history",
    "Is_Obese": "obesity indicator",
    "Risk_Factor_Count": "overall risk-factor count",
    "Age_Group": "age group",
    "BMI_Category": "BMI category"
}


def format_feature_name(feature):

    return FEATURE_LABELS.get(
        feature,
        feature.replace("_", " ").lower()
    )


def generate_explanation(df):

    positive = df[
        df["SHAP_Value"] > 0
    ].copy()

    negative = df[
        df["SHAP_Value"] < 0
    ].copy()

    positive = positive.sort_values(
        "SHAP_Value",
        ascending=False
    )

    negative = negative.sort_values(
        "SHAP_Value",
        ascending=True
    )

    explanation = []

    explanation.append(
        "ONCOGUARD AI — MODEL EXPLANATION"
    )

    explanation.append(
        ""
    )

    explanation.append(
        "This explanation describes which input "
        "features contributed most to the model's "
        "prediction."
    )

    explanation.append(
        "These contributions describe model behavior "
        "and should not be interpreted as causal "
        "evidence or a medical diagnosis."
    )

    # ---------------------------------------------------------
    # Positive contributors
    # ---------------------------------------------------------

    if len(positive) > 0:

        explanation.append("")
        explanation.append(
            "Factors contributing toward a higher "
            "predicted risk:"
        )

        for _, row in positive.head(5).iterrows():

            feature = format_feature_name(
                row["Feature"]
            )

            value = row["Feature_Value"]

            explanation.append(
                f"- {feature} "
                f"(value: {value}) was an important "
                f"positive contributor to the model's "
                f"prediction."
            )

    # ---------------------------------------------------------
    # Negative contributors
    # ---------------------------------------------------------

    if len(negative) > 0:

        explanation.append("")
        explanation.append(
            "Factors contributing toward a lower "
            "predicted risk:"
        )

        for _, row in negative.head(5).iterrows():

            feature = format_feature_name(
                row["Feature"]
            )

            value = row["Feature_Value"]

            explanation.append(
                f"- {feature} "
                f"(value: {value}) contributed toward "
                f"a lower model prediction."
            )

    # ---------------------------------------------------------
    # Strongest contributors
    # ---------------------------------------------------------

    strongest = df.sort_values(
        "Absolute_SHAP",
        ascending=False
    ).head(3)

    if len(strongest) > 0:

        names = [
            format_feature_name(x)
            for x in strongest["Feature"]
        ]

        explanation.append("")

        explanation.append(
            "Most influential features for this "
            "prediction:"
        )

        for name in names:

            explanation.append(
                f"- {name}"
            )

    explanation.append("")

    explanation.append(
        "Important: A feature's contribution indicates "
        "how the trained model used that feature for "
        "this prediction. It does not establish that "
        "the feature caused cancer or that changing "
        "the feature will produce a particular change "
        "in risk."
    )

    return "\n".join(
        explanation
    )


def main():

    print("=" * 70)
    print("ONCOGUARD AI — NATURAL-LANGUAGE EXPLAINABILITY")
    print("=" * 70)

    if not os.path.exists(SHAP_PATH):

        raise FileNotFoundError(
            f"SHAP explanation not found:\n{SHAP_PATH}\n"
            "Run explainability/shap_engine.py first."
        )

    df = pd.read_csv(
        SHAP_PATH
    )

    print(
        f"\nLoaded SHAP explanation: "
        f"{len(df)} features"
    )

    explanation = generate_explanation(
        df
    )

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            explanation
        )

    print("\n" + "=" * 70)
    print("GENERATED EXPLANATION")
    print("=" * 70)

    print(
        "\n" + explanation
    )

    print("\n" + "=" * 70)

    print(
        f"✓ Saved: {OUTPUT_PATH}"
    )

    print("\nSTEP 14 COMPLETE")


if __name__ == "__main__":
    main()