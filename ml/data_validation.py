from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cancer_prediction_dataset.csv"
)


REQUIRED_COLUMNS = [
    "Age",
    "Gender",
    "BMI",
    "Smoking",
    "GeneticRisk",
    "PhysicalActivity",
    "AlcoholIntake",
    "CancerHistory",
    "Diagnosis",
]


def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    return df


def validate_columns(df):

    print("\n[1] COLUMN VALIDATION")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        print("❌ Missing columns:")
        for column in missing_columns:
            print(f"   - {column}")

        return False

    print("✅ All required columns are present.")

    return True


def validate_missing_values(df):

    print("\n[2] MISSING VALUE VALIDATION")

    missing = df.isnull().sum()

    total_missing = missing.sum()

    if total_missing == 0:

        print("✅ No missing values found.")

        return True

    print("❌ Missing values detected:")

    print(missing[missing > 0])

    return False


def validate_duplicates(df):

    print("\n[3] DUPLICATE VALIDATION")

    duplicates = df.duplicated().sum()

    print(f"Duplicate rows: {duplicates}")

    if duplicates == 0:

        print("✅ No duplicate rows found.")

        return True

    print("⚠️ Duplicate rows detected.")

    return False


def validate_target(df):

    print("\n[4] TARGET VALIDATION")

    target = "Diagnosis"

    if target not in df.columns:

        print("❌ Diagnosis column missing.")

        return False

    print("Target distribution:")

    print(df[target].value_counts())

    unique_values = set(
        df[target].dropna().unique()
    )

    print(
        f"Unique target values: {unique_values}"
    )

    if not unique_values.issubset({0, 1}):

        print(
            "⚠️ Target contains values "
            "other than 0 and 1."
        )

        return False

    print("✅ Target is binary.")

    return True


def validate_ranges(df):

    print("\n[5] NUMERICAL RANGE VALIDATION")

    valid = True

    # Age
    if "Age" in df.columns:

        invalid = (
            (df["Age"] < 0)
            | (df["Age"] > 120)
        ).sum()

        print(f"Invalid Age values: {invalid}")

        if invalid > 0:
            valid = False

    # BMI
    if "BMI" in df.columns:

        invalid = (
            (df["BMI"] <= 0)
            | (df["BMI"] > 100)
        ).sum()

        print(f"Invalid BMI values: {invalid}")

        if invalid > 0:
            valid = False

    # Smoking
    if "Smoking" in df.columns:

        invalid = (
            ~df["Smoking"].isin([0, 1])
        ).sum()

        print(
            f"Invalid Smoking values: {invalid}"
        )

        if invalid > 0:
            valid = False

    # GeneticRisk
    if "GeneticRisk" in df.columns:

        invalid = (
            ~df["GeneticRisk"].isin([0, 1, 2])
        ).sum()

        print(
            f"Invalid GeneticRisk values: {invalid}"
        )

        if invalid > 0:
            valid = False

    # CancerHistory
    if "CancerHistory" in df.columns:

        invalid = (
            ~df["CancerHistory"].isin([0, 1])
        ).sum()

        print(
            f"Invalid CancerHistory values: {invalid}"
        )

        if invalid > 0:
            valid = False

    if valid:

        print("✅ Numerical ranges look valid.")

    else:

        print(
            "⚠️ Some values are outside "
            "expected ranges."
        )

    return valid


def validate_data_types(df):

    print("\n[6] DATA TYPE VALIDATION")

    print(df.dtypes)

    print(
        "\n✅ Data types inspected."
    )

    return True


def leakage_check(df):

    print("\n[7] TARGET LEAKAGE CHECK")

    target = "Diagnosis"

    suspicious_columns = []

    for column in df.columns:

        if column == target:
            continue

        correlation = None

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            correlation = (
                df[column]
                .corr(df[target])
            )

        if (
            correlation is not None
            and abs(correlation) > 0.95
        ):

            suspicious_columns.append(
                (column, correlation)
            )

    if suspicious_columns:

        print(
            "⚠️ Potentially suspicious "
            "features:"
        )

        for column, correlation in (
            suspicious_columns
        ):

            print(
                f"   {column}: "
                f"correlation={correlation:.4f}"
            )

        return False

    print(
        "✅ No obvious correlation-based "
        "target leakage detected."
    )

    return True


def generate_summary(df):

    print("\n[8] DATASET SUMMARY")

    print(
        f"Number of records : {len(df)}"
    )

    print(
        f"Number of features: "
        f"{len(df.columns) - 1}"
    )

    print(
        f"Memory usage      : "
        f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
    )


def run_validation():

    print("\n" + "=" * 70)

    print(
        "ONCOGUARD AI — DATA VALIDATION PIPELINE"
    )

    print("=" * 70)

    df = load_dataset()

    print(
        f"\nDataset loaded from:\n{DATA_PATH}"
    )

    print(
        f"\nDataset shape: {df.shape}"
    )

    results = []

    results.append(
        validate_columns(df)
    )

    results.append(
        validate_missing_values(df)
    )

    results.append(
        validate_duplicates(df)
    )

    results.append(
        validate_target(df)
    )

    results.append(
        validate_ranges(df)
    )

    results.append(
        validate_data_types(df)
    )

    results.append(
        leakage_check(df)
    )

    generate_summary(df)

    print("\n" + "=" * 70)

    if all(results):

        print(
            "✅ DATA VALIDATION PASSED"
        )

    else:

        print(
            "⚠️ DATA VALIDATION COMPLETED "
            "WITH WARNINGS"
        )

    print("=" * 70)


if __name__ == "__main__":

    run_validation()