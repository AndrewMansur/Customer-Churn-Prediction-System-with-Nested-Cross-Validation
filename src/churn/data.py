"""Load and clean the Telco customer churn dataset."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

ID_COLUMN = "customerID"
TARGET = "Churn"

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw(path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV exactly as it is on disk."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw data, ready for modelling.

    - Checks that every expected column is present.
    - Drops the customer ID, which carries no signal.
    - Converts TotalCharges from text to numbers. Blank values belong to
      brand-new customers (tenure 0) who have not been billed yet, so they
      become 0. Any other unparseable value is an error.
    - Encodes Churn as 1 (Yes) / 0 (No).
    """
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.drop(columns=[ID_COLUMN], errors="ignore").copy()

    total = pd.to_numeric(df["TotalCharges"], errors="coerce")
    total = total.mask(total.isna() & (df["tenure"] == 0), 0.0)
    if total.isna().any():
        bad_rows = df.index[total.isna()].tolist()
        raise ValueError(f"Unparseable TotalCharges in rows: {bad_rows}")
    df["TotalCharges"] = total.astype(float)

    unknown = set(df[TARGET].unique()) - {"Yes", "No"}
    if unknown:
        raise ValueError(f"Unexpected {TARGET} values: {sorted(unknown)}")
    df[TARGET] = (df[TARGET] == "Yes").astype(int)

    return df[FEATURES + [TARGET]]


def load_dataset(path: Path | str = DEFAULT_DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load, clean and split the data into features (X) and target (y)."""
    df = clean(load_raw(path))
    return df[FEATURES], df[TARGET]
