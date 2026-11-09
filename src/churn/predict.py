"""Load the saved model and turn customer records into churn probabilities."""

from pathlib import Path

import joblib
import pandas as pd

from churn.data import FEATURES
from churn.train import MODEL_PATH

DEFAULT_THRESHOLD = 0.5


def load_model(path: Path | str = MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path}. Train one first with: python -m churn.train"
        )
    return joblib.load(path)


def predict_proba(model, customers: list[dict]) -> list[float]:
    """Churn probability for each customer, in the same order as the input."""
    df = pd.DataFrame(customers, columns=FEATURES)
    return [float(p) for p in model.predict_proba(df)[:, 1]]
