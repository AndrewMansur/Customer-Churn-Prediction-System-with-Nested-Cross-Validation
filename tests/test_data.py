import pandas as pd
import pytest

from churn.data import FEATURES, TARGET, clean, load_dataset


def make_row(**overrides):
    row = {
        "customerID": "0001-TEST",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": "600.0",
        "Churn": "No",
    }
    row.update(overrides)
    return row


def test_converts_total_charges_and_target():
    df = clean(pd.DataFrame([make_row(Churn="Yes")]))
    assert df["TotalCharges"].iloc[0] == 600.0
    assert df[TARGET].iloc[0] == 1


def test_blank_total_charges_for_new_customer_becomes_zero():
    df = clean(pd.DataFrame([make_row(tenure=0, TotalCharges=" ")]))
    assert df["TotalCharges"].iloc[0] == 0.0


def test_blank_total_charges_for_existing_customer_raises():
    with pytest.raises(ValueError, match="TotalCharges"):
        clean(pd.DataFrame([make_row(tenure=5, TotalCharges=" ")]))


def test_drops_customer_id():
    df = clean(pd.DataFrame([make_row()]))
    assert "customerID" not in df.columns


def test_missing_column_raises():
    raw = pd.DataFrame([make_row()]).drop(columns=["Contract"])
    with pytest.raises(ValueError, match="Contract"):
        clean(raw)


def test_unexpected_target_value_raises():
    with pytest.raises(ValueError, match="Churn"):
        clean(pd.DataFrame([make_row(Churn="Maybe")]))


def test_real_dataset_loads_cleanly():
    X, y = load_dataset()
    assert X.shape == (7043, len(FEATURES))
    assert X.isna().sum().sum() == 0
    assert set(y.unique()) == {0, 1}
    assert 0.25 < y.mean() < 0.28
