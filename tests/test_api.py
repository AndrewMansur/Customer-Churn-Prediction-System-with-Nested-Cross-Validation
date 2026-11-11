import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from churn.api import create_app
from churn.data import load_dataset
from churn.schemas import EXAMPLE_CUSTOMER
from churn.train import fit_final_model


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    """Train a small, quick model so the tests don't need the real one."""
    X, y = load_dataset()
    X, _, y, _ = train_test_split(X, y, train_size=600, stratify=y, random_state=0)
    model, _ = fit_final_model("logistic_regression", X, y, inner_folds=2)
    path = tmp_path_factory.mktemp("models") / "model.joblib"
    joblib.dump(model, path)
    return path


@pytest.fixture(scope="module")
def client(model_path):
    with TestClient(create_app(model_path)) as c:
        yield c


def test_home_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Churn Risk Predictor" in response.text


def test_health_ok(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_probability(client):
    response = client.post("/predict", json=EXAMPLE_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["will_churn"] == (body["churn_probability"] >= body["threshold"])


def test_risky_customer_scores_higher_than_loyal_one(client):
    loyal = {**EXAMPLE_CUSTOMER, "tenure": 70, "Contract": "Two year",
             "PaymentMethod": "Bank transfer (automatic)", "TotalCharges": 5950.0}
    risky = client.post("/predict", json=EXAMPLE_CUSTOMER).json()["churn_probability"]
    safe = client.post("/predict", json=loyal).json()["churn_probability"]
    assert risky > safe


def test_batch_keeps_order(client):
    loyal = {**EXAMPLE_CUSTOMER, "tenure": 70, "Contract": "Two year"}
    batch = client.post("/predict/batch", json={"customers": [EXAMPLE_CUSTOMER, loyal]})
    assert batch.status_code == 200
    preds = batch.json()["predictions"]
    single = client.post("/predict", json=loyal).json()
    assert len(preds) == 2
    assert preds[1]["churn_probability"] == single["churn_probability"]


@pytest.mark.parametrize(
    "change",
    [
        {"Contract": "banana"},
        {"tenure": -1},
        {"SeniorCitizen": 2},
        {"MonthlyCharges": "a lot"},
        {"favorite_color": "blue"},
    ],
)
def test_rejects_bad_input(client, change):
    assert client.post("/predict", json={**EXAMPLE_CUSTOMER, **change}).status_code == 422


def test_rejects_missing_field(client):
    customer = {k: v for k, v in EXAMPLE_CUSTOMER.items() if k != "Contract"}
    assert client.post("/predict", json=customer).status_code == 422


def test_rejects_empty_batch(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_missing_model_returns_503(tmp_path):
    with TestClient(create_app(tmp_path / "nope.joblib")) as c:
        assert c.get("/health").json() == {"status": "model_not_loaded"}
        assert c.post("/predict", json=EXAMPLE_CUSTOMER).status_code == 503
