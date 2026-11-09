from sklearn.model_selection import train_test_split

from churn.data import load_dataset
from churn.features import build_preprocessor
from churn.train import METRICS, fit_final_model, nested_cv, pick_best


def small_sample(n=600):
    X, y = load_dataset()
    X_small, _, y_small, _ = train_test_split(
        X, y, train_size=n, stratify=y, random_state=0
    )
    return X_small, y_small


def test_preprocessor_ignores_unseen_categories():
    X, _ = small_sample(200)
    pre = build_preprocessor().fit(X)
    row = X.iloc[[0]].copy()
    row["PaymentMethod"] = "Crypto"
    assert pre.transform(row).shape == pre.transform(X.iloc[[0]]).shape


def test_nested_cv_reports_every_model_and_metric():
    X, y = small_sample()
    results = nested_cv(X, y, outer_folds=2, inner_folds=2)
    assert set(results) == {"logistic_regression", "random_forest", "gradient_boosting"}
    for r in results.values():
        assert set(r["metrics"]) == set(METRICS)
        assert len(r["best_params_per_fold"]) == 2
        assert 0.5 < r["metrics"]["roc_auc"]["mean"] <= 1.0
    assert pick_best(results) in results


def test_final_model_outputs_probabilities():
    X, y = small_sample()
    model, _ = fit_final_model("logistic_regression", X, y, inner_folds=2)
    proba = model.predict_proba(X.head(10))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
