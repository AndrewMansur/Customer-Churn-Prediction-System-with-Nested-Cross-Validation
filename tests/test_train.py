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
    results, oof = nested_cv(X, y, outer_folds=2, inner_folds=2)
    assert set(results) == {"logistic_regression", "random_forest", "gradient_boosting"}
    for name, r in results.items():
        assert set(r["metrics"]) == set(METRICS)
        assert len(r["best_params_per_fold"]) == 2
        assert 0.5 < r["metrics"]["roc_auc"]["mean"] <= 1.0
        assert set(r["permutation_importance"]) == set(X.columns)
        assert oof[name].between(0, 1).all()
    assert len(oof) == len(X)
    assert pick_best(results) in results


def test_final_model_outputs_probabilities():
    X, y = small_sample()
    model, _ = fit_final_model("logistic_regression", X, y, inner_folds=2)
    proba = model.predict_proba(X.head(10))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_charts_are_drawn(tmp_path):
    import json

    from churn.data import clean, load_raw
    from churn.plots import OOF_FILE, RESULTS_FILE, make_all_plots

    X, y = small_sample()
    results, oof = nested_cv(X, y, outer_folds=2, inner_folds=2)
    report = {"selected_model": pick_best(results), "outer_folds": 2, "results": results}
    (tmp_path / RESULTS_FILE).write_text(json.dumps(report))
    oof.to_csv(tmp_path / OOF_FILE, index=False)

    paths = make_all_plots(tmp_path, df=clean(load_raw()))
    assert len(paths) == 4
    assert all(p.exists() and p.stat().st_size > 10_000 for p in paths)
