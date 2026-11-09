"""Candidate models and the hyperparameters to search for each."""

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn.features import build_preprocessor


def get_candidates(random_state: int = 42) -> dict[str, tuple[Pipeline, dict]]:
    """Return {name: (pipeline, param_grid)} for every model we compare.

    Every model uses class_weight="balanced" because only ~26% of customers
    churn. Without it the models lean towards always predicting "no churn".
    """

    def pipeline(model):
        return Pipeline([("preprocess", build_preprocessor()), ("model", model)])

    return {
        "logistic_regression": (
            pipeline(
                LogisticRegression(
                    class_weight="balanced", max_iter=2000, random_state=random_state
                )
            ),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            pipeline(
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                )
            ),
            {
                "model__max_depth": [6, 10, None],
                "model__min_samples_leaf": [1, 5, 20],
            },
        ),
        "gradient_boosting": (
            pipeline(
                HistGradientBoostingClassifier(
                    class_weight="balanced", random_state=random_state
                )
            ),
            {
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [3, 6],
                "model__max_iter": [100, 300],
            },
        ),
    }
