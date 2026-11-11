"""Compare models with nested cross-validation and save the best one.

Outer loop: estimates how well each model does on customers it has never seen.
Inner loop: picks hyperparameters using only the outer training folds.
Keeping the two separate means the reported scores are not inflated by tuning.

Run with:  python -m churn.train
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from churn.data import PROJECT_ROOT, load_dataset
from churn.models import get_candidates
from churn.plots import OOF_FILE, REPORT_DIR, RESULTS_FILE, make_all_plots

SELECTION_METRIC = "roc_auc"
METRICS = ["roc_auc", "average_precision", "f1", "precision", "recall", "accuracy"]

MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"


def nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    outer_folds: int = 5,
    inner_folds: int = 3,
    random_state: int = 42,
) -> tuple[dict, pd.DataFrame]:
    """Run nested CV for every candidate model.

    Returns:
        results: per-model scores, chosen settings and feature importance.
        oof: every customer's churn probability from each model, predicted by
            the outer fold that held that customer out ("out-of-fold"). Used
            for ROC curves without any customer being scored by a model that
            trained on them.
    """
    outer_cv = StratifiedKFold(outer_folds, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(inner_folds, shuffle=True, random_state=random_state)

    results = {}
    oof = pd.DataFrame({"y_true": y.to_numpy()}, index=X.index)
    for name, (pipeline, grid) in get_candidates(random_state).items():
        print(f"Running nested CV for {name}...")
        search = GridSearchCV(
            pipeline, grid, cv=inner_cv, scoring=SELECTION_METRIC, n_jobs=-1
        )
        scores = cross_validate(
            search,
            X,
            y,
            cv=outer_cv,
            scoring=METRICS,
            return_estimator=True,
            return_indices=True,
        )

        proba = np.empty(len(X))
        fold_importances = []
        for est, test_idx in zip(scores["estimator"], scores["indices"]["test"]):
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            proba[test_idx] = est.predict_proba(X_test)[:, 1]
            # How much does the held-out score drop when one column is shuffled?
            imp = permutation_importance(
                est, X_test, y_test, scoring=SELECTION_METRIC,
                n_repeats=5, random_state=random_state, n_jobs=-1,
            )
            fold_importances.append(imp.importances_mean)
        oof[name] = proba
        fold_importances = np.array(fold_importances)

        results[name] = {
            "metrics": {
                m: {
                    "mean": float(np.mean(scores[f"test_{m}"])),
                    "std": float(np.std(scores[f"test_{m}"])),
                    "folds": [float(s) for s in scores[f"test_{m}"]],
                }
                for m in METRICS
            },
            "best_params_per_fold": [est.best_params_ for est in scores["estimator"]],
            "permutation_importance": {
                feature: {
                    "mean": float(fold_importances[:, i].mean()),
                    "std": float(fold_importances[:, i].std()),
                }
                for i, feature in enumerate(X.columns)
            },
        }
        mean = results[name]["metrics"][SELECTION_METRIC]["mean"]
        print(f"  {SELECTION_METRIC}: {mean:.4f}")
    return results, oof


def pick_best(results: dict) -> str:
    """Name of the model with the highest mean outer-fold score."""
    return max(results, key=lambda n: results[n]["metrics"][SELECTION_METRIC]["mean"])


def fit_final_model(name: str, X, y, inner_folds: int = 3, random_state: int = 42):
    """Tune the chosen model on all the data and return the fitted pipeline."""
    pipeline, grid = get_candidates(random_state)[name]
    cv = StratifiedKFold(inner_folds, shuffle=True, random_state=random_state)
    search = GridSearchCV(pipeline, grid, cv=cv, scoring=SELECTION_METRIC, n_jobs=-1)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def write_markdown_report(results: dict, best: str, path: Path) -> None:
    lines = [
        "# Nested cross-validation results",
        "",
        "Mean ± std across outer folds. Model selected by ROC-AUC.",
        "",
        "| Model | " + " | ".join(METRICS) + " |",
        "|---|" + "---|" * len(METRICS),
    ]
    ranked = sorted(
        results.items(), key=lambda kv: -kv[1]["metrics"][SELECTION_METRIC]["mean"]
    )
    for name, r in ranked:
        cells = [
            f"{r['metrics'][m]['mean']:.3f} ± {r['metrics'][m]['std']:.3f}"
            for m in METRICS
        ]
        label = f"**{name}** (selected)" if name == best else name
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the churn model with nested CV.")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    X, y = load_dataset()
    results, oof = nested_cv(X, y, args.outer_folds, args.inner_folds, args.seed)
    best = pick_best(results)
    print(f"Best model: {best}. Fitting on all data...")
    model, params = fit_final_model(best, X, y, args.inner_folds, args.seed)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "selected_model": best,
        "selected_params": params,
        "selection_metric": SELECTION_METRIC,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "seed": args.seed,
        "results": results,
    }
    (REPORT_DIR / RESULTS_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
    oof.to_csv(REPORT_DIR / OOF_FILE, index=False, float_format="%.5f")
    write_markdown_report(results, best, REPORT_DIR / "nested_cv_results.md")
    print(f"Saved model to {MODEL_PATH}")

    make_all_plots(REPORT_DIR)
    print(f"Saved reports and charts to {REPORT_DIR}")


if __name__ == "__main__":
    main()
