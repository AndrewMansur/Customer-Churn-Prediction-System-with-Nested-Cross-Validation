"""Charts for the README, built from the saved nested CV results.

Runs automatically at the end of training. To redraw without retraining:
    python -m churn.plots
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from churn.data import TARGET, clean, load_raw  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parents[2] / "reports"
RESULTS_FILE = "nested_cv_results.json"
OOF_FILE = "oof_predictions.csv"

# Colors
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua, in this order

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 1,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "grid.color": GRID,
            "grid.linewidth": 1,
            "legend.frameon": False,
        }
    )


def _label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def plot_model_comparison(report: dict, path: Path) -> None:
    """Mean ± std of the outer-fold scores, one panel per metric."""
    results = report["results"]
    names = sorted(results, key=lambda n: results[n]["metrics"]["roc_auc"]["mean"])
    panels = [("roc_auc", "ROC-AUC"), ("average_precision", "Average precision")]

    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), sharey=True)
    for ax, (metric, title) in zip(axes, panels):
        means = [results[n]["metrics"][metric]["mean"] for n in names]
        stds = [results[n]["metrics"][metric]["std"] for n in names]
        ax.errorbar(
            means, range(len(names)), xerr=stds, fmt="o", color=SERIES[0],
            ecolor=SERIES[0], elinewidth=2, capsize=0, markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2,
        )
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.annotate(f"{m:.3f}", (m + s, i), xytext=(6, 0),
                        textcoords="offset points", va="center", color=INK_2)
        lo = min(m - s for m, s in zip(means, stds))
        hi = max(m + s for m, s in zip(means, stds))
        pad = (hi - lo) * 0.25
        ax.set_xlim(lo - pad, hi + pad * 2)
        ax.set_title(title)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(
        [_label(n) + ("  (selected)" if n == report["selected_model"] else "") for n in names]
    )
    axes[0].set_ylim(-0.6, len(names) - 0.4)
    fig.suptitle(
        f"Nested CV scores: mean ± std over {report['outer_folds']} outer folds. "
        "Overlapping ranges mean the models are effectively tied.",
        x=0.01, ha="left", fontsize=9, color=INK_2, y=0.02, va="bottom",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curves(oof: pd.DataFrame, models: list[str], path: Path) -> None:
    """ROC curve per model from out-of-fold predictions."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    ax.text(0.68, 0.62, "Random guess", color=MUTED, fontsize=9,
            rotation=45, rotation_mode="anchor", transform_rotates_text=True)
    for color, name in zip(SERIES, models):
        fpr, tpr, _ = roc_curve(oof["y_true"], oof[name])
        auc = roc_auc_score(oof["y_true"], oof[name])
        ax.plot(fpr, tpr, color=color, linewidth=2, solid_capstyle="round",
                label=f"{_label(name)}  (AUC {auc:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("False positive rate (loyal customers wrongly flagged)")
    ax.set_ylabel("True positive rate (churners caught)")
    ax.set_title("ROC curves, out-of-fold predictions")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(report: dict, path: Path, top: int = 10) -> None:
    """Permutation importance of the selected model, from the held-out folds."""
    best = report["selected_model"]
    imp = report["results"][best]["permutation_importance"]
    items = sorted(imp.items(), key=lambda kv: kv[1]["mean"], reverse=True)[:top]
    items.reverse()  # biggest at the top
    names = [k for k, _ in items]
    means = [v["mean"] for _, v in items]

    fig, ax = plt.subplots(figsize=(7, 0.36 * len(items) + 1.2))
    ax.barh(range(len(items)), means, height=0.45, color=SERIES[0])
    for i, m in enumerate(means):
        ax.annotate(f"{m:.3f}", (m, i), xytext=(5, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=9)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(names)
    ax.set_xlim(0, max(means) * 1.15)
    ax.set_xlabel("Drop in ROC-AUC when the column is shuffled")
    ax.set_title(f"What drives churn predictions ({_label(best)})")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_churn_rates(df: pd.DataFrame, path: Path) -> None:
    """Churn rate by contract type and by how long the customer has stayed."""
    overall = df[TARGET].mean()
    by_contract = df.groupby("Contract")[TARGET].mean()
    tenure_group = pd.cut(df["tenure"], [-1, 12, 24, 48, 72],
                          labels=["0–12", "13–24", "25–48", "49–72"])
    by_tenure = df.groupby(tenure_group, observed=True)[TARGET].mean()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    panels = [(by_contract, "By contract type", ""),
              (by_tenure, "By tenure", "Months with the company")]
    for ax, (rates, title, xlabel) in zip(axes, panels):
        x = range(len(rates))
        ax.bar(x, rates.values, width=0.5, color=SERIES[0])
        for i, r in enumerate(rates.values):
            ax.annotate(f"{r:.0%}", (i, r), xytext=(0, 4), textcoords="offset points",
                        ha="center", color=INK_2, fontsize=9)
        ax.axhline(overall, color=MUTED, linewidth=1)
        ax.set_xticks(list(x))
        ax.set_xticklabels(rates.index)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
    axes[1].text(0.99, overall, f"All customers {overall:.0%}  ",
                 transform=axes[1].get_yaxis_transform(), ha="right", va="bottom",
                 color=MUTED, fontsize=9)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[0].set_ylabel("Customers who churned")
    axes[0].set_ylim(0, max(by_contract.max(), by_tenure.max()) * 1.15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_all_plots(report_dir: Path = REPORT_DIR, df: pd.DataFrame | None = None) -> list[Path]:
    """Draw every chart into report_dir and return their paths."""
    _style()
    report_dir = Path(report_dir)
    report = json.loads((report_dir / RESULTS_FILE).read_text(encoding="utf-8"))
    oof = pd.read_csv(report_dir / OOF_FILE)
    if df is None:
        df = clean(load_raw())

    ranked = sorted(report["results"],
                    key=lambda n: -report["results"][n]["metrics"]["roc_auc"]["mean"])
    paths = {
        "model_comparison.png": lambda p: plot_model_comparison(report, p),
        "roc_curves.png": lambda p: plot_roc_curves(oof, ranked, p),
        "feature_importance.png": lambda p: plot_feature_importance(report, p),
        "churn_rates.png": lambda p: plot_churn_rates(df, p),
    }
    out = []
    for filename, draw in paths.items():
        draw(report_dir / filename)
        out.append(report_dir / filename)
    return out


if __name__ == "__main__":
    for p in make_all_plots():
        print(f"Saved {p}")
