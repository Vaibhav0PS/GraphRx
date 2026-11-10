"""Plot evaluation results written by src/train.py.

Reads from the results folder set in config.json (ml.results_dir):

  metrics.json             per-model val/test metrics and the best model
  test_scores.csv.gz       test pairs with labels and one score column per model
  feature_importance.csv   model, feature, importance

Saves to <results_dir>/figures/:

  roc.png                  ROC curves on the test set
  pr.png                   precision-recall curves on the test set
  model_comparison.png     test metrics per model
  feature_importance.png   feature importance per model

Usage:
    python src/plots.py
    python src/plots.py --config other_config.json
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from config import load_config, resolve

DPI = 150
ID_COLUMNS = ["drug_id", "disease_id", "label"]
COUNT_KEYS = {"n_positive", "n_negative"}
POSITIVE_COLOR = "tab:blue"
NEGATIVE_COLOR = "tab:red"


def results_paths(cfg: dict) -> dict[str, Path]:
    try:
        results_dir = resolve(cfg["ml"]["results_dir"])
    except KeyError:
        sys.exit("config.json has no ml.results_dir. Add it, then run `python src/train.py` first.")
    paths = {
        "metrics": results_dir / "metrics.json",
        "scores": results_dir / "test_scores.csv.gz",
        "importance": results_dir / "feature_importance.csv",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        sys.exit("Missing result files:\n  " + "\n  ".join(missing) + "\nRun `python src/train.py` first.")
    paths["figures"] = results_dir / "figures"
    return paths


def edge_label(metrics: dict) -> str:
    edge = metrics.get("label_edge")
    return " ".join(edge) if edge else "link prediction"


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def score_columns(scores: pd.DataFrame, metrics: dict) -> list[str]:
    """Model score columns, in the order models appear in metrics.json when possible."""
    columns = [c for c in scores.columns if c not in ID_COLUMNS]
    order = list(metrics.get("models", {}))
    return sorted(columns, key=lambda c: order.index(c) if c in order else len(order))


def plot_roc(scores: pd.DataFrame, models: list[str], title: str, path: Path) -> None:
    y = scores["label"].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    for model in models:
        s = scores[model].to_numpy()
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, lw=1.8, label=f"{model} (AUROC = {roc_auc_score(y, s):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate", ylabel="True positive rate",
           title=f"ROC curve, test set\n{title}")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    save(fig, path)


def plot_pr(scores: pd.DataFrame, models: list[str], title: str, path: Path) -> None:
    y = scores["label"].to_numpy()
    positive_rate = y.mean()
    fig, ax = plt.subplots(figsize=(6, 6))
    for model in models:
        s = scores[model].to_numpy()
        precision, recall, _ = precision_recall_curve(y, s)
        ax.plot(recall, precision, lw=1.8, label=f"{model} (AUPRC = {average_precision_score(y, s):.3f})")
    ax.axhline(positive_rate, color="k", ls="--", lw=1, label=f"random ({positive_rate:.4f})")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="Recall", ylabel="Precision",
           title=f"Precision-recall curve, test set\n{title}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    save(fig, path)


def float_metric_names(models: dict) -> list[str]:
    """Metric keys present in any model's test dict, excluding integer counts, in first-seen order."""
    names = []
    for result in models.values():
        for key, value in result.get("test", {}).items():
            if key in COUNT_KEYS or not isinstance(value, float):
                continue
            if key not in names:
                names.append(key)
    return names


def plot_model_comparison(metrics: dict, title: str, path: Path) -> None:
    models = metrics.get("models", {})
    names = float_metric_names(models)
    if not models or not names:
        print("No test metrics in metrics.json, skipping model comparison.")
        return
    model_names = list(models)
    width = 0.8 / len(model_names)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(min(16, max(8, 0.35 * len(names) * len(model_names) + 2)), 5))
    for i, model in enumerate(model_names):
        values = [models[model].get("test", {}).get(n, np.nan) for n in names]
        bars = ax.bar(x + (i - (len(model_names) - 1) / 2) * width, values, width, label=model)
        ax.bar_label(bars, labels=["" if np.isnan(v) else f"{v:.2f}" for v in values],
                     fontsize=7, rotation=90, padding=2)
    best = metrics.get("best_model")
    heading = f"Test metrics by model, {title}"
    if best:
        heading += f"\nbest model (chosen on validation): {best}"
    ax.set(xticks=x, ylim=(0, 1.15), ylabel="Score", title=heading)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    save(fig, path)


def plot_feature_importance(importance: pd.DataFrame, path: Path) -> None:
    models = list(dict.fromkeys(importance["model"]))
    if not models:
        print("feature_importance.csv is empty, skipping feature importance.")
        return
    n_features = importance.groupby("model")["feature"].nunique().max()
    height = max(3, 0.28 * n_features + 1)
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), height), squeeze=False)
    for ax, model in zip(axes[0], models):
        rows = importance[importance["model"] == model]
        rows = rows.reindex(rows["importance"].abs().sort_values().index)
        values = rows["importance"].to_numpy()
        signed = (values < 0).any()
        colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in values] if signed else POSITIVE_COLOR
        ax.barh(rows["feature"], values, color=colors)
        if signed:
            ax.axvline(0, color="k", lw=0.8)
            ax.set_xlabel("Coefficient (blue positive, red negative)")
        else:
            ax.set_xlabel("Importance")
        ax.set_title(model)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Feature importance")
    fig.tight_layout()
    save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="path to config.json (default: project config.json)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = results_paths(cfg)
    paths["figures"].mkdir(parents=True, exist_ok=True)

    with open(paths["metrics"], encoding="utf-8") as f:
        metrics = json.load(f)
    scores = pd.read_csv(paths["scores"])
    importance = pd.read_csv(paths["importance"])

    if "label" not in scores.columns:
        sys.exit(f"{paths['scores']} has no 'label' column. Re-run `python src/train.py`.")
    models = score_columns(scores, metrics)
    if not models:
        sys.exit(f"{paths['scores']} has no model score columns. Re-run `python src/train.py`.")
    if scores["label"].nunique() < 2:
        sys.exit(f"{paths['scores']} needs both positive and negative labels to draw ROC/PR curves.")

    title = edge_label(metrics)
    plot_roc(scores, models, title, paths["figures"] / "roc.png")
    plot_pr(scores, models, title, paths["figures"] / "pr.png")
    plot_model_comparison(metrics, title, paths["figures"] / "model_comparison.png")
    plot_feature_importance(importance, paths["figures"] / "feature_importance.png")


if __name__ == "__main__":
    main()
