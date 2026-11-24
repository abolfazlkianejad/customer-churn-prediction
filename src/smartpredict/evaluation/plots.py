"""Evaluation visualizations saved under ``artifacts/plots``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from smartpredict.evaluation.metrics import (
    EvaluationOutput,
    precision_recall_points,
    roc_points,
)


def _load_plotting() -> tuple[Any, Any, Any]:
    """Import plotting libraries only when a figure is actually generated."""
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    return plt, np, sns


def _save_figure(figure: Any, path: Path, plt: Any) -> None:
    """Save and close one matplotlib figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(output: EvaluationOutput, path: Path) -> Path:
    """Write a labeled confusion-matrix heatmap."""
    plt, np, sns = _load_plotting()
    matrix = np.asarray(output.result.confusion)
    figure, axes = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted negative", "Predicted positive"],
        yticklabels=["Actual negative", "Actual positive"],
        ax=axes,
    )
    axes.set_title(
        f"Confusion Matrix — {output.result.model_name} ({output.result.split})"
    )
    axes.set_xlabel("Predicted label")
    axes.set_ylabel("Actual label")
    _save_figure(figure, path, plt)
    return path


def plot_roc_curve(output: EvaluationOutput, path: Path) -> Path:
    """Write a ROC curve, or an explanatory figure when scores are absent."""
    plt, _, _ = _load_plotting()
    figure, axes = plt.subplots(figsize=(6, 5))
    axes.plot([0, 1], [0, 1], "k--", label="Random classifier")
    if output.y_prob is not None:
        points = roc_points(output.y_true, output.y_prob)
        if points is not None:
            false_positive, true_positive = points
            axes.plot(
                false_positive,
                true_positive,
                linewidth=2,
                label=f"Model (AUC = {output.result.metrics.get('roc_auc')!s})",
            )
    else:
        axes.text(
            0.5,
            0.5,
            "ROC curve unavailable: no probability predictions",
            ha="center",
            va="center",
            transform=axes.transAxes,
        )
    axes.set_title(f"ROC Curve — {output.result.model_name} ({output.result.split})")
    axes.set_xlabel("False positive rate")
    axes.set_ylabel("True positive rate")
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.legend(loc="lower right")
    _save_figure(figure, path, plt)
    return path


def plot_precision_recall_curve(output: EvaluationOutput, path: Path) -> Path:
    """Write a precision-recall curve with a graceful probability fallback."""
    plt, _, _ = _load_plotting()
    figure, axes = plt.subplots(figsize=(6, 5))
    baseline = float(output.y_true.mean()) if output.y_true.size else 0.0
    axes.axhline(baseline, color="k", linestyle="--", label="Baseline")
    if output.y_prob is not None:
        points = precision_recall_points(output.y_true, output.y_prob)
        if points is not None:
            precision, recall = points
            axes.plot(
                recall,
                precision,
                linewidth=2,
                label=(
                    "Model (AP = "
                    f"{output.result.metrics.get('average_precision')!s})"
                ),
            )
    else:
        axes.text(
            0.5,
            0.5,
            "Precision-recall curve unavailable: no probability predictions",
            ha="center",
            va="center",
            transform=axes.transAxes,
        )
    axes.set_title(
        f"Precision-Recall Curve — {output.result.model_name} "
        f"({output.result.split})"
    )
    axes.set_xlabel("Recall")
    axes.set_ylabel("Precision")
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.05)
    axes.legend(loc="lower left")
    _save_figure(figure, path, plt)
    return path


def generate_evaluation_plots(
    output: EvaluationOutput,
    plots_dir: Path,
    *,
    confusion_file: str = "confusion_matrix.png",
    roc_file: str = "roc_curve.png",
    precision_recall_file: str = "precision_recall_curve.png",
) -> dict[str, str]:
    """Generate all required evaluation plots for the final test result."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "confusion_matrix": plots_dir / confusion_file,
        "roc_curve": plots_dir / roc_file,
        "precision_recall_curve": plots_dir / precision_recall_file,
    }
    return {
        "confusion_matrix": str(
            plot_confusion_matrix(output, paths["confusion_matrix"])
        ),
        "roc_curve": str(plot_roc_curve(output, paths["roc_curve"])),
        "precision_recall_curve": str(
            plot_precision_recall_curve(output, paths["precision_recall_curve"])
        ),
    }
