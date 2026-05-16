"""
Visualize results from the deepfake detection experiments.
Usage: uv run python src/data/visualize_results.py <metrics_json>
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

CLASSES = ["REAL", "SYNTHETIC", "TAMPERED"]


def load_metrics(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    with open(p) as f:
        return json.load(f)


def plot_conf_matrix(ax, conf_matrix, title="Confusion Matrix"):
    cm = np.array(conf_matrix)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    annot = np.array([
        [f"{cm[i,j]:,}\n({cm_pct[i,j]:.1f}%)" for j in range(3)]
        for i in range(3)
    ])

    sns.heatmap(
        cm_pct,
        ax=ax,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        linewidths=0.5,
        cbar_kws={"label": "% of true class"},
        vmin=0,
        vmax=100,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)


def plot_metrics_per_class(ax, report, title="Per-class Metrics"):
    metrics = ["precision", "recall", "f1-score"]
    labels = ["Precision", "Recall", "F1-Score"]
    x = np.arange(len(CLASSES))
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        values = [report[c][metric] * 100 for c in CLASSES]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASSES)
    min_val = min(report[c][m] * 100 for c in CLASSES for m in ["precision", "recall", "f1-score"])
    ax.set_ylim(max(0, min_val - 10), 102)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)


def plot_summary(ax, metrics_dict, model_name):
    accuracy = metrics_dict["accuracy"] * 100
    f1_macro = metrics_dict["f1_macro"] * 100

    ax.axis("off")
    text = (
        f"Model: {model_name}\n\n"
        f"Accuracy:  {accuracy:.2f}%\n"
        f"F1-Macro:  {f1_macro:.2f}%\n\n"
        f"F1 per class:\n"
        + "\n".join(
            f"  {c}: {metrics_dict['report'][c]['f1-score']*100:.2f}%"
            for c in CLASSES
        )
    )
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        fontsize=12,
        va="center",
        ha="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f0f4f8", edgecolor="#aab4c4", linewidth=1.5),
    )
    ax.set_title("Summary", fontsize=13, fontweight="bold", pad=12)


def generate_visualization(file: str, save: bool = True):
    data = load_metrics(file)
    name = data["name"]
    metrics = data["metrics"]
    report = metrics["report"]
    conf_matrix = metrics["conf_matrix"]

    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    ax_cm = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])
    ax_summary = fig.add_subplot(gs[2])

    plot_conf_matrix(ax_cm, conf_matrix, title=f"Confusion Matrix\n{name}")
    plot_metrics_per_class(ax_bar, report, title=f"Per-class Metrics\n{name}")
    plot_summary(ax_summary, metrics, name)

    fig.suptitle("Results — Deepfake Detection", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        out_path = Path(file).parent / "visualization.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved to: {out_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to the metrics JSON (e.g. experiments/cnn_baseline_01/metrics_20260510_193136.json)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    generate_visualization(args.file, save=not args.no_save)
