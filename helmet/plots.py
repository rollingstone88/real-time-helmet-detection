"""Matplotlib figures for training curves and evaluation reports."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_training_history(history: List[dict], path: str | Path) -> None:
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    loss_keys = [k for k in history[0] if k.startswith("train_")]
    for k in loss_keys:
        axes[0].plot(epochs, [h[k] for h in history], label=k.replace("train_", ""))
    axes[0].set_title("Training losses"); axes[0].set_xlabel("epoch"); axes[0].legend(fontsize=8)
    for k, label in (("val_map50", "mAP@0.5"), ("val_map50_95", "mAP@0.5:0.95")):
        axes[1].plot(epochs, [h.get(k, np.nan) for h in history], marker="o", label=label)
    axes[1].set_title("Validation mAP"); axes[1].set_xlabel("epoch"); axes[1].set_ylim(0, 1); axes[1].legend()
    for k, label in (("val_precision", "precision"), ("val_recall", "recall"), ("val_f1", "F1")):
        axes[2].plot(epochs, [h.get(k, np.nan) for h in history], marker="o", label=label)
    axes[2].set_title("Validation P / R / F1"); axes[2].set_xlabel("epoch"); axes[2].set_ylim(0, 1)
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_pr_curves(curves: Dict[str, dict], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, c in curves.items():
        ap = c["ap50"]
        label = f"{name} (AP50={ap:.3f})" if ap == ap else f"{name} (no GT)"
        ax.plot(c["recall"], c["precision"], label=label)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall @ IoU 0.5")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_f1_curve(thresholds, precision, recall, f1, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(thresholds, precision, label="precision")
    ax.plot(thresholds, recall, label="recall")
    ax.plot(thresholds, f1, label="F1", linewidth=2)
    if len(f1) and np.max(f1) > 0:
        i = int(np.argmax(f1))
        ax.axvline(thresholds[i], color="gray", linestyle="--")
        ax.set_title(f"Best F1 {f1[i]:.3f} at confidence {thresholds[i]:.2f}")
    ax.set_xlabel("Confidence threshold"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_confusion_matrix(cm: Sequence[Sequence[int]], labels: Sequence[str], path: str | Path) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2, 1.4 * len(labels) + 1.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth"); ax.set_title("Confusion matrix (IoU 0.5)")
    vmax = cm.max() if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if vmax and cm[i, j] > vmax / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
