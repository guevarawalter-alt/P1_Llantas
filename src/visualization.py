# -*- coding: utf-8 -*-
"""
visualization.py — Funciones de visualización: curvas, matrices, ROC, ablación, distribución.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

from src.config import RESULTS_DIR


# ── Curvas de entrenamiento ───────────────────────────────────────────────────

def plot_training_curves(
    hist: Dict[str, List[float]],
    tag: str,
    out_dir: Path,
) -> None:
    """Guarda curvas de pérdida y accuracy/F1."""
    ep_range = range(1, len(hist["tr_loss"]) + 1)
    best_epoch = int(np.argmin(hist["vl_loss"])) + 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(ep_range, hist["tr_loss"], label="Train", linewidth=1.5)
    axes[0].plot(ep_range, hist["vl_loss"], label="Val",   linewidth=1.5)
    axes[0].axvline(best_epoch, color="green", linestyle="--", alpha=0.6,
                    label=f"Mejor época ({best_epoch})")
    axes[0].set_title(f"Pérdida — {tag}", fontweight="bold")
    axes[0].set_xlabel("Época"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep_range, hist["tr_acc"],  label="Train acc", linewidth=1.5)
    axes[1].plot(ep_range, hist["vl_acc"],  label="Val acc",   linewidth=1.5)
    axes[1].plot(ep_range, hist["vl_f1"],   label="Val F1",    linewidth=1.5, linestyle="--")
    axes[1].set_title(f"Accuracy & F1 — {tag}", fontweight="bold")
    axes[1].set_xlabel("Época"); axes[1].set_ylabel("Valor")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=130, bbox_inches="tight")
    plt.close()


# ── Matriz de confusión ───────────────────────────────────────────────────────

def plot_confusion_matrix(
    labels: List[int],
    preds: List[int],
    class_names: List[str],
    tag: str,
    out_dir: Path,
) -> None:
    """Guarda la matriz de confusión como heatmap."""
    cm_arr = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_arr, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusión — {tag}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=130, bbox_inches="tight")
    plt.close()


# ── Distribución de clases ────────────────────────────────────────────────────

def plot_class_distribution(
    data_dir: Path,
    results_dir: Path = RESULTS_DIR,
    splits: Tuple[str, ...] = ("train", "val", "test"),
) -> Dict:
    """Analiza y grafica la distribución de clases por split."""
    from torchvision import datasets, transforms
    from src.config import IMG_SIZE

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    palette = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
    report  = {}
    fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 4))

    for ax, split in zip(axes, splits):
        path = data_dir / split
        if not path.exists():
            continue
        ds     = datasets.ImageFolder(str(path), tf)
        counts = np.bincount(ds.targets)
        classes = ds.classes
        ratio   = counts.max() / counts.min()
        report[split] = {
            "classes": classes,
            "counts":  counts.tolist(),
            "ratio":   round(float(ratio), 2),
        }
        bars = ax.bar(classes, counts, color=palette[: len(classes)],
                      edgecolor="white", linewidth=1.5)
        for bar, cnt in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                f"{cnt}\n({100 * cnt / counts.sum():.0f}%)",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )
        ax.set_title(f"{split.capitalize()}\nRatio desbalance: {ratio:.1f}:1",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Imágenes")
        ax.set_ylim(0, counts.max() * 1.2)
        ax.grid(axis="y", alpha=0.3)

        print(f"[{split}] {dict(zip(classes, counts))} | Ratio: {ratio:.2f}:1")
        if ratio > 2:
            print(f"  ⚠️  Desbalance significativo → usar WeightedSampler + FocalLoss")

    plt.suptitle("Distribución de Clases por Split", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_dir = results_dir / "imbalance"
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    return report


# ── Curvas ROC y Precision-Recall comparativas ────────────────────────────────

def plot_roc_pr_curves(
    all_results: Dict,
    results_dir: Path = RESULTS_DIR,
) -> None:
    """Grafica curvas ROC y Precision-Recall para todos los modelos."""
    colors_map = {
        "CustomCNN (scratch)":  "#F44336",
        "ResNet-50 (FT)":       "#2196F3",
        "EfficientNet-B0 (FT)": "#4CAF50",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, data in all_results.items():
        color = colors_map.get(name, "#9E9E9E")
        fpr, tpr, _ = roc_curve(data["labels"], data["probs"])
        auc_sc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc_sc:.4f})", color=color, lw=2)

        prec, rec, _ = precision_recall_curve(data["labels"], data["probs"])
        ap = average_precision_score(data["labels"], data["probs"])
        axes[1].plot(rec, prec, label=f"{name} (AP={ap:.4f})", color=color, lw=2)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("Curvas ROC", fontweight="bold")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Curvas Precision-Recall", fontweight="bold")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_dir = results_dir / "comparison"
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / "roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_metrics_table(all_results: Dict, results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Guarda y devuelve la tabla comparativa de métricas."""
    rows = []
    for name, data in all_results.items():
        m = data["metrics"]
        rows.append({
            "Modelo":    name,
            "Accuracy":  f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall":    f"{m['recall']:.4f}",
            "F1-Score":  f"{m['f1']:.4f}",
            "AUC-ROC":   f"{m['auc_roc']:.4f}",
        })
    df = pd.DataFrame(rows).set_index("Modelo")
    print("\n" + "=" * 60)
    print("  COMPARACIÓN FINAL — CONJUNTO DE PRUEBA")
    print("=" * 60)
    print(df.to_string())
    out_dir = results_dir / "comparison"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "metrics_table.csv")
    return df


# ── Ablación ─────────────────────────────────────────────────────────────────

def plot_ablation(
    ablation_results: List[Dict],
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """Grafica y guarda el estudio de ablación."""
    df_abl = pd.DataFrame(ablation_results).set_index("Variante")
    print("\n" + "=" * 50)
    print("  RESULTADOS DE ABLACIÓN")
    print("=" * 50)
    print(df_abl.to_string())

    groups = [
        ("Augmentación",  ["A_aug_ON", "A_aug_OFF"]),
        ("Loss Function",  ["B_focal", "B_ce_weights"]),
        ("Class Sampler",  ["C_sampler_ON", "C_sampler_OFF"]),
        ("Learning Rate",  ["D_lr_1e3", "D_lr_1e4", "D_lr_5e5"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for ax, (grp_name, variants) in zip(axes, groups):
        sub = df_abl.loc[variants]
        x   = np.arange(len(variants))
        w   = 0.25
        for i, (col, c) in enumerate(
            zip(["F1", "Recall", "AUC"], ["#2196F3", "#4CAF50", "#FF5722"])
        ):
            bars = ax.bar(x + i * w, sub[col], w, label=col, color=c, alpha=0.85)
            for b in bars:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.005,
                    f"{b.get_height():.3f}",
                    ha="center", va="bottom", fontsize=7,
                )
        ax.set_xticks(x + w)
        ax.set_xticklabels(variants, rotation=15, ha="right", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(grp_name, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Estudio de Ablación — ResNet-50", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_dir = results_dir / "ablation"
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / "ablation_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    df_abl.to_csv(out_dir / "ablation_results.csv")
    return df_abl


# ── Mitigación del desbalance ─────────────────────────────────────────────────

def plot_mitigation(
    mit_results: List[Dict],
    results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    """Grafica y guarda la comparación de estrategias de mitigación."""
    df_mit = pd.DataFrame(mit_results).set_index("Estrategia")
    print("\n" + "=" * 60)
    print("  COMPARACIÓN DE ESTRATEGIAS DE MITIGACIÓN")
    print("=" * 60)
    print(df_mit.to_string())

    metrics_cols = ["Recall (damaged)", "F1", "AUC-ROC"]
    cols_c = ["#F44336", "#2196F3", "#4CAF50"]
    names  = df_mit.index.tolist()
    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (mc, c) in enumerate(zip(metrics_cols, cols_c)):
        bars = ax.bar(x + i * w, df_mit[mc], w, label=mc, color=c, alpha=0.85)
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.004,
                f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=8,
            )
    ax.set_xticks(x + w)
    ax.set_xticklabels(names, rotation=10, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title(
        "Comparación de Estrategias de Mitigación del Desbalance",
        fontsize=12, fontweight="bold",
    )
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_dir = results_dir / "imbalance"
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / "mitigation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    df_mit.to_csv(out_dir / "mitigation_comparison.csv")
    return df_mit
