# -*- coding: utf-8 -*-
"""
train.py — Funciones de entrenamiento, evaluación y pipeline completo run_training.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
)

from src.config import DEVICE, EPOCHS, PATIENCE, LR, RESULTS_DIR, NUM_CLASSES, set_seed
from src.dataset import get_dataloaders
from src.models import build_model, count_parameters
from src.losses import build_criterion
from src.visualization import plot_training_curves, plot_confusion_matrix


# ── Early Stopping ─────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Para el entrenamiento si la pérdida de validación no mejora en `patience` épocas.
    Guarda el mejor checkpoint automáticamente.
    """

    def __init__(self, patience: int = PATIENCE, delta: float = 1e-4, path: str = "best.pth"):
        self.patience   = patience
        self.delta      = delta
        self.path       = path
        self.counter    = 0
        self.best       = np.inf
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best - self.delta:
            self.best    = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ── Una época de entrenamiento ────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler=None,
) -> Tuple[float, float]:
    """
    Ejecuta una época de entrenamiento.

    Returns:
        (loss_promedio, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    n          = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += imgs.size(0)

    return total_loss / n, correct / n


# ── Evaluación ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[Dict[str, float], List[int], List[int], List[float]]:
    """
    Evalúa el modelo en el loader dado.

    Returns:
        (métricas_dict, predicciones, etiquetas_reales, probabilidades_clase_positiva)
    """
    model.eval()
    total_loss = 0.0
    preds:  List[int]   = []
    labels: List[int]   = []
    probs:  List[float] = []

    for imgs, lbs in loader:
        imgs, lbs = imgs.to(device), lbs.to(device)
        logits = model(imgs)
        loss   = criterion(logits, lbs)
        total_loss += loss.item() * imgs.size(0)

        prob = torch.softmax(logits, dim=1)[:, 1]
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(lbs.cpu().numpy())
        probs.extend(prob.cpu().numpy())

    n = len(labels)
    metrics = {
        "loss":      total_loss / n,
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "auc_roc":   roc_auc_score(labels, probs),
    }
    return metrics, preds, labels, probs


# ── Pipeline completo ─────────────────────────────────────────────────────────

def run_training(
    arch: str,
    device: torch.device,
    data_dir: Path = None,
    results_dir: Path = RESULTS_DIR,
    loss_fn: str = "focal",
    augment: bool = True,
    use_sampler: bool = True,
    lr: float = LR,
    batch_size: int = 32,
    epochs: int = EPOCHS,
    tag: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, float], List[int], List[float], Dict[str, List]]:
    """
    Entrena un modelo completo: train → val (early stopping) → test.

    Args:
        arch        : 'custom_cnn' | 'resnet50' | 'efficientnet_b0'
        device      : torch.device
        data_dir    : directorio raíz del dataset (con train/val/test)
        results_dir : carpeta raíz de resultados
        loss_fn     : 'focal' | 'ce'
        augment     : usar data augmentation en entrenamiento
        use_sampler : usar WeightedRandomSampler
        lr          : learning rate
        batch_size  : tamaño del batch
        epochs      : número máximo de épocas
        tag         : nombre del experimento (subcarpeta en results_dir)

    Returns:
        (model, test_metrics, test_labels, test_probs, history)
    """
    from src.config import DATA_DIR
    if data_dir is None:
        data_dir = DATA_DIR

    set_seed()
    tag = tag or arch
    out_dir = results_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # DataLoaders
    loader_tr, loader_vl, loader_ts, classes, class_weights = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        augment=augment,
        use_weighted_sampler=use_sampler,
    )

    # Modelo
    model = build_model(arch, device=device)
    tr_params = count_parameters(model)

    print(f"\n{'='*60}")
    print(f" {arch} | loss={loss_fn} | aug={augment} | sampler={use_sampler} | lr={lr}")
    print(f" Parámetros entrenables : {tr_params:,}")
    print(f"{'='*60}")

    # Pérdida, optimizer, scheduler
    criterion = build_criterion(loss_fn, class_weights, device, NUM_CLASSES)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    es = EarlyStopping(PATIENCE, path=str(out_dir / "best.pth"))

    # Historial
    hist: Dict[str, List] = {
        "tr_loss": [], "vl_loss": [], "tr_acc": [], "vl_acc": [], "vl_f1": [],
    }

    for ep in range(1, epochs + 1):
        tl, ta       = train_epoch(model, loader_tr, optimizer, criterion, device, scaler)
        vm, _, _, _  = evaluate(model, loader_vl, criterion, device)
        scheduler.step()

        hist["tr_loss"].append(tl)
        hist["vl_loss"].append(vm["loss"])
        hist["tr_acc"].append(ta)
        hist["vl_acc"].append(vm["accuracy"])
        hist["vl_f1"].append(vm["f1"])

        print(
            f"Ep {ep:02d}/{epochs} | tr_loss={tl:.4f} tr_acc={ta:.4f} "
            f"| vl_f1={vm['f1']:.4f} vl_auc={vm['auc_roc']:.4f}",
            end=" ",
        )
        es(vm["loss"], model)
        print("✓" if not es.early_stop else "⏹")
        if es.early_stop:
            print(f"   Early stopping en época {ep}.")
            break

    # Cargar mejor checkpoint y evaluar en test
    model.load_state_dict(torch.load(out_dir / "best.pth", map_location=device, weights_only=True))
    test_m, test_p, test_l, test_pr = evaluate(model, loader_ts, criterion, device)

    print(f"\n[TEST] {tag}")
    print(classification_report(test_l, test_p, target_names=classes, digits=4))
    print(f"  AUC-ROC: {test_m['auc_roc']:.4f}")

    # Guardar métricas
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({k: round(float(v), 4) for k, v in test_m.items()}, f, indent=2)

    # Guardar gráficas
    plot_training_curves(hist, tag, out_dir)
    plot_confusion_matrix(test_l, test_p, classes, tag, out_dir)

    return model, test_m, test_l, test_pr, hist
