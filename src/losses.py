# -*- coding: utf-8 -*-
"""
losses.py — Funciones de pérdida personalizadas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss para clasificación con desbalance de clases.
    FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)

    Referencia: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce    = F.cross_entropy(inputs, targets, reduction="none")
        p_t   = torch.exp(-ce)
        alpha_t = self.alpha * targets.float() + (1 - self.alpha) * (1 - targets.float())
        loss  = alpha_t * (1 - p_t) ** self.gamma * ce
        return loss.mean()


def build_criterion(
    loss_fn: str,
    class_weights=None,
    device: torch.device = None,
    num_classes: int = 2,
) -> nn.Module:
    """
    Construye la función de pérdida según el nombre indicado.

    Args:
        loss_fn       : 'focal' | 'ce'
        class_weights : array de pesos por clase (para CrossEntropy ponderada)
        device        : torch.device
        num_classes   : número de clases

    Returns:
        nn.Module criterion listo para usar.
    """
    if device is None:
        device = torch.device("cpu")

    if loss_fn == "focal":
        return FocalLoss(alpha=0.25, gamma=2.0)

    elif loss_fn == "ce":
        if class_weights is not None:
            import numpy as np
            cw = class_weights / class_weights.sum() * num_classes
            weight = torch.tensor(cw, dtype=torch.float).to(device)
            return nn.CrossEntropyLoss(weight=weight)
        return nn.CrossEntropyLoss()

    else:
        raise ValueError(f"Función de pérdida desconocida: '{loss_fn}'. Opciones: 'focal', 'ce'.")
