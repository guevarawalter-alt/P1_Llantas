# -*- coding: utf-8 -*-
"""
models.py — Definición de las tres arquitecturas CNN:
  · CustomCNN  — diseñada desde cero con ResBlocks y SEBlocks
  · ResNet-50  — fine-tuning desde ImageNet
  · EfficientNet-B0 — fine-tuning desde ImageNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.config import NUM_CLASSES


# ════════════════════════════════════════════════════════════════════
#  Módulos base
# ════════════════════════════════════════════════════════════════════

class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, ks: int = 3, stride: int = 1, pad: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ks, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    """Bloque residual de 2 capas con shortcut de proyección opcional."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: atención adaptativa de canal (Hu et al., CVPR 2018)."""

    def __init__(self, ch: int, r: int = 16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x).unsqueeze(-1).unsqueeze(-1)


# ════════════════════════════════════════════════════════════════════
#  CustomCNN — arquitectura desde cero
# ════════════════════════════════════════════════════════════════════

class CustomCNN(nn.Module):
    """
    CNN para reconocimiento de textura de llantas.
    4 etapas: 64→128→256→512 canales con ResBlocks + SEBlocks.
    ~6.5M parámetros entrenables.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        # Stem: 224 → 56
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage1 = nn.Sequential(ResBlock(32, 64),            SEBlock(64))
        self.stage2 = nn.Sequential(ResBlock(64, 128, 2),  ResBlock(128, 128), SEBlock(128))
        self.stage3 = nn.Sequential(ResBlock(128, 256, 2), ResBlock(256, 256), SEBlock(256))
        self.stage4 = nn.Sequential(ResBlock(256, 512, 2),        SEBlock(512))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
            x = stage(x)
        return self.classifier(self.gap(x))


# ════════════════════════════════════════════════════════════════════
#  Factory — build_model
# ════════════════════════════════════════════════════════════════════

def build_model(
    arch: str,
    num_classes: int = NUM_CLASSES,
    freeze_backbone: bool = True,
    device: torch.device = None,
) -> nn.Module:
    """
    Construye y devuelve el modelo indicado, enviándolo al device.

    Args:
        arch            : 'custom_cnn' | 'resnet50' | 'efficientnet_b0'
        num_classes     : número de clases de salida
        freeze_backbone : si True congela el backbone (solo aplica a modelos preentrenados)
        device          : torch.device destino (None → CPU)

    Returns:
        Modelo nn.Module listo para entrenar.
    """
    if device is None:
        device = torch.device("cpu")

    if arch == "custom_cnn":
        model = CustomCNN(num_classes)

    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for name, param in model.named_parameters():
                if "layer4" not in name and "fc" not in name:
                    param.requires_grad_(False)
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for name, param in model.named_parameters():
                if not any(x in name for x in ("features.7", "features.8", "classifier")):
                    param.requires_grad_(False)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes),
        )

    else:
        raise ValueError(
            f"Arquitectura desconocida: '{arch}'. "
            "Opciones: 'custom_cnn', 'resnet50', 'efficientnet_b0'."
        )

    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """Devuelve el número de parámetros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
