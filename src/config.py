# -*- coding: utf-8 -*-
"""
config.py — Constantes y configuración global del proyecto.
Todas las rutas son relativas a la raíz del proyecto.
"""

import random
from pathlib import Path

import numpy as np
import torch

# ── Rutas del proyecto ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "tire_texture"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR  = PROJECT_ROOT / "results"

# ── Hiperparámetros por defecto ───────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 3
PATIENCE   = 7
SEED       = 42
LR         = 1e-4

# ── Clases ────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["good_tire", "damaged_tire"]
NUM_CLASSES = len(CLASS_NAMES)

# ── Mapeo categorías Kaggle → binario ─────────────────────────────────────────
CATEGORY_MAP = {
    "good":       "good_tire",
    "normal":     "good_tire",
    "cracked":    "damaged_tire",
    "broken":     "damaged_tire",
    "slash_cut":  "damaged_tire",
    "bulge":      "damaged_tire",
    "worn":       "damaged_tire",
}

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ── Normalización ImageNet ────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def set_seed(seed: int = SEED) -> None:
    """Fija la semilla para reproducibilidad completa."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Devuelve CUDA si está disponible, de lo contrario CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def print_device_info(device: torch.device) -> None:
    print(f"💻 Dispositivo : {device}")
    if device.type == "cuda":
        print(f"🖥️  GPU         : {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"💾 VRAM        : {mem_gb:.1f} GB")
    else:
        print("⚠️  Sin GPU detectada — el entrenamiento será lento.")


# Instancia de device disponible como constante de módulo
DEVICE: torch.device = get_device()
