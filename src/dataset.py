# -*- coding: utf-8 -*-
"""
dataset.py — Descarga del dataset de Kaggle, preparación (split train/val/test)
y construcción de DataLoaders con augmentación y WeightedRandomSampler.
"""

import os
import shutil
import random
import zipfile
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_DIR, RAW_DATA_DIR, CATEGORY_MAP, VALID_EXTS,
    IMG_SIZE, BATCH_SIZE, SEED, IMAGENET_MEAN, IMAGENET_STD,
)


# ── Descarga del dataset ───────────────────────────────────────────────────────

def download_dataset(raw_dir: Path = RAW_DATA_DIR) -> Path:
    """
    Descarga el dataset de Kaggle usando la CLI.
    Requiere ~/.kaggle/kaggle.json configurado previamente.

    Args:
        raw_dir: Carpeta donde se descargará el zip y se extraerá.

    Returns:
        Ruta al directorio con los datos crudos (antes del split).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_slug = "jehanbhathena/tire-texture-image-recognition"
    zip_path = raw_dir / "tire-texture-image-recognition.zip"

    if not zip_path.exists():
        print("📥 Descargando dataset de Kaggle...")
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(raw_dir)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Error al descargar el dataset:\n{result.stderr}\n"
                "Asegúrate de que ~/.kaggle/kaggle.json esté configurado.\n"
                "Descárgalo desde https://www.kaggle.com/settings → API → Create New Token"
            )
        print("✅ Descarga completada.")
    else:
        print(f"✅ Zip ya existe en {zip_path}, omitiendo descarga.")

    # Extraer
    extract_dir = raw_dir / "extracted"
    if not extract_dir.exists():
        print("📦 Extrayendo zip...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        print("✅ Extracción completada.")
    else:
        print(f"✅ Dataset ya extraído en {extract_dir}.")

    return extract_dir


# ── Recolección de imágenes ────────────────────────────────────────────────────

def collect_images(kaggle_dir: Path) -> Dict[str, List[Path]]:
    """
    Recorre el directorio extraído de Kaggle y agrupa las imágenes
    en las dos clases binarias (good_tire / damaged_tire).

    Args:
        kaggle_dir: Directorio raíz de los datos crudos extraídos.

    Returns:
        Dict {clase_binaria: [lista de rutas de imagen]}
    """
    if not kaggle_dir.is_dir():
        raise FileNotFoundError(
            f"El directorio '{kaggle_dir}' no existe.\n"
            "Ejecuta primero: python prepare_data.py"
        )

    collected: Dict[str, List[Path]] = defaultdict(list)

    # Buscar directorios training_data / testing_data
    data_roots = [
        entry for entry in kaggle_dir.rglob("*")
        if entry.is_dir() and entry.name in ("training_data", "testing_data", "Tire Textures")
    ]
    if not data_roots:
        # Fallback: buscar categorías directamente
        data_roots = [kaggle_dir]

    for data_root in sorted(data_roots):
        print(f"  📂 Explorando: {data_root.relative_to(kaggle_dir.parent)}/")
        for subdir in sorted(data_root.iterdir()):
            if not subdir.is_dir():
                continue
            cat = subdir.name.lower()
            cls = CATEGORY_MAP.get(cat)
            if cls is None:
                # Coincidencia parcial
                for k, v in CATEGORY_MAP.items():
                    if k in cat:
                        cls = v
                        break
            if cls is None:
                print(f"  ⚠️  Categoría no mapeada: '{cat}' — omitida.")
                continue
            imgs = [p for p in subdir.rglob("*") if p.suffix.lower() in VALID_EXTS]
            collected[cls].extend(imgs)
            print(f"  {cat:20s} → {cls:15s}: {len(imgs):5d} imágenes")

    return dict(collected)


# ── Split y copia ──────────────────────────────────────────────────────────────

def split_and_copy(
    collected: Dict[str, List[Path]],
    out_dir: Path,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = SEED,
) -> Dict[str, Dict[str, int]]:
    """
    Divide las imágenes en train/val/test y las copia a la estructura
    esperada por torchvision.datasets.ImageFolder.

    Args:
        collected : salida de collect_images()
        out_dir   : destino del split (DATA_DIR por defecto)
        val       : fracción de validación
        test      : fracción de test
        seed      : semilla aleatoria

    Returns:
        Estadísticas {clase: {split: n}}
    """
    if not collected:
        raise ValueError("No se encontraron imágenes para dividir.")

    out_dir = Path(out_dir)
    for sp in ("train", "val", "test"):
        for cls in collected:
            (out_dir / sp / cls).mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Dict[str, int]] = {}
    for cls, paths in collected.items():
        random.shuffle(paths)
        tr, tmp = train_test_split(paths, test_size=val + test, random_state=seed)
        vl, ts  = train_test_split(tmp, test_size=test / (val + test), random_state=seed)

        for sp, ps in [("train", tr), ("val", vl), ("test", ts)]:
            for src in ps:
                dst = out_dir / sp / cls / src.name
                if dst.exists():
                    dst = out_dir / sp / cls / f"{src.stem}_{id(src)}{src.suffix}"
                shutil.copy2(src, dst)

        stats[cls] = {"train": len(tr), "val": len(vl), "test": len(ts)}
        print(f"  {cls}: train={len(tr)}  val={len(vl)}  test={len(ts)}")

    return stats


def prepare_dataset(raw_dir: Path = RAW_DATA_DIR, out_dir: Path = DATA_DIR) -> None:
    """Pipeline completo: recolectar → dividir → copiar."""
    if out_dir.exists() and any(out_dir.rglob("*.jpg")):
        print(f"✅ Dataset ya preparado en {out_dir}.")
        return

    print("\n🗂️  Preparando dataset...")
    collected = collect_images(raw_dir)
    total = sum(len(v) for v in collected.values())
    print(f"\n   Total imágenes recolectadas: {total}")

    print("\n   Dividiendo en train / val / test...")
    split_and_copy(collected, out_dir)
    print(f"\n✅ Dataset preparado en {out_dir}.")


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Devuelve (transform_train, transform_val).

    Args:
        augment: Si True aplica augmentación en train.
    """
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    if augment:
        train_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            norm,
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            norm,
        ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        norm,
    ])

    return train_tf, val_tf


# ── DataLoaders ───────────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir: Path = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    augment: bool = True,
    use_weighted_sampler: bool = True,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], np.ndarray]:
    """
    Construye y devuelve los DataLoaders de train, val y test.

    Returns:
        (train_loader, val_loader, test_loader, class_names, class_weights)
    """
    train_tf, val_tf = get_transforms(augment)

    train_ds = datasets.ImageFolder(str(data_dir / "train"), train_tf)
    val_ds   = datasets.ImageFolder(str(data_dir / "val"),   val_tf)
    test_ds  = datasets.ImageFolder(str(data_dir / "test"),  val_tf)

    counts = np.bincount(train_ds.targets).astype(float)
    class_weights = 1.0 / counts

    if use_weighted_sampler:
        sample_weights = torch.tensor([class_weights[t] for t in train_ds.targets])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    classes = train_ds.classes
    print(f"   Clases       : {classes}")
    print(f"   Train        : {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"   Distribución : {dict(zip(classes, counts.astype(int)))}")

    return train_loader, val_loader, test_loader, classes, class_weights
