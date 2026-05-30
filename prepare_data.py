#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_data.py — Descarga y prepara el dataset de Kaggle para el proyecto.

Uso:
    # Descargar automáticamente con Kaggle API:
    python prepare_data.py

    # Usar un directorio con los datos ya descargados:
    python prepare_data.py --raw-dir data/raw/extracted
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Prepara el dataset de llantas desde Kaggle."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help=(
            "Directorio con los datos crudos ya descargados y extraídos. "
            "Si no se indica, se descarga automáticamente desde Kaggle."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directorio de salida para el dataset split. Por defecto: data/tire_texture/",
    )
    args = parser.parse_args()

    # Importar aquí para que los errores de importación sean claros
    from src.config import DATA_DIR, RAW_DATA_DIR
    from src.dataset import download_dataset, collect_images, split_and_copy

    out_dir = args.out_dir or DATA_DIR

    # ── Si ya está preparado, salir ───────────────────────────────────────────
    if out_dir.exists() and any(out_dir.rglob("*.jpg")):
        print(f"✅ Dataset ya preparado en '{out_dir}'. No se hace nada.")
        print("   Elimina esa carpeta si quieres volver a prepararlo.")
        return

    # ── Obtener ruta a los datos crudos ──────────────────────────────────────
    if args.raw_dir is not None:
        raw_dir = args.raw_dir
        if not raw_dir.is_dir():
            print(f"❌ El directorio '{raw_dir}' no existe.", file=sys.stderr)
            sys.exit(1)
        print(f"📂 Usando datos crudos de: {raw_dir}")
    else:
        print("📥 Iniciando descarga desde Kaggle...")
        raw_dir = download_dataset(RAW_DATA_DIR)

    # ── Recolectar imágenes ───────────────────────────────────────────────────
    print("\n🔍 Escaneando imágenes...")
    collected = collect_images(raw_dir)
    total = sum(len(v) for v in collected.values())
    print(f"\n   Total imágenes encontradas: {total}")

    if total == 0:
        print(
            "❌ No se encontraron imágenes. Verifica que el directorio de datos "
            "contenga las categorías correctas.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Split y copia ─────────────────────────────────────────────────────────
    print(f"\n✂️  Dividiendo en train / val / test → '{out_dir}'...")
    stats = split_and_copy(collected, out_dir)

    print("\n✅ Dataset preparado correctamente.")
    print(f"   Ruta: {out_dir.resolve()}")
    print(f"   Estadísticas finales: {stats}")


if __name__ == "__main__":
    main()
