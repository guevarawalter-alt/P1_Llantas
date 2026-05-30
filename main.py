#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — Punto de entrada principal del proyecto de clasificación de llantas.

Ejecuta el pipeline completo: entrenamiento de los tres modelos, evaluación
comparativa, estudio de ablación y Grad-CAM.

Uso:
    python main.py                          # Pipeline completo
    python main.py --arch resnet50          # Solo un modelo
    python main.py --ablation              # Solo ablación
    python main.py --mitigation            # Solo mitigación de desbalance
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline de clasificación de llantas dañadas — RNAPD 2026-I"
    )
    parser.add_argument(
        "--arch",
        default="all",
        choices=["custom_cnn", "resnet50", "efficientnet_b0", "all"],
        help="Arquitectura a entrenar (default: all)",
    )
    parser.add_argument("--epochs",     type=int,   default=3,   help="Épocas de entrenamiento")
    parser.add_argument("--batch-size", type=int,   default=32,   help="Tamaño del batch")
    parser.add_argument("--lr",         type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--loss",
        default="focal",
        choices=["focal", "ce"],
        help="Función de pérdida (default: focal)",
    )
    parser.add_argument("--no-augment", action="store_true", help="Desactivar data augmentation")
    parser.add_argument("--no-sampler", action="store_true", help="Desactivar WeightedRandomSampler")
    parser.add_argument("--ablation",   action="store_true", help="Ejecutar estudio de ablación")
    parser.add_argument("--mitigation", action="store_true", help="Ejecutar análisis de mitigación")
    parser.add_argument("--seed",       type=int,   default=42,           help="Semilla aleatoria")
    parser.add_argument("--data-dir",   type=Path,  default=None,         help="Ruta al dataset preparado")
    parser.add_argument("--results-dir",type=Path,  default=Path("results"), help="Carpeta de resultados")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Importaciones principales ─────────────────────────────────────────────
    from src.config import (
        DATA_DIR, RESULTS_DIR, set_seed, get_device, print_device_info,
    )
    from src.train import run_training
    from src.dataset import get_dataloaders
    from src.visualization import (
        plot_class_distribution, plot_roc_pr_curves,
        save_metrics_table, plot_ablation, plot_mitigation,
    )

    set_seed(args.seed)
    device      = get_device()
    data_dir    = args.data_dir or DATA_DIR
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  🛞  Clasificación de Llantas Dañadas — RNAPD 2026-I")
    print("=" * 60)
    print_device_info(device)

    # ── Verificar que el dataset existe ──────────────────────────────────────
    if not data_dir.exists() or not any(data_dir.rglob("*.jpg")):
        print(
            f"\n❌ No se encontraron imágenes en '{data_dir}'.\n"
            "   Ejecuta primero:  python prepare_data.py",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Análisis de distribución de clases ───────────────────────────────────
    print("\n📊 Analizando distribución de clases...")
    plot_class_distribution(data_dir, results_dir)

    # ── Selección de arquitecturas a entrenar ─────────────────────────────────
    archs_to_train = (
        ["custom_cnn", "resnet50", "efficientnet_b0"]
        if args.arch == "all"
        else [args.arch]
    )

    all_results = {}
    arch_model_map = {}

    for arch in archs_to_train:
        model, metrics, labels, probs, hist = run_training(
            arch=arch,
            device=device,
            data_dir=data_dir,
            results_dir=results_dir,
            loss_fn=args.loss,
            augment=not args.no_augment,
            use_sampler=not args.no_sampler,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            tag=arch,
        )
        display_name = {
            "custom_cnn":     "CustomCNN (scratch)",
            "resnet50":       "ResNet-50 (FT)",
            "efficientnet_b0": "EfficientNet-B0 (FT)",
        }.get(arch, arch)

        all_results[display_name] = {
            "metrics": metrics,
            "labels":  labels,
            "probs":   probs,
        }
        arch_model_map[arch] = model

    # ── Evaluación comparativa ────────────────────────────────────────────────
    if len(all_results) > 1:
        print("\n📈 Generando evaluación comparativa...")
        plot_roc_pr_curves(all_results, results_dir)
        save_metrics_table(all_results, results_dir)

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    if arch_model_map:
        print("\n🔍 Generando visualizaciones Grad-CAM...")
        from src.gradcam import show_gradcam, analyze_failures
        from src.dataset import get_dataloaders

        _, _, test_loader, class_names, _ = get_dataloaders(
            data_dir=data_dir, batch_size=args.batch_size,
            augment=False, use_weighted_sampler=False,
        )

        # EfficientNet-B0 (si está disponible)
        if "efficientnet_b0" in arch_model_map:
            model_eff  = arch_model_map["efficientnet_b0"]
            target_eff = model_eff.features[8][0]
            show_gradcam(model_eff, target_eff, test_loader, class_names,
                         device, results_dir=results_dir)
            analyze_failures(model_eff, target_eff, test_loader, class_names,
                             device, results_dir=results_dir)

        # CustomCNN (si está disponible)
        if "custom_cnn" in arch_model_map:
            model_cnn  = arch_model_map["custom_cnn"]
            target_cnn = model_cnn.stage4[0].conv2
            show_gradcam(model_cnn, target_cnn, test_loader, class_names,
                         device, results_dir=results_dir,
                         filename="gradcam_custom_cnn.png")

    # ── Estudio de ablación ───────────────────────────────────────────────────
    if args.ablation:
        print("\n🔬 Ejecutando estudio de ablación...")
        ABLATION_GRID = [
            ("A_aug_ON",      "resnet50", "focal", True,  True,  1e-4),
            ("A_aug_OFF",     "resnet50", "focal", False, True,  1e-4),
            ("B_focal",       "resnet50", "focal", True,  True,  1e-4),
            ("B_ce_weights",  "resnet50", "ce",    True,  True,  1e-4),
            ("C_sampler_ON",  "resnet50", "focal", True,  True,  1e-4),
            ("C_sampler_OFF", "resnet50", "focal", True,  False, 1e-4),
            ("D_lr_1e3",      "resnet50", "focal", True,  True,  1e-3),
            ("D_lr_1e4",      "resnet50", "focal", True,  True,  1e-4),
            ("D_lr_5e5",      "resnet50", "focal", True,  True,  5e-5),
        ]
        ablation_results = []
        for tag, arch, loss, aug, samp, lr in ABLATION_GRID:
            print(f"\n>>> Ablación: {tag}")
            _, m, _, _, _ = run_training(
                arch=arch, device=device, data_dir=data_dir,
                results_dir=results_dir, loss_fn=loss,
                augment=aug, use_sampler=samp, lr=lr,
                epochs=15, tag=f"ablation/{tag}",
            )
            ablation_results.append({
                "Variante": tag,
                "F1":       round(m["f1"], 4),
                "Recall":   round(m["recall"], 4),
                "AUC":      round(m["auc_roc"], 4),
            })
        plot_ablation(ablation_results, results_dir)

    # ── Análisis de mitigación ────────────────────────────────────────────────
    if args.mitigation:
        print("\n📋 Ejecutando análisis de estrategias de mitigación...")
        MITIGATION_GRID = [
            ("Sin mitigación",         "resnet50", "ce",    False, False),
            ("CE + pesos de clase",    "resnet50", "ce",    True,  False),
            ("WeightedRandomSampler",  "resnet50", "ce",    False, True),
            ("Focal Loss",             "resnet50", "focal", True,  True),
        ]
        mit_results = []
        for name, arch, loss, aug, samp in MITIGATION_GRID:
            tag_name = name.replace(" ", "_").replace("+", "").lower()
            print(f"\n>>> Estrategia: {name}")
            _, m, _, _, _ = run_training(
                arch=arch, device=device, data_dir=data_dir,
                results_dir=results_dir, loss_fn=loss,
                augment=aug, use_sampler=samp, lr=1e-4,
                epochs=15, tag=f"mitigation/{tag_name}",
            )
            mit_results.append({
                "Estrategia":       name,
                "Accuracy":         round(m["accuracy"], 4),
                "Precision":        round(m["precision"], 4),
                "Recall (damaged)": round(m["recall"], 4),
                "F1":               round(m["f1"], 4),
                "AUC-ROC":          round(m["auc_roc"], 4),
            })
        plot_mitigation(mit_results, results_dir)

    print("\n" + "=" * 60)
    print("  🏁 Pipeline completado.")
    print(f"  📁 Resultados en: {results_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
