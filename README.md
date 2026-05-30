# 🛞 Clasificación de Llantas Dañadas — RNAPD 2026-I

**Pregunta 1 | Ph.D. Aldo Camargo**

Pipeline completo de clasificación binaria de llantas (buena/dañada) usando técnicas de visión por computador con PyTorch.

## 📋 Descripción

Este proyecto implementa y compara tres arquitecturas de redes neuronales:
- **CustomCNN** — diseñada desde cero con ResBlocks y SEBlocks (~6.5M parámetros)
- **ResNet-50** — con fine-tuning desde ImageNet
- **EfficientNet-B0** — con fine-tuning desde ImageNet

Además incluye:
- Análisis del desbalance de clases y estrategias de mitigación
- Estudio de ablación (augmentación, función de pérdida, sampler, learning rate)
- Interpretabilidad con Grad-CAM
- Análisis cualitativo de fallos

## 📁 Estructura del Proyecto

```
tire_classification/
├── data/
│   └── tire_texture/          # Dataset (no incluido en el repo)
│       ├── train/
│       │   ├── good_tire/
│       │   └── damaged_tire/
│       ├── val/
│       └── test/
├── results/                   # Generado automáticamente al ejecutar
│   ├── custom_cnn/
│   ├── resnet50/
│   ├── efficientnet_b0/
│   ├── comparison/
│   ├── ablation/
│   ├── imbalance/
│   └── gradcam/
├── src/
│   ├── config.py              # Constantes y configuración global
│   ├── dataset.py             # Descarga, preparación y DataLoaders
│   ├── models.py              # Arquitecturas CNN
│   ├── losses.py              # FocalLoss y CombinedLoss
│   ├── train.py               # Funciones de entrenamiento y evaluación
│   ├── gradcam.py             # Implementación Grad-CAM
│   └── visualization.py      # Gráficos y reportes
├── main.py                    # Punto de entrada principal
├── prepare_data.py            # Script separado para preparar el dataset
├── requirements.txt
└── README.md
```

## ⚙️ Instalación

```bash
git clone https://github.com/tu-usuario/tire-classification.git
cd tire-classification
pip install -r requirements.txt
```

## 🗂️ Preparación del Dataset

El dataset proviene de [Tire Texture Image Recognition](https://www.kaggle.com/datasets/jehanbhathena/tire-texture-image-recognition) en Kaggle.

**Opción A — Descarga automática vía Kaggle API:**

1. Obtén tu token en [kaggle.com/settings](https://www.kaggle.com/settings) → API → *Create New Token*
2. Coloca `kaggle.json` en `~/.kaggle/kaggle.json` (Linux/Mac) o `%USERPROFILE%\.kaggle\kaggle.json` (Windows)
3. Ejecuta:

```bash
python prepare_data.py
```

**Opción B — Descarga manual:**

1. Descarga el zip desde Kaggle y descomprímelo en `data/raw/`
2. Ejecuta:

```bash
python prepare_data.py --raw-dir data/raw
```

## 🚀 Ejecución

```bash
# Pipeline completo (entrenamiento + evaluación + Grad-CAM)
python main.py

# Solo entrenamiento de un modelo específico
python main.py --arch resnet50

# Estudio de ablación
python main.py --ablation

# Análisis de estrategias de mitigación de desbalance
python main.py --mitigation
```

### Argumentos disponibles

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--arch` | `all` | Arquitectura: `custom_cnn`, `resnet50`, `efficientnet_b0`, `all` |
| `--epochs` | `30` | Número de épocas |
| `--batch-size` | `32` | Tamaño del batch |
| `--lr` | `1e-4` | Learning rate |
| `--loss` | `focal` | Función de pérdida: `focal`, `ce` |
| `--no-augment` | — | Desactivar data augmentation |
| `--no-sampler` | — | Desactivar WeightedRandomSampler |
| `--ablation` | — | Ejecutar estudio de ablación |
| `--mitigation` | — | Ejecutar análisis de mitigación |
| `--seed` | `42` | Semilla para reproducibilidad |
| `--data-dir` | `data/tire_texture` | Ruta al dataset preparado |
| `--results-dir` | `results` | Carpeta de salida |

## 📊 Resultados Generados

| Entregable | Ruta |
|---|---|
| Modelos entrenados | `results/{arch}/best.pth` |
| Curvas de entrenamiento | `results/{arch}/training_curves.png` |
| Matrices de confusión | `results/{arch}/confusion_matrix.png` |
| Métricas test | `results/{arch}/test_metrics.json` |
| Curvas ROC + PR | `results/comparison/roc_pr_curves.png` |
| Tabla comparativa | `results/comparison/metrics_table.csv` |
| Ablación | `results/ablation/ablation_plot.png` + `.csv` |
| Distribución clases | `results/imbalance/class_distribution.png` |
| Mitigación desbalance | `results/imbalance/mitigation_comparison.png` |
| Grad-CAM | `results/gradcam/gradcam_analysis.png` |
| Análisis de fallos | `results/gradcam/failure_analysis.png` |

## 🔧 Requisitos

- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA (recomendado) o CPU

Ver `requirements.txt` para la lista completa.
