# -*- coding: utf-8 -*-
"""
gradcam.py — Implementación de Grad-CAM y análisis cualitativo de fallos.

Referencia: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from src.config import IMAGENET_MEAN, IMAGENET_STD, RESULTS_DIR, NUM_CLASSES


# ── Desnormalización ──────────────────────────────────────────────────────────

_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD  = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def denorm(tensor: torch.Tensor) -> torch.Tensor:
    """Revierte la normalización ImageNet de un tensor (C, H, W) → [0, 1]."""
    return (tensor.cpu() * _STD + _MEAN).clamp(0, 1)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convierte tensor (C, H, W) normalizado → PIL Image RGB."""
    return transforms.ToPILImage()(denorm(tensor))


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM: calcula mapas de activación ponderados por gradiente.

    Uso:
        gcam = GradCAM(model, target_layer)
        cam, pred_class = gcam(img_tensor)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model  = model
        self._grad  = None
        self._act   = None

        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_act", o.detach())
        )
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_grad", go[0].detach())
        )

    @torch.enable_grad()
    def __call__(
        self,
        img_t: torch.Tensor,
        cls: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Args:
            img_t : tensor (1, C, H, W) en el device del modelo
            cls   : clase objetivo (None → clase predicha)

        Returns:
            (cam_array_normalizado [H, W] en [0, 1], clase_usada)
        """
        self.model.eval()
        img_t  = img_t.requires_grad_(True)
        logits = self.model(img_t)
        cls    = cls if cls is not None else logits.argmax(1).item()
        self.model.zero_grad()
        logits[0, cls].backward()

        # Pesos: media global de los gradientes por canal
        w   = self._grad.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((w * self._act).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, int(cls)


def overlay_cam(img_pil: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    Combina la imagen original con el heatmap Grad-CAM.

    Args:
        img_pil : imagen PIL original (RGB)
        cam     : array numpy [H, W] normalizado en [0, 1]
        alpha   : peso del heatmap (0 = solo imagen, 1 = solo heatmap)

    Returns:
        PIL Image con el overlay.
    """
    h, w = np.array(img_pil).shape[:2]
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ) / 255.0
    heat = (cm.jet(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    orig = np.array(img_pil.convert("RGB")).astype(float)
    blended = np.clip((1 - alpha) * orig + alpha * heat, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# ── Visualización Grad-CAM ────────────────────────────────────────────────────

def show_gradcam(
    model: nn.Module,
    target_layer: nn.Module,
    loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    n_per_class: int = 3,
    results_dir: Path = RESULTS_DIR,
    filename: str = "gradcam_analysis.png",
) -> None:
    """
    Genera y guarda el panel Grad-CAM con n_per_class ejemplos por clase.
    """
    gcam = GradCAM(model, target_layer)
    collected: Dict[int, List[Dict]] = {i: [] for i in range(NUM_CLASSES)}

    for imgs, labels in loader:
        for img, lbl in zip(imgs, labels):
            l = lbl.item()
            if len(collected[l]) >= n_per_class:
                continue
            img_t       = img.unsqueeze(0).to(device)
            cam, pred   = gcam(img_t)
            pil         = tensor_to_pil(img)
            collected[l].append({
                "orig":    pil,
                "overlay": overlay_cam(pil, cam),
                "cam":     cam,
                "pred":    pred,
                "true":    l,
            })
        if all(len(v) >= n_per_class for v in collected.values()):
            break

    rows = NUM_CLASSES * n_per_class
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]

    r = 0
    for cls_i, samples in collected.items():
        for s in samples:
            axes[r, 0].imshow(s["orig"])
            axes[r, 0].set_title(
                f"GT: {class_names[s['true']]} | Pred: {class_names[s['pred']]}", fontsize=8
            )
            axes[r, 0].axis("off")
            axes[r, 1].imshow(s["overlay"])
            axes[r, 1].set_title("Grad-CAM Overlay", fontsize=8)
            axes[r, 1].axis("off")
            axes[r, 2].imshow(s["cam"], cmap="jet")
            axes[r, 2].set_title("Activation Map", fontsize=8)
            axes[r, 2].axis("off")
            r += 1

    plt.suptitle("Grad-CAM — Interpretabilidad", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_dir = results_dir / "gradcam"
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Grad-CAM guardado en {out_dir / filename}")


# ── Análisis de fallos ────────────────────────────────────────────────────────

def analyze_failures(
    model: nn.Module,
    target_layer: nn.Module,
    loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    n: int = 5,
    results_dir: Path = RESULTS_DIR,
) -> None:
    """
    Identifica los n ejemplos mal clasificados con mayor confianza y
    los visualiza con Grad-CAM.
    """
    gcam     = GradCAM(model, target_layer)
    failures = []

    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs_d = imgs.to(device)
            logits = model(imgs_d)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(1)
            for i in range(len(labels)):
                if preds[i].item() != labels[i].item():
                    failures.append({
                        "img":  imgs[i],
                        "true": labels[i].item(),
                        "pred": preds[i].item(),
                        "conf": probs[i, preds[i]].item(),
                    })
            if len(failures) >= n * 3:
                break

    failures = sorted(failures, key=lambda x: x["conf"], reverse=True)[:n]

    print(f"\n{'='*60}")
    print(f"  ANÁLISIS DE FALLOS ({n} más confiados mal clasificados)")
    print(f"{'='*60}")

    fig, axes = plt.subplots(len(failures), 3, figsize=(12, 4 * len(failures)))
    if len(failures) == 1:
        axes = axes[np.newaxis, :]

    for ri, fail in enumerate(failures):
        img_t       = fail["img"].unsqueeze(0).to(device)
        cam, _      = gcam(img_t)
        pil         = tensor_to_pil(fail["img"])
        ov          = overlay_cam(pil, cam)

        axes[ri, 0].imshow(pil)
        axes[ri, 0].set_title(
            f"GT: {class_names[fail['true']]}  →  Pred: {class_names[fail['pred']]}  "
            f"(conf={fail['conf']:.2f})",
            fontsize=9, color="red",
        )
        axes[ri, 0].axis("off")
        axes[ri, 1].imshow(ov)
        axes[ri, 1].set_title("Grad-CAM Overlay", fontsize=9)
        axes[ri, 1].axis("off")
        axes[ri, 2].imshow(cam, cmap="jet")
        axes[ri, 2].set_title("Activation Map", fontsize=9)
        axes[ri, 2].axis("off")

        err = (
            "FN (daño no detectado)"
            if fail["true"] == 1
            else "FP (buena clasificada como dañada)"
        )
        print(f"  #{ri + 1} | {err} | conf={fail['conf']:.3f}")

    plt.suptitle(
        f"Análisis de Fallos — {n} ejemplos más confiados",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out_dir = results_dir / "gradcam"
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / "failure_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Análisis de fallos guardado en {out_dir / 'failure_analysis.png'}")
