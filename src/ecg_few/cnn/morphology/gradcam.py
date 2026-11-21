"""Gradcam for ECG experiments."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ecg_few.cnn.data import (
    PreprocessConfig,
    load_image_tensor,
    load_raw_image,
)
from ecg_few.protocols.loocv import (
    BrugadaImageRow,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model: nn.Module, target_module: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_module.register_forward_hook(self._forward_hook)
        self.backward_handle = target_module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output.detach()

    def _backward_hook(
        self,
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def compute(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        logits[:, 0].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM did not capture activations and gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze(0).squeeze(0)
        cam = cam / cam.max().clamp_min(1e-8)
        return cam.detach().cpu().numpy()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def save_gradcam_panel(
    *,
    model: nn.Module,
    rows: Sequence[BrugadaImageRow],
    dataset_root: Path,
    preprocess: PreprocessConfig,
    device: torch.device,
    output_path: Path,
    title: str,
) -> None:
    target_module = gradcam_target_module(model)
    if target_module is None:
        return
    model.eval()
    gradcam = GradCAM(model=model, target_module=target_module)
    try:
        fig, axes = plt.subplots(len(rows), 3, figsize=(10.5, 3.2 * len(rows)))
        if len(rows) == 1:
            axes = np.expand_dims(axes, axis=0)
        for row_index, row in enumerate(rows):
            image_path = dataset_root / row.image_path
            input_tensor = (
                load_image_tensor(
                    image_path,
                    image_size=preprocess.image_size,
                    mean=preprocess.mean,
                    std=preprocess.std,
                )
                .unsqueeze(0)
                .to(device)
            )
            display = load_raw_image(image_path, preprocess.image_size)
            with torch.enable_grad():
                heatmap = gradcam.compute(input_tensor)
            axes[row_index, 0].imshow(display, cmap="gray", vmin=0, vmax=1)
            axes[row_index, 0].set_title(f"{row.lead} input")
            axes[row_index, 1].imshow(heatmap, cmap="inferno", vmin=0, vmax=1)
            axes[row_index, 1].set_title("Grad-CAM")
            axes[row_index, 2].imshow(display, cmap="gray", vmin=0, vmax=1)
            axes[row_index, 2].imshow(heatmap, cmap="inferno", alpha=0.45, vmin=0, vmax=1)
            axes[row_index, 2].set_title("Overlay")
            for col in range(3):
                axes[row_index, col].axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
    finally:
        gradcam.close()


def gradcam_target_module(model: nn.Module) -> nn.Module | None:
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    return None
