"""Clinical gradcam for ECG experiments."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn

from ecg_few.cnn.data import PreprocessConfig, load_image_tensor
from ecg_few.protocols.loocv import (
    BrugadaImageRow,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_binary_gradcam_panel(
    *,
    model: nn.Module,
    rows: Sequence[BrugadaImageRow],
    dataset_root: Path,
    preprocess: PreprocessConfig,
    device: torch.device,
    output_path: Path,
    title: str,
) -> None:
    target_module = model.layer4[-1] if hasattr(model, "layer4") else None
    if target_module is None:
        return
    activations: torch.Tensor | None = None
    gradients: torch.Tensor | None = None

    def forward_hook(
        _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        nonlocal activations
        activations = output.detach()

    def backward_hook(
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        nonlocal gradients
        gradients = grad_output[0].detach()

    forward_handle = target_module.register_forward_hook(forward_hook)
    backward_handle = target_module.register_full_backward_hook(backward_hook)
    try:
        fig, axes = plt.subplots(len(rows), 3, figsize=(9.5, max(2.8, 2.8 * len(rows))))
        axes = np.atleast_2d(axes)
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
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logit = model(input_tensor)
                logit[:, 0].sum().backward()
            if activations is None or gradients is None:
                continue
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
            cam = torch.nn.functional.interpolate(
                cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
            )
            heatmap = cam.squeeze().detach().cpu().numpy()
            heatmap /= max(float(heatmap.max()), 1e-8)
            display = input_tensor[0, 0].detach().cpu().numpy()
            axes[row_index, 0].imshow(display, cmap="gray")
            axes[row_index, 1].imshow(heatmap, cmap="inferno", vmin=0, vmax=1)
            axes[row_index, 2].imshow(display, cmap="gray")
            axes[row_index, 2].imshow(heatmap, cmap="inferno", alpha=0.45, vmin=0, vmax=1)
            for col in range(3):
                axes[row_index, col].axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
    finally:
        forward_handle.remove()
        backward_handle.remove()
