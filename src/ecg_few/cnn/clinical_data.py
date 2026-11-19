"""Clinical data for ECG experiments."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from ecg_few.cnn.data import PreprocessConfig, load_image_tensor
from ecg_few.protocols.loocv import (
    BrugadaImageRow,
)


class ClinicalImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        rows: Sequence[BrugadaImageRow],
        dataset_root: Path,
        preprocess: PreprocessConfig,
    ) -> None:
        self.rows = list(rows)
        self.dataset_root = dataset_root
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        if row.clinical_brugada is None:
            raise ValueError(f"Clinical label is missing for {row.patient_id}.")
        image = load_image_tensor(
            self.dataset_root / row.image_path,
            image_size=self.preprocess.image_size,
            mean=self.preprocess.mean,
            std=self.preprocess.std,
        )
        label = torch.tensor(float(row.clinical_brugada), dtype=torch.float32)
        return image, label


def make_clinical_loader(
    rows: Sequence[BrugadaImageRow],
    *,
    dataset_root: Path,
    preprocess: PreprocessConfig,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ClinicalImageDataset(rows, dataset_root, preprocess),
        batch_size=max(1, batch_size),
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
