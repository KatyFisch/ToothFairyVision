"""
PanoraDent preprocessing utilities.

Expected dataset layout:
PanoraDent/
    images/OPG_NNNN.jpg
    labels/OPG_NNNN.txt
    splits/train.txt, valid.txt, test.txt
    data.yaml

YOLO label format: class_id x_center y_center width height (normalised 0-1).
"""

from pathlib import Path
from typing import Callable, Optional, Union

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_yolo_labels(label_path: Union[str, Path]) -> torch.Tensor:
    """
    Read one YOLO label file.

    Returns a float32 tensor of shape [N, 5]:
    [class_id, x_center, y_center, width, height].
    Empty/missing files return a tensor with shape [0, 5].
    """
    label_path = Path(label_path)
    rows = []
    if not label_path.exists():
        return torch.zeros((0, 5), dtype=torch.float32)

    with label_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            values = line.split()
            if len(values) != 5:
                raise ValueError(
                    f"{label_path}:{line_number}: expected 5 values, got {len(values)}"
                )
            rows.append([float(value) for value in values])

    if not rows:
        return torch.zeros((0, 5), dtype=torch.float32)

    labels = torch.tensor(rows, dtype=torch.float32)
    if not torch.isfinite(labels).all():
        raise ValueError(f"Non-finite values found in {label_path}")
    if (labels[:, 1:] < 0).any() or (labels[:, 1:] > 1).any():
        raise ValueError(f"Normalised box coordinates outside [0, 1] in {label_path}")
    return labels


class PanoraDentDataset(Dataset):
    """
    PyTorch dataset for PanoraDent YOLO-format detection annotations.

    Images are returned as float tensors [C, H, W] in [0, 1].
    Targets are dictionaries containing:
      - boxes: [N, 4] in normalised xywh format
      - labels: [N] integer class IDs
      - image_id: dataset index
      - image_path: image path (string)

    No resizing is performed by default. This preserves the original image
    dimensions and avoids silently changing the geometry of the radiographs.
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        image_list: Optional[Union[str, Path]] = None,
    ):
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.label_dir = self.root / "labels"
        self.split = split
        self.transform = transform

        if image_list is None:
            if split not in {"train", "valid", "test"}:
                raise ValueError("split must be 'train', 'valid', or 'test'")
            image_list = self.root / "splits" / f"{split}.txt"

        image_list = Path(image_list)
        if not image_list.is_absolute():
            image_list = self.root / image_list
        if not image_list.exists():
            raise FileNotFoundError(f"Image list not found: {image_list}")

        entries = [
            line.strip() for line in image_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.image_paths = []
        for entry in entries:
            candidate = Path(entry)
            if not candidate.is_absolute():
                # Split files may contain either OPG_0001.jpg or images/OPG_0001.jpg.
                candidate = self.root / candidate
                if not candidate.exists():
                    candidate = self.image_dir / Path(entry).name
            if not candidate.exists():
                raise FileNotFoundError(f"Image from split list not found: {entry}")
            self.image_paths.append(candidate)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is None:
            # Avoid requiring torchvision for the basic preprocessing path.
            import numpy as np
            image_tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255.0
        else:
            image_tensor = self.transform(image)

        label_path = self.label_dir / f"{image_path.stem}.txt"
        yolo = load_yolo_labels(label_path)
        target = {
            "boxes": yolo[:, 1:5],
            "labels": yolo[:, 0].long(),
            "image_id": torch.tensor(index, dtype=torch.int64),
            "image_path": str(image_path),
        }
        return image_tensor, target


def detection_collate_fn(batch):
    """
    Collate detection samples with potentially different image dimensions.

    Returns a list of images and a list of target dictionaries. This is
    compatible with common object-detection training APIs.
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


def create_dataloader(
    root: Union[str, Path],
    split: str = "train",
    batch_size: int = 1,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    transform: Optional[Callable] = None,
    pin_memory: Optional[bool] = None,
):
    """Create a DataLoader; batch_size=1 is a safe starting point for large OPGs."""
    device = get_device()
    dataset = PanoraDentDataset(root=root, split=split, transform=transform)
    if shuffle is None:
        shuffle = split == "train"
    if pin_memory is None:
        pin_memory = device.type == "cuda"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=detection_collate_fn,
    )
