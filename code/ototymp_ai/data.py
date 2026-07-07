from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def label_index(labels: list[int] | list[float]) -> int:
    if not labels:
        return 0
    return int(max(range(len(labels)), key=lambda i: labels[i]))


def read_test_ids(path: str | Path | None) -> set[str]:
    """Read video IDs from CSV, TXT, or XLSX."""
    if path is None:
        return set()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
        col = "ID" if "ID" in df.columns else df.columns[0]
        return set(df[col].dropna().astype(str).str.strip())
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        preferred = ["ID", "Video_ID", "video_id", "used_video_id"]
        col = next((c for c in preferred if c in df.columns), df.columns[0])
        return set(df[col].dropna().astype(str).str.strip())

    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def load_manifest_samples(
    manifest: str | Path,
    keep_video_ids: set[str] | None = None,
    exclude_label_names: set[str] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    with open(manifest, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = list(data.get("label_list", []))
    samples = list(data.get("samples", data.get("videos", [])))

    if exclude_label_names:
        drop_idx = {labels.index(name) for name in exclude_label_names if name in labels}
    else:
        drop_idx = set()

    kept = []
    for sample in samples:
        video_id = str(sample.get("video_id", sample.get("video_ID", sample.get("id", "")))).strip()
        if keep_video_ids is not None and video_id not in keep_video_ids:
            continue

        label_vec = list(sample.get("labels", []))
        if drop_idx and any(i < len(label_vec) and int(label_vec[i]) == 1 for i in drop_idx):
            continue

        paths = sample.get("hq_img_paths", sample.get("img_paths", []))
        if paths:
            kept.append({**sample, "video_id": video_id, "hq_img_paths": paths})
    return labels, kept


class FrameDataset(Dataset):
    def __init__(self, video_samples: list[dict[str, Any]], img_size: int = 224):
        self.rows: list[dict[str, Any]] = []
        for sample in video_samples:
            y = label_index(list(sample.get("labels", [])))
            video_id = str(sample.get("video_id", "")).strip()
            for image_path in sample.get("hq_img_paths", []):
                self.rows.append({"image_path": image_path, "video_id": video_id, "label": y})

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(row["label"], dtype=torch.long), row["image_path"], row["video_id"]


def make_frame_loader(
    manifest: str | Path,
    test_ids: str | Path | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: int = 224,
) -> tuple[list[str], DataLoader]:
    keep_ids = read_test_ids(test_ids) if test_ids else None
    labels, samples = load_manifest_samples(manifest, keep_video_ids=keep_ids)
    dataset = FrameDataset(samples, img_size=img_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return labels, loader


def split_videos(samples: list[dict[str, Any]], val_fraction: float = 0.15, seed: int = 42):
    samples = list(samples)
    random.Random(seed).shuffle(samples)
    n_val = int(round(len(samples) * val_fraction))
    return samples[n_val:], samples[:n_val]

