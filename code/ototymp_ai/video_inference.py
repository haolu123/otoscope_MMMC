from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

from .data import make_frame_loader
from .metrics import multiclass_metrics
from .resnet import load_resnet_checkpoint


def _plot_roc(y_true: np.ndarray, probs: np.ndarray, labels: list[str], out_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    for idx, label in enumerate(labels):
        y_bin = (y_true == idx).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin, probs[:, idx])
        plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Video-level ROC curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def run_video_inference(
    manifest: str,
    checkpoint: str,
    output_dir: str,
    test_ids: str | None = None,
    backbone: str = "resnet50",
    img_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    device: str | None = None,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    labels, loader = make_frame_loader(
        manifest=manifest,
        test_ids=test_ids,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
    )
    if not labels:
        raise ValueError("No labels found in manifest.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_resnet_checkpoint(
        checkpoint=checkpoint,
        backbone=backbone,
        num_classes=len(labels),
        device=device,
    )

    probs_sum = defaultdict(lambda: np.zeros(len(labels), dtype=np.float64))
    counts = defaultdict(int)
    y_true_by_video = {}

    with torch.no_grad():
        for images, targets, _paths, video_ids in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            targets_np = targets.numpy()

            for prob, target, video_id in zip(probs, targets_np, video_ids):
                video_id = str(video_id)
                probs_sum[video_id] += prob
                counts[video_id] += 1
                y_true_by_video.setdefault(video_id, int(target))

    rows = []
    y_true = []
    prob_rows = []
    for video_id in sorted(counts):
        prob = probs_sum[video_id] / float(counts[video_id])
        prob = prob / prob.sum() if prob.sum() > 0 else np.ones(len(labels)) / len(labels)
        pred_idx = int(np.argmax(prob))
        true_idx = int(y_true_by_video[video_id])
        y_true.append(true_idx)
        prob_rows.append(prob)

        row = {
            "video_id": video_id,
            "num_frames": int(counts[video_id]),
            "y_true": true_idx,
            "y_true_name": labels[true_idx] if 0 <= true_idx < len(labels) else str(true_idx),
            "y_pred": pred_idx,
            "y_pred_name": labels[pred_idx],
        }
        for i, label in enumerate(labels):
            row[f"prob_{label}"] = float(prob[i])
        rows.append(row)

    probs_arr = np.vstack(prob_rows) if prob_rows else np.zeros((0, len(labels)), dtype=float)
    y_true_arr = np.asarray(y_true, dtype=int)

    pd.DataFrame(rows).to_csv(output / "video_probs.csv", index=False)
    metrics = multiclass_metrics(y_true_arr, probs_arr, labels)
    with open(output / "video_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if len(y_true_arr):
        _plot_roc(y_true_arr, probs_arr, labels, output / "roc_curves_all_classes.png")

    print(f"[done] Saved video probabilities to {output / 'video_probs.csv'}")
    print(f"[done] Saved video metrics to {output / 'video_metrics.json'}")
    return metrics

