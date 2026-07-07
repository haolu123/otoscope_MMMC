from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ototymp_ai.data import FrameDataset, load_manifest_samples, split_videos
from ototymp_ai.resnet import build_resnet


def _run_epoch(model, loader, criterion, optimizer, device: str, train: bool) -> tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    total = 0
    correct = 0
    for images, targets, _paths, _video_ids in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, targets)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * len(targets)
        pred = logits.argmax(dim=1)
        correct += int((pred == targets).sum().item())
        total += int(len(targets))

    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a compact ResNet otoscopy frame classifier.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    labels, samples = load_manifest_samples(args.manifest)
    train_samples, val_samples = split_videos(samples, val_fraction=args.val_fraction, seed=args.seed)
    train_ds = FrameDataset(train_samples, img_size=args.img_size)
    val_ds = FrameDataset(val_samples, img_size=args.img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_resnet(args.backbone, num_classes=len(labels), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    history = []
    for epoch in range(args.epochs):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        print(json.dumps(row))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "label_list": labels,
                    "backbone": args.backbone,
                    "best_val_acc": best_val_acc,
                },
                output / "best_model.pt",
            )

    with open(output / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[done] Best validation accuracy: {best_val_acc:.4f}")
    print(f"[done] Saved checkpoint to {output / 'best_model.pt'}")


if __name__ == "__main__":
    main()
