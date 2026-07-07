from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ototymp_ai.video_inference import run_video_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ResNet frame inference and aggregate to video-level probabilities.")
    parser.add_argument("--manifest", required=True, help="Dataset JSON containing video IDs, frame paths, and labels.")
    parser.add_argument("--checkpoint", required=True, help="PyTorch or PyTorch-Lightning checkpoint.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-ids", default=None, help="Optional CSV/TXT/XLSX list of video IDs to evaluate.")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    run_video_inference(
        manifest=args.manifest,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        test_ids=args.test_ids,
        backbone=args.backbone,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
