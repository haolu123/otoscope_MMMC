from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ototymp_ai.analysis import run_multimodal_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OtoTymp-AI rule-based multimodal analysis.")
    parser.add_argument("--metadata", required=True, help="Excel file with ground truth and tympanometry fields.")
    parser.add_argument("--video-probs", required=True, help="CSV produced by run_video_inference.py.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--sheet-name", default="dataset")
    parser.add_argument("--threshold-sweep", type=float, nargs="*", default=None)
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--n-one-ear", type=int, default=100)
    parser.add_argument("--strategy", choices=["paper_v3", "simple"], default="paper_v3")
    args = parser.parse_args()

    run_multimodal_analysis(
        metadata=args.metadata,
        video_probs=args.video_probs,
        output_dir=args.output_dir,
        threshold=args.threshold,
        threshold_sweep=args.threshold_sweep,
        sheet_name=args.sheet_name,
        n_boot=args.n_boot,
        n_one_ear=args.n_one_ear,
        strategy=args.strategy,
    )


if __name__ == "__main__":
    main()
