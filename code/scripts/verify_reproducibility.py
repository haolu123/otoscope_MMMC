#!/usr/bin/env python3
"""Verify that the public package reproduces the core paired-cohort results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = ["aom", "effusion", "normal", "perforation", "retraction", "tube", "tympanosclerosis"]


def normalize_label(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_")


def hybrid_contains_truth(prediction: object, truth: object) -> bool:
    truth_norm = normalize_label(truth)
    pred_labels = [normalize_label(part) for part in str(prediction).split("+")]
    return truth_norm in pred_labels


def assert_close(name: str, observed: float, expected: float, tol: float = 1e-12) -> None:
    if abs(float(observed) - float(expected)) > tol:
        raise AssertionError(f"{name}: observed={observed} expected={expected}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--write-json", type=Path, default=None)
    args = ap.parse_args()

    root = args.package_root.resolve()
    embeddings = np.load(root / "embeddings" / "video_embeddings.npy")
    emb_index = pd.read_csv(root / "embeddings" / "embedding_index.csv")
    probs = pd.read_csv(root / "derived_tables" / "video_probs.csv")
    hybrid = pd.read_csv(root / "derived_tables" / "cnn_hybrid_predictions_table.csv")
    overall = pd.read_csv(root / "derived_tables" / "overall_accuracy_with_95CI.csv")

    if embeddings.shape != (104, 2048):
        raise AssertionError(f"Expected embeddings shape (104, 2048), got {embeddings.shape}")
    if len(emb_index) != 104 or len(probs) != 104 or len(hybrid) != 104:
        raise AssertionError(
            f"Expected 104 rows in index/probs/hybrid, got {len(emb_index)}/{len(probs)}/{len(hybrid)}"
        )
    if list(emb_index["video_id"]) != list(probs["video_id"]):
        raise AssertionError("embedding_index.csv and video_probs.csv video_id order do not match")
    if set(probs["video_id"]) != set(hybrid["video_id"]):
        raise AssertionError("video_probs.csv and cnn_hybrid_predictions_table.csv video_id sets do not match")

    prob_cols = [f"prob_{label}" for label in LABELS]
    missing_prob_cols = [col for col in prob_cols if col not in probs.columns]
    if missing_prob_cols:
        raise AssertionError(f"Missing probability columns: {missing_prob_cols}")
    recomputed_pred_idx = probs[prob_cols].to_numpy().argmax(axis=1)
    recomputed_pred_name = [LABELS[i] for i in recomputed_pred_idx]
    if not (probs["y_pred"].astype(int).to_numpy() == recomputed_pred_idx).all():
        raise AssertionError("y_pred does not match argmax of prob_* columns")
    if [normalize_label(x) for x in probs["y_pred_name"]] != recomputed_pred_name:
        raise AssertionError("y_pred_name does not match argmax of prob_* columns")

    merged = probs.merge(hybrid, on="video_id", how="inner", validate="one_to_one")
    if len(merged) != 104:
        raise AssertionError(f"Expected 104 merged rows, got {len(merged)}")
    if [normalize_label(x) for x in merged["y_true_name"]] != [normalize_label(x) for x in merged["Ground Truth"]]:
        raise AssertionError("Ground Truth does not match y_true_name")
    if [normalize_label(x) for x in merged["y_pred_name"]] != [normalize_label(x) for x in merged["CNN Predict"]]:
        raise AssertionError("CNN Predict does not match video_probs y_pred_name")

    cnn_accuracy = float((merged["CNN Predict"].map(normalize_label) == merged["Ground Truth"].map(normalize_label)).mean())
    hybrid_accuracy = float(
        np.mean([hybrid_contains_truth(pred, truth) for pred, truth in zip(merged["Hybrid Predict"], merged["Ground Truth"])])
    )

    expected_cnn = float(overall.loc[overall["Method"] == "CNN Only", "Estimate"].iloc[0])
    expected_hybrid = float(overall.loc[overall["Method"] == "Hybrid", "Estimate"].iloc[0])
    assert_close("CNN Only overall accuracy", cnn_accuracy, expected_cnn)
    assert_close("Hybrid overall accuracy", hybrid_accuracy, expected_hybrid)

    result = {
        "status": "passed",
        "package_root": ".",
        "num_videos": 104,
        "embedding_shape": list(embeddings.shape),
        "cnn_accuracy_recomputed": cnn_accuracy,
        "hybrid_accuracy_recomputed": hybrid_accuracy,
        "cnn_accuracy_expected": expected_cnn,
        "hybrid_accuracy_expected": expected_hybrid,
        "checks": [
            "embedding_index row order matches video_probs",
            "video_probs video_id set matches cnn_hybrid_predictions_table",
            "CNN predictions match argmax(prob_*)",
            "Ground Truth matches y_true_name",
            "CNN Predict matches y_pred_name",
            "CNN and Hybrid overall accuracies reproduce overall_accuracy_with_95CI.csv",
        ],
    }
    print(json.dumps(result, indent=2))
    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
