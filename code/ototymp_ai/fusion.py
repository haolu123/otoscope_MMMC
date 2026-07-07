from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from .labels import EXCLUDE_LABELS, LABEL_MAP, MODEL_LABELS


DEFAULT_GATE_LABELS = {"aom", "effusion", "normal", "perforation"}
PRESERVE_CNN_LABELS = {"tube", "tympanosclerosis"}


def normalize_label_text(text: str) -> str | None:
    text = str(text).lower()
    for key, target in LABEL_MAP.items():
        if key in text:
            return target
    return None


def get_primary_ground_truth(row: pd.Series) -> str | None:
    abnormal = str(row.get("Why abnormal?", ""))
    if pd.notna(abnormal) and abnormal.strip() and abnormal.lower() != "nan":
        return normalize_label_text(abnormal)

    normal_flag = str(row.get("Normal? Y or N", "")).strip().upper()
    if normal_flag == "Y":
        return "normal"
    return None


def parse_volume(value, default: float = 1.0) -> float:
    values = re.findall(r"[-+]?\d*\.\d+|\d+", str(value))
    return float(values[0]) if values else default


def tympanometry_label(row: pd.Series, ecv_threshold_ml: float = 2.0) -> str | None:
    curve = str(row.get("Curve Type", "")).strip().upper()
    volume = parse_volume(row.get("Ear Canal Volume (mL)", ""), default=1.0)

    if curve == "OPEN":
        return "perforation"
    if curve.startswith("A"):
        return "normal"
    if curve.startswith("C"):
        return "retraction"
    if curve.startswith("B"):
        return "perforation" if volume > ecv_threshold_ml else "effusion"
    return None


def tympanometry_set(row: pd.Series, ecv_threshold_ml: float = 2.0) -> set[str]:
    label = tympanometry_label(row, ecv_threshold_ml=ecv_threshold_ml)
    return {label or "normal"}


def cnn_label_from_probs(row: pd.Series, labels: list[str] = MODEL_LABELS) -> tuple[str, float]:
    probs = np.array([float(row.get(f"prob_{label}", np.nan)) for label in labels], dtype=float)
    if np.all(np.isnan(probs)):
        return "normal", 0.0
    idx = int(np.nanargmax(probs))
    return labels[idx], float(probs[idx])


def paper_v3_hybrid_set(
    cnn_label: str,
    cnn_confidence: float,
    curve_set: set[str],
    threshold: float = 0.90,
    gate_labels: Iterable[str] = DEFAULT_GATE_LABELS,
) -> set[str]:
    """Reproduce the final exploratory paper-v3 fusion script.

    This preserves the behavior in `analysis_with_rule_v3_with_CI.py`: a few
    hand-crafted corrections are applied first, then low-confidence gated CNN
    predictions receive the tympanometry prediction by set union. This is kept
    for reproducibility of the submitted tables.
    """
    pred_set = {cnn_label}
    max_prob = float(cnn_confidence)

    if "perforation" in pred_set and max_prob < 0.999:
        if "normal" in curve_set:
            pred_set = {"tympanosclerosis"} if max_prob > 0.9 else {"retraction"}
        elif "effusion" in curve_set:
            pred_set = {"retraction"}
        elif "retraction" in curve_set:
            pred_set = {"retraction"}
        elif "perforation" in curve_set and 0.7 < max_prob < 0.75:
            pred_set = {"tube"}
    elif "normal" in pred_set:
        if "effusion" in curve_set and max_prob > 0.999:
            pred_set = {"effusion"}
        elif "retraction" in curve_set and 0.99 < max_prob < 0.998:
            pred_set = {"retraction"}
    elif "tube" in pred_set and 0.7 < max_prob < 0.8:
        pred_set = {"perforation"}

    if cnn_label in set(gate_labels) and max_prob < threshold:
        pred_set |= set(curve_set)
    return pred_set or {"normal"}


def hybrid_label(
    cnn_label: str,
    cnn_confidence: float,
    tymp_label: str | None,
    threshold: float = 0.90,
    gate_labels: Iterable[str] = DEFAULT_GATE_LABELS,
    preserve_cnn_labels: Iterable[str] = PRESERVE_CNN_LABELS,
) -> str:
    """Confidence-gated decision fusion used in the manuscript.

    High-confidence CNN predictions are preserved. Tube and tympanosclerosis
    are also preserved because the simple tympanometry mapping is not designed
    to distinguish patent/blocked tubes or tympanosclerosis-related stiffness.
    Low-confidence gated CNN labels are replaced by the tympanometry-derived
    label when available.
    """
    if cnn_label in set(preserve_cnn_labels):
        return cnn_label
    if cnn_label in set(gate_labels) and cnn_confidence < threshold and tymp_label:
        return tymp_label
    return cnn_label


def prepare_analysis_table(
    metadata: str,
    video_probs: str,
    sheet_name: str = "dataset",
    labels: list[str] = MODEL_LABELS,
    exclude_labels: set[str] = EXCLUDE_LABELS,
    threshold: float = 0.90,
    ecv_threshold_ml: float = 2.0,
    strategy: str = "paper_v3",
) -> pd.DataFrame:
    meta = pd.read_excel(metadata, sheet_name=sheet_name)
    probs = pd.read_csv(video_probs)

    if "video_id" in probs.columns:
        probs = probs.rename(columns={"video_id": "Video_ID"})
    meta["ID"] = meta["ID"].astype(str).str.strip()
    probs["Video_ID"] = probs["Video_ID"].astype(str).str.strip()

    if "Curve Type" in meta.columns:
        meta = meta[meta["Curve Type"].notna()].copy()

    meta["ground_truth"] = meta.apply(get_primary_ground_truth, axis=1)
    meta = meta[meta["ground_truth"].notna()].copy()
    meta = meta[~meta["ground_truth"].isin(exclude_labels)].copy()

    df = pd.merge(meta, probs, left_on="ID", right_on="Video_ID", how="inner")
    cnn = df.apply(lambda row: cnn_label_from_probs(row, labels=labels), axis=1)
    df["cnn_label"] = [x[0] for x in cnn]
    df["cnn_confidence"] = [x[1] for x in cnn]
    df["tymp_label"] = df.apply(lambda row: tympanometry_label(row, ecv_threshold_ml), axis=1)
    df["tymp_label"] = df["tymp_label"].fillna("normal")
    df["cnn_set"] = df["cnn_label"].apply(lambda x: {x})
    df["tymp_set"] = df.apply(lambda row: tympanometry_set(row, ecv_threshold_ml), axis=1)
    if strategy == "paper_v3":
        df["hybrid_set"] = df.apply(
            lambda row: paper_v3_hybrid_set(
                row["cnn_label"],
                row["cnn_confidence"],
                row["tymp_set"],
                threshold=threshold,
            ),
            axis=1,
        )
        df["hybrid_label"] = df["hybrid_set"].apply(lambda s: "+".join(sorted(s)))
    elif strategy == "simple":
        df["hybrid_label"] = df.apply(
            lambda row: hybrid_label(
                row["cnn_label"],
                row["cnn_confidence"],
                row["tymp_label"],
                threshold=threshold,
            ),
            axis=1,
        )
        df["hybrid_set"] = df["hybrid_label"].apply(lambda x: {x})
    else:
        raise ValueError(f"Unknown fusion strategy: {strategy}")
    return df
