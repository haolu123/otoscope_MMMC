from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .fusion import paper_v3_hybrid_set, prepare_analysis_table
from .labels import EXTERNAL_EVAL_LABELS, MODEL_LABELS
from .metrics import bootstrap_ci, percentile_ci, safe_div, wilson_ci


METHOD_COLUMNS = {
    "CNN Only": "cnn_set",
    "Tympanometry Only": "tymp_set",
    "Hybrid": "hybrid_set",
}


def add_patient_id_ear_side(df: pd.DataFrame, id_col: str = "Video_ID") -> pd.DataFrame:
    out = df.copy()
    text = out[id_col].astype(str).str.strip()
    out["ear_side"] = text.str.extract(r"([LR])(?:_|$)", expand=False)
    out["patient_id"] = text.str.replace(r"([LR])(?:_|$).*", "", regex=True)
    return out


def _contains_label(value, label: str) -> bool:
    if isinstance(value, set):
        return label in value
    if isinstance(value, (list, tuple)):
        return label in set(value)
    return str(value) == label


def _correct_series(df: pd.DataFrame, pred_col: str) -> pd.Series:
    return df.apply(lambda row: _contains_label(row[pred_col], row["ground_truth"]), axis=1)


def _overall_accuracy(df: pd.DataFrame, pred_col: str) -> float:
    return float(_correct_series(df, pred_col).mean()) if len(df) else np.nan


def _ovr_counts(df: pd.DataFrame, pred_col: str, label: str) -> dict[str, int]:
    true_pos = df["ground_truth"] == label
    pred_pos = df[pred_col].apply(lambda value: _contains_label(value, label))
    return {
        "tp": int((true_pos & pred_pos).sum()),
        "tn": int((~true_pos & ~pred_pos).sum()),
        "fp": int((~true_pos & pred_pos).sum()),
        "fn": int((true_pos & ~pred_pos).sum()),
    }


def _balanced_accuracy(counts: dict[str, int]) -> float:
    sens = safe_div(counts["tp"], counts["tp"] + counts["fn"])
    spec = safe_div(counts["tn"], counts["tn"] + counts["fp"])
    return float(np.nanmean([sens, spec]))


def save_prediction_table(df: pd.DataFrame, output_dir: Path) -> None:
    keep = [
        "Video_ID",
        "ground_truth",
        "cnn_label",
        "cnn_confidence",
        "tymp_label",
        "hybrid_label",
        "hybrid_set",
        "Curve Type",
        "Ear Canal Volume (mL)",
    ]
    keep = [c for c in keep if c in df.columns]
    df[keep].to_csv(output_dir / "cnn_hybrid_predictions_table.csv", index=False)


def save_overall_accuracy(df: pd.DataFrame, output_dir: Path, n_boot: int = 100) -> pd.DataFrame:
    rows = []
    for method, col in METHOD_COLUMNS.items():
        n = len(df)
        k = int(_correct_series(df, col).sum())
        lo, hi = wilson_ci(k, n)
        rows.append({
            "Method": method,
            "Correct": k,
            "Total": n,
            "Accuracy": safe_div(k, n),
            "Accuracy_95CI_low": lo,
            "Accuracy_95CI_high": hi,
        })

    diff = _overall_accuracy(df, "hybrid_set") - _overall_accuracy(df, "cnn_set")
    diff_lo, diff_hi = bootstrap_ci(
        df,
        lambda sample: _overall_accuracy(sample, "hybrid_set") - _overall_accuracy(sample, "cnn_set"),
        n_boot=n_boot,
    )
    rows.append({
        "Method": "Hybrid minus CNN",
        "Correct": np.nan,
        "Total": len(df),
        "Accuracy": diff,
        "Accuracy_95CI_low": diff_lo,
        "Accuracy_95CI_high": diff_hi,
    })

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "overall_accuracy_with_95CI.csv", index=False)
    return out


def save_per_class_metrics(df: pd.DataFrame, output_dir: Path, labels: list[str], n_boot: int = 100) -> pd.DataFrame:
    rows = []
    for method, col in METHOD_COLUMNS.items():
        for label in labels:
            c = _ovr_counts(df, col, label)
            n = c["tp"] + c["tn"] + c["fp"] + c["fn"]
            pos = c["tp"] + c["fn"]
            neg = c["tn"] + c["fp"]
            acc = safe_div(c["tp"] + c["tn"], n)
            sens = safe_div(c["tp"], pos)
            spec = safe_div(c["tn"], neg)
            bal = _balanced_accuracy(c)

            acc_l, acc_h = wilson_ci(c["tp"] + c["tn"], n)
            sens_l, sens_h = wilson_ci(c["tp"], pos)
            spec_l, spec_h = wilson_ci(c["tn"], neg)
            bal_l, bal_h = bootstrap_ci(
                df,
                lambda sample, p=col, l=label: _balanced_accuracy(_ovr_counts(sample, p, l)),
                n_boot=n_boot,
            )

            rows.append({
                "Method": method,
                "Class": label,
                "n_positive": pos,
                "n_negative": neg,
                **c,
                "Accuracy": acc,
                "Accuracy_95CI_low": acc_l,
                "Accuracy_95CI_high": acc_h,
                "Sensitivity": sens,
                "Sensitivity_95CI_low": sens_l,
                "Sensitivity_95CI_high": sens_h,
                "Specificity": spec,
                "Specificity_95CI_low": spec_l,
                "Specificity_95CI_high": spec_h,
                "Balanced_accuracy": bal,
                "Balanced_accuracy_95CI_low": bal_l,
                "Balanced_accuracy_95CI_high": bal_h,
            })
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "per_class_metrics_3methods_with_95CI.csv", index=False)
    return out


def save_auroc_table(df: pd.DataFrame, output_dir: Path, labels: list[str], n_boot: int = 100) -> pd.DataFrame:
    rows = []
    for label in labels:
        y_true = (df["ground_truth"] == label).astype(int).to_numpy()
        if len(np.unique(y_true)) < 2:
            continue

        cnn_scores = df[f"prob_{label}"].astype(float).to_numpy()
        hybrid_scores = df["hybrid_set"].apply(lambda value: _contains_label(value, label)).astype(float).to_numpy()
        tymp_scores = df["tymp_set"].apply(lambda value: _contains_label(value, label)).astype(float).to_numpy()

        for method, scores in [
            ("CNN Only", cnn_scores),
            ("Tympanometry Only", tymp_scores),
            ("Hybrid", hybrid_scores),
        ]:
            auc = float(roc_auc_score(y_true, scores))
            lo, hi = bootstrap_ci(
                df,
                lambda sample, l=label, m=method: _auroc_for_sample(sample, l, m),
                n_boot=n_boot,
            )
            rows.append({
                "Method": method,
                "Class": label,
                "AUROC": auc,
                "AUROC_95CI_low": lo,
                "AUROC_95CI_high": hi,
            })

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "auroc_with_95CI.csv", index=False)
    return out


def _auroc_for_sample(sample: pd.DataFrame, label: str, method: str) -> float:
    y_true = (sample["ground_truth"] == label).astype(int).to_numpy()
    if len(np.unique(y_true)) < 2:
        return np.nan
    if method == "CNN Only":
        scores = sample[f"prob_{label}"].astype(float).to_numpy()
    elif method == "Tympanometry Only":
        scores = sample["tymp_set"].apply(lambda value: _contains_label(value, label)).astype(float).to_numpy()
    else:
        scores = sample["hybrid_set"].apply(lambda value: _contains_label(value, label)).astype(float).to_numpy()
    return float(roc_auc_score(y_true, scores))


def save_threshold_ablation(df: pd.DataFrame, output_dir: Path, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    cnn_acc = _overall_accuracy(df, "cnn_set")
    tymp_acc = _overall_accuracy(df, "tymp_set")
    for threshold in thresholds:
        tmp = df.copy()
        tmp["hybrid_set"] = tmp.apply(
            lambda row: paper_v3_hybrid_set(
                row["cnn_label"],
                row["cnn_confidence"],
                row["tymp_set"],
                threshold=threshold,
            ),
            axis=1,
        )
        tmp["hybrid_label"] = tmp["hybrid_set"].apply(lambda s: "+".join(sorted(s)))
        hyb_acc = _overall_accuracy(tmp, "hybrid_set")
        rows.append({
            "threshold": threshold,
            "cnn_acc": cnn_acc,
            "tympanometry_acc": tymp_acc,
            "hybrid_acc": hyb_acc,
            "hybrid_minus_cnn": hyb_acc - cnn_acc,
            "hybrid_minus_tympanometry": hyb_acc - tymp_acc,
        })
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "ablation_threshold_results.csv", index=False)
    return out


def _cluster_sample(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    patients = df["patient_id"].dropna().unique()
    sampled = rng.choice(patients, size=len(patients), replace=True)
    return pd.concat([df[df["patient_id"] == pid] for pid in sampled], ignore_index=True)


def save_patient_level_sensitivity(df: pd.DataFrame, output_dir: Path, n_boot: int = 100, n_one_ear: int = 100) -> None:
    df = add_patient_id_ear_side(df)
    patient_list = df.groupby("patient_id", dropna=False).agg(
        n_ear_examinations=("Video_ID", "size"),
        ear_sides=("ear_side", lambda x: "+".join(sorted(set(x.dropna().astype(str))))),
        video_ids=("Video_ID", lambda x: "+".join(map(str, x))),
    ).reset_index()
    patient_list["has_bilateral_ears"] = patient_list["ear_sides"].apply(lambda x: {"L", "R"}.issubset(set(str(x).split("+"))))

    summary = pd.DataFrame([{
        "n_ear_level_examinations": len(df),
        "n_unique_patients": int(df["patient_id"].nunique(dropna=False)),
        "n_patients_with_bilateral_ears": int(patient_list["has_bilateral_ears"].sum()),
        "n_patients_with_one_ear": int((~patient_list["has_bilateral_ears"]).sum()),
        "n_right_ears": int((df["ear_side"] == "R").sum()),
        "n_left_ears": int((df["ear_side"] == "L").sum()),
    }])
    summary.to_csv(output_dir / "patient_level_cohort_summary.csv", index=False)
    patient_list.to_csv(output_dir / "patient_level_patient_list.csv", index=False)

    rng = np.random.default_rng(2026)
    rows = []
    for method, col in METHOD_COLUMNS.items():
        vals = []
        for _ in range(n_boot):
            sample = _cluster_sample(df, rng)
            vals.append(_overall_accuracy(sample, col))
        lo, hi = percentile_ci(vals)
        rows.append({
            "Analysis": "patient-clustered bootstrap",
            "Metric": "Overall accuracy",
            "Method": method,
            "Estimate": _overall_accuracy(df, col),
            "95CI_low": lo,
            "95CI_high": hi,
        })
    vals = []
    for _ in range(n_boot):
        sample = _cluster_sample(df, rng)
        vals.append(_overall_accuracy(sample, "hybrid_set") - _overall_accuracy(sample, "cnn_set"))
    lo, hi = percentile_ci(vals)
    rows.append({
        "Analysis": "patient-clustered bootstrap",
        "Metric": "Paired accuracy improvement",
        "Method": "Hybrid minus CNN",
        "Estimate": _overall_accuracy(df, "hybrid_set") - _overall_accuracy(df, "cnn_set"),
        "95CI_low": lo,
        "95CI_high": hi,
    })
    pd.DataFrame(rows).to_csv(output_dir / "patient_level_clustered_bootstrap_overall.csv", index=False)

    one_ear_rows = []
    for rep in range(n_one_ear):
        selected = df.groupby("patient_id", group_keys=False).sample(n=1, random_state=2026 + rep)
        for method, col in METHOD_COLUMNS.items():
            one_ear_rows.append({
                "replicate": rep,
                "Metric": "Overall accuracy",
                "Method": method,
                "Estimate": _overall_accuracy(selected, col),
            })
        one_ear_rows.append({
            "replicate": rep,
            "Metric": "Paired accuracy improvement",
            "Method": "Hybrid minus CNN",
            "Estimate": _overall_accuracy(selected, "hybrid_set") - _overall_accuracy(selected, "cnn_set"),
        })
    reps = pd.DataFrame(one_ear_rows)
    reps.to_csv(output_dir / "patient_level_one_ear_per_patient_replicates.csv", index=False)
    summary = reps.groupby(["Metric", "Method"], as_index=False).agg(
        n_replicates=("Estimate", "size"),
        mean_estimate=("Estimate", "mean"),
        p2_5=("Estimate", lambda s: float(np.percentile(s, 2.5))),
        p97_5=("Estimate", lambda s: float(np.percentile(s, 97.5))),
    )
    summary.to_csv(output_dir / "patient_level_one_ear_per_patient_summary.csv", index=False)


def run_multimodal_analysis(
    metadata: str,
    video_probs: str,
    output_dir: str,
    threshold: float = 0.90,
    threshold_sweep: list[float] | None = None,
    sheet_name: str = "dataset",
    eval_labels: list[str] = EXTERNAL_EVAL_LABELS,
    n_boot: int = 100,
    n_one_ear: int = 100,
    strategy: str = "paper_v3",
) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df = prepare_analysis_table(
        metadata=metadata,
        video_probs=video_probs,
        sheet_name=sheet_name,
        labels=MODEL_LABELS,
        threshold=threshold,
        strategy=strategy,
    )
    df.to_csv(output / "analysis_dataset.csv", index=False)
    save_prediction_table(df, output)
    save_overall_accuracy(df, output, n_boot=n_boot)
    save_per_class_metrics(df, output, labels=eval_labels, n_boot=n_boot)
    save_auroc_table(df, output, labels=eval_labels, n_boot=n_boot)
    save_patient_level_sensitivity(df, output, n_boot=n_boot, n_one_ear=n_one_ear)
    if threshold_sweep:
        save_threshold_ablation(df, output, threshold_sweep)

    print(f"[done] Saved multimodal analysis outputs to {output}")
    return df
