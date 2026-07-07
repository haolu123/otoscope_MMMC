# OtoTymp-AI clean code package

This folder contains a compact, reproducible version of the code used for the
PLOS ONE revision of manuscript `PONE-D-26-23620`.

It keeps only the core workflow:

1. Train or load a ResNet otoscopy frame classifier.
2. Aggregate frame-level predictions to video-level probabilities.
3. Combine video probabilities with tympanometry using the paper's
   confidence-guided rule-based fusion.
4. Generate the main reproducibility tables: overall accuracy, per-class
   sensitivity/specificity/balanced accuracy, AUROC, threshold ablation, and
   patient-level sensitivity summaries.

The original source folders were:

- original `classification_v2` ResNet image classification code
- original `multi_modal` ResNet + rule-based classification code

Large intermediate outputs, logs, CAM figures, old exploratory scripts, and
temporary result files are intentionally not included here.

## Install

```bash
pip install -r requirements.txt
```

## 1. Train the ResNet frame classifier

```bash
python scripts/train_resnet.py ^
  --manifest path/to/train_dataset.json ^
  --output-dir outputs/resnet_train ^
  --backbone resnet50
```

The best checkpoint is saved as `outputs/resnet_train/best_model.pt`.

## 2. Video-level ResNet inference

The input manifest should be a JSON file with this structure:

```json
{
  "label_list": ["aom", "effusion", "normal", "perforation", "retraction", "tube", "tympanosclerosis"],
  "samples": [
    {
      "video_id": "V000001",
      "hq_img_paths": ["path/to/frame1.png", "path/to/frame2.png"],
      "labels": [0, 1, 0, 0, 0, 0, 0]
    }
  ]
}
```

Run inference and aggregate probabilities by video:

```bash
python scripts/run_video_inference.py ^
  --manifest path/to/train_dataset.json ^
  --checkpoint path/to/model.ckpt ^
  --output-dir outputs/video_infer ^
  --test-ids path/to/used_video_ids_only.csv
```

Outputs:

- `video_probs.csv`
- `video_metrics.json`
- `roc_curves_all_classes.png`

## 3. Multimodal fusion and paper tables

```bash
python scripts/run_multimodal_analysis.py ^
  --metadata "Data collection form Prospective_video_collection 6.11.25.xlsx" ^
  --video-probs outputs/video_infer/video_probs.csv ^
  --output-dir outputs/multimodal ^
  --threshold 0.90 ^
  --strategy paper_v3 ^
  --threshold-sweep 0.50 0.60 0.70 0.75 0.80 0.85 0.90 0.95 1.00
```

Outputs:

- `cnn_hybrid_predictions_table.csv`
- `overall_accuracy_with_95CI.csv`
- `per_class_metrics_3methods_with_95CI.csv`
- `auroc_with_95CI.csv`
- `ablation_threshold_results.csv`
- `patient_level_cohort_summary.csv`
- `patient_level_clustered_bootstrap_overall.csv`
- `patient_level_one_ear_per_patient_summary.csv`

## Notes

- The default threshold is `0.90`, matching the best observed operating point
  in the external paired VUMC cohort. In the manuscript this is treated as
  exploratory, not prespecified.
- The default strategy is `paper_v3`, which reproduces the final exploratory
  analysis script and the manuscript's 85/104 hybrid accuracy. Use
  `--strategy simple` for a stricter single-label rule that is easier to
  interpret but does not reproduce the submitted tables.
- Tympanometry rules are intentionally transparent:
  - Type A, As, Ad -> normal
  - Type C -> retraction
  - Type B with ECV <= 2.0 mL -> effusion
  - Type B with ECV > 2.0 mL or Open -> perforation
- Tube and tympanosclerosis CNN predictions are preserved by default because
  tympanometry alone is reductive for those classes.
