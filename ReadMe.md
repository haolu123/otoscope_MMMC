# OtoTymp AI Reproducibility Package

This repository contains the public reproducibility package for the PLOS ONE revision:

**Confidence-guided integration of otoscopy videos and tympanometry for middle ear disease diagnosis: A multi-center otoscopy study with external paired-cohort evaluation**

The package includes cleaned analysis code, anonymized intermediate ResNet video embeddings, CNN video-level probabilities, CNN-hybrid prediction tables, aggregate result tables, and metadata dictionaries. Raw clinical images, videos, patient identifiers, filesystem paths, and private ID mappings are not included.

## Repository Contents

- `code/`: cleaned Python implementation for CNN-only and CNN-hybrid analyses.
- `code/scripts/verify_reproducibility.py`: verification script for the packaged intermediate outputs.
- `embeddings/video_embeddings.npy`: anonymized video-level ResNet-50 penultimate embeddings for the 104-video paired tympanometry analysis subset.
- `embeddings/embedding_index.csv`: row index linking each embedding vector to an anonymized video ID.
- `embeddings/embedding_metadata.json`: checkpoint/config hashes and embedding-generation details.
- `derived_tables/video_probs.csv`: CNN video-level probabilities filtered to the same 104 videos as `cnn_hybrid_predictions_table.csv`.
- `derived_tables/cnn_hybrid_predictions_table.csv`: paired tympanometry subset used for CNN-hybrid comparison.
- `derived_tables/*_with_95CI.csv`: aggregate accuracy, per-class, and AUROC tables with 95% confidence intervals.
- `metadata_dictionary/`: class-label mappings, column descriptions, and dataset split/cohort summaries.
- `docs/`: reproducibility notes, citation metadata, license, and saved verification output.

## Reproducibility Check

Create a Python environment using `code/requirements.txt`, then run:

```bash
python code/scripts/verify_reproducibility.py --package-root .
```

The verification script checks that `video_embeddings.npy`, `embedding_index.csv`, `video_probs.csv`, and `cnn_hybrid_predictions_table.csv` contain the same 104 anonymized videos. It then recomputes the core paired-cohort results from the packaged files:

- CNN-only overall accuracy: `0.6346153846153846`
- Hybrid overall accuracy: `0.8173076923076923`

The saved verification record is available at `docs/reproducibility_check.json`.

## Notes On Embeddings

The released embeddings are video-level means of frame-level ResNet-50 penultimate feature vectors. They are restricted to the 104 external videos with paired tympanometry used for the CNN-hybrid comparison. The corresponding CNN probabilities in `derived_tables/video_probs.csv` are filtered to the same 104 videos.

## Data Availability

The clinical dataset is not publicly available due to IRB and institutional privacy restrictions. Qualified researchers may request access from the corresponding author subject to institutional approval and data-use agreement.

## Citation And License

Citation metadata is provided in `docs/CITATION.cff`. The released code is distributed under the license in `docs/LICENSE`.
