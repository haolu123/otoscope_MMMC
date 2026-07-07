# OtoTymp AI Reproducibility Package

This package contains code, intermediate ResNet video embeddings, video-level CNN probabilities, CNN-hybrid prediction tables, aggregate metrics, and metadata dictionaries for the PLOS ONE revision of:

**Confidence-guided integration of otoscopy videos and tympanometry for middle ear disease diagnosis: A multi-center otoscopy study with external paired-cohort evaluation**

## Contents

- `code/`: cleaned Python implementation for CNN-only and CNN-hybrid analyses.
- `embeddings/video_embeddings.npy`: anonymized video-level ResNet-50 penultimate embeddings for the paired tympanometry analysis subset.
- `embeddings/embedding_index.csv`: row index linking each embedding vector to an anonymized video ID.
- `embeddings/embedding_metadata.json`: checkpoint/config hashes and embedding-generation details.
- `derived_tables/video_probs.csv`: CNN video-level probabilities filtered to the same paired tympanometry subset as `cnn_hybrid_predictions_table.csv`.
- `derived_tables/cnn_hybrid_predictions_table.csv`: paired tympanometry subset used for CNN-hybrid comparison.
- `metadata_dictionary/`: class-label mappings, column descriptions, and split/cohort summaries.

## Reproduction

Create an environment with `code/requirements.txt`, then run the cleaned scripts in `code/scripts/`.
The original analysis used a ResNet-50 checkpoint and averaged frame-level softmax probabilities to obtain video-level CNN probabilities. The released embeddings are video-level means of the penultimate ResNet-50 feature vectors across the same HQ frames, restricted to the 104 videos with paired tympanometry used in the CNN-hybrid comparison.

To verify the packaged intermediate outputs reproduce the core paired-cohort results:

```bash
python code/scripts/verify_reproducibility.py --package-root .
```

This checks that `video_embeddings.npy`, `embedding_index.csv`, `video_probs.csv`, and `cnn_hybrid_predictions_table.csv` contain the same 104 anonymized videos, then recomputes the CNN-only and hybrid overall accuracies reported in `derived_tables/overall_accuracy_with_95CI.csv`. The saved verification record is `docs/reproducibility_check.json`.

## Privacy

Video identifiers are anonymized as `V000001`, `V000002`, etc. No raw images, videos, patient identifiers, filesystem paths, or private mapping files are included in the public package.
