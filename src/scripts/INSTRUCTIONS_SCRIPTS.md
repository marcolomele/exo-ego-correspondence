## Purpose

Miscellaneous scripts for data processing, feature extraction, and pipeline utilities.

## Scripts

### Data Processing

- **`download_and_process_data.py`** - Downloads EgoExo4D videos, extracts annotated frames, and organizes them for training. Supports cooking and health scenarios.

- **`filter_annotations.py`** - Extracts take UIDs from relation annotation files based on filtering criteria (scenario type, presence of both ego/exo cameras, valid object masks).

- **`create_pairs.py`** - Generates ego-exo frame pair JSON files from processed annotations. Creates `{split}_{setting}_pairs.json` files for train/val/test splits.

- **`create_split_from_features.py`** - Creates data splits at frame level, outputting new `split.json` and `{split}_exoego_pairs.json`.

### Feature Extraction

- **`precompute_features.py`** - Pre-extracts DINOv3 features for all dataset images (basic version without pair sampling).

- **`precompute_features_dinov2.py`** - Pre-extracts DINOv2 features with random pair sampling (~13K pairs). Updates JSON files and `split.json` to reflect sampled data.

- **`precompute_features_dinov3.py`** - Pre-extracts DINOv3 features with random pair sampling (~13K pairs). Same functionality as DINOv2 version but uses DINOv3 backbone.

- **`precompute_features_resnet50.py`** - Pre-extracts ResNet-50 (DINO pretrained) features with random pair sampling. Alternative backbone for comparison experiments.

### Utilities

- **`load_relation_masks.py`** - Helper to load and decode relation masks from EgoExo4D annotations. Demonstrates mask decoding from LZString compressed format to binary numpy arrays.

- **`upload_to_drive.py`** - Uploads directories to Google Drive using rclone with subfolder-level progress tracking.
