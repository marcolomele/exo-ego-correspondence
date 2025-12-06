There are two approaches to create the `root/` folder for O-MaMa.

## Approach 1: Pre-processed Data

1. Download pre-processed data from [Google Drive](https://drive.google.com/drive/folders/1iH0zl6lrLcFze9L25g9JUVA6CBWjOGnJ?usp=drive_link) (Health scenario in full, Cooking in part). Access requires confirmation. Place contents in `processed/` folder.
2. Run `../../src/scripts/FastSAM_masks_creation.py` (ensure `fastsam_extraction/` exists in `src/`).
3. Run `../../src/scripts/precompute_features_{feature_extractor}.py` (ensure appropriate feature extractor structure exists).

## Approach 2: From Scratch

1. Follow instructions from the [O-MaMa official repository](https://github.com/Maria-SanVil/O-MaMa/tree/main?tab=readme-ov-file).
2. Run `../../src/scripts/download_and_process_data.py` and rename the output folder to `processed/`.
3. Run `../../src/fastsam_extraction/extract_masks_FastSAM.py` (ensure `fastsam_extraction/` exists in `src/`). This creates `Masks_{split}_{source}2{destination}/` folders.
4. Run `../../src/scripts/precompute_features_{feature_extractor}.py` (ensure appropriate feature extractor structure exists).

**Notes**:
- Use non-downscaled data for optimal O-MaMa performance.
- DINOv2 and ResNet50 are loaded with `torch.load()`; DINOv3 requires cloning its official repository.

## Directory Structure

Following either approach, `root/` will contain:

```
root/
├── processed/
│   ├── split.json
│   └── {take_UID}/
│       ├── annotation.json
│       └── {cam}/
│           └── {frame}.jpg
├── Masks_TRAIN_EXO2EGO/
│   └── {take_UID}/
│       └── {cam}/
│           ├── {idx}_boxes.npy
│           └── {idx}_masks.npz
├── Masks_VAL_EXO2EGO/
├── Masks_TEST_EXO2EGO/
├── dataset_jsons/
│   ├── train_exoego_pairs.json
│   ├── val_exoego_pairs.json
│   └── test_exoego_pairs.json
└── precomputed_features/
    └── {take_UID}/
        └── {cam}/
            └── {frame}.npz
```
where
```
└── precomputed_features/
│   └── {take_UID_1}/
│       └── {cam_1}/
│           └── {frame_1}.npz
│           └── ...
│       └── {cam_2}/
│           └── {frame_1}.npz
│           └── ...```
```