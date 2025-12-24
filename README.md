#Object_Recognition_Pipelines

Overview
- This repository contains lightweight Bag-of-Visual-Words (BoVW) experiments under the `BoVW/` folder. The core flow: local descriptors (SIFT / dense SIFT) -> k-means visual dictionary -> per-image histograms (catalog) -> k-NN classification.

Key files and roles
- `BoVW/main.py`: simple dataset-driven pipeline using `data/` (expects `data/train/<class>/<0001.png>` and `data/validation/...`). Produces `clusters_k_{k}_probably.pkl` and `catalog_k_{k}_probably.pkl` files and appends accuracy to `results.txt`.
- `BoVW/parameter_search_caltech.py`: variant that expects a `./caltech-101/` folder (train/validation split done by file list). Produces `resultsdeep_c101.txt` and `*_deep_c101.pkl` artifacts.
- `BoVW/test.py`: script for running inference on images in `data/test`; writes `test.txt` and `catalog_k_{k}_test.pkl`.
- `BoVW/kNN_model.py`: distance calculation and majority voting used by all scripts. (FROM SCRATCH)



Running / developer workflows (examples)
- Run a basic train+test on the dataset:

  python BoVW/main.py

- Run test on `data/test` images:

  python BoVW/test.py



# Spatial Pyramid Matching (SPM) + Baselines

# Spatial Pyramid Matching (SPM) + Baselines

# Spatial Pyramid Matching (SPM) + Baselines

- **Purpose:** Classical object recognition using **SIFT + Bag of Visual Words (BoVW) + Spatial Pyramid Matching (SPM)**, with simple deep-learning baselines (**VGG16, ResNet50**).
- **Dataset format:** Single **folder-of-classes** under `data/dataset_name/` (e.g., `class_x/*.jpg`); **no on-disk train/val/test split**.
- **Splitting:** Performed **internally and deterministically** by the code.
- **Main pipeline:** `spm2.py` (recommended) implements SIFT extraction, BoVW encoding, and multi-level SPM.
- **Lightweight runner:** `spm.py` provides a compact end-to-end experiment with a small config block.
- **Outputs:** Accuracy, macro-F1, precision, recall, and confusion matrices saved under `save_<dataset_name>/`.
- **Utilities:** `collect_results.py` aggregates metrics into CSV; `plot_class_distribution.py` produces class-count tables and plots.
- **Deep baselines:** `vgg_classifier.py` (VGG16) and `resnet.py` (ResNet50) reuse the same internal splitting logic and run on CPU by default.
- **Requirements:** Python 3.10+, `numpy`, `opencv-python` or `opencv-contrib-python` (SIFT), `scikit-learn`, `matplotlib`, `tqdm` (optional: `torch`, `torchvision`).
- **Notes:** Larger vocabularies (`M`) and deeper pyramids (`L`) increase memory usage; fix seeds for reproducibility.
- **Acknowledgment:** The implementation and structure of the repository  
  **https://github.com/MercuriXito/Spm** were helpful in guiding the development of our own SPM and dataset-handling code.
- **Reference:** Lazebnik et al., *Beyond Bags of Features: Spatial Pyramid Matching for Recognizing Natural Scene Categories*, CVPR 2006.


 

HOG pipeline
- `HOG/main.py`: HOG feature extraction + classifier training with hyperparameter search; uses `data/<dataset>` layout with train/validation/test splits.
- Run with default dataset from `HOG/config.py`:

  python HOG/main.py

- Run with a specific dataset name under `data/` or a full path:

  python HOG/main.py --dataset dataset1
  python HOG/main.py --dataset /path/to/dataset_root
