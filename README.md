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

This repository implements a classical object recognition pipeline based on **SIFT + Bag of Visual Words (BoVW) + Spatial Pyramid Matching (SPM)**, together with simple deep-learning baselines (**VGG16, ResNet50**). Datasets are expected in a single **folder-of-classes** format under `data/dataset_name/` (e.g., `data/dataset_name/class_x/*.jpg`), with **no on-disk train/val/test split**; all splits are handled internally and deterministically. The main SPM implementation is in `spm2.py` (recommended), with a lightweight runner in `spm.py`; results (accuracy, macro-F1, precision, recall) and confusion matrices are saved under `save_<dataset_name>/`. Utilities include `collect_results.py` for CSV aggregation and `plot_class_distribution.py` for class-count tables and plots. Requirements include Python 3.10+, `numpy`, `opencv-python` or `opencv-contrib-python` (SIFT), `scikit-learn`, `matplotlib`, and `tqdm` (optional: `torch`, `torchvision` for baselines). Larger vocabularies and deeper pyramids increase memory usage; fix seeds for reproducibility. Reference: Lazebnik et al., *CVPR 2006*.
 

HOG pipeline
- `HOG/main.py`: HOG feature extraction + classifier training with hyperparameter search; uses `data/<dataset>` layout with train/validation/test splits.
- Run with default dataset from `HOG/config.py`:

  python HOG/main.py

- Run with a specific dataset name under `data/` or a full path:

  python HOG/main.py --dataset dataset1
  python HOG/main.py --dataset /path/to/dataset_root
