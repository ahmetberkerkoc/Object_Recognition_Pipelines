# Copilot / AI agent instructions — Object_Recognition_Pipelines

Overview
- This repository contains lightweight Bag-of-Visual-Words (BoVW) experiments under the `BoVW/` folder. The core flow: local descriptors (SIFT / dense SIFT) -> k-means visual dictionary -> per-image histograms (catalog) -> k-NN classification.

Key files and roles
- `BoVW/main.py`: simple dataset-driven pipeline using `data/` (expects `data/train/<class>/<0001.png>` and `data/validation/...`). Produces `clusters_k_{k}_probably.pkl` and `catalog_k_{k}_probably.pkl` files and appends accuracy to `results.txt`.
- `BoVW/parameter_search_caltech.py`: variant that expects a `./caltech-101/` folder (train/validation split done by file list). Produces `resultsdeep_c101.txt` and `*_deep_c101.pkl` artifacts.
- `BoVW/test.py`: script for running inference on images in `data/test`; writes `test.txt` and `catalog_k_{k}_test.pkl`.
- `BoVW/kNN_model.py`: distance calculation and majority voting used by all scripts.



Running / developer workflows (examples)
- Run a basic train+test on the dataset:

  python BoVW/main.py

- Run test on `data/test` images:

  python BoVW/test.py

