import numpy as np
import pathlib
import time
from joblib import Parallel, delayed
from config import CACHE_DIR
from features import process_single_image

def cache_filename(cell_size, block_size, bins, spm_levels):
    spm_str = "none" if spm_levels is None else "".join(map(str, spm_levels))
    return CACHE_DIR / (
        f"hog_cell{cell_size}_block{block_size}_bins{bins}_spm{spm_str}_structured.npz"
    )

def resolve_split_dir(dataset_root: pathlib.Path, split_name: str) -> pathlib.Path | None:
    if split_name == "validation":
        for candidate in ("validation", "val"):
            split_dir = dataset_root / candidate
            if split_dir.exists():
                return split_dir
        return None
    split_dir = dataset_root / split_name
    return split_dir if split_dir.exists() else None

def list_images(class_dir: pathlib.Path):
    if not class_dir.exists(): return []
    patterns = ("*.png", "*.jpg", "*.jpeg")
    images = []
    for pattern in patterns:
        images.extend(class_dir.glob(pattern))
    return sorted(images)

def load_structured_dataset_parallel(dataset_root, cell_size, block_size, bins, spm_levels=None):
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root {dataset_root} not found.")

    train_dir = resolve_split_dir(dataset_root, "train")
    val_dir = resolve_split_dir(dataset_root, "validation")
    test_dir = resolve_split_dir(dataset_root, "test")

    if train_dir is None or val_dir is None:
        raise FileNotFoundError(f"Expected 'train' and 'validation' folders inside {dataset_root}")

    class_names = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
    all_tasks = []
    split_dirs = [("train", train_dir), ("validation", val_dir)]
    if test_dir: split_dirs.append(("test", test_dir))

    for split_name, split_dir in split_dirs:
        for label, cls in enumerate(class_names):
            class_dir = split_dir / cls
            for p in list_images(class_dir):
                all_tasks.append((p, label, split_name))

    print(f"\n[INFO] Extracting features for {len(all_tasks)} images (Parallel)...")
    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(process_single_image)(t[0], cell_size, block_size, bins, spm_levels)
        for t in all_tasks
    )

    Xtr, ytr, Xval, yval, Xte, yte = [], [], [], [], [], []
    for i, (_, label, split) in enumerate(all_tasks):
        feat = results[i]
        if feat is None: continue
        
        if split == "train": Xtr.append(feat); ytr.append(label)
        elif split == "validation": Xval.append(feat); yval.append(label)
        elif split == "test": Xte.append(feat); yte.append(label)

    return (
        np.vstack(Xtr), np.array(ytr),
        np.vstack(Xval), np.array(yval),
        np.vstack(Xte) if Xte else np.empty((0, Xtr[0].shape[0])), np.array(yte),
        class_names,
    )

def load_data_cached(dataset_root, cell, bins, spm):
    block = 2
    cache_path = cache_filename(cell, block, bins, spm)

    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return (data["Xtr"], data["ytr"], data["Xval"], data["yval"], 
                data["Xte"], data["yte"], data["class_names"].tolist())

    t_start = time.time()
    data = load_structured_dataset_parallel(dataset_root, cell, block, bins, spm)
    print(f"[INFO] Extraction finished in {time.time() - t_start:.2f}s")
    
    np.savez_compressed(cache_path, Xtr=data[0], ytr=data[1], Xval=data[2], yval=data[3], 
                        Xte=data[4], yte=data[5], class_names=np.array(data[6], dtype=object))
    return data