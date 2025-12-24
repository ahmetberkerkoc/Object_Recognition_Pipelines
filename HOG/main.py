import argparse
import itertools
import pathlib
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from config import (
    MODEL_TYPE, CELL_SIZES, BINS_LIST, SPM_OPTIONS, 
    SVM_C_VALUES, RF_ESTIMATORS, KNN_NEIGHBORS, C_LOGISTIC_REGRESSION,
    DATASET_DIR, USE_PCA, PROJECT_ROOT
)
from dataset import load_data_cached
from model import get_classifier
from tracker import ExperimentTracker

def get_hyperparams(model_type):
    if model_type == "SVM": return SVM_C_VALUES
    if model_type == "RF": return RF_ESTIMATORS
    if model_type == "KNN": return KNN_NEIGHBORS
    if model_type == "LR": return C_LOGISTIC_REGRESSION
    return []

def hyperparameter_search(dataset_root, tracker):
    best_acc = -1
    best_cfg = None
    hyperparams = get_hyperparams(MODEL_TYPE)
    param_grid = itertools.product(CELL_SIZES, BINS_LIST, SPM_OPTIONS, hyperparams)

    for cell, bins, spm, param in param_grid:
        current_cfg = (cell, bins, spm, param)
        
        t0 = time.time()
        Xtr, ytr, Xval, yval, _, _, _ = load_data_cached(dataset_root, cell, bins, spm)
        
        scaler = StandardScaler(with_mean=False)
        Xtr = scaler.fit_transform(Xtr)
        Xval = scaler.transform(Xval)
        t_load = time.time() - t0

        t1 = time.time()
        clf = get_classifier(MODEL_TYPE, param)
        clf.fit(Xtr, ytr)
        
        if USE_PCA:
            n_in = Xtr.shape[1]
            n_out = clf.named_steps['pca'].n_components_
            print(f"  [PCA] {n_in} -> {n_out} feats")
            
        t_train = time.time() - t1

        preds = clf.predict(Xval)
        acc = accuracy_score(yval, preds)
        metrics = (
            acc,
            precision_score(yval, preds, average="macro", zero_division=0),
            recall_score(yval, preds, average="macro", zero_division=0),
            f1_score(yval, preds, average="macro", zero_division=0)
        )

        tracker.log_run(current_cfg, metrics, (t_load, t_train))

        if acc > best_acc:
            best_acc = acc
            best_cfg = current_cfg

    tracker.log_best(best_cfg, best_acc)
    return best_cfg

def final_test(dataset_root, best_cfg, tracker):
    cell, bins, spm, param = best_cfg
    print(f"\n[INFO] Starting Final Test ({MODEL_TYPE}) with Best Config: {best_cfg}")

    Xtr, ytr, Xval, yval, Xte, yte, class_names = load_data_cached(dataset_root, cell, bins, spm)
    
    # Merge Train + Val
    Xtrain = np.vstack([Xtr, Xval])
    ytrain = np.hstack([ytr, yval])

    scaler = StandardScaler(with_mean=False)
    Xtrain = scaler.fit_transform(Xtrain)
    Xte = scaler.transform(Xte)

    clf = get_classifier(MODEL_TYPE, param)
    clf.fit(Xtrain, ytrain)
    
    if USE_PCA:
        n_in = Xtrain.shape[1]
        n_out = clf.named_steps['pca'].n_components_
        var = np.sum(clf.named_steps['pca'].explained_variance_ratio_)
        print(f"[INFO] Final PCA: {n_in} -> {n_out} feats (Var: {var:.2%})")

    preds = clf.predict(Xte)
    acc = accuracy_score(yte, preds)
    report = classification_report(yte, preds, target_names=class_names, digits=4)
    cm = confusion_matrix(yte, preds)

    tracker.log_final_test(acc, report, cm)
    tracker.save_confusion_matrix(cm, class_names)
    print(f"Final Test Accuracy: {acc:.4f}")

def resolve_dataset_dir(dataset_arg):
    if not dataset_arg:
        return DATASET_DIR

    candidate = pathlib.Path(dataset_arg)
    if candidate.exists():
        return candidate

    return PROJECT_ROOT.parent / "data" / dataset_arg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HOG training and evaluation.")
    parser.add_argument(
        "--dataset",
        help="Dataset name under data/ or a path to a dataset root.",
        default=None,
    )
    args = parser.parse_args()

    dataset_root = resolve_dataset_dir(args.dataset)
    if not dataset_root.exists():
        print(f"[ERROR] Dataset not found at {dataset_root}")
    else:
        tracker = ExperimentTracker(dataset_root.name, MODEL_TYPE)
        best_config = hyperparameter_search(dataset_root, tracker)
        final_test(dataset_root, best_config, tracker)
