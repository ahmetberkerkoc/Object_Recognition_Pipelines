import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Assuming these are your local helpers
from utils import test_and_make_dir, currentTime, json_dump


cfg = {
    "M": 512,               # Vocab size
    "L": 2,                 # Pyramid levels
    "k": 10,                # kNN neighbors
    "img_size": 224,        # Resize dimension
    "sift_step": 8,
    "sift_patch": 16,
    "max_kmeans_samples": 30000,
    "dataset_path": "dataset2",
    "save_root": "experiments"
}


def get_dense_sift(imgs, step, patch):
    sift = cv2.SIFT_create()
    h, w = imgs.shape[1:3]
    kp = [cv2.KeyPoint(x, y, patch) for y in range(step, h, step) for x in range(step, w, step)]
    
    all_descs = []
    for img in tqdm(imgs, desc="Extracting SIFT"):
        _, des = sift.compute(img, kp)
        if des is None:
            des = np.zeros((len(kp), 128), dtype=np.float32)
        all_descs.append(des.astype(np.float32))
    return np.array(all_descs)



def build_spm_features(sift_feats, km_model, cfg):
    N, P, _ = sift_feats.shape
    # Calculate grid size based on image dimensions and step
    gh = len(range(cfg['sift_step'], cfg['img_size'], cfg['sift_step']))
    gw = len(range(cfg['sift_step'], cfg['img_size'], cfg['sift_step']))
    
    # Visual word assignment
    words = km_model.predict(sift_feats.reshape(-1, 128)).reshape(N, gh, gw)
    
    pyramid_feats = []
    for i in range(N):
        level_hists = []
        for lv in range(cfg['L'] + 1):
            # Statistical weighting for SPM
            weight = 1.0/(2**cfg['L']) if lv == 0 else 1.0/(2**(cfg['L']-lv+1))
            
            grid_parts = 2**lv
            for r_block in np.array_split(words[i], grid_parts, axis=0):
                for cell in np.array_split(r_block, grid_parts, axis=1):
                    h, _ = np.histogram(cell, bins=cfg['M'], range=(0, cfg['M']))
                    level_hists.append(h.astype(float) * weight)
        
        pyramid_feats.append(np.concatenate(level_hists))
    
    return np.array(pyramid_feats)



def run_experiment(c):
    from data import CustomDataset # Internal loader
    
    # Setup directory for this specific run
    run_name = f"run_M{c['M']}_L{c['L']}_{currentTime()}"
    out_dir = os.path.join(c['save_root'], run_name)
    test_and_make_dir(out_dir)
    
    # Load data
    print(f"--- Starting Experiment: {run_name} ---")
    ds = CustomDataset(c['dataset_path'], resize_h=c['img_size'], resize_w=c['img_size'])
    X_tr, y_tr, X_val, y_val, X_te, y_te = ds.xy(split=(0.6, 0.2, 0.2))

    # 1. features
    s_tr = get_dense_sift(X_tr, c['sift_step'], c['sift_patch'])
    s_val = get_dense_sift(X_val, c['sift_step'], c['sift_patch'])
    s_te = get_dense_sift(X_te, c['sift_step'], c['sift_patch'])

    # 2.vocab
    flat = s_tr.reshape(-1, 128)
    if len(flat) > c['max_kmeans_samples']:
        flat = flat[np.random.choice(len(flat), c['max_kmeans_samples'], replace=False)]
    
    print(f"Clustering {len(flat)} descriptors...")
    km = MiniBatchKMeans(n_clusters=c['M'], batch_size=1000, n_init='auto').fit(flat)

    # 3. spm transfoprmation
    print("Building Spatial Pyramids...")
    X_tr_spm = build_spm_features(s_tr, km, c)
    X_val_spm = build_spm_features(s_val, km, c)
    X_te_spm = build_spm_features(s_te, km, c)

    # 4. Classifier
    print(f"Training kNN (k={c['k']})...")
    clf = KNeighborsClassifier(n_neighbors=c['k'], n_jobs=-1)
    clf.fit(X_tr_spm, y_tr)

    # 5. Evaluation and Logging
    results = {"config": c, "metrics": {}}
    
    for name, feat, label in [("val", X_val_spm, y_val), ("test", X_te_spm, y_te)]:
        preds = clf.predict(feat)
        acc = accuracy_score(label, preds)
        print(f"Result for {name}: {acc:.4f}")
        
        # Save classification report to dict
        results["metrics"][name] = classification_report(label, preds, output_dict=True)
        
        # Quick ConfMat Plot
        cm = confusion_matrix(label, preds)
        plt.figure()
        plt.imshow(cm, cmap='Blues')
        plt.title(f"{name} - Acc: {acc:.2f}")
        plt.savefig(os.path.join(out_dir, f"cm_{name}.png"))
        plt.close()

    # Save the config and results so we dont forget what we ran
    with open(os.path.join(out_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"--- Done. Results saved to {out_dir} ---")

if __name__ == "__main__":
    run_experiment(cfg)