import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from datetime import datetime
from config import RESULTS_DIR, USE_PCA, PCA_VARIANCE

class ExperimentTracker:
    def __init__(self, filename_prefix="caltech101", model_type="SVM"):
        self.model_type = model_type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.txt_path = RESULTS_DIR / f"{filename_prefix}_{model_type}_{timestamp}.log"
        self.csv_path = RESULTS_DIR / f"{filename_prefix}_{model_type}_{timestamp}.csv"
        self.plot_path = RESULTS_DIR / f"{filename_prefix}_{model_type}_{timestamp}_cm.png"

        with open(self.txt_path, "w") as f:
            f.write(f"Experiment Log ({model_type}) - {timestamp}\n")
            f.write(f"PCA Enabled: {USE_PCA} (Variance: {PCA_VARIANCE})\n" + "=" * 100 + "\n")

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Cell", "Bins", "SPM", "Param", "Val_Acc", "Val_F1", 
                             "Val_Prec", "Val_Rec", "Time_Load(s)", "Time_Train(s)"])

        print(f"Running Experiment: {model_type} | PCA: {USE_PCA}")
        print(f"{'Cell':<6} | {'Bins':<6} | {'SPM':<10} | {'Param':<6} | {'Val Acc':<8} | {'Val F1':<8} | {'Time(s)':<8}")
        print("-" * 80)

    def log_run(self, cfg, metrics, times):
        cell, bins, spm, param = cfg
        acc, prec, rec, f1 = metrics
        t_load, t_train = times
        spm_str = str(spm) if spm else "None"

        print(f"{cell:<6} | {bins:<6} | {spm_str:<10} | {param:<6} | {acc:.4f}   | {f1:.4f}   | {t_train + t_load:.1f}")

        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([cell, bins, spm_str, param, acc, f1, prec, rec, 
                                    f"{t_load:.2f}", f"{t_train:.2f}"])

    def log_best(self, cfg, acc):
        with open(self.txt_path, "a") as f:
            f.write(f"\nBEST CONFIG: {cfg} -> Acc: {acc:.4f}\n")
        print("-" * 80 + f"\nBest Config: {cfg} | Acc: {acc:.4f}")

    def log_final_test(self, acc, report, cm):
        with open(self.txt_path, "a") as f:
            f.write(f"\nFINAL TEST RESULTS\nAccuracy: {acc:.4f}\n{report}\n{np.array2string(cm)}\n")

    def save_confusion_matrix(self, cm, class_names):
        print(f"[INFO] Saving Confusion Matrix Plot...")
        fig, ax = plt.subplots(figsize=(30, 30))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical', values_format='d')
        plt.title(f"Test Confusion Matrix - {self.model_type}", fontsize=24)
        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.close(fig)
        print(f"[INFO] Plot saved to: {self.plot_path}")