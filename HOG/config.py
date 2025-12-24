import pathlib

# --- MODEL SELECTION ---
MODEL_TYPE = "SVM"

# --- PCA SETTINGS ---
USE_PCA = False
PCA_VARIANCE = 0.95

# --- HYPERPARAMETERS ---
CELL_SIZES = [4, 6, 8, 10, 12]
BINS_LIST = [6, 9, 12, 15]
SPM_OPTIONS = [None, [0, 1]]

# Model-specific parameters
SVM_C_VALUES = [0.1, 1.0, 10.0]
RF_ESTIMATORS = [100, 300, 500]
KNN_NEIGHBORS = [1, 3, 5, 10]
C_LOGISTIC_REGRESSION = [0.1, 1.0, 10.0]

# --- PATHS ---
PROJECT_ROOT = pathlib.Path(__file__).parent
DATASET_DIR = PROJECT_ROOT.parent / "data/dataset1"
CACHE_DIR = PROJECT_ROOT / "hog_cache"
RESULTS_DIR = PROJECT_ROOT / "results"

CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)