import math
from collections import Counter
from typing import List, Iterable, Any
import numpy as np
from typing import Any, Iterable, List, Optional, Tuple
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt




class KNN:
    def __init__(self, k: int = 3, metric: str = "euclidean"):
        if k < 1:
            raise ValueError("k must be >= 1")
        if metric not in ("euclidean", "manhattan"):
            raise ValueError("metric must be 'euclidean' or 'manhattan'")
        self.k = k
        self.metric = metric
        self._X: Optional[np.ndarray] = None
        self._y_int: Optional[np.ndarray] = None
        self._label_map: Optional[dict] = None
        self._inv_label_map: Optional[dict] = None

    def fit(self, X: Iterable[Iterable[float]], y: Iterable[Any]) -> None:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(list(y))
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if X_arr.shape[0] == 0:
            raise ValueError("Training data is empty")

        # map labels to integers for efficient bincount usage
        classes, y_int = np.unique(y_arr, return_inverse=True)
        self._label_map = {cls: int(i) for i, cls in enumerate(classes)}
        self._inv_label_map = {int(i): cls for i, cls in enumerate(classes)}
        self._X = X_arr
        self._y_int = y_int

        if self.k > self._X.shape[0]:
            raise ValueError("k cannot be larger than number of training samples")

    def _pairwise_distance(self, x: np.ndarray) -> np.ndarray:
        """Return distances between all training samples and single test sample x."""
        if self._X is None:
            raise ValueError("Classifier has not been fitted")
        if x.shape != (self._X.shape[1],):
            # allow 1-D shaped input otherwise raise
            try:
                x = x.ravel()
            except Exception:
                raise ValueError("Test sample has wrong shape")
        if self.metric == "euclidean":
            # squared euclidean is enough for ranking (faster), no sqrt
            return np.sum((self._X - x) ** 2, axis=1)
        else:  # manhattan
            return np.sum(np.abs(self._X - x), axis=1)

    def _nearest_indices(self, x: np.ndarray) -> np.ndarray:
        dists = self._pairwise_distance(x)
        return np.argsort(dists)[: self.k]

    def predict(self, X: Iterable[Iterable[float]]) -> np.ndarray:
        """Predict labels for samples in X. Returns original label types."""
        if self._X is None or self._y_int is None:
            raise ValueError("Classifier has not been fitted")
        X_arr = np.asarray(X, dtype=float)
        preds_int = np.empty(X_arr.shape[0], dtype=int)
        for i in range(X_arr.shape[0]):
            idx = self._nearest_indices(X_arr[i])
            counts = np.bincount(self._y_int[idx], minlength=self._y_int.max() + 1)
            preds_int[i] = int(np.argmax(counts))  # tie-break: smaller label index
        # map back to original labels
        return np.array([self._inv_label_map[int(v)] for v in preds_int])

    def predict_int(self, X: Iterable[Iterable[float]]) -> np.ndarray:
        """Return integer-encoded predictions (internal representation)."""
        if self._X is None or self._y_int is None:
            raise ValueError("Classifier has not been fitted")
        X_arr = np.asarray(X, dtype=float)
        preds_int = np.empty(X_arr.shape[0], dtype=int)
        for i in range(X_arr.shape[0]):
            idx = self._nearest_indices(X_arr[i])
            counts = np.bincount(self._y_int[idx], minlength=self._y_int.max() + 1)
            preds_int[i] = int(np.argmax(counts))
        return preds_int


def accuracy(true: Iterable[Any], pred: Iterable[Any]) -> float:
    t = np.asarray(list(true))
    p = np.asarray(list(pred))
    return float(np.mean(t == p)) * 100.0


def confusion_matrix_plot(true: Iterable[Any], pred: Iterable[Any], title: str = "Confusion Matrix"):
    cm = confusion_matrix(true, pred, labels=np.unique(true))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(true))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # tiny usage example
    X_train = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]])
    y_train = np.array(["A", "A", "B", "B", "A", "B"])
    clf = KNN(k=3)
    clf.fit(X_train, y_train)
    X_test = np.array([[1.0, 1.0], [8.5, 9.0]])
    preds = clf.predict(X_test)
    print("Predictions:", preds)