from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from config import USE_PCA, PCA_VARIANCE

def get_classifier(model_type, param):
    """Factory to create classifier, optionally wrapped in PCA pipeline."""
    if model_type == "SVM":
        clf = LinearSVC(C=param, dual="auto", max_iter=1000)
    elif model_type == "RF":
        clf = RandomForestClassifier(n_estimators=int(param), n_jobs=-1, random_state=42)
    elif model_type == "KNN":
        clf = KNeighborsClassifier(n_neighbors=int(param), metric='euclidean', n_jobs=-1)
    elif model_type == "LR":
        clf = LogisticRegression(C=param, max_iter=1000, solver="saga", n_jobs=-1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if USE_PCA:
        return Pipeline([('pca', PCA(n_components=PCA_VARIANCE)), ('clf', clf)])
    return clf