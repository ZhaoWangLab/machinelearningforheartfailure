import os
import numpy as np
import pandas as pd
import scipy.io
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

BASE = "path/to/mainfolder"


def load_data(base=BASE):
    os.chdir(base)
    os.makedirs("result/top1000_results", exist_ok=True)

    X = scipy.io.mmread("data/CM_RNA.mtx").tocsr().T.astype(np.float32)
    cells = pd.read_csv("data/CM_cells.txt", header=None)[0].astype(str).tolist()
    genes = pd.read_csv("data/CM_genes.txt", header=None)[0].astype(str).to_numpy()
    labels = pd.read_csv("data/CM_labels.txt",sep="\t", header=None, names=["cell", "label"])
    label_map = dict(zip(labels["cell"].astype(str), labels["label"].astype(str)))
    y = np.array([label_map[c] for c in cells], dtype=str)
    y = LabelEncoder().fit_transform(y)

    return X, y, genes, cells

def compute_class_weights(y):
    n_classes = len(np.unique(y))
    counts = np.bincount(y, minlength=n_classes)
    counts = np.maximum(counts, 1)
    class_weights = y.size / (n_classes * counts)
    return class_weights


if __name__ == "__main__":
    topK = 1000
    outpath = "result/top1000_results/xgb_top1000.csv"

    X, y, genes, cells = load_data(BASE)

    class_weights = compute_class_weights(y)
    sample_w = class_weights[y].astype(np.float32)

    n_classes = len(np.unique(y))

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.4,
        reg_lambda=1.0,
        n_jobs=12,
        random_state=42,
        tree_method="hist",
        eval_metric="mlogloss",
        verbosity=1
    )

    model.fit(X, y, sample_weight=sample_w)

    imp = model.feature_importances_  # length = n_features

    top_idx = np.argsort(-imp)[:topK]
    pd.DataFrame({
        "gene_index": top_idx,
        "gene": genes[top_idx],
        "importance": imp[top_idx],
    }).to_csv(outpath, index=False)

    print("Saved:", outpath, flush=True)
