import argparse
import os
import numpy as np
import pandas as pd
import scipy.io
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

BASE = "path/to/mainfolder"

STEP = 5

def load_labels(cells):
    labels_df = pd.read_csv("data/CM_labels.txt", sep="\t", header=None, names=["cell", "label"])
    label_map = dict(zip(labels_df["cell"].astype(str), labels_df["label"].astype(str)))
    y = np.array([label_map[c] for c in cells], dtype=str)
    y = LabelEncoder().fit_transform(y)
    return y


def main():
    # Get input from command line 
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        required=True,
        choices=["RF", "XGB", "ET", "CAT"],
        help="feature importance source"
    )
    args = ap.parse_args()
    SOURCE = args.source

    os.chdir(BASE)
    os.makedirs("result/IFS_results", exist_ok=True)

    # load top100 genes 
    top_path = f"result/top100_results/FI_{SOURCE}_top100.csv"
    out_csv = f"result/IFS_results/IFS_RF_{SOURCE}_top100_step{STEP}.csv"

    top_df = pd.read_csv(top_path)
    gene_idx = top_df["gene_index"].values.astype(np.int32)

    # load expression matrix
    X = scipy.io.mmread("data/scp1303_RNA.mtx").tocsr().T.astype(np.float32)
    cells = pd.read_csv("data/scp1303_cells.txt", header=None)[0].astype(str).tolist()
    X_top = X[:, gene_idx].toarray().astype(np.float32)

    # load the data 
    y = load_labels(cells)
    
    # create 5-fold cross validation
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Perform incremental feature selection (IFS) using a Random Forest classifier.
    # Starting from STEP{5} genes, progressively increase feature number by STEP{5}.
    # For each feature subset, evaluate performance using 5-fold stratified CV.
    # Mean ACC, Macro-F1, Weighted-F1, and MCC are recorded.
    
    results = []

    for k in range(STEP, len(gene_idx) + 1, STEP):
        Xk = X_top[:, :k]

        accs, macrof1s, wtf1s, mccs = [], [], [], []

        for fold_i, (tr, te) in enumerate(skf.split(Xk, y), start=1):
            rf = RandomForestClassifier(
                n_estimators=200,
                max_features="sqrt",
                min_samples_leaf=3,
                n_jobs=16,
                random_state=42 + fold_i,
                class_weight="balanced_subsample"
            )
            rf.fit(Xk[tr], y[tr])
            pred = rf.predict(Xk[te])

            accs.append(accuracy_score(y[te], pred))
            macrof1s.append(f1_score(y[te], pred, average="macro"))
            wtf1s.append(f1_score(y[te], pred, average="weighted"))
            mccs.append(matthews_corrcoef(y[te], pred))

        results.append({
            "source_top100": SOURCE,
            "k_features": k,
            "acc": np.mean(accs),
            "macro_f1": np.mean(macrof1s),
            "weighted_f1": np.mean(wtf1s),
            "mcc": np.mean(mccs)
        })

        print(
            f"RF({SOURCE}) k={k}: "
            f"ACC={np.mean(accs):.4f} "
            f"MacroF1={np.mean(macrof1s):.4f} "
            f"WF1={np.mean(wtf1s):.4f} "
            f"MCC={np.mean(mccs):.4f}",
            flush=True
        )

    pd.DataFrame(results).to_csv(out_csv, index=False)
    print("Saved:", out_csv, flush=True)


if __name__ == "__main__":
    main()
