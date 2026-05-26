import os
import numpy as np
import pandas as pd
import scipy.io
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

BASE = "path/to/mainfolder"


def main():
    os.chdir(BASE)
    os.makedirs("result/top100_results", exist_ok=True)

    ## Load the data
    # load RF top1000 gene indices
    top_df = pd.read_csv("result/top1000_results/top1000_vote_ge2_candidates.csv")
    gene_idx = top_df["gene_index"].values.astype(np.int32)
    top_genes = top_df["gene"].astype(str).values
    topK = len(gene_idx)

    # load expression matrix
    X = scipy.io.mmread("data/CM_RNA.mtx").tocsr().T.astype(np.float32)
    cells = pd.read_csv("data/CM_cells.txt", header=None)[0].astype(str).tolist()

    # subset by gene_index
    X = X[:, gene_idx]

    # load labels
    labels_df = pd.read_csv("data/CM_labels.txt", sep="\t", header=None, names=["cell", "label"])
    label_map = dict(zip(labels_df["cell"], labels_df["label"]))

    y = np.array([label_map[c] for c in cells], dtype=str)
    y = LabelEncoder().fit_transform(y)

    ## Repeat training 50 times using balanced subsampling. Compute gene importance ranking in each run
    ## Select top gene based on average rank across runs
    rng = np.random.default_rng(42)
    n_classes = len(np.unique(y))
    n_repeats = 50
    counts = np.bincount(y, minlength=n_classes)
    n_per_class = counts.min()

    all_ranks = np.zeros((n_repeats, topK), dtype=np.int32)

    for r in range(n_repeats):
        print(f"Repeat {r+1}/{n_repeats}", flush=True)

        idx = np.concatenate([
            rng.choice(np.where(y == c)[0], size=n_per_class, replace=False)
            for c in range(n_classes)
        ])
        rng.shuffle(idx)

        X_sub = X[idx].toarray().astype(np.float32)
        y_sub = y[idx]

        rf = RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            n_jobs=16,
            random_state=42 + r
        )

        rf.fit(X_sub, y_sub)
        imp = rf.feature_importances_

        order = np.argsort(-imp)
        ranks = np.empty(topK, dtype=np.int32)
        ranks[order] = np.arange(1, topK + 1)
        all_ranks[r] = ranks

    repeat_ids = [f"repeat_{i+1}" for i in range(n_repeats)]
    df_ranks = pd.DataFrame(all_ranks, index=repeat_ids, columns=top_genes)
    df_ranks.to_csv("result/top100_results/FI_RF_top100_all_ranks.csv")

    avg_rank = all_ranks.mean(axis=0)
    std_rank = all_ranks.std(axis=0)

    summary = pd.DataFrame({
        "gene_index": gene_idx,
        "gene": top_genes,
        "avg_rank": avg_rank,
        "std_rank": std_rank
    }).sort_values("avg_rank", ascending=True)

    summary.to_csv("result/top100_results/FI_RF_top1000_summary.csv", index=False)
    summary.head(100).to_csv("result/top100_results/FI_RF_top100.csv", index=False)

    print("Saved RF top100 results", flush=True)


if __name__ == "__main__":
    main()
