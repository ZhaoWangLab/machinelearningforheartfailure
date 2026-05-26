import numpy as np
import pandas as pd
import scipy.io
from sklearn.preprocessing import LabelEncoder
import os

os.chdir("path/to/mainfolder/result")

# Load the data
X = scipy.io.mmread("data/CM_RNA.mtx").tocsr().T
genes = np.loadtxt("data/CM_genes.txt", dtype=str)
cells = pd.read_csv("data/CM_cells.txt", header=None)[0].astype(str).tolist()

labels_df = pd.read_csv("data/CM_labels.txt", sep="\t", header=None, names=["cell", "label"])
label_map = dict(zip(labels_df["cell"].astype(str), labels_df["label"].astype(str)))
y = np.array([label_map[c] for c in cells], dtype=str)
y_encoded = LabelEncoder().fit_transform(y)

# Extract the selected genes from IFS results.
selected_genes =  pd.read_csv("result/IFS_results/genes_ACC80_rank.csv")["gene"].to_list()
gene_to_idx = {g: i for i, g in enumerate(genes)}
idx = [gene_to_idx[g] for g in selected_genes]
X_selected = X[:, idx]   # cells × selected genes
X_selected = X_selected.toarray().astype(np.float32)

# Build a final cells × selected genes expression table.
df = pd.DataFrame(X_selected, columns=selected_genes)
df["label"] = y_encoded.astype(int)  
df["class"] = y     

# Save the processed dataset for downstream SHAP analysis.
df.to_csv("data/HF_dataset_shap_genes.csv", index=False)