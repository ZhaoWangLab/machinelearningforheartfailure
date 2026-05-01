"""
Single-cell RNA-seq disease classification: machine-learning pipeline.

Loads cell-by-gene expression from Parquet (split metadata / long-format
expression as produced by ``run_pipeline.R``, or legacy wide Parquet), performs
univariate feature selection, and evaluates XGBoost, random forest, and an MLP
with stratified *k*-fold cross-validation. Writes per-cell predictions to
``*_results.csv`` and ``*_results.parquet``.

CLI usage::

    python sc_pipeline.py <path/to/base.parquet> [--output-dir DIR]

When invoked from R, ``parquet_path`` is typically ``OUTPUT_PARQUET`` (e.g.
``SCP1303_CM.parquet``); companion files ``<base>_meta.parquet`` and
``<base>_expr.parquet`` are read when present.

If you use this code in published work, please cite the original study and
document package versions
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

__all__ = [
    "CLASS_MAP",
    "DATA_PATH",
    "LABEL_COLUMN_NAME",
    "METADATA_COLS",
    "MLP",
    "load_data",
    "main",
    "parse_args",
    "preprocess_data",
    "run_pipeline",
    "select_features",
]

# -----------------------------------------------------------------------------
# Data paths and labels (must stay consistent with run_pipeline.R)
# -----------------------------------------------------------------------------
#: Default base Parquet path when no CLI argument is given.
DATA_PATH: str = "SCP1303_CM.parquet"
#: Column in metadata holding disease / condition string labels.
LABEL_COLUMN_NAME: str = "disease_state"
#: Metadata columns expected in exports (order not required in wide tables).
METADATA_COLS: List[str] = ["cell_id", "disease_state", "sample_id"]

#: String label -> 0-based integer; order must match ``CLASS_LABELS`` in R.
CLASS_MAP: Dict[str, int] = {
    "normal": 0,
    "hypertrophic cardiomyopathy": 1,
    "dilated cardiomyopathy": 2,
}

# -----------------------------------------------------------------------------
# Cross-validation and reproducibility
# -----------------------------------------------------------------------------
RANDOM_STATE: int = 42
N_FOLDS: int = 5
STRATIFY_SPLIT: bool = True

# -----------------------------------------------------------------------------
# Feature selection
# -----------------------------------------------------------------------------
TOP_N_FEATURES: int = 1000

# -----------------------------------------------------------------------------
# MLP hyperparameters
# -----------------------------------------------------------------------------
MLP_HIDDEN_SIZES: List[int] = [512, 256, 128]
MLP_DROPOUT_RATE: float = 0.3
MLP_LEARNING_RATE: float = 0.001
MLP_BATCH_SIZE: int = 32
MLP_EPOCHS: int = 200
MLP_EARLY_STOPPING_PATIENCE: int = 10
MLP_WEIGHT_DECAY: float = 1e-10

_LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed ``parquet_path`` (optional) and ``output_dir`` (optional).
    """
    parser = argparse.ArgumentParser(
        description=(
            "scRNA ML pipeline: XGBoost, random forest, MLP. Expects "
            "<base>_meta.parquet and <base>_expr.parquet, or legacy single Parquet."
        )
    )
    parser.add_argument(
        "parquet_path",
        nargs="?",
        default=None,
        help=(
            "Base Parquet path (e.g. SCP1303_CM.parquet). Loads companion "
            "<base>_meta.parquet and <base>_expr.parquet. If omitted, uses DATA_PATH."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for *_results.csv and *_results.parquet. "
            "Default: directory of the Parquet file, or current working directory."
        ),
    )
    return parser.parse_args()


def _base_and_paths(data_path: str) -> Tuple[str, str]:
    """
    Split a data path into directory and file stem.

    Parameters
    ----------
    data_path
        Path to the base ``.parquet`` file (or any path sharing that stem).

    Returns
    -------
    tuple of str
        ``(directory, base_name)`` such that ``<dir>/<base>_meta.parquet`` matches
        R exports.
    """
    path = os.path.abspath(data_path)
    dir_path = os.path.dirname(path)
    base = os.path.splitext(os.path.basename(path))[0]
    return dir_path, base


def extract_dataset_name(data_path: str) -> str:
    """
    Derive the results file stem from the base Parquet path.

    Parameters
    ----------
    data_path
        Base Parquet path (e.g. ``.../SCP1303_CM.parquet``).

    Returns
    -------
    str
        Stem used for ``{stem}_results.csv`` (``_Final`` suffix stripped).
    """
    _, base = _base_and_paths(data_path)
    return base.replace("_Final", "")


def _load_meta_expr_parquets(
    data_path: str,
) -> Optional[Tuple[pd.DataFrame, csr_matrix, List[str]]]:
    """
    Load split metadata and long-format expression Parquet files.

    Parameters
    ----------
    data_path
        Base path; expects ``<dir>/<base>_meta.parquet`` and
        ``<dir>/<base>_expr.parquet``.

    Returns
    -------
    tuple or None
        ``(meta_df, X_sparse, gene_names)`` if both files exist; otherwise
        ``None``.
    """
    dir_path, base = _base_and_paths(data_path)
    meta_path = os.path.join(dir_path, f"{base}_meta.parquet")
    expr_path = os.path.join(dir_path, f"{base}_expr.parquet")
    if not (os.path.isfile(meta_path) and os.path.isfile(expr_path)):
        return None
    meta_df = pd.read_parquet(meta_path)
    meta_df = meta_df.loc[:, ~meta_df.columns.str.contains("^Unnamed")]
    expr_long = pd.read_parquet(expr_path)
    cells = meta_df["cell_id"].values
    genes = np.sort(expr_long["gene"].unique())
    cell_to_idx = {c: i for i, c in enumerate(cells)}
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    row = expr_long["cell_id"].map(cell_to_idx)
    col = expr_long["gene"].map(gene_to_idx)
    valid = row.notna() & col.notna()
    x_sparse = csr_matrix(
        (
            expr_long.loc[valid, "value"].values,
            (row[valid].astype(int), col[valid].astype(int)),
        ),
        shape=(len(cells), len(genes)),
    )
    return meta_df, x_sparse, list(genes)


def load_data(data_path: str) -> Tuple[pd.DataFrame, csr_matrix, List[str]]:
    """
    Load expression and metadata from Parquet (split or wide) or CSV.

    Parameters
    ----------
    data_path
        Path to ``.parquet`` or ``.csv``. Split layout is preferred when both
        companion files exist.

    Returns
    -------
    meta_df : pandas.DataFrame
        Cell-level metadata including ``METADATA_COLS``.
    X_sparse : scipy.sparse.csr_matrix
        Cells × genes sparse matrix.
    gene_names : list of str
        Gene identifiers aligned with columns of ``X_sparse``.

    Raises
    ------
    ValueError
        If the file extension is not ``.parquet`` or ``.csv``.
    """
    ext = os.path.splitext(data_path)[1].lower()
    if ext == ".parquet":
        split = _load_meta_expr_parquets(data_path)
        if split is not None:
            return split
        df = pd.read_parquet(data_path)
    elif ext == ".csv":
        df = pd.read_csv(data_path)
        df.to_parquet(f"{data_path[: len(data_path) - 4]}.parquet")
    else:
        raise ValueError(f"Unsupported format: {ext}. Use .parquet or .csv.")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    meta_cols = [c for c in METADATA_COLS if c in df.columns]
    gene_names = [c for c in df.columns if c not in meta_cols]
    meta_df = df[meta_cols].copy()
    x_sparse = csr_matrix(df[gene_names].values)
    return meta_df, x_sparse, gene_names


def drop_zero_genes_sparse(
    x: csr_matrix, gene_names: List[str]
) -> Tuple[csr_matrix, List[str], int]:
    """
    Remove genes (columns) that are identically zero.

    Parameters
    ----------
    x
        Sparse cell × gene matrix.
    gene_names
        Gene names parallel to columns of ``x``.

    Returns
    -------
    x_sub : scipy.sparse.csr_matrix
        Filtered matrix.
    gene_names_sub : list of str
        Genes retained.
    n_dropped : int
        Number of all-zero columns removed.
    """
    col_sums = np.array(x.sum(axis=0)).flatten()
    keep = col_sums > 0
    n_dropped = int((~keep).sum())
    kept = [g for g, k in zip(gene_names, keep) if k]
    return x[:, keep], kept, n_dropped


def preprocess_data(
    meta_df: pd.DataFrame,
    x: csr_matrix,
    gene_names: List[str],
    label_column: str,
    class_map: Dict[str, int],
    _metadata_cols: List[str],
) -> Tuple[pd.DataFrame, csr_matrix, List[str], int]:
    """
    Restrict to rows with labels in ``class_map`` and drop zero-variance genes.

    Parameters
    ----------
    meta_df
        Metadata including ``label_column``.
    x
        Sparse matrix aligned row-wise with ``meta_df``.
    gene_names
        Gene identifiers for columns of ``x``.
    label_column
        Column name for phenotype labels.
    class_map
        Mapping from label strings to integer class indices.
    _metadata_cols
        Reserved for API symmetry with callers that pass ``METADATA_COLS``.

    Returns
    -------
    meta_df_f : pandas.DataFrame
        Filtered metadata with added ``class_label`` column.
    x_f : scipy.sparse.csr_matrix
        Filtered matrix.
    gene_names_f : list of str
        Genes after removing all-zero columns.
    n_zero : int
        Count of removed all-zero genes.
    """
    valid_classes = list(class_map.keys())
    mask = meta_df[label_column].isin(valid_classes).values
    meta_df = meta_df.loc[mask].copy()
    meta_df["class_label"] = meta_df[label_column].replace(class_map)
    x = x[mask]
    x, gene_names, n_zero = drop_zero_genes_sparse(x, gene_names)
    return meta_df, x, gene_names, n_zero


def select_features(
    x: csr_matrix, gene_names: List[str], y: np.ndarray, top_n: int
) -> List[str]:
    """
    Rank genes by ANOVA F-test and return the top ``top_n`` names.

    Parameters
    ----------
    x
        Sparse cell × gene matrix.
    gene_names
        Gene identifiers.
    y
        Integer class labels.
    top_n
        Number of genes to retain.

    Returns
    -------
    list of str
        Selected gene names in descending score order.
    """
    _f_scores, p_vals = f_classif(x, y)
    p_vals = np.maximum(p_vals, 1e-300)
    scores = -np.log10(p_vals)
    ranked = sorted(zip(gene_names, scores), key=lambda t: t[1], reverse=True)
    return [g for g, _ in ranked[:top_n]]


class MLP(nn.Module):
    """
    Fully connected classifier with ReLU, dropout, and cross-entropy training.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        num_classes: int,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        dims = [input_size] + list(hidden_sizes) + [num_classes]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.extend([nn.ReLU(), nn.Dropout(dropout_rate)])
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def run_pipeline(
    data_path: str, output_dir: Optional[str] = None
) -> Tuple[str, str]:
    """
    Execute loading, preprocessing, feature selection, CV, and export.

    Parameters
    ----------
    data_path
        Base Parquet path (split or wide layout).
    output_dir
        Directory for result files. Default: directory of ``data_path``, or
        ``"."`` if ``data_path`` has no directory component.

    Returns
    -------
    results_csv : str
        Path to written CSV.
    results_parquet : str
        Path to written Parquet.
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(data_path)) or "."
    os.makedirs(output_dir, exist_ok=True)

    t0 = time.perf_counter()

    def log(msg: str, *args: Any) -> None:
        _LOG.info("[%8.4fs] " + msg, time.perf_counter() - t0, *args)

    log("Loading: %s", data_path)
    meta_df, x_full, gene_names = load_data(data_path)
    log(
        "Data loaded: %d cells × %d genes (sparse CSR)",
        x_full.shape[0],
        x_full.shape[1],
    )

    meta_df, x_full, gene_names, n_zero = preprocess_data(
        meta_df, x_full, gene_names, LABEL_COLUMN_NAME, CLASS_MAP, METADATA_COLS
    )
    log("Preprocessed; removed %d zero-variance genes", n_zero)

    y = meta_df["class_label"].values
    selected_features = select_features(x_full, gene_names, y, TOP_N_FEATURES)
    log("Feature selection: %d genes retained", len(selected_features))

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    sel_idx = np.array([gene_to_idx[g] for g in selected_features])
    x = x_full[:, sel_idx]

    n_samples = x.shape[0]
    all_nn_pred = np.zeros(n_samples, dtype=int)
    all_xgb_pred = np.zeros(n_samples, dtype=int)
    all_rf_pred = np.zeros(n_samples, dtype=int)
    all_nn_probs = np.zeros((n_samples, len(CLASS_MAP)))
    all_xgb_probs = np.zeros((n_samples, len(CLASS_MAP)))
    all_rf_probs = np.zeros((n_samples, len(CLASS_MAP)))

    if STRATIFY_SPLIT:
        kfold = StratifiedKFold(
            n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
        )
        splits: Iterator[Tuple[np.ndarray, np.ndarray]] = kfold.split(x, y)
    else:
        kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        splits = kfold.split(x)

    log("Starting %d-fold cross-validation", N_FOLDS)
    device = torch.device("cpu")
    num_classes = len(CLASS_MAP)

    for fold_num, (train_idx, test_idx) in enumerate(splits, start=1):
        log("Fold %d / %d", fold_num, N_FOLDS)
        x_train_sp = x[train_idx]
        x_test_sp = x[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        log("  Fold %d: training XGBoost", fold_num)
        xgb_model = xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=num_classes,
            eval_metric="logloss",
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_STATE,
        )
        xgb_model.fit(x_train_sp, y_train)
        all_xgb_pred[test_idx] = xgb_model.predict(x_test_sp)
        all_xgb_probs[test_idx] = xgb_model.predict_proba(x_test_sp)
        log(
            "  Fold %d: XGBoost hold-out accuracy = %.4f",
            fold_num,
            accuracy_score(y_test, all_xgb_pred[test_idx]),
        )

        log("  Fold %d: training random forest", fold_num)
        rf_model = RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        )
        rf_model.fit(x_train_sp, y_train)
        all_rf_pred[test_idx] = rf_model.predict(x_test_sp)
        all_rf_probs[test_idx] = rf_model.predict_proba(x_test_sp)
        log(
            "  Fold %d: random forest hold-out accuracy = %.4f",
            fold_num,
            accuracy_score(y_test, all_rf_pred[test_idx]),
        )

        log("  Fold %d: training MLP", fold_num)
        x_train_dense = x_train_sp.toarray()
        x_test_dense = x_test_sp.toarray()
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_dense)
        x_test_scaled = scaler.transform(x_test_dense)
        del x_train_dense, x_test_dense
        x_train_tensor = torch.FloatTensor(x_train_scaled)
        x_test_tensor = torch.FloatTensor(x_test_scaled)
        y_train_tensor = torch.LongTensor(y_train)
        y_test_tensor = torch.LongTensor(y_test)
        train_loader = DataLoader(
            TensorDataset(x_train_tensor, y_train_tensor),
            batch_size=MLP_BATCH_SIZE,
            shuffle=True,
        )
        model = MLP(
            len(selected_features), MLP_HIDDEN_SIZES, num_classes, MLP_DROPOUT_RATE
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=MLP_LEARNING_RATE, weight_decay=MLP_WEIGHT_DECAY
        )
        best_val_acc = 0.0
        patience_counter = 0
        best_state = copy.deepcopy(model.state_dict())

        for _epoch in range(MLP_EPOCHS):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_outputs = model(x_test_tensor.to(device))
                val_acc = (
                    (val_outputs.argmax(dim=1) == y_test_tensor.to(device))
                    .float()
                    .mean()
                    .item()
                )
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                if patience_counter >= MLP_EARLY_STOPPING_PATIENCE:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            logits = model(x_test_tensor.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
        all_nn_pred[test_idx] = preds
        all_nn_probs[test_idx] = probs
        log(
            "  Fold %d: MLP hold-out accuracy = %.4f",
            fold_num,
            accuracy_score(y_test, all_nn_pred[test_idx]),
        )

    assert all_nn_probs.shape == all_xgb_probs.shape == all_rf_probs.shape
    assert all_nn_probs.shape[1] == num_classes

    log("Cross-validation complete - overall accuracy:")
    log("  XGBoost:        %.4f", accuracy_score(y, all_xgb_pred))
    log("  Random forest:  %.4f", accuracy_score(y, all_rf_pred))
    log("  MLP:            %.4f", accuracy_score(y, all_nn_pred))

    dataset_name = extract_dataset_name(data_path)
    res = pd.DataFrame(
        {
            "cell_id": meta_df["cell_id"].values,
            "sample_id": meta_df["sample_id"].values,
            "y_true": y,
            "nn_pred": all_nn_pred,
            "xgb_pred": all_xgb_pred,
            "rf_pred": all_rf_pred,
        }
    )
    results_csv = os.path.join(output_dir, f"{dataset_name}_results.csv")
    results_parquet = os.path.join(output_dir, f"{dataset_name}_results.parquet")
    res.to_csv(results_csv, index=False)
    res.to_parquet(results_parquet, index=False)
    log("Wrote %s and %s", results_csv, results_parquet)
    return results_csv, results_parquet


def main() -> None:
    """CLI entry point: configure logging and run :func:`run_pipeline`."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    data_path = args.parquet_path if args.parquet_path else DATA_PATH
    run_pipeline(data_path, args.output_dir)


if __name__ == "__main__":
    main()
