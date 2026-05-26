import os
import pandas as pd

BASE = "path/to/mainfolder/result/top1000_results"

files = {
    "RF": "rf_top1000.csv",
    "CatBoost": "catboost_top1000.csv",
    "LightGBM": "lightgbm_top1000.csv",
    "XGBoost": "xgb_top1000.csv"
}

dfs = []

for model, fname in files.items():
    path = os.path.join(BASE, fname)
    df = pd.read_csv(path)[["gene", "gene_index"]].copy()
    df["model"] = model
    dfs.append(df)

# concatenate
all_df = pd.concat(dfs, axis=0)

# vote count per gene
vote_df = (
    all_df.groupby("gene")
          .agg({
              "model": "nunique",
              "gene_index": "first"
          })
          .rename(columns={"model": "vote"})
          .reset_index()
)

# split by vote
df_4 = vote_df[vote_df["vote"] == 4].sort_values("gene_index")
df_3 = vote_df[vote_df["vote"] == 3].sort_values("gene_index")
df_2 = vote_df[vote_df["vote"] == 2].sort_values("gene_index")
df_1 = vote_df[vote_df["vote"] == 1].sort_values("gene_index")

# vote ≥ 2
df_ge2 = vote_df[vote_df["vote"] >= 2].sort_values("gene_index")

# print sizes
print("Genes in all 4 models:", df_4.shape[0])
print("Genes in any 3 models:", df_3.shape[0])
print("Genes in any 2 models:", df_2.shape[0])
print("Genes in only 1 model:", df_1.shape[0])
print("Genes with vote ≥ 2:", df_ge2.shape[0])
print("Total union:", vote_df.shape[0])

# save outputs
#df_4.to_csv(os.path.join(BASE, "top1000_vote4_all4.csv"), index=False)
#df_3.to_csv(os.path.join(BASE, "top1000_vote3_any3.csv"), index=False)
#df_2.to_csv(os.path.join(BASE, "top1000_vote2_any2.csv"), index=False)
#df_1.to_csv(os.path.join(BASE, "top1000_vote1_any1.csv"), index=False)
df_ge2.to_csv(os.path.join(BASE, "top1000_vote_ge2_candidates.csv"), index=False)
#vote_df.to_csv(os.path.join(BASE, "top1000_union_with_votes.csv"), index=False)
