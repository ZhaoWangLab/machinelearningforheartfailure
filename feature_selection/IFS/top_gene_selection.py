import os
import pandas as pd

os.chdir("path/to/mainfolder/result")

ALGOS = ["RF", "XGB", "ET", "CAT"]

tables = []
for algo in ALGOS:
    # load the IFS result 
    ifs = pd.read_csv(f"IFS_results/IFS_RF_{algo}_top100_step5.csv")              
    topN = ifs.loc[ifs["acc"] >= 0.8, "k_features"].iloc[0]

    # load the top 100 genes 
    top100 = pd.read_csv(f"top100_results/FI_{algo}_top100.csv")         
    genes = top100["gene"].iloc[:topN].astype(str).tolist()

    # create the table with gene and rank at ACC
    tables.append(pd.DataFrame({
        "algo": algo,
        "topN_at_ACC80": topN,
        "rank": range(1, topN + 1),
        "gene": genes
    }))

# long matrix 
long_df = pd.concat(tables, ignore_index=True)

# wide matrix
all_genes = sorted(long_df["gene"].unique())
wide_df = pd.DataFrame({"gene": all_genes})
for algo in ALGOS:
    s = set(long_df.loc[long_df["algo"] == algo, "gene"])
    wide_df[algo] = wide_df["gene"].isin(s).astype(int)

# genes x algo rank table 
rank_df = (
    long_df[["gene", "algo", "rank"]]
    .pivot(index="gene", columns="algo", values="rank")
    .reset_index()
)
rank_df["n_algos"] = rank_df[ALGOS].notna().sum(axis=1)

# save the result 
long_df.to_csv("IFS_results/genes_ACC80_long.csv", index=False)
wide_df.to_csv("IFS_results/genes_ACC80_upset.csv", index=False)
rank_df.to_csv("IFS_results/genes_ACC80_rank.csv", index=False)

