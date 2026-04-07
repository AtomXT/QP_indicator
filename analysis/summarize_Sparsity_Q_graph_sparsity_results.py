# analysis/summarize_results.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_results(results_dir):
    all_files = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".csv") and f.startswith("Sparsity_Q_graph_sparsity")
    ]

    dfs = []
    for file in all_files:
        df = pd.read_csv(file)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "..", "experiments_results")
    df_all = load_results(results_dir)

    # If it solved in one node then the root gap is zero.
    mask = df_all["node_count"] <= 1
    df_all.loc[mask, ["root_ub", "root_lb"]] = df_all.loc[mask, ["end_ub", "end_lb"]].to_numpy()
    df_all.loc[mask, "root_gap"] = 0
    avg = df_all.groupby(['delta', 'formulation'])[["root_ub","root_lb","root_gap","end_ub", "end_lb", "end_gap", "nnz", "node_count", "time"]].mean()
    print(avg)

    # save
    output_path = os.path.join(current_dir, "averaged_results_graph_sparsity.csv")
    avg.to_csv(output_path, index=True)




if __name__ == "__main__":
    main()