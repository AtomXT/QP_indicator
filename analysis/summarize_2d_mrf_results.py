# analysis/summarize_results.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_results(results_dir):
    all_files = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".csv") and f.startswith("2D_MRF")
    ]

    dfs = []
    for file in all_files:
        df = pd.read_csv(file)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all

def keep_only_paired_trials(df_all):
    """
    Keep only trials for which both formulations are present
    for the same (n, sigma2, rep).
    """

    pair_cols = ["n", "sigma2", "rep"]
    expected_formulations = set(df_all["formulation"].unique())

    complete_trials = (
        df_all.groupby(pair_cols)["formulation"]
        .apply(lambda x: set(x) == expected_formulations)
        .reset_index(name="complete")
    )

    complete_trials = complete_trials[complete_trials["complete"]][pair_cols]

    df_filtered = df_all.merge(
        complete_trials,
        on=pair_cols,
        how="inner"
    )

    return df_filtered


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "..", "experiments_results")
    df_all = load_results(results_dir)

    # Keep only paired trials before averaging
    df_all = keep_only_paired_trials(df_all)

    # If it solved in one node then the root gap is zero.
    mask = df_all["node_count"] <= 1
    df_all.loc[mask, ["root_ub", "root_lb"]] = df_all.loc[mask, ["end_ub", "end_lb"]].to_numpy()
    df_all.loc[mask, "root_gap"] = 0
    avg = df_all.groupby(['n', 'formulation','sigma2'])[["root_ub","root_lb","root_gap","end_ub", "end_lb", "end_gap", "TPR", "FPR", "nnz", "node_count", "time"]].mean()
    print(avg)

    # save
    output_path = os.path.join(current_dir, "averaged_results_2D_MRF.csv")
    avg.to_csv(output_path, index=True)




if __name__ == "__main__":
    main()