# analysis/summarize_results.py

import os
import pandas as pd

def load_results(results_dir):
    all_files = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".csv") and f.startswith("Sparsity_Q")
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

    avg = df_all[df_all.tau == 0.1].groupby(['n','delta', 'formulation'])[["end_ub", "end_lb", "end_gap", "nnz", "node_count", "time"]].mean()
    print(avg)

    # save
    output_path = os.path.join(current_dir, "averaged_results_tau_0.1.csv")
    avg.to_csv(output_path, index=True)


if __name__ == "__main__":
    main()