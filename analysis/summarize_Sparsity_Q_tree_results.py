# analysis/summarize_results.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_results(results_dir):
    all_files = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".csv") and f.startswith("Tree_Q")
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
    avg = df_all.groupby(['n', 'formulation'])[["root_ub","root_lb","root_gap","end_ub", "end_lb", "end_gap", "nnz", "node_count", "time"]].mean()
    avg = df_all.groupby(['n', 'formulation'])[["root_ub","root_lb","root_gap","end_ub", "end_lb", "end_gap", "nnz", "node_count", "time"]].mean()
    print(avg)

    # save
    output_path = os.path.join(current_dir, "averaged_results_tree_Q.csv")
    avg.to_csv(output_path, index=True)


    # Example: replace with your actual data
    n_values = np.array([100, 200, 300, 500, 1000, 2000, 5000, 10000])
    avg = avg.reset_index()
    core_times = avg[avg['formulation'] == 'opt'].time
    tree_times = avg[avg['formulation'] == 'tree'].time

    # Fit slope in log-log space
    log_n = np.log(n_values)
    log_t = np.log(core_times)
    log_t_parametric = np.log(tree_times)

    slope, intercept = np.polyfit(log_n, log_t, 1)
    slope1, intercept1 = np.polyfit(log_n, log_t_parametric, 1)

    print(f"Estimated slope: {slope:.3f}")

    plt.figure(figsize=(6, 4))

    fitted = np.exp(intercept) * n_values ** slope
    fitted1 = np.exp(intercept1) * n_values ** slope1
    plt.loglog(n_values, fitted, linestyle='--', label=f'Fit slope for CORe={slope:.2f}')
    plt.loglog(n_values, fitted1, linestyle='--', label=f'Fit slope for parametric={slope1:.2f}')
    plt.loglog(n_values, core_times, marker='o', linewidth=2, label='CORe')
    plt.loglog(n_values, tree_times, marker='s', linewidth=2, label='parametric')

    plt.xlabel('Problem size n (log scale)')
    plt.ylabel('Solution time (seconds, log scale)')
    plt.title('Log-Log Plot: CORe vs parametric (Tree Graph)')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)

    plt.legend()
    plt.tight_layout()
    plt.savefig("core_tree_plot.pdf", format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()