# analysis/summarize_results.py

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(results_dir):
    all_files = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".csv") and f.startswith("star_Q")
    ]

    dfs = []
    for file in all_files:
        df = pd.read_csv(file)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No star_Q result files found in experiments_results.")

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def majority_filtered_mean_time(group, timelimit=600.0, time_tol=1e-3):
    """
    Average time with the user's requested rule:

    - If most trials solve within the timelimit, average only solved trials.
    - If most trials hit the timelimit, average only timeout/unsolved trials.
    - If tied, average all trials.
    """
    valid = group.dropna(subset=["time"])
    if valid.empty:
        return np.nan

    timeout_mask = valid["time"] >= (timelimit - time_tol)
    solved = valid.loc[~timeout_mask, "time"]
    timed_out = valid.loc[timeout_mask, "time"]

    if len(solved) == 0:
        return timed_out.mean()
    if len(timed_out) == 0:
        return solved.mean()
    if len(solved) > len(timed_out):
        return solved.mean()
    if len(timed_out) > len(solved):
        return timed_out.mean()
    return valid["time"].mean()


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "..", "experiments_results")
    df_all = load_results(results_dir)
    df_all["Active (%)"] = 100 * df_all["nnz"] / df_all["n"]

    # If it solved in one node then the root gap is zero.
    mask = df_all["node_count"] <= 1
    df_all.loc[mask, ["root_ub", "root_lb"]] = df_all.loc[mask, ["end_ub", "end_lb"]].to_numpy()
    df_all.loc[mask, "root_gap"] = 0

    avg = df_all.groupby(["n", "formulation"])[
        ["root_ub", "root_lb", "root_gap", "end_ub", "end_lb", "end_gap", "Active (%)", "node_count", "time"]
    ].mean()

    avg_time = df_all.groupby(["n", "formulation"]).apply(majority_filtered_mean_time)
    avg["time"] = avg_time

    print(avg)

    output_path = os.path.join(current_dir, "averaged_results_star_Q.csv")
    avg.to_csv(output_path, index=True)

    avg = avg.reset_index()
    plt.figure(figsize=(6, 4))

    methods = [
        ("opt", "CORe", "o"),
        ("original", "Big-M", "s"),
        ("tree", "parametric", "^"),
    ]
    for formulation, label, marker in methods:
        method_df = avg[avg["formulation"] == formulation].sort_values("n")
        n_values = method_df["n"].to_numpy()
        times = method_df["time"].to_numpy()
        valid = np.isfinite(times) & (times > 0)

        if np.count_nonzero(valid) >= 2:
            slope, intercept = np.polyfit(np.log(n_values[valid]), np.log(times[valid]), 1)
            fitted = np.exp(intercept) * n_values[valid] ** slope
            plt.loglog(n_values[valid], fitted, linestyle="--", label=f"Fit slope for {label}={slope:.2f}")

        if np.count_nonzero(valid) > 0:
            plt.loglog(n_values[valid], times[valid], marker=marker, linewidth=2, label=label)

    plt.xlabel("Problem size n (log scale)")
    plt.ylabel("Solution time (seconds, log scale)")
    plt.title("Log-Log Plot: CORe vs Big-M vs parametric (Star Graph)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, "core_star_plot.pdf"), format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
