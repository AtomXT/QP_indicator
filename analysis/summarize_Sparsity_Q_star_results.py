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

    # If it solved in one node then the root gap is zero.
    mask = df_all["node_count"] <= 1
    df_all.loc[mask, ["root_ub", "root_lb"]] = df_all.loc[mask, ["end_ub", "end_lb"]].to_numpy()
    df_all.loc[mask, "root_gap"] = 0

    avg = df_all.groupby(["n", "formulation"])[
        ["root_ub", "root_lb", "root_gap", "end_ub", "end_lb", "end_gap", "nnz", "node_count", "time"]
    ].mean()

    avg_time = df_all.groupby(["n", "formulation"]).apply(majority_filtered_mean_time)
    avg["time"] = avg_time

    print(avg)

    output_path = os.path.join(current_dir, "averaged_results_star_Q.csv")
    avg.to_csv(output_path, index=True)

    avg = avg.reset_index()
    core_df = avg[avg["formulation"] == "opt"].sort_values("n")
    bigm_df = avg[avg["formulation"] == "original"].sort_values("n")

    n_core = core_df["n"].to_numpy()
    t_core = core_df["time"].to_numpy()
    n_bigm = bigm_df["n"].to_numpy()
    t_bigm = bigm_df["time"].to_numpy()

    plt.figure(figsize=(6, 4))

    valid_core = np.isfinite(t_core) & (t_core > 0)
    valid_bigm = np.isfinite(t_bigm) & (t_bigm > 0)

    if np.count_nonzero(valid_core) >= 2:
        slope_core, intercept_core = np.polyfit(np.log(n_core[valid_core]), np.log(t_core[valid_core]), 1)
        fitted_core = np.exp(intercept_core) * n_core[valid_core] ** slope_core
        plt.loglog(n_core[valid_core], fitted_core, linestyle="--", label=f"Fit slope for CORe={slope_core:.2f}")

    if np.count_nonzero(valid_bigm) >= 2:
        slope_bigm, intercept_bigm = np.polyfit(np.log(n_bigm[valid_bigm]), np.log(t_bigm[valid_bigm]), 1)
        fitted_bigm = np.exp(intercept_bigm) * n_bigm[valid_bigm] ** slope_bigm
        plt.loglog(n_bigm[valid_bigm], fitted_bigm, linestyle="--", label=f"Fit slope for Big-M={slope_bigm:.2f}")

    plt.loglog(n_core[valid_core], t_core[valid_core], marker="o", linewidth=2, label="CORe")
    plt.loglog(n_bigm[valid_bigm], t_bigm[valid_bigm], marker="s", linewidth=2, label="Big-M")

    plt.xlabel("Problem size n (log scale)")
    plt.ylabel("Solution time (seconds, log scale)")
    plt.title("Log-Log Plot: CORe vs Big-M (Star Graph)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, "core_star_plot.pdf"), format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
