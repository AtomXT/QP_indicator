# analysis/summarize_results.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir):
    all_files = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".csv") and f.startswith("path_Q_adversarial")
    ]

    dfs = []
    for file in all_files:
        df = pd.read_csv(file)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No path_Q_adversarial result files found in experiments_results.")

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "..", "experiments_results")
    df_all = load_results(results_dir)

    if "active_pct" in df_all.columns:
        df_all["Active (%)"] = df_all["active_pct"]
    else:
        df_all["Active (%)"] = 100 * df_all["nnz"] / df_all["n"]

    # If it solved in one node then the root gap is zero.
    mask = df_all["node_count"] <= 1
    df_all.loc[mask, ["root_ub", "root_lb"]] = df_all.loc[mask, ["end_ub", "end_lb"]].to_numpy()
    df_all.loc[mask, "root_gap"] = 0

    group_cols = ["variant", "rho", "perturbation_level", "tau", "n", "formulation"]
    metric_cols = [
        "root_ub",
        "root_lb",
        "root_gap",
        "end_ub",
        "end_lb",
        "end_gap",
        "nnz",
        "Active (%)",
        "node_count",
        "time",
        "lambda_value",
    ]

    avg = df_all.groupby(group_cols)[metric_cols].mean()

    print(avg)

    # save
    output_path = os.path.join(current_dir, "averaged_results_path_Q_adversarial.csv")
    avg.to_csv(output_path, index=True)

    avg_plot = avg.reset_index()
    plt.figure(figsize=(7, 4))

    methods = [
        ("opt", "CORe", "o"),
        ("original", "Big-M", "s"),
        ("tree", "parametric", "^"),
    ]

    for n in sorted(avg_plot["n"].unique()):
        for formulation, label, marker in methods:
            method_df = avg_plot[
                (avg_plot["n"] == n) & (avg_plot["formulation"] == formulation)
            ].sort_values("tau")
            if method_df.empty:
                continue
            plt.plot(
                method_df["tau"],
                method_df["time"],
                marker=marker,
                linewidth=2,
                label=f"{label}, n={n}",
            )

    plt.xlabel("CORe threshold tau")
    plt.ylabel("Solution time (seconds)")
    plt.title("Path Adversarial: Solution Time vs tau")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, "core_path_adversarial_tau_plot.pdf"), format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
