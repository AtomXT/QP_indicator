#!/usr/bin/env python3

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/private/tmp", "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--job_name", type=str, default="accelerometer_default")
    p.add_argument("--plot_formulation", type=str, default="opt")
    p.add_argument("--results_dir", type=str, default=None)
    p.add_argument("--circle_outliers", action="store_true")
    return p.parse_args()


def pick_formulation(df, preferred_formulation):
    if preferred_formulation in set(df["formulation"]):
        return preferred_formulation
    return df["formulation"].iloc[0]


def block_averages(y_df):
    return (
        y_df.groupby("t", as_index=False)
        .agg(
            start_reading=("start_reading", "min"),
            y=("y", "mean"),
        )
        .sort_values("t")
    )


def outlier_blocks(mode_outliers, avg_df):
    outlier_t = mode_outliers[["t"]].drop_duplicates()
    return outlier_t.merge(avg_df[["t", "start_reading", "y"]], on="t", how="left").sort_values("t")


def main():
    args = parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = args.results_dir or os.path.join(current_dir, "..", "experiments_results")
    prefix = os.path.join(results_dir, f"accelerometer_GHMM_{args.job_name}")

    hidden_df = pd.read_csv(f"{prefix}_hidden_signal.csv")
    outlier_df = pd.read_csv(f"{prefix}_outlier_flags.csv")

    if hidden_df.empty:
        raise ValueError("Hidden signal CSV is empty.")

    y_df = outlier_df.sort_values("reading_index").drop_duplicates("reading_index")
    avg_df = block_averages(y_df)

    modes = ["robust", "nonrobust"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, mode in zip(axes, modes):
        mode_hidden = hidden_df[hidden_df["inference_mode"] == mode]
        if mode_hidden.empty:
            ax.set_visible(False)
            continue

        formulation = pick_formulation(mode_hidden, args.plot_formulation)
        signal = mode_hidden[mode_hidden["formulation"] == formulation].sort_values("t")

        ax.plot(avg_df["start_reading"], avg_df["y"], label="Avg_Signal")
        ax.plot(signal["start_reading"], signal["hidden_state"], label="x")

        mode_outliers = outlier_df[
            (outlier_df["inference_mode"] == mode)
            & (outlier_df["formulation"] == formulation)
            & (outlier_df["outlier_flag"] == 1)
        ]
        if args.circle_outliers and not mode_outliers.empty:
            block_outliers = outlier_blocks(mode_outliers, avg_df)
            ax.scatter(
                block_outliers["start_reading"],
                block_outliers["y"],
                facecolors="none",
                edgecolors="red",
                linewidths=1.0,
                s=35,
                label="Outliers",
            )

        ax.set_title(f"{mode.capitalize()} Signal Recovery")
        ax.set_xlabel("Time")
        ax.set_ylabel("Signal")
        ax.legend()
        ax.grid()

    fig.tight_layout()

    output_prefix = os.path.join(current_dir, f"accelerometer_GHMM_{args.job_name}_robust_nonrobust")
    fig.savefig(f"{output_prefix}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved robust/non-robust plot to {output_prefix}.pdf")


if __name__ == "__main__":
    main()
