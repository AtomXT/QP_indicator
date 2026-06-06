#!/usr/bin/env python3

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/private/tmp", "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--job_name", type=str, default="accelerometer_default")
    p.add_argument("--plot_formulation", type=str, default="opt")
    p.add_argument("--results_dir", type=str, default=None)
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


def main():
    args = parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = args.results_dir or os.path.join(current_dir, "..", "experiments_results")
    prefix = os.path.join(results_dir, f"accelerometer_GHMM_{args.job_name}")

    hidden_df = pd.read_csv(f"{prefix}_hidden_signal.csv")
    outlier_df = pd.read_csv(f"{prefix}_outlier_flags.csv")

    mode_hidden = hidden_df[hidden_df["inference_mode"] == "nonrobust"]
    if mode_hidden.empty:
        raise ValueError("No nonrobust hidden signal rows found.")

    formulation = pick_formulation(mode_hidden, args.plot_formulation)
    signal = mode_hidden[mode_hidden["formulation"] == formulation].sort_values("t")

    y_df = outlier_df.sort_values("reading_index").drop_duplicates("reading_index")
    avg_df = block_averages(y_df)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(avg_df["start_reading"], avg_df["y"], label="Original Signal")
    ax.plot(signal["start_reading"], signal["hidden_state"], label="Estimated Signal")
    ax.set_title("Signal Recovery")
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal Magnitude")
    ax.legend()
    ax.grid()
    fig.tight_layout()

    output_prefix = os.path.join(current_dir, f"accelerometer_GHMM_{args.job_name}_nonrobust")
    fig.savefig(f"{output_prefix}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved nonrobust plot to {output_prefix}.pdf")


if __name__ == "__main__":
    main()
