#!/usr/bin/env python3
"""
Plot validation accuracy vs version from a CSV with columns:
timestamp,version,split,num_samples,loss,acc,f1_macro,madds_M,flops_M,latency_ms,ckpt_path
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ---- Fixed constants ----
MAX_XTICKS = 12
MARK_PARTS = 20

def main(csv_path: str, out_path: str | None):
    df = pd.read_csv(csv_path)

    # Basic dtype parsing
    if "version" in df.columns:
        df["version"] = pd.to_numeric(df["version"], errors="coerce")
    if "acc" in df.columns:
        df["acc"] = pd.to_numeric(df["acc"], errors="coerce")

    # Keep only validation split
    if "split" not in df.columns:
        raise ValueError("CSV must contain a 'split' column.")
    val = df[df["split"].astype(str).str.lower() == "val"].copy()

    # Drop rows with missing version/acc
    val = val.dropna(subset=["version", "acc"])

    if val.empty:
        raise ValueError("No rows with split == 'val' found.")

    # If acc is in [0,1], scale to percentage
    if val["acc"].max() <= 1.0:
        val["acc"] = val["acc"] * 100.0

    # If multiple rows per version exist, average them
    plot_df = (
        val.groupby("version", as_index=False)
           .agg(acc_mean=("acc", "mean"))
           .sort_values("version")
    )

    if plot_df.empty:
        raise ValueError("No validation rows with valid 'version' and 'acc' found in the CSV.")
    
    x = plot_df["version"].astype(int).to_numpy()
    y = plot_df["acc_mean"].to_numpy()

    plt.figure(figsize=(10, 5))

    plt.plot(x, y, linewidth=1.5)

    # Draw line with sparse markers
    # if len(x) > MARK_PARTS:
    #     mark_every = max(1, len(x) // MARK_PARTS)
    #     plt.plot(x, y, marker="o", markevery=mark_every, linewidth=1.5)
    # else:
    #     plt.plot(x, y, marker="o", linewidth=1.5)

    # --- Plot ---
    # plt.figure(figsize=(8, 5))
    # Bar chart; you can switch to line by using plt.plot(...)
    # plt.plot(plot_df["version"].astype(int), plot_df["acc_mean"])
    plt.xlabel("version")
    plt.ylabel("accuracy (%)")
    plt.title("Validation Accuracy by Version")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # X tick skipping: show at most ~MAX_XTICKS ticks
    if len(x) > 1:
        step = max(1, len(x) // MAX_XTICKS)
        ticks = x[::step]
        plt.xticks(ticks)

    # Nice x ticks as integers
    # plt.xticks(plot_df["version"].astype(int))

    # Ensure integer tick formatting
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"[saved] {out_path}")
    else:
        plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to the CSV file.")
    ap.add_argument("--out", default=None, help="Optional path to save PNG. If omitted, shows interactively.")
    args = ap.parse_args()
    main(args.csv, args.out)
