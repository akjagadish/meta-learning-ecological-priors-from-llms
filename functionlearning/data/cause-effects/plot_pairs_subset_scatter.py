from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot scatter subplots for Cause-Effect pairs subset."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(
            "/Users/aj9225/Local/meta-learning-ecological-priors-from-llms/functionlearning/data/cause-effects/pairs_subset_xy_with_metadata.csv"),
        help="Path to consolidated x/y metadata CSV.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(
            "/Users/aj9225/Local/meta-learning-ecological-priors-from-llms/functionlearning/data/cause-effects/pairs_subset_scatter_grid_with_axis_labels"
        ),
        help="Output path prefix (without extension). Saves PNG and PDF.",
    )
    parser.add_argument(
        "--exclude-task-ids",
        type=int,
        nargs="*",
        default=[47, 69, 82, 83, 94, 95, 96],
        help="Task IDs to exclude from the plot.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of subplot columns.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="DPI for PNG output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    if args.exclude_task_ids:
        df = df[~df["task_id"].isin(args.exclude_task_ids)].copy()

    task_ids = sorted(df["task_id"].unique())
    n_tasks = len(task_ids)
    if n_tasks == 0:
        raise ValueError("No tasks to plot after filtering.")

    rows = math.ceil(n_tasks / args.cols)
    fig, axes = plt.subplots(
        rows,
        args.cols,
        figsize=(args.cols * 4.5, rows * 3.8),
        constrained_layout=True,
    )
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for i, task_id in enumerate(task_ids):
        ax = axes[i]
        task_df = df[df["task_id"] == task_id]
        x_name = str(task_df["x_name"].iloc[0])
        y_name = str(task_df["y_name"].iloc[0])

        ax.scatter(task_df["x"], task_df["y"], s=8, alpha=0.55, linewidths=0)
        ax.set_title(f"Task {task_id}", fontsize=9)
        ax.set_xlabel(x_name, fontsize=8)
        ax.set_ylabel(y_name, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.15)

    for j in range(n_tasks, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Cause-Effect Pairs Subset Scatter Plots (with axis labels)", fontsize=13)

    png_path = args.output_prefix.with_suffix(".png")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Plotted tasks: {n_tasks}")


if __name__ == "__main__":
    main()
